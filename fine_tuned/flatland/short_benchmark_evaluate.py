#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

BENCHMARL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARL_ROOT))

from benchmarl.experiment import Experiment

SHORT_BENCHMARK_PATH = Path(__file__).resolve().parent / "short_benchmark.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate checkpoints produced by short_benchmark.py / "
            "short_benchmark_curriculum.py with the correct task/model wiring."
        )
    )
    parser.add_argument("checkpoint_file", type=Path, help="Checkpoint .pt file")
    parser.add_argument(
        "--env-pickle",
        type=Path,
        default=None,
        help="Optional Flatland .pkl environment to evaluate on.",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=None,
        help="Override evaluation episodes.",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override experiment.render during evaluation.",
    )
    parser.add_argument(
        "--save-folder",
        type=Path,
        default=None,
        help=(
            "Optional folder root for evaluation logs. "
            "Defaults to <experiment_folder>/eval_<timestamp>."
        ),
    )
    return parser.parse_args()


def _load_short_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "flatland_short_benchmark", SHORT_BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import short_benchmark from {SHORT_BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_checkpoint_metadata(checkpoint_file: Path):
    experiment_folder = checkpoint_file.parent.parent.resolve()
    config_pkl = experiment_folder / "config.pkl"
    if not config_pkl.exists():
        raise FileNotFoundError(f"config.pkl not found near checkpoint: {config_pkl}")

    with open(config_pkl, "rb") as f:
        _task = pickle.load(f)
        task_config = pickle.load(f)
        algorithm_config = pickle.load(f)
        model_config = pickle.load(f)
        seed = pickle.load(f)
        experiment_config = pickle.load(f)
        critic_model_config = pickle.load(f)
        _callbacks = pickle.load(f)

    if not isinstance(task_config, dict):
        raise ValueError("Unexpected task_config type in config.pkl")

    return (
        experiment_folder,
        task_config,
        algorithm_config,
        model_config,
        seed,
        experiment_config,
        critic_model_config,
    )


def _infer_model_name(model_config: Any) -> str:
    name = type(model_config).__name__.lower()
    if "treelstm" in name:
        return "treelstm"
    if "treetransformer" in name or "tree_transformer" in name:
        return "treetransformer"
    if "treegnn" in name:
        return "treegnn"
    if name.startswith("lstm"):
        return "lstm_mlp"
    if name.startswith("gru"):
        return "gru_mlp"
    if "mlp" in name:
        return "mlp"
    raise ValueError(f"Could not infer model family from {type(model_config).__name__}")


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_eval_save_folder(
    checkpoint_file: Path,
    save_folder_arg: Path | None,
) -> Path:
    if save_folder_arg is not None:
        root = save_folder_arg.resolve()
    else:
        experiment_folder = checkpoint_file.parent.parent.resolve()
        root = experiment_folder / f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _checkpoint_frame_count(checkpoint_file: Path) -> int:
    m = re.search(r"checkpoint_(\d+)\.pt$", checkpoint_file.name)
    if not m:
        return 0
    return int(m.group(1))


def main() -> None:
    args = _parse_args()
    checkpoint_file = args.checkpoint_file.resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    sb = _load_short_benchmark_module()

    (
        _experiment_folder,
        task_config,
        algorithm_config,
        model_config,
        seed,
        experiment_config,
        critic_model_config,
    ) = _load_checkpoint_metadata(checkpoint_file)

    model_name = _infer_model_name(model_config)
    task = sb._build_task(model_name)
    _deep_update(task.config, task_config)

    if args.env_pickle is not None:
        env_pickle = args.env_pickle.resolve()
        if not env_pickle.exists():
            raise FileNotFoundError(f"env-pickle not found: {env_pickle}")

        from flatland.envs.persistence import RailEnvPersister

        env, _ = RailEnvPersister.load_new(str(env_pickle))
        task.config["env_pickle"] = str(env_pickle)
        task.config["num_agents"] = int(env.get_num_agents())

    experiment_config.restore_file = str(checkpoint_file)
    experiment_config.evaluation_only = True
    experiment_config.max_n_frames = _checkpoint_frame_count(checkpoint_file)
    if args.evaluation_episodes is not None:
        experiment_config.evaluation_episodes = args.evaluation_episodes
    if args.render is not None:
        experiment_config.render = args.render

    eval_save_folder = _resolve_eval_save_folder(checkpoint_file, args.save_folder)
    experiment_config.save_folder = str(eval_save_folder)

    print("\nEvaluating with short_benchmark wiring\n")
    sb._print_hydra_config(
        experiment_config=experiment_config,
        algorithm_config=algorithm_config,
        task=task,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
    )

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
    )

    try:
        experiment.evaluate()
    finally:
        experiment.close()

    print("\nEvaluation complete.")
    print(f"Logs written under: {eval_save_folder}")


if __name__ == "__main__":
    main()
