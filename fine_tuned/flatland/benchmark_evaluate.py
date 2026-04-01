#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

BENCHMARL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARL_ROOT))

from benchmarl.experiment import Experiment

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate checkpoints produced by benchmark.py / "
            "benchmark_curriculum.py with the correct task/model wiring."
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
    parser.add_argument(
        "--evaluation-runs",
        type=int,
        default=1,
        help=(
            "Number of repeated evaluation runs. "
            "A per-run CSV and an aggregate mean/std CSV will be written."
        ),
    )
    return parser.parse_args()


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "flatland_benchmark", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import benchmark from {BENCHMARK_PATH}")
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

    if "sequence" in name and hasattr(model_config, "model_configs"):
        model_layers = list(getattr(model_config, "model_configs") or [])
        if len(model_layers) >= 2:
            first = type(model_layers[0]).__name__.lower()
            second = type(model_layers[1]).__name__.lower()
            if "flatlandtreelstmfeature" in first:
                if second.startswith("gru"):
                    return "treelstm_gru"
                if second.startswith("mlp"):
                    return "treelstm_mlp"
            if first.startswith("gnn") and second.startswith("mlp"):
                return "gnn"

    if name.startswith("lstm"):
        return "lstm"
    if name.startswith("gru"):
        return "gru"
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


def _find_eval_json(run_save_folder: Path, experiment_name: str) -> Path:
    expected = run_save_folder / experiment_name / f"{experiment_name}.json"
    if expected.exists():
        return expected

    candidates = sorted(run_save_folder.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No evaluation json found under {run_save_folder}")
    return candidates[0]


def _load_latest_step_metrics(json_path: Path) -> tuple[int, dict[str, float]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    env = next(iter(data))
    task = next(iter(data[env]))
    algo = next(iter(data[env][task]))
    seed = next(iter(data[env][task][algo]))
    run_data = data[env][task][algo][seed]

    step_keys = [k for k in run_data.keys() if k.startswith("step_")]
    if not step_keys:
        raise ValueError(f"No step_* keys found in {json_path}")
    step_keys.sort(key=lambda k: int(k.split("_")[1]))
    last_step_key = step_keys[-1]
    step_payload = run_data[last_step_key]

    metrics: dict[str, float] = {}
    for key, value in step_payload.items():
        if key == "step_count":
            continue
        if isinstance(value, list):
            if not value:
                continue
            metrics[key] = float(statistics.fmean(float(v) for v in value))
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)

    return int(step_payload.get("step_count", 0)), metrics


def _write_per_run_csv(path: Path, rows: list[dict[str, Any]], metric_keys: list[str]) -> None:
    fieldnames = ["run_idx", "seed", "step_count", "json_path", *metric_keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]], metric_keys: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "n_runs", "mean", "std", "plus_minus"]
        )
        writer.writeheader()

        for metric in metric_keys:
            vals = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if not vals:
                continue
            mean_val = statistics.fmean(vals)
            std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
            writer.writerow(
                {
                    "metric": metric,
                    "n_runs": len(vals),
                    "mean": f"{mean_val:.6f}",
                    "std": f"{std_val:.6f}",
                    "plus_minus": f"{mean_val:.6f} ± {std_val:.6f}",
                }
            )


def main() -> None:
    args = _parse_args()
    if args.evaluation_runs < 1:
        raise ValueError("--evaluation-runs must be >= 1")

    checkpoint_file = args.checkpoint_file.resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    sb = _load_benchmark_module()

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
    checkpoint_experiment_name = checkpoint_file.parent.parent.name

    print("\nEvaluating with benchmark wiring\n")
    sb._print_hydra_config(
        experiment_config=experiment_config,
        algorithm_config=algorithm_config,
        task=task,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
    )

    per_run_rows: list[dict[str, Any]] = []
    metric_keys_union: set[str] = set()

    for run_idx in range(1, args.evaluation_runs + 1):
        run_seed = seed + (run_idx - 1)
        run_save_folder = eval_save_folder / f"run_{run_idx:03d}"
        run_save_folder.mkdir(parents=True, exist_ok=True)
        experiment_config.save_folder = str(run_save_folder)

        print(
            f"\n--- Evaluation run {run_idx}/{args.evaluation_runs} "
            f"(seed={run_seed}) ---"
        )
        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=run_seed,
            config=experiment_config,
        )

        try:
            experiment.evaluate()
        finally:
            experiment.close()

        json_path = _find_eval_json(run_save_folder, checkpoint_experiment_name)
        step_count, metrics = _load_latest_step_metrics(json_path)
        metric_keys_union.update(metrics.keys())
        row: dict[str, Any] = {
            "run_idx": run_idx,
            "seed": run_seed,
            "step_count": step_count,
            "json_path": str(json_path),
        }
        row.update(metrics)
        per_run_rows.append(row)

    metric_keys = sorted(metric_keys_union)
    per_run_csv = eval_save_folder / "evaluation_runs.csv"
    summary_csv = eval_save_folder / "evaluation_summary.csv"
    _write_per_run_csv(per_run_csv, per_run_rows, metric_keys)
    _write_summary_csv(summary_csv, per_run_rows, metric_keys)

    print("\nEvaluation complete.")
    print(f"Logs written under: {eval_save_folder}")
    print(f"Per-run metrics CSV: {per_run_csv}")
    print(f"Summary mean±std CSV: {summary_csv}")


if __name__ == "__main__":
    main()
