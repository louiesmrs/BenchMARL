#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

BENCHMARL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARL_ROOT))

from benchmarl.algorithms import IppoConfig
from benchmarl.benchmark import Benchmark
from benchmarl.experiment import Experiment

DEFAULT_CURRICULUM = (
    Path(__file__).resolve().parent
    / "conf"
    / "curriculum"
    / "short_benchmark_curriculum.yaml"
)
SHORT_BENCHMARK_PATH = Path(__file__).resolve().parent / "short_benchmark.py"
DEFAULT_CHECKPOINT_INTERVAL_FRAMES = 2_000_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Flatland IPPO curriculum phases sequentially from checkpoints. "
            "Each phase continues from the previous phase checkpoint."
        )
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=DEFAULT_CURRICULUM,
        help=f"Curriculum YAML file (default: {DEFAULT_CURRICULUM}).",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint (.pt) to start/resume from. "
            "If omitted, phase 1 starts from scratch."
        ),
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        default=1,
        help="1-indexed phase number to start from (default: 1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=[
            "mlp",
            "lstm",
            "lstm_mlp",
            "gru",
            "gru_mlp",
            "gnn",
            "gnn_mlp",
            "treelstm",
            "treeltsm",
            "treelstm_gru",
            "treelstm_mlp",
            "treetransformer",
            "treegnn",
        ],
        help="Model family to run for each curriculum phase.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional label for output folder naming.",
    )
    parser.add_argument(
        "--weights-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, restores only model/loss weights from --initial-checkpoint and "
            "resets replay buffer/collector state. Use this for transfer runs "
            "(e.g. 2-agent checkpoint -> 10-agent curriculum)."
        ),
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML in {path}, got {type(data).__name__}")
    return data


def _load_short_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "flatland_short_benchmark", SHORT_BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import short_benchmark module from {SHORT_BENCHMARK_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_") or "phase"


def _checkpoint_frames(checkpoint: Path) -> int:
    match = re.search(r"checkpoint_(\d+)\.pt$", checkpoint.name)
    if not match:
        raise ValueError(
            f"Could not infer frame count from checkpoint filename: {checkpoint.name}"
        )
    return int(match.group(1))


def _latest_checkpoint_recursive(root: Path) -> Path | None:
    candidates = []
    for path in root.rglob("checkpoint_*.pt"):
        match = re.search(r"checkpoint_(\d+)\.pt$", path.name)
        if match:
            candidates.append((int(match.group(1)), path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _align_interval(desired_frames: int, frames_per_batch: int) -> int:
    if desired_frames <= 0:
        return 0
    if frames_per_batch <= 0:
        raise ValueError(f"frames_per_batch must be positive, got {frames_per_batch}")
    return int(math.ceil(desired_frames / frames_per_batch) * frames_per_batch)


def _make_run_dir(run_name: str | None) -> Path:
    root = Path(__file__).resolve().parent / "short_benchmark_runs"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = _slug(run_name) if run_name else "curriculum"
    run_dir = root / f"{timestamp}__{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_weights_only_checkpoint(
    experiment: "Experiment",
    checkpoint_file: Path,
    map_location: Any = None,
) -> None:
    loaded_dict = torch.load(str(checkpoint_file), map_location=map_location)

    loaded_groups = []
    for group in experiment.group_map.keys():
        key = f"loss_{group}"
        if key not in loaded_dict:
            raise KeyError(
                f"Weights-only restore could not find '{key}' in checkpoint {checkpoint_file}"
            )
        experiment.losses[group].load_state_dict(loaded_dict[key])
        loaded_groups.append(group)

    state = loaded_dict.get("state", {})
    if "total_frames" in state:
        experiment.total_frames = int(state["total_frames"])
    if "n_iters_performed" in state:
        experiment.n_iters_performed = int(state["n_iters_performed"])
    if "mean_return" in state:
        experiment.mean_return = state["mean_return"]

    print(
        "Weights-only restore complete: "
        f"loaded groups={loaded_groups}, "
        f"total_frames={experiment.total_frames}, "
        f"n_iters={experiment.n_iters_performed}."
    )


def main() -> None:
    args = _parse_args()
    sb = _load_short_benchmark_module()
    model_name = sb._normalize_model_name(args.model)

    curriculum_path = args.curriculum.resolve()
    if not curriculum_path.exists():
        raise FileNotFoundError(f"Curriculum YAML not found: {curriculum_path}")

    initial_checkpoint: Path | None = None
    if args.initial_checkpoint is not None:
        initial_checkpoint = args.initial_checkpoint.resolve()
        if not initial_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {initial_checkpoint}")

    cfg = _load_yaml(curriculum_path)
    phases = cfg.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("Curriculum YAML must contain a non-empty 'phases' list")

    if args.start_phase < 1 or args.start_phase > len(phases):
        raise ValueError(
            f"--start-phase must be in [1, {len(phases)}], got {args.start_phase}"
        )
    if args.start_phase > 1 and initial_checkpoint is None:
        raise ValueError("--initial-checkpoint is required when --start-phase > 1")

    run_dir = _make_run_dir(args.run_name)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "curriculum_yaml": str(curriculum_path),
        "model_name": model_name,
        "weights_only": args.weights_only,
        "initial_checkpoint": str(initial_checkpoint)
        if initial_checkpoint is not None
        else None,
        "start_phase": args.start_phase,
        "phases_total": len(phases),
        "run_dir": str(run_dir),
        "phases": [],
    }
    _write_manifest(run_dir / "manifest.yaml", manifest)

    current_checkpoint: Path | None = initial_checkpoint
    for idx in range(args.start_phase, len(phases) + 1):
        phase = phases[idx - 1]
        if not isinstance(phase, dict):
            raise ValueError(f"Phase {idx} must be a mapping")

        phase_name = str(phase.get("name") or f"phase_{idx}")
        phase_slug = _slug(phase_name)
        phase_dir = run_dir / f"phase_{idx:02d}__{phase_slug}"
        phase_dir.mkdir(parents=True, exist_ok=False)

        phase_frames = int(phase.get("total_timesteps", 0))
        if phase_frames <= 0:
            raise ValueError(
                f"Phase {idx} has invalid total_timesteps={phase.get('total_timesteps')}"
            )

        task_overrides = phase.get("task", {})
        if not isinstance(task_overrides, dict):
            raise ValueError(f"Phase {idx} field 'task' must be a mapping")

        exp_overrides = phase.get("experiment", {})
        if not isinstance(exp_overrides, dict):
            raise ValueError(f"Phase {idx} field 'experiment' must be a mapping")

        base_frames = (
            _checkpoint_frames(current_checkpoint)
            if current_checkpoint is not None
            else 0
        )
        target_frames = base_frames + phase_frames

        task = sb._build_task(model_name)
        _deep_update(task.config, task_overrides)

        model_config, critic_model_config = sb._build_model_configs(model_name)
        algorithm_config = IppoConfig.get_from_yaml()
        if hasattr(algorithm_config, "minibatch_advantage"):
            algorithm_config.minibatch_advantage = True
        if hasattr(algorithm_config, "entropy_coef"):
            algorithm_config.entropy_coef = 0.0005
        sb._apply_tree_algorithm_overrides(algorithm_config, model_name)

        experiment_config = sb._build_experiment_config(
            model_name=model_name,
            run_dir=phase_dir,
            algorithm_name="ippo",
        )
        experiment_config.restore_file = (
            str(current_checkpoint)
            if (current_checkpoint is not None and not args.weights_only)
            else None
        )
        experiment_config.max_n_frames = target_frames
        experiment_config.evaluation_only = False

        # Default checkpointing policy for curriculum runs:
        # save roughly every 2.0M frames (aligned to collector batch size), keep all.
        frames_per_batch = int(experiment_config.on_policy_collected_frames_per_batch)
        resolved_interval = _align_interval(
            DEFAULT_CHECKPOINT_INTERVAL_FRAMES, frames_per_batch
        )
        if "checkpoint_interval" not in exp_overrides:
            experiment_config.checkpoint_interval = resolved_interval
        if "keep_checkpoints_num" not in exp_overrides:
            experiment_config.keep_checkpoints_num = None

        for key, value in exp_overrides.items():
            if not hasattr(experiment_config, key):
                raise ValueError(
                    f"Phase {idx} experiment override '{key}' is not a valid ExperimentConfig field"
                )
            setattr(experiment_config, key, value)

        print("\n" + "=" * 88)
        print(f"Phase {idx}/{len(phases)}: {phase_name}")
        print(
            f"Start checkpoint: {current_checkpoint}"
            if current_checkpoint is not None
            else "Start checkpoint: <fresh start>"
        )
        if args.weights_only and current_checkpoint is not None:
            print("Restore mode: weights-only (buffer/collector reset)")
        print(f"Frames: base={base_frames}, add={phase_frames}, target={target_frames}")
        print(
            "Checkpoint interval: "
            f"{experiment_config.checkpoint_interval} frames "
            f"(requested≈{DEFAULT_CHECKPOINT_INTERVAL_FRAMES}, "
            f"collector_batch={frames_per_batch})"
        )
        print(f"Output root: {phase_dir}")

        sb._print_hydra_config(
            experiment_config=experiment_config,
            algorithm_config=algorithm_config,
            task=task,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=0,
        )

        if args.weights_only and current_checkpoint is not None:
            experiment = Experiment(
                task=task,
                algorithm_config=algorithm_config,
                model_config=model_config,
                critic_model_config=critic_model_config,
                seed=0,
                config=experiment_config,
            )
            _load_weights_only_checkpoint(
                experiment,
                current_checkpoint,
                map_location=experiment_config.restore_map_location,
            )
            experiment.run()
        else:
            benchmark = Benchmark(
                algorithm_configs=[algorithm_config],
                model_config=model_config,
                critic_model_config=critic_model_config,
                tasks=[task],
                seeds={0},
                experiment_config=experiment_config,
            )
            benchmark.run_sequential()

        next_checkpoint = _latest_checkpoint_recursive(phase_dir)
        if next_checkpoint is None:
            raise RuntimeError(f"No checkpoint found under {phase_dir}")

        phase_record = {
            "phase_index": idx,
            "phase_name": phase_name,
            "input_checkpoint": str(current_checkpoint)
            if current_checkpoint is not None
            else None,
            "output_folder": str(next_checkpoint.parent.parent),
            "output_checkpoint": str(next_checkpoint),
            "base_frames": base_frames,
            "phase_timesteps": phase_frames,
            "target_frames": target_frames,
            "model_name": model_name,
            "task_config": sb._serialize_config(task.config),
            "experiment_overrides": exp_overrides,
            "checkpoint_interval": experiment_config.checkpoint_interval,
            "keep_checkpoints_num": experiment_config.keep_checkpoints_num,
        }
        manifest["phases"].append(phase_record)
        _write_manifest(run_dir / "manifest.yaml", manifest)

        print(f"Completed phase {idx}: {next_checkpoint}")
        current_checkpoint = next_checkpoint

    print("\nCurriculum complete.")
    print(f"Run folder: {run_dir}")
    print(f"Final checkpoint: {current_checkpoint}")


if __name__ == "__main__":
    main()
