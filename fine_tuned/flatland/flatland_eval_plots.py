#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

BENCHMARL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARL_ROOT))

if importlib.util.find_spec("marl_eval") is None:
    raise ImportError(
        "marl_eval is not installed. Run this script with:\n"
        "  uv run --with id-marl-eval python benchmarl_ext/fine_tuned/flatland/flatland_eval_plots.py"
    )

from benchmarl.eval_results import Plotting, load_and_merge_json_dicts

DEFAULT_METRICS = ("return", "arrival_ratio", "deadlock_ratio")
HIGHER_IS_BETTER = {
    "return": True,
    "arrival_ratio": True,
    "deadlock_ratio": False,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate marl-eval plots for Flatland benchmark runs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("benchmarl_ext/fine_tuned/flatland/short_benchmark_runs"),
        help="Directory containing BenchMARL run folders with marl-eval JSON outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where merged JSON and plots are written.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="flatland",
        help="Environment name stored in the marl-eval JSON.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="flatland",
        help="Task name stored in the marl-eval JSON.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_METRICS),
        help="Metrics to plot. Defaults to return arrival_ratio deadlock_ratio.",
    )
    return parser.parse_args()


def _find_json_files(input_dir: Path, output_dir: Path) -> List[str]:
    json_files = []
    for path in sorted(input_dir.rglob("*.json")):
        if output_dir in path.parents:
            continue
        if "wandb" in path.parts:
            continue
        json_files.append(str(path))
    return json_files


def _discover_step_metrics(node: Dict) -> Set[str]:
    metrics: Set[str] = set()
    if not isinstance(node, dict):
        return metrics
    for key, value in node.items():
        if key.startswith("step_") and isinstance(value, dict):
            metrics.update(metric for metric in value.keys() if metric != "step_count")
        elif isinstance(value, dict):
            metrics.update(_discover_step_metrics(value))
    return metrics


def _save_current_figure(path: Path) -> None:
    plt.gcf().savefig(path, dpi=200, bbox_inches="tight")
    print(f"Wrote {path}")


def _plot_metric(
    processed_data,
    env_name: str,
    task_name: str,
    metric_name: str,
    output_dir: Path,
) -> None:
    metrics_to_normalize = Plotting.metrics_to_normalize_for(metric_name)
    env_matrix, sample_matrix = Plotting.create_matrices(
        processed_data,
        env_name=env_name,
        metrics_to_normalize=metrics_to_normalize,
    )

    metric_slug = metric_name.replace("/", "_")

    plt.close("all")
    Plotting.environemnt_sample_efficiency_curves(
        sample_matrix,
        metric_name=metric_name,
        metrics_to_normalize=metrics_to_normalize,
    )
    _save_current_figure(output_dir / f"{metric_slug}_environment_sample_efficiency_curves.png")

    plt.close("all")
    Plotting.task_sample_efficiency_curves(
        processed_data,
        env=env_name,
        task=task_name,
        metric_name=metric_name,
        metrics_to_normalize=metrics_to_normalize,
    )
    _save_current_figure(output_dir / f"{metric_slug}_task_sample_efficiency_curves.png")

    if HIGHER_IS_BETTER.get(metric_name, True):
        plt.close("all")
        Plotting.performance_profile_figure(
            env_matrix,
            metric_name=metric_name,
            metrics_to_normalize=metrics_to_normalize,
        )
        _save_current_figure(output_dir / f"{metric_slug}_performance_profile.png")

        plt.close("all")
        Plotting.aggregate_scores(
            env_matrix,
            metric_name=metric_name,
            metrics_to_normalize=metrics_to_normalize,
        )
        _save_current_figure(output_dir / f"{metric_slug}_aggregate_scores.png")
    else:
        print(
            f"Skipping performance-profile and aggregate-score plots for {metric_name} "
            "because lower values are better. Sample-efficiency curves were generated instead."
        )


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "marl_eval_flatland"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = _find_json_files(input_dir, output_dir)
    if not json_files:
        raise FileNotFoundError(f"No marl-eval JSON files found under {input_dir}")

    merged_json = output_dir / "merged.json"
    raw_data = load_and_merge_json_dicts(json_files, json_output_file=str(merged_json))
    print(f"Merged {len(json_files)} json files into {merged_json}")

    available_metrics = _discover_step_metrics(raw_data)
    print(f"Available metrics: {sorted(available_metrics)}")

    processed_data = Plotting.process_data(raw_data)
    requested_metrics = list(dict.fromkeys(args.metrics))
    for metric_name in requested_metrics:
        if metric_name not in available_metrics:
            print(f"Skipping {metric_name}: not present in merged JSON")
            continue
        _plot_metric(
            processed_data=processed_data,
            env_name=args.env_name,
            task_name=args.task_name,
            metric_name=metric_name,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
