#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import importlib.util
import json
import pickle
import platform
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

BENCHMARL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARL_ROOT))


def _load_eval_results_symbols():
    eval_results_path = BENCHMARL_ROOT / "benchmarl" / "eval_results.py"
    spec = importlib.util.spec_from_file_location(
        "benchmarl_eval_results_standalone", eval_results_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load eval_results module from {eval_results_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Plotting


Plotting = _load_eval_results_symbols()

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
        default=None,
        help="Benchmark batch folder to plot. Defaults to the latest run folder under benchmark_runs.",
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
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include incomplete runs instead of excluding them by default.",
    )
    parser.add_argument(
        "--algorithm-label-source",
        choices=["folder", "json"],
        default="folder",
        help=(
            "How to name algorithms when merging JSONs. "
            "'folder' uses the top-level subfolder under --input-dir (e.g. ippo, ippo2), "
            "which avoids overwriting when multiple runs use the same json algorithm key."
        ),
    )
    parser.add_argument(
        "--normalize-step-keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize per-run step_* keys to step_0..step_N based on numeric order. "
            "This is useful for curriculum/resume runs where one phase may emit step_7..step_13."
        ),
    )
    parser.add_argument(
        "--rebase-resumed-step-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When normalizing step keys, also rebase step_count to a phase-local axis for "
            "resumed runs (detected when original step keys start at step_K with K>0). "
            "This keeps the x-axis aligned with plotted points for phase-only folders."
        ),
    )
    parser.add_argument(
        "--truncate-to-shortest",
        action="store_true",
        help=(
            "Include all discovered runs and truncate each run to the shortest common "
            "number of step_* entries before plotting. Useful for curriculum runs where "
            "phase 1 has one extra initial eval point."
        ),
    )
    parser.add_argument(
        "--plot-training-efficiency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate an additional training-efficiency graph based on "
            "seconds_per_million_frames = (collection+training wall seconds) / (frames/1e6)."
        ),
    )
    parser.add_argument(
        "--efficiency-baseline-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with baseline_seconds_per_million_frames. Supported columns: "
            "machine_id, task_signature, baseline_seconds_per_million_frames. "
            "Use '*' in machine_id/task_signature as wildcard."
        ),
    )
    parser.add_argument(
        "--efficiency-global-baseline",
        type=float,
        default=None,
        help=(
            "Optional global baseline seconds_per_million_frames used when no machine/task baseline is found."
        ),
    )
    parser.add_argument(
        "--machine-id",
        type=str,
        default=None,
        help=(
            "Optional machine id override for training-efficiency normalization. "
            "Defaults to host parsed from event files, else platform.node()."
        ),
    )
    return parser.parse_args()


def _find_run_dirs(root_dir: Path) -> List[Path]:
    return sorted(
        [path for path in root_dir.iterdir() if path.is_dir() and (path / "manifest.yaml").exists()],
        key=lambda path: path.name,
    )



def _resolve_input_dir(input_dir: Path | None) -> Path:
    if input_dir is not None:
        resolved = input_dir.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input directory does not exist: {resolved}")
        if (resolved / "manifest.yaml").exists():
            return resolved
        run_dirs = _find_run_dirs(resolved)
        if run_dirs:
            latest = run_dirs[-1]
            print(f"Using latest benchmark run folder inside {resolved}: {latest}")
            return latest
        return resolved

    root_dir = Path("benchmarl_ext/fine_tuned/flatland/benchmark_runs").resolve()
    run_dirs = _find_run_dirs(root_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No benchmark run folders with manifest.yaml found under {root_dir}")
    latest = run_dirs[-1]
    print(f"Using latest benchmark run folder: {latest}")
    return latest



def _find_json_files(input_dir: Path, output_dir: Path) -> List[Path]:
    json_files = []
    for path in sorted(input_dir.rglob("*.json")):
        if output_dir in path.parents:
            continue
        if "wandb" in path.parts:
            continue
        if path.name == "merged.json":
            continue
        json_files.append(path)
    return json_files



def _extract_step_keys(run_data: Dict) -> Tuple[str, ...]:
    return tuple(
        key
        for key, _ in sorted(
            (
                (key, value)
                for key, value in run_data.items()
                if key.startswith("step_") and isinstance(value, dict)
            ),
            key=lambda item: _step_sort_key(item[0]),
        )
    )


def _step_sort_key(step_key: str) -> tuple[int, str]:
    match = re.match(r"^step_(\d+)$", step_key)
    if match:
        return (int(match.group(1)), step_key)
    return (10**12, step_key)



def _get_single_run_payload(data: Dict) -> Tuple[str, Tuple[str, ...]]:
    env = next(iter(data))
    task = next(iter(data[env]))
    algo = next(iter(data[env][task]))
    seed = next(iter(data[env][task][algo]))
    run_data = data[env][task][algo][seed]
    run_id = f"{algo}/{seed}"
    return run_id, _extract_step_keys(run_data)



def _filter_incomplete_json_files(json_files: List[Path]) -> List[Path]:
    if not json_files:
        return []

    step_sets: Dict[Path, Tuple[str, ...]] = {}
    normalized_signatures: Dict[Path, Tuple[str, ...]] = {}
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _, step_keys = _get_single_run_payload(data)
        step_sets[path] = step_keys
        normalized_signatures[path] = tuple(
            f"step_{idx}" for idx, _ in enumerate(step_keys)
        )

    reference_steps = max(step_sets.values(), key=lambda steps: (len(steps), steps))
    reference_signature = tuple(f"step_{idx}" for idx, _ in enumerate(reference_steps))
    filtered: List[Path] = []
    excluded = []
    for path, step_keys in step_sets.items():
        if normalized_signatures[path] == reference_signature:
            filtered.append(path)
        else:
            observed_first = step_keys[0] if step_keys else "<none>"
            observed_last = step_keys[-1] if step_keys else "<none>"
            expected_first = reference_steps[0] if reference_steps else "<none>"
            expected_last = reference_steps[-1] if reference_steps else "<none>"
            excluded.append(
                (
                    path,
                    len(step_keys),
                    len(reference_steps),
                    observed_first,
                    observed_last,
                    expected_first,
                    expected_last,
                )
            )

    if excluded:
        print("Excluding incomplete runs:")
        for (
            path,
            observed,
            expected,
            observed_first,
            observed_last,
            expected_first,
            expected_last,
        ) in excluded:
            print(
                f"  - {path} (steps={observed}, expected={expected}, "
                f"observed_range={observed_first}..{observed_last}, "
                f"expected_range={expected_first}..{expected_last})"
            )

    return filtered


def _algo_alias_from_path(path: Path, input_dir: Path) -> str | None:
    rel = path.resolve().relative_to(input_dir.resolve())
    if not rel.parts:
        return None
    alias = rel.parts[0]
    return alias if alias and alias not in {".", ".."} else None


def _rename_single_algo_payload(data: Dict, algo_alias: str) -> Dict:
    env = next(iter(data))
    task = next(iter(data[env]))
    algo_dict = data[env][task]
    if not isinstance(algo_dict, dict) or len(algo_dict) != 1:
        return data

    old_algo = next(iter(algo_dict))
    if old_algo == algo_alias:
        return data

    payload = copy.deepcopy(data)
    payload[env][task][algo_alias] = payload[env][task].pop(old_algo)
    return payload


def _step_index_or_none(step_key: str) -> int | None:
    match = re.match(r"^step_(\d+)$", step_key)
    if not match:
        return None
    return int(match.group(1))


def _normalize_step_keys_payload(
    data: Dict,
    *,
    rebase_resumed_step_counts: bool,
) -> tuple[Dict, int]:
    payload = copy.deepcopy(data)
    rebased_runs = 0

    for env_data in payload.values():
        if not isinstance(env_data, dict):
            continue
        for task_data in env_data.values():
            if not isinstance(task_data, dict):
                continue
            for algo_data in task_data.values():
                if not isinstance(algo_data, dict):
                    continue
                for seed, run_data in list(algo_data.items()):
                    if not isinstance(run_data, dict):
                        continue

                    step_items = [
                        (k, v)
                        for k, v in run_data.items()
                        if k.startswith("step_") and isinstance(v, dict)
                    ]
                    if not step_items:
                        continue

                    step_items.sort(key=lambda item: _step_sort_key(item[0]))
                    first_step_idx = _step_index_or_none(step_items[0][0])
                    rebase_offset: float | int = 0
                    should_rebase = bool(
                        rebase_resumed_step_counts
                        and first_step_idx is not None
                        and first_step_idx > 0
                    )
                    if should_rebase:
                        first_step_count = step_items[0][1].get("step_count")
                        if isinstance(first_step_count, (int, float)):
                            rebase_offset = first_step_count
                        else:
                            should_rebase = False

                    non_step_items = [
                        (k, v)
                        for k, v in run_data.items()
                        if not (k.startswith("step_") and isinstance(v, dict))
                    ]

                    normalized_run: Dict = {}
                    for k, v in non_step_items:
                        normalized_run[k] = v
                    for idx, (_, step_payload) in enumerate(step_items):
                        normalized_step_payload = copy.deepcopy(step_payload)
                        if should_rebase:
                            step_count = normalized_step_payload.get("step_count")
                            if isinstance(step_count, (int, float)):
                                rebased = step_count - rebase_offset
                                if isinstance(step_count, int):
                                    rebased = int(round(rebased))
                                normalized_step_payload["step_count"] = rebased

                        normalized_run[f"step_{idx}"] = normalized_step_payload

                    algo_data[seed] = normalized_run
                    if should_rebase:
                        rebased_runs += 1

    return payload, rebased_runs


def _iter_run_dicts(payload: Dict) -> Iterable[Dict]:
    for env_data in payload.values():
        if not isinstance(env_data, dict):
            continue
        for task_data in env_data.values():
            if not isinstance(task_data, dict):
                continue
            for algo_data in task_data.values():
                if not isinstance(algo_data, dict):
                    continue
                for run_data in algo_data.values():
                    if isinstance(run_data, dict):
                        yield run_data


def _step_count_in_payload(payload: Dict) -> int:
    counts = []
    for run_data in _iter_run_dicts(payload):
        step_count = len(
            [
                k
                for k, v in run_data.items()
                if k.startswith("step_") and isinstance(v, dict)
            ]
        )
        if step_count:
            counts.append(step_count)
    return min(counts) if counts else 0


def _truncate_step_keys_payload(payload: Dict, keep_steps: int) -> Dict:
    truncated = copy.deepcopy(payload)
    if keep_steps <= 0:
        return truncated

    for run_data in _iter_run_dicts(truncated):
        step_items = [
            (k, v)
            for k, v in run_data.items()
            if k.startswith("step_") and isinstance(v, dict)
        ]
        if not step_items:
            continue

        step_items.sort(key=lambda item: _step_sort_key(item[0]))
        non_step_items = [
            (k, v)
            for k, v in run_data.items()
            if not (k.startswith("step_") and isinstance(v, dict))
        ]

        run_data.clear()
        for k, v in non_step_items:
            run_data[k] = v
        for idx, (_, step_payload) in enumerate(step_items[:keep_steps]):
            run_data[f"step_{idx}"] = step_payload

    return truncated


def _merge_json_dicts(
    json_files: List[Path],
    *,
    input_dir: Path,
    algorithm_label_source: str,
    normalize_step_keys: bool,
    rebase_resumed_step_counts: bool,
    truncate_to_shortest: bool,
    json_output_file: Path,
) -> Dict:
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    prepared_payloads: List[Dict] = []
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if algorithm_label_source == "folder":
            alias = _algo_alias_from_path(path, input_dir)
            if alias is not None:
                data = _rename_single_algo_payload(data, alias)

        if normalize_step_keys:
            data, rebased_runs = _normalize_step_keys_payload(
                data,
                rebase_resumed_step_counts=rebase_resumed_step_counts,
            )
            if rebased_runs > 0:
                print(
                    f"Rebased step_count for {rebased_runs} resumed run(s) in {path}"
                )

        prepared_payloads.append(data)

    if truncate_to_shortest and prepared_payloads:
        min_steps = min(_step_count_in_payload(payload) for payload in prepared_payloads)
        if min_steps <= 0:
            raise ValueError(
                "--truncate-to-shortest was requested but no step_* entries were found."
            )
        print(f"Truncating all runs to shortest common step count: {min_steps}")
        prepared_payloads = [
            _truncate_step_keys_payload(payload, min_steps)
            for payload in prepared_payloads
        ]

    merged: Dict = {}
    for data in prepared_payloads:
        update(merged, data)

    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4)

    return merged


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


def _safe_read_scalar_values(path: Path) -> list[float]:
    if not path.exists():
        return []
    values: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                values.append(float(row[1]))
            except ValueError:
                continue
    return values


def _infer_machine_id(run_folder: Path, machine_override: str | None) -> str:
    if machine_override:
        return machine_override

    event_files = sorted(run_folder.glob("events.out.tfevents.*"))
    event_re = re.compile(r"^events\.out\.tfevents\.\d+\.(.+)\.\d+\.\d+$")
    for event_file in event_files:
        match = event_re.match(event_file.name)
        if match:
            return match.group(1)
    return platform.node() or "unknown_machine"


def _load_task_and_model_signature(config_pkl: Path) -> tuple[str, str]:
    if not config_pkl.exists():
        return "unknown_task", "unknown_model"

    try:
        with open(config_pkl, "rb") as f:
            _task = pickle.load(f)
            task_config = pickle.load(f)
            _algorithm_config = pickle.load(f)
            model_config = pickle.load(f)
    except Exception:
        return "unknown_task", "unknown_model"

    model_name = type(model_config).__name__.lower()

    if not isinstance(task_config, dict):
        return "unknown_task", model_name

    task_signature_payload = {
        "map_width": task_config.get("map_width"),
        "map_height": task_config.get("map_height"),
        "num_agents": task_config.get("num_agents"),
        "num_steps": task_config.get("num_steps"),
        "reward_coefs": task_config.get("reward_coefs"),
        "env_pickle": bool(task_config.get("env_pickle")),
    }
    task_signature = json.dumps(task_signature_payload, sort_keys=True, separators=(",", ":"))
    return task_signature, model_name


def _label_for_json(path: Path, input_dir: Path, algorithm_label_source: str) -> str:
    if algorithm_label_source == "folder":
        alias = _algo_alias_from_path(path, input_dir)
        if alias:
            return alias.upper()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    env = next(iter(data))
    task = next(iter(data[env]))
    algo = next(iter(data[env][task]))
    return str(algo).upper()


def _collect_training_efficiency_records(
    json_files: List[Path],
    *,
    input_dir: Path,
    algorithm_label_source: str,
    machine_id_override: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_path in json_files:
        experiment_folder = json_path.parent
        run_folder = experiment_folder / json_path.stem
        scalars_folder = run_folder / "scalars"

        collection_vals = _safe_read_scalar_values(
            scalars_folder / "timers_collection_time.csv"
        )
        training_vals = _safe_read_scalar_values(
            scalars_folder / "timers_training_time.csv"
        )

        if not collection_vals and not training_vals:
            continue

        wall_seconds = float(sum(collection_vals) + sum(training_vals))

        total_frames_vals = _safe_read_scalar_values(
            scalars_folder / "counters_total_frames.csv"
        )
        if total_frames_vals:
            total_frames = float(total_frames_vals[-1])
        else:
            current_frames_vals = _safe_read_scalar_values(
                scalars_folder / "counters_current_frames.csv"
            )
            total_frames = float(sum(current_frames_vals)) if current_frames_vals else 0.0

        if total_frames <= 0:
            continue

        seconds_per_million_frames = wall_seconds / (total_frames / 1_000_000.0)
        task_signature, model_name = _load_task_and_model_signature(
            experiment_folder / "config.pkl"
        )
        machine_id = _infer_machine_id(run_folder, machine_id_override)
        label = _label_for_json(json_path, input_dir, algorithm_label_source)

        records.append(
            {
                "label": label,
                "json_path": str(json_path),
                "machine_id": machine_id,
                "task_signature": task_signature,
                "model_name": model_name,
                "wall_clock_seconds": wall_seconds,
                "total_frames": total_frames,
                "seconds_per_million_frames": seconds_per_million_frames,
            }
        )

    return records


def _load_efficiency_baseline_map(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Efficiency baseline CSV not found: {path}")

    mapping: dict[tuple[str, str], float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            machine = (row.get("machine_id") or "*").strip() or "*"
            task_sig = (row.get("task_signature") or "*").strip() or "*"
            raw = (
                row.get("baseline_seconds_per_million_frames")
                or row.get("baseline_spm")
            )
            if raw is None:
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if val > 0:
                mapping[(machine, task_sig)] = val
    return mapping


def _resolve_baseline_for_record(
    record: dict[str, Any],
    *,
    explicit_map: dict[tuple[str, str], float],
    auto_map: dict[tuple[str, str], float],
    global_baseline: float | None,
) -> float | None:
    machine = record["machine_id"]
    task_sig = record["task_signature"]

    for key in [(machine, task_sig), (machine, "*"), ("*", task_sig), ("*", "*")]:
        if key in explicit_map:
            return explicit_map[key]

    if (machine, task_sig) in auto_map:
        return auto_map[(machine, task_sig)]

    if global_baseline is not None and global_baseline > 0:
        return global_baseline
    return None


def _plot_training_efficiency(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    explicit_baseline_map: dict[tuple[str, str], float],
    global_baseline: float | None,
) -> None:
    if not records:
        print("Skipping training efficiency plot: no timer/frame data found.")
        return

    auto_baseline_map: dict[tuple[str, str], float] = {}
    grouped_mlp: dict[tuple[str, str], list[float]] = {}
    for record in records:
        if "mlp" not in str(record.get("model_name", "")).lower():
            continue
        key = (record["machine_id"], record["task_signature"])
        grouped_mlp.setdefault(key, []).append(record["seconds_per_million_frames"])
    for key, vals in grouped_mlp.items():
        if vals:
            auto_baseline_map[key] = float(statistics.median(vals))

    for record in records:
        baseline = _resolve_baseline_for_record(
            record,
            explicit_map=explicit_baseline_map,
            auto_map=auto_baseline_map,
            global_baseline=global_baseline,
        )
        record["baseline_seconds_per_million_frames"] = baseline
        record["normalized_seconds_per_million_frames"] = (
            record["seconds_per_million_frames"] / baseline
            if baseline is not None and baseline > 0
            else None
        )

    records_sorted = sorted(records, key=lambda r: r["label"])

    csv_path = output_dir / "training_efficiency.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "machine_id",
                "task_signature",
                "model_name",
                "total_frames",
                "wall_clock_seconds",
                "seconds_per_million_frames",
                "baseline_seconds_per_million_frames",
                "normalized_seconds_per_million_frames",
                "json_path",
            ],
        )
        writer.writeheader()
        for record in records_sorted:
            writer.writerow(record)
    print(f"Wrote {csv_path}")

    labels = [record["label"] for record in records_sorted]
    spm_vals = [record["seconds_per_million_frames"] for record in records_sorted]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(labels)), 4.8))
    ax.bar(labels, spm_vals)
    ax.set_ylabel("seconds per million frames (lower is better)")
    ax.set_title("Training efficiency")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_current_figure(output_dir / "training_efficiency_seconds_per_million_frames.png")

    normalized_pairs = [
        (record["label"], record["normalized_seconds_per_million_frames"])
        for record in records_sorted
        if record["normalized_seconds_per_million_frames"] is not None
    ]
    if normalized_pairs:
        n_labels = [x[0] for x in normalized_pairs]
        n_vals = [x[1] for x in normalized_pairs]
        plt.close("all")
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(n_labels)), 4.8))
        ax.bar(n_labels, n_vals)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_ylabel("normalized seconds per million frames")
        ax.set_title("Training efficiency normalized by machine/task baseline")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        _save_current_figure(
            output_dir / "training_efficiency_normalized_seconds_per_million_frames.png"
        )
    else:
        print(
            "Skipping normalized training-efficiency plot: "
            "no matching baseline (explicit, auto MLP median, or global) was found."
        )


def _plot_metric(
    processed_data,
    env_name: str,
    task_name: str,
    metric_name: str,
    output_dir: Path,
) -> None:
    metrics_to_normalize = Plotting.metrics_to_normalize_for(metric_name)
    metric_processed_data = copy.deepcopy(processed_data)
    env_matrix, sample_matrix = Plotting.create_matrices(
        metric_processed_data,
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
        metric_processed_data,
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
    if importlib.util.find_spec("marl_eval") is None:
        raise ImportError(
            "marl_eval is not installed. Run this script with:\n"
            "  uv run --with id-marl-eval --with 'numpy<2' "
            "python fine_tuned/flatland/flatland_eval_plots.py"
        )

    input_dir = _resolve_input_dir(args.input_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "marl_eval_flatland"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered_json_files = _find_json_files(input_dir, output_dir)
    if not discovered_json_files:
        raise FileNotFoundError(f"No marl-eval JSON files found under {input_dir}")

    if args.truncate_to_shortest:
        json_files = discovered_json_files
        print(
            "Including all discovered runs and truncating to shortest common step count "
            "(--truncate-to-shortest)."
        )
    else:
        json_files = (
            discovered_json_files
            if args.include_incomplete
            else _filter_incomplete_json_files(discovered_json_files)
        )
    if not json_files:
        raise FileNotFoundError(
            f"No complete marl-eval JSON files remained after filtering under {input_dir}"
        )

    merged_json = output_dir / "merged.json"
    raw_data = _merge_json_dicts(
        json_files,
        input_dir=input_dir,
        algorithm_label_source=args.algorithm_label_source,
        normalize_step_keys=args.normalize_step_keys,
        rebase_resumed_step_counts=args.rebase_resumed_step_counts,
        truncate_to_shortest=args.truncate_to_shortest,
        json_output_file=merged_json,
    )
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

    if args.plot_training_efficiency:
        efficiency_records = _collect_training_efficiency_records(
            json_files,
            input_dir=input_dir,
            algorithm_label_source=args.algorithm_label_source,
            machine_id_override=args.machine_id,
        )
        baseline_map = _load_efficiency_baseline_map(args.efficiency_baseline_csv)
        _plot_training_efficiency(
            efficiency_records,
            output_dir=output_dir,
            explicit_baseline_map=baseline_map,
            global_baseline=args.efficiency_global_baseline,
        )


if __name__ == "__main__":
    main()
