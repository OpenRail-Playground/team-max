"""Parallel grid-search runner.

Runs a :class:`~popupsim.optimizer.model.SearchSpace` in parallel using a
``ProcessPoolExecutor``, writes results incrementally to a CSV file, and
returns a clean ``pandas.DataFrame`` for analysis.

Usage::

    from pathlib import Path
    from popupsim.optimizer.model import SearchSpace
    from popupsim.optimizer.runner import load_tracks_from_scenario, run_parallel

    scenario_dir = Path("Data/examples/ten_trains_two_days_priority_dispatch")
    tracks = load_tracks_from_scenario(scenario_dir)
    space  = SearchSpace.coarse(tracks)

    df = run_parallel(
        space=space,
        scenario_dir=scenario_dir,
        output_csv=Path("results/coarse_run.csv"),
        max_workers=8,      # defaults to os.cpu_count()
    )
    print(df.sort_values("score", ascending=False).head(10))
"""

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from popupsim.optimizer.model import Track
from popupsim.optimizer.model import TrackType


# ---------------------------------------------------------------------------
# Track loader
# ---------------------------------------------------------------------------


# Map known raw type strings from tracks.json to TrackType enum values.
# Handles the typo "rescource_parking" present in existing scenario files.
_RAW_TYPE_MAP: dict[str, TrackType] = {
    t.value: t for t in TrackType
}
_RAW_TYPE_MAP["rescource_parking"] = TrackType.RESOURCE_PARKING  # typo in data


def load_tracks_from_scenario(scenario_dir: Path) -> list[Track]:
    """Read ``tracks.json`` referenced by ``scenario.json`` and return a list of :class:`Track`.

    Resolves the ``references.tracks`` path from ``scenario.json``, loads the
    referenced file, and converts each entry to a :class:`~popupsim.optimizer.model.Track`.
    Unknown track types are silently skipped.

    Args:
        scenario_dir: Directory containing ``scenario.json``.

    Returns:
        List of :class:`Track` objects matching the scenario's track configuration.

    Raises:
        FileNotFoundError: If ``scenario.json`` or the referenced tracks file is missing.
        KeyError: If ``scenario.json`` has no ``references.tracks`` entry.
    """
    scenario_json = scenario_dir / "scenario.json"
    with open(scenario_json, encoding="utf-8") as f:
        scenario_data: dict[str, Any] = json.load(f)

    tracks_filename: str = scenario_data["references"]["tracks"]
    tracks_json = scenario_dir / tracks_filename
    with open(tracks_json, encoding="utf-8") as f:
        tracks_data: dict[str, Any] = json.load(f)

    tracks: list[Track] = []
    for entry in tracks_data["tracks"]:
        raw_type: str = entry.get("type", "")
        track_type = _RAW_TYPE_MAP.get(raw_type)
        if track_type is not None:
            tracks.append(Track(type=track_type))

    return tracks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_parallel(
    space: Any,
    scenario_dir: Path,
    output_csv: Path,
    max_workers: int | None = None,
    w_wagons: float = 0.9,
    w_loco: float = 0.1,
    backend_src: str = "popupsim/backend/src",
    chunk_size: int = 1,
) -> pd.DataFrame:
    """Evaluate every configuration in *space* in parallel, write to CSV.

    Args:
        space: Any iterable of :class:`~popupsim.optimizer.model.Configuration`
               objects (e.g. ``SearchSpace.coarse(tracks)``).
        scenario_dir: Base scenario directory passed to the harness.
        output_csv: Path where results are written (appended row-by-row so
                    progress survives interruption).  Parent directories are
                    created automatically.
        max_workers: Number of worker processes.  Defaults to ``os.cpu_count()``.
        w_wagons: Weight for wagon completion in cost function (default 0.9).
        w_loco: Weight for locomotive utilisation penalty (default 0.1).
        backend_src: Relative path to the backend ``src/`` directory.  Added to
                     ``sys.path`` in each worker process.
        chunk_size: How many configs to submit per ``Future``.  1 gives the
                    finest-grained progress reporting; larger values reduce
                    inter-process overhead.

    Returns:
        ``pd.DataFrame`` with one row per configuration, sorted by *score*
        descending.  All task-rule parameters are flattened into columns so
        the result is immediately filterable.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    configs = list(space)  # materialise once (model objects are tiny)
    total = len(configs)

    workers = max_workers or os.cpu_count() or 1
    print(f"Running {total:,} configurations on {workers} workers → {output_csv}")

    _write_header = not output_csv.exists() or output_csv.stat().st_size == 0

    with open(output_csv, "a", newline="", encoding="utf-8") as fh:
        writer: csv.DictWriter | None = None
        completed = 0
        errors = 0

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _worker,
                    cfg.to_scenario_overrides(),
                    str(scenario_dir),
                    backend_src,
                    w_wagons,
                    w_loco,
                    _config_to_flat_dict(cfg),
                ): cfg
                for cfg in configs
            }

            for future in as_completed(futures):
                completed += 1
                row = future.result()

                if row.get("error"):
                    errors += 1

                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    if _write_header:
                        writer.writeheader()

                writer.writerow(row)
                fh.flush()

                if completed % max(1, total // 100) == 0 or completed == total:
                    pct = completed / total * 100
                    print(
                        f"  {completed:>7,} / {total:,}  ({pct:.1f}%)  errors: {errors}",
                        end="\r",
                        flush=True,
                    )

    print(f"\nDone. {completed:,} evaluated, {errors} errors → {output_csv}")
    return load_results(output_csv)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load a results CSV written by :func:`run_parallel` into a sorted DataFrame.

    Args:
        csv_path: Path to the CSV file produced by :func:`run_parallel`.

    Returns:
        DataFrame sorted by ``score`` descending, index reset.
    """
    df = pd.read_csv(csv_path)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Worker (runs in a subprocess)
# ---------------------------------------------------------------------------


def _worker(
    overrides: dict[str, Any],
    scenario_dir_str: str,
    backend_src: str,
    w_wagons: float,
    w_loco: float,
    flat_params: dict[str, Any],
) -> dict[str, Any]:
    """Simulate one configuration and return a flat result dict.

    Runs in a subprocess — must be importable at module level (no lambdas).
    """
    # Ensure backend is importable in the worker process
    abs_backend = str(Path(scenario_dir_str).parent / backend_src)
    if abs_backend not in sys.path:
        sys.path.insert(0, abs_backend)

    # Also try relative path fallback
    if backend_src not in sys.path:
        sys.path.insert(0, backend_src)

    row: dict[str, Any] = dict(flat_params)
    row["error"] = None
    row["score"] = float("-inf")
    row["wagons_parked"] = 0
    row["wagons_processable"] = 0
    row["wagon_completion_pct"] = 0.0
    row["locomotive_active_time_min"] = float("nan")
    row["loco_utilization_pct"] = float("nan")

    try:
        from popupsim.optimizer.harness import compute_cost
        from popupsim.optimizer.harness import run_simulation

        sim = run_simulation(Path(scenario_dir_str), scenario_overrides=overrides)
        row["wagons_parked"] = sim.wagons_parked
        row["wagons_processable"] = sim.wagons_processable
        row["wagon_completion_pct"] = round(sim.wagon_completion_pct, 4)
        row["locomotive_active_time_min"] = sim.locomotive_active_time_min
        row["loco_utilization_pct"] = round(sim.loco_utilization_pct, 4)
        row["score"] = round(compute_cost(sim, w_wagons=w_wagons, w_loco=w_loco), 6)
    except Exception:
        row["error"] = traceback.format_exc(limit=3).replace("\n", " | ")

    return row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_to_flat_dict(cfg: Any) -> dict[str, Any]:
    """Flatten a :class:`~popupsim.optimizer.model.Configuration` into a row dict.

    Task-rule parameters are prefixed with the task abbreviation, e.g.
    ``ctr_base_priority``, ``rtw_hold_condition``, etc.
    """
    # Max rules per task is 5 (TaskPriorityRule.rules max_length=5). Always emit
    # all slots so every row has the same fixed-width schema regardless of how
    # many rules are active in a given configuration.
    MAX_RULES = 5
    ABBREV = {
        "collection_to_retrofit": "ctr",
        "retrofit_to_workshop": "rtw",
        "workshop_to_retrofitted": "wtr",
        "retrofitted_to_parking": "rtp",
    }
    row: dict[str, Any] = {}

    tp = cfg.task_priorities
    for task_name, prefix in ABBREV.items():
        task: Any = getattr(tp, task_name)
        row[f"{prefix}_base_priority"] = task.base_priority
        row[f"{prefix}_n_rules"] = len(task.rules)
        row[f"{prefix}_hold_condition"] = (
            task.hold_until.condition.value if task.hold_until else None
        )
        row[f"{prefix}_hold_threshold"] = (
            task.hold_until.threshold if task.hold_until else None
        )
        row[f"{prefix}_max_hold_time"] = task.max_hold_time
        for i in range(MAX_RULES):
            rule = task.rules[i] if i < len(task.rules) else None
            row[f"{prefix}_r{i}_condition"] = rule.condition.value if rule else None
            row[f"{prefix}_r{i}_threshold"] = rule.threshold if rule else None
            row[f"{prefix}_r{i}_priority"] = rule.priority if rule else None

    row["collection_track_strategy"] = cfg.collection_track_strategy.value
    row["retrofit_selection_strategy"] = cfg.retrofit_selection_strategy.value
    row["retrofitted_selection_strategy"] = cfg.retrofitted_selection_strategy.value
    row["workshop_selection_strategy"] = cfg.workshop_selection_strategy.value
    row["parking_selection_strategy"] = cfg.parking_selection_strategy.value

    return row


def row_to_configuration(row: pd.Series) -> Any:
    """Reconstruct a :class:`~popupsim.optimizer.model.Configuration` from a DataFrame row.

    Pass a row directly from the results DataFrame, e.g.::

        cfg = row_to_configuration(df_adaptive.iloc[0])

    This avoids the ordering mismatch that occurs when reading the CSV file
    independently (the CSV is written in evaluation order, not score order).

    Args:
        row: A single row from a results DataFrame produced by
            :func:`run_parallel` or :func:`run_adaptive`.

    Returns:
        A :class:`~popupsim.optimizer.model.Configuration` instance.
    """
    from popupsim.optimizer.model import (  # noqa: PLC0415
        Condition,
        Configuration,
        HoldCondition,
        PriorityRule,
        SelectionStrategy,
        TaskPriorities,
        TaskPriorityRule,
    )

    ABBREV = {
        "collection_to_retrofit": "ctr",
        "retrofit_to_workshop": "rtw",
        "workshop_to_retrofitted": "wtr",
        "retrofitted_to_parking": "rtp",
    }
    MAX_RULES = 5

    def _parse_task(prefix: str) -> TaskPriorityRule:
        rules: list[PriorityRule] = []
        for i in range(MAX_RULES):
            cond_val = row.get(f"{prefix}_r{i}_condition")
            thr_val = row.get(f"{prefix}_r{i}_threshold")
            pri_val = row.get(f"{prefix}_r{i}_priority")
            if pd.isna(cond_val) or pd.isna(pri_val):
                break
            rules.append(
                PriorityRule(
                    condition=Condition(cond_val),
                    threshold=float(thr_val) if thr_val is not None else 0.0,
                    priority=int(pri_val),
                )
            )

        hold_cond_val = row.get(f"{prefix}_hold_condition")
        hold_thr_val = row.get(f"{prefix}_hold_threshold")
        hold_until: HoldCondition | None = None
        if not pd.isna(hold_cond_val):
            hold_until = HoldCondition(
                condition=Condition(hold_cond_val),
                threshold=float(hold_thr_val) if hold_thr_val is not None else 0.0,
            )

        max_hold_val = row.get(f"{prefix}_max_hold_time")
        max_hold_time: float | None = None if pd.isna(max_hold_val) else float(max_hold_val)

        return TaskPriorityRule(
            base_priority=int(row[f"{prefix}_base_priority"]),
            rules=rules,
            hold_until=hold_until,
            max_hold_time=max_hold_time,
        )

    task_priorities = TaskPriorities(
        **{task: _parse_task(prefix) for task, prefix in ABBREV.items()}
    )

    return Configuration(
        tracks=[],  # tracks are scenario-level, not encoded in the CSV
        task_priorities=task_priorities,
        collection_track_strategy=SelectionStrategy(row["collection_track_strategy"]),
        retrofit_selection_strategy=SelectionStrategy(row["retrofit_selection_strategy"]),
        retrofitted_selection_strategy=SelectionStrategy(row["retrofitted_selection_strategy"]),
        workshop_selection_strategy=SelectionStrategy(row["workshop_selection_strategy"]),
        parking_selection_strategy=SelectionStrategy(row["parking_selection_strategy"]),
    )

