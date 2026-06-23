"""Two-phase adaptive search algorithm for the PopUpSim optimizer.

Phase 1 — Random exploration
    Sample ``n_random`` configurations uniformly at random from the space and
    evaluate them in parallel.  This maps the landscape cheaply (~0.2 % of the
    coarse grid for the default budget).

Phase 2 — Coordinate descent from top-K starts
    Take the ``k_starts`` best results from Phase 1.  For each starting
    configuration perform repeated sweeps over the four task dimensions
    (``ctr``, ``rtw``, ``wtr``, ``rtp``): fix three tasks, enumerate all
    variants for the fourth, keep the best-scoring one.  Repeat until
    convergence or ``max_rounds`` is reached.  Already-evaluated configurations
    are skipped (no duplicate simulation runs).

Typical budget
--------------
With default parameters on the coarse space (259,200 configurations):

    Phase 1: 500 evals
    Phase 2: k_starts=5 × max_rounds=5 × ≤224 neighbors ≈ 5,600 evals
    Total:   ~6,100 evals  vs  259,200  →  ~42× faster than exhaustive search

Usage::

    import sys
    sys.path.insert(0, 'popupsim/backend/src')

    from pathlib import Path
    from popupsim.optimizer.model import SearchSpace
    from popupsim.optimizer.runner import load_tracks_from_scenario
    from popupsim.optimizer.adaptive import run_adaptive

    scenario_dir = Path("Data/examples/ten_trains_two_days_priority_dispatch")
    tracks = load_tracks_from_scenario(scenario_dir)
    space = SearchSpace.coarse(tracks)

    df = run_adaptive(
        space=space,
        scenario_dir=scenario_dir,
        output_csv=Path("results/adaptive_run.csv"),
        n_random=500,
        k_starts=5,
        max_rounds=5,
        max_workers=10,
        seed=42,
    )
    print(df.head(10))
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from popupsim.optimizer.model import Configuration
from popupsim.optimizer.model import SearchSpace
from popupsim.optimizer.runner import _config_to_flat_dict
from popupsim.optimizer.runner import _worker
from popupsim.optimizer.runner import load_results

logger = logging.getLogger(__name__)

# Task names in the order used by the search space
_TASK_NAMES: tuple[str, ...] = (
    "collection_to_retrofit",
    "retrofit_to_workshop",
    "workshop_to_retrofitted",
    "retrofitted_to_parking",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eval_batch(
    configs: list[Configuration],
    scenario_dir: Path,
    backend_src: str,
    w_wagons: float,
    w_loco: float,
    max_workers: int,
    seen: dict[str, float],
    *,
    label: str = "batch",
) -> list[dict[str, Any]]:
    """Evaluate a list of configurations in parallel, skipping already-seen ones.

    Updates *seen* in-place (key = model_dump JSON, value = score).

    Args:
        configs: Configurations to evaluate.
        scenario_dir: Simulation scenario directory.
        backend_src: Path to ``popupsim/backend/src`` added to worker ``sys.path``.
        w_wagons: Weight for wagon completion in the cost function.
        w_loco: Weight for locomotive utilisation in the cost function.
        max_workers: Process-pool worker count.
        seen: Mutable dict mapping config JSON keys to scores (dedup cache).
        label: Log label for progress reporting.

    Returns:
        List of result row dicts (same schema as :func:`~runner.run_parallel`).
    """
    # Dedup: skip configs already in seen
    to_run: list[Configuration] = []
    for cfg in configs:
        key = cfg.model_dump_json()
        if key not in seen:
            to_run.append(cfg)

    if not to_run:
        return []

    results: list[dict[str, Any]] = []
    total = len(to_run)
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
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
            for cfg in to_run
        }

        for future in as_completed(futures):
            row = future.result()
            completed += 1
            if row.get("error"):
                errors += 1
            results.append(row)

            # Update the dedup cache
            cfg = futures[future]
            seen[cfg.model_dump_json()] = float(row.get("score", float("-inf")))

            if completed % max(1, total // 20) == 0 or completed == total:
                pct = completed / total * 100
                print(
                    f"  [{label}] {completed:>5,}/{total:,} ({pct:.0f}%)  errors: {errors}",
                    end="\r",
                    flush=True,
                )

    print()  # newline after \r progress
    return results


def _write_rows(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Append *rows* to *output_csv* (writes header if the file is new/empty)."""
    import csv  # noqa: PLC0415

    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with open(output_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_adaptive(
    space: SearchSpace,
    scenario_dir: Path,
    output_csv: Path,
    n_random: int = 500,
    k_starts: int = 5,
    max_rounds: int = 5,
    max_workers: int | None = None,
    w_wagons: float = 0.9,
    w_loco: float = 0.1,
    backend_src: str = "popupsim/backend/src",
    seed: int | None = None,
) -> pd.DataFrame:
    """Run a two-phase adaptive search over *space*.

    Phase 1 evaluates ``n_random`` uniformly sampled configurations in
    parallel.  Phase 2 runs coordinate descent from the ``k_starts`` best
    Phase 1 results.

    All evaluated configurations are written incrementally to *output_csv*
    (same schema as :func:`~runner.run_parallel`), so runs can be resumed
    and results analysed with the same visualization tools.

    Args:
        space: The search space to explore (e.g. ``SearchSpace.coarse(tracks)``).
        scenario_dir: Scenario directory passed to the simulation harness.
        output_csv: Path for the incremental results CSV.
        n_random: Number of random configurations to evaluate in Phase 1.
        k_starts: Number of top Phase 1 results to use as Phase 2 starting
            points.
        max_rounds: Maximum coordinate-descent rounds per starting point.
        max_workers: Worker processes for parallel evaluation.  Defaults to
            ``os.cpu_count()``.
        w_wagons: Weight for wagon completion (passed to cost function).
        w_loco: Weight for locomotive utilisation (passed to cost function).
        backend_src: Path to ``popupsim/backend/src`` added to worker
            ``sys.path``.
        seed: Optional random seed for reproducibility of Phase 1 sampling.

    Returns:
        DataFrame of all evaluated configurations sorted by ``score`` descending.
    """
    import random  # noqa: PLC0415

    workers = max_workers or os.cpu_count() or 1
    rng = random.Random(seed)

    # seen: config JSON → score (dedup cache shared across both phases)
    seen: dict[str, float] = {}

    # ── Phase 1: random exploration ──────────────────────────────────────────
    sizes = space.grid_size()
    total_space = sizes["total"]
    actual_n = min(n_random, total_space)
    print(
        f"Phase 1 — random exploration: {actual_n:,} configs "
        f"({actual_n / total_space * 100:.1f}% of {total_space:,})"
    )
    phase1_configs = space.sample(actual_n, rng=rng)
    phase1_rows = _eval_batch(
        phase1_configs, scenario_dir, backend_src, w_wagons, w_loco, workers, seen, label="P1"
    )
    _write_rows(phase1_rows, output_csv)
    print(f"  Phase 1 done: {len(phase1_rows):,} evaluated")

    # ── Identify top-K starting points ───────────────────────────────────────
    if not phase1_rows:
        logger.warning("Phase 1 produced no results; skipping Phase 2.")
        return load_results(output_csv)

    scored_phase1 = sorted(
        (r for r in phase1_rows if not r.get("error")),
        key=lambda r: float(r.get("score", float("-inf"))),
        reverse=True,
    )
    starts: list[Configuration] = []
    used_keys: set[str] = set()
    for row in scored_phase1:
        if len(starts) >= k_starts:
            break
        # Re-hydrate config from the flat CSV row
        from popupsim.optimizer.runner import row_to_configuration  # noqa: PLC0415

        try:
            # We need the CSV on disk; write Phase 1 first (already done above)
            # Instead, reconstruct directly from the matching Phase 1 config
            cfg = phase1_configs[phase1_rows.index(row)]
            key = cfg.model_dump_json()
            if key not in used_keys:
                starts.append(cfg)
                used_keys.add(key)
        except (IndexError, ValueError):
            continue

    print(f"\nPhase 2 — coordinate descent from {len(starts)} starting point(s)")

    # ── Phase 2: coordinate descent ───────────────────────────────────────────
    all_phase2_rows: list[dict[str, Any]] = []

    for start_idx, start_cfg in enumerate(starts):
        current_cfg = start_cfg
        current_score = seen.get(current_cfg.model_dump_json(), float("-inf"))
        print(
            f"\n  Start {start_idx + 1}/{len(starts)}  "
            f"(initial score={current_score:.6f})"
        )

        for round_idx in range(max_rounds):
            improved = False

            for task_name in _TASK_NAMES:
                neighbors = space.task_neighbors(current_cfg, task_name)
                new_rows = _eval_batch(
                    neighbors,
                    scenario_dir,
                    backend_src,
                    w_wagons,
                    w_loco,
                    workers,
                    seen,
                    label=f"P2 s{start_idx + 1} r{round_idx + 1} {task_name[:3]}",
                )
                all_phase2_rows.extend(new_rows)
                _write_rows(new_rows, output_csv)

                # Find best neighbor (may have been skipped if already in seen)
                best_neighbor: Configuration | None = None
                best_neighbor_score = current_score
                for neighbor in neighbors:
                    s = seen.get(neighbor.model_dump_json(), float("-inf"))
                    if s > best_neighbor_score:
                        best_neighbor_score = s
                        best_neighbor = neighbor

                if best_neighbor is not None:
                    print(
                        f"    [{task_name[:3]}] improved: "
                        f"{current_score:.6f} → {best_neighbor_score:.6f}"
                    )
                    current_cfg = best_neighbor
                    current_score = best_neighbor_score
                    improved = True

            if not improved:
                print(f"  Converged after round {round_idx + 1}")
                break
        else:
            print(f"  Reached max_rounds={max_rounds}")

        print(f"  Final score for start {start_idx + 1}: {current_score:.6f}")

    print(
        f"\nAdaptive search complete. "
        f"Total evaluated: {len(seen):,}  "
        f"({len(seen) / total_space * 100:.1f}% of space)"
    )
    return load_results(output_csv)
