"""Simulation harness: runs a scenario in a temp dir and returns objective metrics."""

import csv
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from application.simulation_service import SimulationApplicationService
from contexts.configuration.domain.configuration_builder import ConfigurationBuilder
from shared.infrastructure.simpy_time_converters import timedelta_to_sim_ticks


@dataclass(frozen=True)
class SimulationResult:
    """Objective metrics returned after one simulation run."""

    wagons_parked: int
    """Number of wagons that completed the full retrofit workflow (arrived at parking)."""

    wagons_processable: int
    """Number of wagons that could theoretically be processed (denominator for completion rate)."""

    locomotive_active_time_min: float
    """Total minutes all locomotives spent in non-parking states (MOVING + ALLOCATED), summed across locos."""

    n_locomotives: int
    """Number of locomotives present in the simulation."""

    sim_duration_min: float
    """Total simulated timespan in minutes (end_date − start_date)."""

    @property
    def wagon_completion_pct(self) -> float:
        """Percentage of processable wagons that were fully retrofitted and parked (0–100)."""
        if self.wagons_processable == 0:
            return 0.0
        return self.wagons_parked / self.wagons_processable * 100.0

    @property
    def loco_utilization_pct(self) -> float:
        """Average locomotive utilisation as a percentage of total available time (0–100).

        Each locomotive is weighted equally:
            utilisation = (total_active_time / n_locos) / sim_duration × 100
        """
        if self.n_locomotives == 0 or self.sim_duration_min == 0:
            return 0.0
        return (self.locomotive_active_time_min / self.n_locomotives) / self.sim_duration_min * 100.0


def compute_cost(
    result: "SimulationResult",
    w_wagons: float = 0.9,
    w_loco: float = 0.1,
) -> float:
    """Compute a normalised objective score (higher = better).

    The score combines two percentage-based metrics, both normalised to [0, 1]:

    * **Wagon completion** — fraction of processable wagons successfully parked.
      Maximised. Weight ``w_wagons`` (default 0.9).
    * **Locomotive utilisation** — average fraction of sim time each loco is active.
      Minimised (penalised). Weight ``w_loco`` (default 0.1).

    Formula::

        score = w_wagons * wagon_completion_pct/100
              − w_loco   * loco_utilization_pct/100

    Score range: ``[−w_loco, w_wagons]``.  Perfect throughput with zero loco
    use → ``w_wagons``.  Zero throughput with full loco use → ``−w_loco``.

    Args:
        result: SimulationResult from the harness.
        w_wagons: Weight for wagon completion (should dominate, default 0.9).
        w_loco: Weight for locomotive utilisation penalty (default 0.1).

    Returns:
        Scalar objective score to maximise.
    """
    return (
        w_wagons * result.wagon_completion_pct / 100.0
        - w_loco * result.loco_utilization_pct / 100.0
    )


def run_simulation(scenario_dir: Path, scenario_overrides: dict | None = None) -> SimulationResult:
    """Run the simulation in an isolated temporary directory.

    Args:
        scenario_dir: Path to the base scenario directory (must contain scenario.json).
        scenario_overrides: Optional dict of top-level fields to merge into scenario.json
                            before running (e.g. task_priorities, track type settings).

    Returns:
        SimulationResult with wagons_parked and locomotive_active_time_min.

    Raises:
        RuntimeError: If the simulation reports a failure.
    """
    with tempfile.TemporaryDirectory(prefix="popupsim_run_") as tmp:
        tmp_path = Path(tmp)
        scenario_tmp = tmp_path / "scenario"
        output_tmp = tmp_path / "output"
        output_tmp.mkdir()

        shutil.copytree(scenario_dir, scenario_tmp)

        if scenario_overrides:
            _apply_scenario_overrides(scenario_tmp / "scenario.json", scenario_overrides)

        _suppress_simulation_logging()

        scenario = ConfigurationBuilder(scenario_tmp).build()
        service = SimulationApplicationService(scenario, output_tmp)
        until = timedelta_to_sim_ticks(scenario.end_date - scenario.start_date)
        result = service.execute(until)

        if not result.success:
            raise RuntimeError(f"Simulation failed for scenario in {scenario_dir}")

        retrofit_context = service.contexts.get("retrofit_workflow")
        if retrofit_context is not None and hasattr(retrofit_context, "export_events"):
            retrofit_context.export_events(str(output_tmp))

        wagons_parked = _read_wagons_parked(output_tmp)
        wagons_processable = _read_wagons_processable(output_tmp)
        loco_active_time, n_locomotives = _read_locomotive_active_time(output_tmp)
        sim_duration_min = (scenario.end_date - scenario.start_date).total_seconds() / 60.0

        return SimulationResult(
            wagons_parked=wagons_parked,
            wagons_processable=wagons_processable,
            locomotive_active_time_min=loco_active_time,
            n_locomotives=n_locomotives,
            sim_duration_min=sim_duration_min,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_scenario_overrides(scenario_json: Path, overrides: dict) -> None:
    """Deep-merge *overrides* into scenario.json on disk."""
    with open(scenario_json, encoding="utf-8") as f:
        data: dict = json.load(f)
    _deep_merge(data, overrides)
    with open(scenario_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* in-place.

    Dicts are merged recursively; all other types (including lists) are
    replaced wholesale so that e.g. a ``rules: []`` override correctly
    clears any rules present in the base scenario.
    """
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            # Lists (e.g. "rules") and scalars are always replaced, never merged.
            base[key] = value


def _suppress_simulation_logging() -> None:
    """Prevent the simulation from polluting stdout/stderr during batch runs."""
    logging.getLogger().setLevel(logging.CRITICAL)
    for name in ("simpy", "popupsim", "application", "contexts", "infrastructure"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _read_wagons_parked(output_path: Path) -> int:
    """Read wagons_parked from summary_metrics.json."""
    summary_file = output_path / "summary_metrics.json"
    with open(summary_file, encoding="utf-8") as f:
        data: dict = json.load(f)
    return int(data.get("wagons_parked", 0))


def _read_wagons_processable(output_path: Path) -> int:
    """Read wagons_processable from summary_metrics.json."""
    summary_file = output_path / "summary_metrics.json"
    with open(summary_file, encoding="utf-8") as f:
        data: dict = json.load(f)
    return int(data.get("wagons_processable", 0))


def _read_locomotive_active_time(output_path: Path) -> tuple[float, int]:
    """Compute total active (non-parking) locomotive time from locomotive_movements.csv.

    For each locomotive, the events are sorted by timestamp. The duration of a state
    equals the time until the next event. All intervals where the preceding event is
    NOT ``PARKING`` are summed across all locomotives.

    Returns:
        Tuple of (total_active_minutes_across_all_locos, n_locomotives).
    """
    movements_file = output_path / "locomotive_movements.csv"

    # Group (timestamp, event) pairs by locomotive_id
    by_loco: dict[str, list[tuple[float, str]]] = {}
    with open(movements_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            loco_id: str = row["locomotive_id"]
            by_loco.setdefault(loco_id, []).append((float(row["timestamp"]), row["event"]))

    total_active: float = 0.0
    for events in by_loco.values():
        events.sort(key=lambda pair: pair[0])
        for i in range(len(events) - 1):
            event_type = events[i][1]
            duration = events[i + 1][0] - events[i][0]
            if event_type != "PARKING":
                total_active += duration

    return total_active, len(by_loco)
