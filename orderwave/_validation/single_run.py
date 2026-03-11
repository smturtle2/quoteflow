from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

import pandas as pd

from orderwave.history import DEBUG_COLUMNS, EVENT_COLUMNS
from orderwave.market import Market

from .shared import MemoryTracker, ValidationRun


def run_market_validation(
    *,
    preset: str,
    seed: int,
    steps: int,
    config_overrides: Mapping[str, object] | None = None,
    warmup_fraction: float = 0.10,
) -> ValidationRun:
    """Run one market simulation and capture its outputs."""

    config: dict[str, object] = {"preset": preset, "logging_mode": "full"}
    if config_overrides:
        config.update(config_overrides)

    market: Market | None = None
    start_snapshot: dict[str, Any] | None = None
    end_snapshot: dict[str, Any] | None = None
    history = pd.DataFrame()
    event_history = pd.DataFrame()
    debug_history = pd.DataFrame()
    error_message: str | None = None
    run_failed = False
    elapsed = 0.0

    tracker = MemoryTracker()
    try:
        market = Market(seed=seed, config=config)
        start_snapshot = market.get()
        with tracker:
            started = perf_counter()
            market.gen(steps=steps)
            elapsed = perf_counter() - started
        end_snapshot = market.get()
        history = market.get_history()
        if market.config.logging_mode == "full":
            event_history = market.get_event_history()
            debug_history = market.get_debug_history()
        else:
            event_history = pd.DataFrame(columns=EVENT_COLUMNS)
            debug_history = pd.DataFrame(columns=DEBUG_COLUMNS)
    except Exception as exc:  # pragma: no cover - failure-path guard
        run_failed = True
        error_message = f"{type(exc).__name__}: {exc}"
        if market is not None:
            try:
                end_snapshot = market.get()
                history = market.get_history()
                if market.config.logging_mode == "full":
                    event_history = market.get_event_history()
                    debug_history = market.get_debug_history()
                else:
                    event_history = pd.DataFrame(columns=EVENT_COLUMNS)
                    debug_history = pd.DataFrame(columns=DEBUG_COLUMNS)
            except Exception:
                history = pd.DataFrame()
                event_history = pd.DataFrame()
                debug_history = pd.DataFrame()

    warmup_cutoff = int(max(0, min(steps, int(float(steps) * float(warmup_fraction)))))
    return ValidationRun(
        market=market,
        history=history,
        event_history=event_history,
        debug_history=debug_history,
        start_snapshot=start_snapshot,
        end_snapshot=end_snapshot,
        elapsed_seconds=float(elapsed),
        peak_memory_mb=tracker.peak_memory_mb,
        peak_memory_increase_mb=tracker.peak_memory_increase_mb,
        memory_metric=tracker.metric_name,
        memory_growth_without_recovery=tracker.growth_without_recovery,
        requested_steps=int(steps),
        warmup_fraction=float(warmup_fraction),
        warmup_cutoff_step=warmup_cutoff,
        run_failed=bool(run_failed),
        error_message=error_message,
    )
