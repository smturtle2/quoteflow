from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from orderwave.market import Market

from .metrics import compute_run_metrics
from .shared import deterministic_metric_view, stable_frame_hash, stable_object_hash
from .single_run import run_market_validation


def run_reproducibility_checks(
    *,
    presets: Sequence[str],
    seed: int,
    steps: int,
    warmup_fraction: float = 0.10,
) -> pd.DataFrame:
    """Run repeated same-seed checks and compare against step-by-step execution."""

    rows: list[dict[str, object]] = []
    for preset in presets:
        repeated_runs = [
            run_market_validation(
                preset=preset,
                seed=seed,
                steps=steps,
                warmup_fraction=warmup_fraction,
            )
            for _ in range(3)
        ]
        step_market = Market(seed=seed, config={"preset": preset})
        for _ in range(steps):
            step_market.step()

        repeated_hashes = []
        metric_hashes = []
        for run in repeated_runs:
            metrics = compute_run_metrics(
                run,
                stage="reproducibility",
                preset=preset,
                seed=seed,
                config_label="repeat",
            )
            repeated_hashes.append(
                {
                    "history": stable_frame_hash(run.history),
                    "events": stable_frame_hash(run.event_history),
                    "debug": stable_frame_hash(run.debug_history),
                }
            )
            metric_hashes.append(stable_object_hash(deterministic_metric_view(metrics)))

        base_hash = repeated_hashes[0]
        history_hash_equal = all(item["history"] == base_hash["history"] for item in repeated_hashes[1:])
        event_hash_equal = all(item["events"] == base_hash["events"] for item in repeated_hashes[1:])
        debug_hash_equal = all(item["debug"] == base_hash["debug"] for item in repeated_hashes[1:])
        metrics_hash_equal = all(metric_hash == metric_hashes[0] for metric_hash in metric_hashes[1:])

        gen_history = repeated_runs[0].history
        gen_events = repeated_runs[0].event_history
        gen_debug = repeated_runs[0].debug_history

        rows.append(
            {
                "preset": preset,
                "seed": int(seed),
                "steps": int(steps),
                "history_hash_equal": bool(history_hash_equal),
                "event_hash_equal": bool(event_hash_equal),
                "debug_hash_equal": bool(debug_hash_equal),
                "metrics_hash_equal": bool(metrics_hash_equal),
                "gen_vs_step_history_equal": bool(gen_history.equals(step_market.get_history())),
                "gen_vs_step_event_equal": bool(gen_events.equals(step_market.get_event_history())),
                "gen_vs_step_debug_equal": bool(gen_debug.equals(step_market.get_debug_history())),
                "all_reproducible": bool(
                    history_hash_equal
                    and event_hash_equal
                    and debug_hash_equal
                    and metrics_hash_equal
                    and gen_history.equals(step_market.get_history())
                    and gen_events.equals(step_market.get_event_history())
                    and gen_debug.equals(step_market.get_debug_history())
                ),
            }
        )
    return pd.DataFrame(rows)
