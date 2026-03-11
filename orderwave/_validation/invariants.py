from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .shared import INVARIANT_FAILURE_COLUMNS, OPTIONAL_NUMERIC_COLUMNS, ValidationRun, failure_row


def collect_invariant_failures(
    run: ValidationRun,
    *,
    stage: str,
    preset: str,
    seed: int,
) -> pd.DataFrame:
    """Collect normalized invariant failures for one run."""

    failures: list[dict[str, Any]] = []

    if run.run_failed:
        failures.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=np.nan,
                event_idx=np.nan,
                invariant_name="run_failed",
                details=run.error_message or "run failed",
            )
        )

    if not event_order_ok(run.event_history):
        failures.extend(event_order_failures(run.event_history, preset=preset, seed=seed, stage=stage))

    if not run.event_history.empty:
        crossing = run.event_history.loc[run.event_history["best_bid_after"] >= run.event_history["best_ask_after"]]
        for row in crossing.itertuples(index=False):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=row.event_idx,
                    invariant_name="crossed_or_locked_book_event",
                    details=f"best_bid_after={row.best_bid_after} best_ask_after={row.best_ask_after}",
                )
            )

        market_events = run.event_history.loc[run.event_history["event_type"] == "market"]
        fill_sums = market_events["fills"].apply(lambda fills: float(sum(qty for _, qty in fills)))
        mismatched = market_events.loc[
            ~np.isclose(
                market_events["fill_qty"].to_numpy(dtype=float),
                fill_sums.to_numpy(dtype=float),
            )
        ]
        for row in mismatched.itertuples(index=False):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=row.event_idx,
                    invariant_name="market_fill_mismatch",
                    details=f"fill_qty={row.fill_qty}",
                )
            )

    if not run.history.empty:
        crossing = run.history.loc[run.history["best_bid"] >= run.history["best_ask"]]
        for row in crossing.itertuples(index=False):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=np.nan,
                    invariant_name="crossed_or_locked_book_history",
                    details=f"best_bid={row.best_bid} best_ask={row.best_ask}",
                )
            )

    failures.extend(debug_alignment_failures(run.debug_history, run.event_history, preset=preset, seed=seed, stage=stage))

    for label, snapshot in (("start_snapshot", run.start_snapshot), ("end_snapshot", run.end_snapshot)):
        failures.extend(snapshot_failures(snapshot, preset=preset, seed=seed, stage=stage, label=label))

    visual_rows = getattr(run.market, "_visual_history", []) if run.market is not None else []
    for index, row in enumerate(visual_rows):
        bid_values = np.asarray(row.bid_qty, dtype=float)
        ask_values = np.asarray(row.ask_qty, dtype=float)
        if np.any(bid_values[np.isfinite(bid_values)] < 0.0):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=np.nan,
                    invariant_name="negative_visual_bid_depth",
                    details=f"visual_row={index}",
                )
            )
        if np.any(ask_values[np.isfinite(ask_values)] < 0.0):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=np.nan,
                    invariant_name="negative_visual_ask_depth",
                    details=f"visual_row={index}",
                )
            )

    return pd.DataFrame.from_records(failures, columns=INVARIANT_FAILURE_COLUMNS)


def event_order_ok(events: pd.DataFrame) -> bool:
    if len(events) < 2:
        return True
    steps = events["step"].to_numpy(dtype=int)
    event_idx = events["event_idx"].to_numpy(dtype=int)
    return bool(np.all((steps[1:] > steps[:-1]) | ((steps[1:] == steps[:-1]) & (event_idx[1:] > event_idx[:-1]))))


def event_order_failures(
    events: pd.DataFrame,
    *,
    preset: str,
    seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(events) < 2:
        return rows
    steps = events["step"].to_numpy(dtype=int)
    event_idx = events["event_idx"].to_numpy(dtype=int)
    invalid = np.where(~((steps[1:] > steps[:-1]) | ((steps[1:] == steps[:-1]) & (event_idx[1:] > event_idx[:-1]))))[0]
    for index in invalid:
        row = events.iloc[index + 1]
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=row["step"],
                event_idx=row["event_idx"],
                invariant_name="event_order_nonmonotonic",
                details="(step, event_idx) ordering broken",
            )
        )
    return rows


def debug_alignment_failures(
    debug: pd.DataFrame,
    events: pd.DataFrame,
    *,
    preset: str,
    seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(debug) != len(events):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=np.nan,
                event_idx=np.nan,
                invariant_name="debug_length_mismatch",
                details=f"events={len(events)} debug={len(debug)}",
            )
        )
        return rows
    if debug.empty and events.empty:
        return rows
    joined = events.merge(debug, on=["step", "event_idx"], how="outer", indicator=True, suffixes=("_event", "_debug"))
    mismatched = joined.loc[joined["_merge"] != "both"]
    for row in mismatched.itertuples(index=False):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=row.step,
                event_idx=row.event_idx,
                invariant_name="debug_event_join_mismatch",
                details=str(row._merge),
            )
        )
    both = joined.loc[joined["_merge"] == "both"]
    if not both.empty:
        phase_mismatch = both.loc[both["session_phase_event"] != both["session_phase_debug"]]
        for row in phase_mismatch.itertuples(index=False):
            rows.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=row.event_idx,
                    invariant_name="debug_phase_mismatch",
                    details=f"event={row.session_phase_event} debug={row.session_phase_debug}",
                )
            )
    return rows


def snapshot_failures(
    snapshot: dict[str, Any] | None,
    *,
    preset: str,
    seed: int,
    stage: str,
    label: str,
) -> list[dict[str, Any]]:
    if snapshot is None:
        return [
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=np.nan,
                event_idx=np.nan,
                invariant_name=f"{label}_missing",
                details="snapshot missing",
            )
        ]

    rows: list[dict[str, Any]] = []
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    bid_prices = [level["price"] for level in bids]
    ask_prices = [level["price"] for level in asks]
    bid_qty = [level["qty"] for level in bids]
    ask_qty = [level["qty"] for level in asks]
    step = snapshot.get("step", np.nan)

    if any(quantity < 0.0 for quantity in bid_qty):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_negative_bid_depth",
                details="negative bid depth",
            )
        )
    if any(quantity < 0.0 for quantity in ask_qty):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_negative_ask_depth",
                details="negative ask depth",
            )
        )
    if bid_prices != sorted(bid_prices, reverse=True):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_bid_sorting",
                details="bid prices not descending",
            )
        )
    if ask_prices != sorted(ask_prices):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_ask_sorting",
                details="ask prices not ascending",
            )
        )
    if bids and asks and bid_prices[0] >= ask_prices[0]:
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_crossed_book",
                details=f"best_bid={bid_prices[0]} best_ask={ask_prices[0]}",
            )
        )
    return rows


def table_nonfinite_failures(frame: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    failures: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=INVARIANT_FAILURE_COLUMNS)
    numeric = frame.select_dtypes(include=[np.number])
    if stage == "run_metrics":
        numeric = numeric.drop(columns=list(OPTIONAL_NUMERIC_COLUMNS & set(numeric.columns)), errors="ignore")
    if numeric.empty:
        return pd.DataFrame(columns=INVARIANT_FAILURE_COLUMNS)
    invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
    invalid_rows, invalid_cols = np.where(invalid)
    for row_index, col_index in zip(invalid_rows, invalid_cols):
        column = numeric.columns[col_index]
        seed_value = -1
        if "seed" in frame.columns and pd.notna(frame.iloc[row_index].get("seed")):
            seed_value = int(frame.iloc[row_index]["seed"])
        failures.append(
            failure_row(
                preset=str(frame.iloc[row_index].get("preset", "n/a")),
                seed=seed_value,
                stage=stage,
                step=np.nan,
                event_idx=np.nan,
                invariant_name="summary_nonfinite",
                details=f"{column} is non-finite",
            )
        )
    return pd.DataFrame.from_records(failures, columns=INVARIANT_FAILURE_COLUMNS)


def reproducibility_failures(reproducibility: pd.DataFrame) -> pd.DataFrame:
    failures: list[dict[str, Any]] = []
    for row in reproducibility.itertuples(index=False):
        if row.all_reproducible:
            continue
        for field_name in (
            "history_hash_equal",
            "event_hash_equal",
            "debug_hash_equal",
            "metrics_hash_equal",
            "gen_vs_step_history_equal",
            "gen_vs_step_event_equal",
            "gen_vs_step_debug_equal",
        ):
            if getattr(row, field_name):
                continue
            failures.append(
                failure_row(
                    preset=row.preset,
                    seed=row.seed,
                    stage="reproducibility",
                    step=np.nan,
                    event_idx=np.nan,
                    invariant_name=field_name,
                    details="reproducibility mismatch",
                )
            )
    return pd.DataFrame.from_records(failures, columns=INVARIANT_FAILURE_COLUMNS)
