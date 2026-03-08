from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import tracemalloc
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from orderwave.history import DEBUG_COLUMNS, EVENT_COLUMNS
from orderwave.market import Market

try:
    import psutil
except ImportError:  # pragma: no cover - exercised through fallback path
    psutil = None


PHASE_ORDER: tuple[str, ...] = ("open", "mid", "close")
DEFAULT_PRESETS: tuple[str, ...] = ("balanced", "trend", "volatile")
DEFAULT_SENSITIVITY_KNOBS: tuple[str, ...] = (
    "limit_rate_scale",
    "market_rate_scale",
    "cancel_rate_scale",
    "fair_price_vol_scale",
    "regime_transition_scale",
    "seasonality_scale",
    "excitation_scale",
    "meta_order_scale",
    "shock_scale",
)
DEFAULT_SENSITIVITY_SCALES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
CORE_SENSITIVITY_KNOBS: tuple[str, ...] = (
    "market_rate_scale",
    "cancel_rate_scale",
    "fair_price_vol_scale",
    "excitation_scale",
    "meta_order_scale",
    "shock_scale",
)
INVARIANT_FAILURE_COLUMNS: list[str] = [
    "preset",
    "seed",
    "stage",
    "step",
    "event_idx",
    "invariant_name",
    "details",
]
EPSILON = 1e-9
BASELINE_THROUGHPUT_FLOOR = {
    "balanced": 300.0,
    "trend": 225.0,
    "volatile": 200.0,
}
SOAK_PEAK_MEMORY_BUDGET_MB = {
    "balanced": 2_048.0,
    "trend": 3_072.0,
    "volatile": 3_584.0,
}
BYTES_PER_LOGGED_EVENT_BUDGET = 300.0
OPTIONAL_NUMERIC_COLUMNS = {"knob_scale", "repeat_idx"}
VALIDATION_BASELINE_SCHEMA_VERSION = 1
VALIDATION_BASELINE_METRIC_RULES: dict[str, dict[str, tuple[str, float]]] = {
    "baseline": {
        "mean_spread_mean": ("abs", 0.003),
        "realized_vol_mean": ("abs", 0.003),
        "abs_return_acf1_mean": ("abs", 0.08),
        "events_per_step_mean": ("abs", 8.0),
        "market_buy_share_mean": ("abs", 0.08),
    },
    "soak": {
        "peak_memory_mb_mean": ("max", 256.0),
        "bytes_per_logged_event_mean": ("max", 2048.0),
    },
}


@dataclass(frozen=True)
class ValidationRun:
    """Container for a single simulated run and its output tables."""

    market: Market | None
    history: pd.DataFrame
    event_history: pd.DataFrame
    debug_history: pd.DataFrame
    start_snapshot: dict[str, Any] | None
    end_snapshot: dict[str, Any] | None
    elapsed_seconds: float
    peak_memory_mb: float
    peak_memory_increase_mb: float
    memory_metric: str
    memory_growth_without_recovery: bool
    requested_steps: int
    warmup_fraction: float
    warmup_cutoff_step: int
    run_failed: bool
    error_message: str | None


@dataclass(frozen=True)
class ValidationPipelineResult:
    """Collected outputs from the final validation pipeline."""

    run_metrics: pd.DataFrame
    preset_summary: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    invariant_failures: pd.DataFrame
    reproducibility: pd.DataFrame
    acceptance: dict[str, Any]
    diagnostics_paths: dict[str, Path]
    artifact_paths: dict[str, Path]


class MemoryTracker:
    def __init__(self) -> None:
        self.metric_name = "rss_mb" if psutil is not None else "python_heap_mb"
        self._process = psutil.Process() if psutil is not None else None
        self._stop = threading.Event()
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None
        self._peak_memory_mb = 0.0
        self._start_memory_mb = 0.0

    def __enter__(self) -> "MemoryTracker":
        if self._process is not None:
            self._start_memory_mb = self._sample_rss()
            self._samples.append(self._start_memory_mb)
            self._thread = threading.Thread(target=self._rss_loop, daemon=True)
            self._thread.start()
        else:
            self._start_memory_mb = 0.0
            tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._process is not None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            self._samples.append(self._sample_rss())
            self._peak_memory_mb = max(self._samples, default=0.0)
        else:
            _, peak = tracemalloc.get_traced_memory()
            self._peak_memory_mb = float(peak) / (1024.0 * 1024.0)
            tracemalloc.stop()

    @property
    def peak_memory_mb(self) -> float:
        return float(self._peak_memory_mb)

    @property
    def peak_memory_increase_mb(self) -> float:
        if self.metric_name == "rss_mb":
            return max(0.0, float(self._peak_memory_mb - self._start_memory_mb))
        return float(self._peak_memory_mb)

    @property
    def growth_without_recovery(self) -> bool:
        if self.metric_name != "rss_mb" or len(self._samples) < 8:
            return False
        samples = np.array(self._samples, dtype=float)
        anchor = samples[max(1, len(samples) // 4) :]
        if len(anchor) < 4:
            return False
        diffs = np.diff(anchor)
        mostly_nonnegative = float(np.mean(diffs >= -2.0)) >= 0.9
        terminal_growth = anchor[-1] >= (np.nanmin(anchor) * 1.25)
        return bool(mostly_nonnegative and terminal_growth)

    def _rss_loop(self) -> None:
        while not self._stop.wait(0.05):
            self._samples.append(self._sample_rss())

    def _sample_rss(self) -> float:
        assert self._process is not None
        return float(self._process.memory_info().rss) / (1024.0 * 1024.0)


def failure_row(
    *,
    preset: str,
    seed: int,
    stage: str,
    step: float,
    event_idx: float,
    invariant_name: str,
    details: str,
) -> dict[str, Any]:
    return {
        "preset": preset,
        "seed": int(seed),
        "stage": stage,
        "step": step,
        "event_idx": event_idx,
        "invariant_name": invariant_name,
        "details": details,
    }


def concat_failures(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if not frame.empty]
    if not valid:
        return pd.DataFrame(columns=INVARIANT_FAILURE_COLUMNS)
    return pd.concat(valid, ignore_index=True, sort=False).reindex(columns=INVARIANT_FAILURE_COLUMNS)


def tick_size_from_run(run: ValidationRun) -> float:
    if run.market is not None:
        return float(run.market.tick_size)
    if not run.history.empty and len(run.history) > 1:
        spread = run.history["spread"].replace(0.0, np.nan).dropna()
        if not spread.empty:
            return float(spread.min())
    return 0.01


def stable_frame_hash(frame: pd.DataFrame) -> str:
    payload = {
        "columns": list(frame.columns),
        "records": [normalize_value(record) for record in frame.to_dict(orient="records")],
    }
    return stable_object_hash(payload)


def deterministic_metric_view(metrics: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "steps_per_second",
        "events_logged_per_second",
        "peak_memory_mb",
        "bytes_per_logged_event",
        "memory_metric",
        "memory_growth_without_recovery",
        "error_message",
    }
    return {key: value for key, value in metrics.items() if key not in excluded}


def stable_object_hash(value: Any) -> str:
    serialized = json.dumps(normalize_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, np.generic):
        return normalize_value(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if pd.isna(value):
        return "NaN"
    return value


def sensitivity_target_metric(knob_name: str) -> str:
    mapping = {
        "limit_rate_scale": "limit_share_mean",
        "market_rate_scale": "events_per_step_market_mean",
        "cancel_rate_scale": "cancel_share_mean",
        "fair_price_vol_scale": "realized_vol_mean",
        "regime_transition_scale": "regime_switch_rate_mean",
        "seasonality_scale": "phase_structure_signal_mean",
        "excitation_scale": "event_clustering_mean",
        "meta_order_scale": "meta_active_directional_ratio_mean",
        "shock_scale": "shock_to_calm_ratio_mean",
    }
    return mapping[knob_name]


def quasi_monotonic(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return False
    diffs = np.diff(finite)
    signs = np.sign(diffs[np.abs(diffs) > EPSILON])
    if len(signs) <= 1:
        return True
    reversals = int(np.sum(signs[1:] < signs[:-1]))
    return reversals <= 1


def preset_compare(frame: pd.DataFrame, left: str, right: str, column: str, *, comparison: str = "gt") -> bool:
    if left not in frame.index or right not in frame.index:
        return False
    left_value = float(frame.loc[left, column])
    right_value = float(frame.loc[right, column])
    if comparison == "ge":
        return bool(left_value >= right_value)
    return bool(left_value > right_value)


def directional_ratio(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    values = values.astype(float)
    return float(abs(values.sum()) / max(len(values), 1))


def safe_mean(values: pd.Series | Sequence[float]) -> float:
    series = pd.Series(values, dtype=float)
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0
    return float(finite.mean())


def safe_std(values: pd.Series | Sequence[float]) -> float:
    series = pd.Series(values, dtype=float)
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 2:
        return 0.0
    return float(finite.std(ddof=0))


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired) < 2:
        return 0.0
    corr = float(paired["left"].corr(paired["right"]))
    return 0.0 if np.isnan(corr) else corr


def safe_autocorr(values: pd.Series | Sequence[float], lag: int) -> float:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) <= lag:
        return 0.0
    corr = float(series.autocorr(lag=lag))
    return 0.0 if np.isnan(corr) else corr


def coefficient_of_variation(mean_value: float, std_value: float) -> float:
    if not np.isfinite(mean_value) or abs(mean_value) <= EPSILON:
        return float("inf") if std_value > 0.0 else 0.0
    return float(abs(std_value) / abs(mean_value))


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_no data_"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def markdown_bullets(values: Mapping[str, Any]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- `{key}`: {value}" for key, value in values.items())


def pass_fail(value: object) -> str:
    return "PASS" if bool(value) else "FAIL"
