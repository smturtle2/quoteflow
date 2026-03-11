from __future__ import annotations

from typing import Sequence

import pandas as pd

from .shared import DEFAULT_SENSITIVITY_KNOBS, DEFAULT_SENSITIVITY_SCALES
from .tasks import execute_sensitivity_grid


def run_sensitivity_grid(
    *,
    preset: str,
    seeds: Sequence[int],
    steps: int,
    warmup_fraction: float = 0.10,
    knobs: Sequence[str] = DEFAULT_SENSITIVITY_KNOBS,
    scales: Sequence[float] = DEFAULT_SENSITIVITY_SCALES,
    jobs: int = 1,
) -> pd.DataFrame:
    """Run one-at-a-time sensitivity experiments for one preset."""

    metrics, _ = execute_sensitivity_grid(
        preset=preset,
        seeds=seeds,
        steps=steps,
        warmup_fraction=warmup_fraction,
        knobs=knobs,
        scales=scales,
        jobs=jobs,
    )
    return metrics


__all__ = ["run_sensitivity_grid"]
