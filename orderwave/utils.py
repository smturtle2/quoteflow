from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

EPSILON = 1e-9


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@lru_cache(maxsize=64)
def infer_price_precision(tick_size: float) -> int:
    text = f"{tick_size:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", maxsplit=1)[1])


def round_price(value: float, tick_size: float) -> float:
    precision = max(6, infer_price_precision(tick_size) + 4)
    return round(float(value), precision)


def price_to_tick(price: float, tick_size: float) -> int:
    return int(round(float(price) / float(tick_size)))


def tick_to_price(tick: float, tick_size: float) -> float:
    return round_price(float(tick) * float(tick_size), tick_size)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return np.full(scores.shape, 1.0 / max(1, scores.size))
    max_score = np.max(scores[finite_mask])
    exps = np.zeros_like(scores, dtype=float)
    exps[finite_mask] = np.exp(scores[finite_mask] - max_score)
    total = exps.sum()
    if total <= 0.0:
        exps[finite_mask] = 1.0
        total = exps.sum()
    return exps / total


def clipped_exp(log_value: float, low: float = -6.0, high: float = 4.0) -> float:
    return math.exp(clamp(log_value, low, high))


def coerce_quantity(value: float) -> int:
    return max(1, int(round(value)))
