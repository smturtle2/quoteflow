from __future__ import annotations

"""Shared numeric helpers for ``orderwave``."""

from functools import lru_cache


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


def compute_depth_imbalance(bid_depth: int, ask_depth: int) -> float:
    total = bid_depth + ask_depth
    if total <= 0:
        return 0.0
    return (float(bid_depth) - float(ask_depth)) / float(total)


def bounded_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(value))))


__all__ = [
    "bounded_int",
    "clamp",
    "compute_depth_imbalance",
    "infer_price_precision",
    "price_to_tick",
    "round_price",
    "tick_to_price",
]
