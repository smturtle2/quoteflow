from __future__ import annotations

"""Internal placeholder module.

`orderwave` intentionally exposes only `Market` as its supported public API.
The engine helpers live under the private `orderwave._model` package and are
not re-exported from this module anymore.
"""

__all__: tuple[str, ...] = ()


def __getattr__(name: str) -> object:
    raise AttributeError(
        f"orderwave.model is internal and does not expose {name!r}; "
        "use 'from orderwave import Market'"
    )
