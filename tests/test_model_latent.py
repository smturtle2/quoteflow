from __future__ import annotations

import pandas as pd

from orderwave import Market
from orderwave._model.latent import resolve_session_phase, seasonality_multipliers


def test_resolve_session_phase_covers_open_mid_close() -> None:
    assert resolve_session_phase(0, 390)[0] == "open"
    assert resolve_session_phase(120, 390)[0] == "mid"
    assert resolve_session_phase(389, 390)[0] == "close"


def test_advance_latent_state_does_not_mutate_public_outputs() -> None:
    market = Market(seed=42)
    initial_snapshot = market.get()
    initial_history = market.get_history().copy()

    step_state = market._engine._advance_latent_state(market._compute_features())

    assert market._regime in {"calm", "directional", "stressed"}
    assert step_state.context.session_phase in {"open", "mid", "close"}
    assert market.get() == initial_snapshot
    pd.testing.assert_frame_equal(market.get_history(), initial_history)


def test_seasonality_multipliers_are_positive() -> None:
    multipliers = seasonality_multipliers("open", session_progress=0.1, scale=1.0)

    assert set(multipliers) == {"limit", "market", "cancel", "depth", "meta", "shock"}
    assert all(value > 0.0 for value in multipliers.values())
