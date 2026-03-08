from __future__ import annotations

from orderwave import Market
from orderwave._model.samplers import sample_participant_events


def test_canonical_budget_sampler_returns_valid_event_shapes() -> None:
    market = Market(seed=17, config={"preset": "balanced"})
    features = market._compute_features()
    step_state = market.advance_latent_state(features)

    events = sample_participant_events(
        book=market._book,
        features=step_state.previous_features,
        hidden_fair_tick=step_state.hidden_fair_tick,
        regime=step_state.regime,
        context=step_state.context,
        rng=market._rng,
        config=market.config,
        params=market._params,
    )

    assert all(event["event_type"] in {"limit", "market", "cancel"} for event in events)
    assert all(event["participant_type"] in {"passive_lp", "inventory_mm", "noise_taker", "informed_meta"} for event in events)
    assert all(event["qty"] > 0 for event in events)
