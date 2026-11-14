"""Jednotkové testy pre ratingové systémy Elo, Glicko a Glicko-2."""

from __future__ import annotations

import pytest

from src.ratings import (
    START_RATING,
    START_RD,
    EloSystem,
    Glicko2System,
    Glicko2Team,
    GlickoSystem,
    GlickoTeam,
)


def test_elo_winner_gains_and_loser_loses() -> None:
    elo = EloSystem()
    new_h, new_a = elo.update(1500.0, 1500.0, score_home=1.0)
    assert new_h > 1500.0
    assert new_a < 1500.0
    assert (new_h + new_a) == pytest.approx(3000.0)


def test_glicko_uncertainty_decreases_after_match() -> None:
    g1 = GlickoSystem()
    home = GlickoTeam(START_RATING, START_RD)
    away = GlickoTeam(START_RATING, START_RD)
    new_h, new_a = g1.update(home, away, score_home=1.0)
    assert new_h.rating > START_RATING
    assert new_a.rating < START_RATING
    assert new_h.rd < START_RD
    assert new_a.rd < START_RD


def test_glicko2_probabilities_sum_to_one() -> None:
    g2 = Glicko2System()
    home = Glicko2Team(1650.0, 80.0, 0.06)
    away = Glicko2Team(1450.0, 100.0, 0.06)
    p_h, p_d, p_a = g2.predict_proba(home, away)
    assert (p_h + p_d + p_a) == pytest.approx(1.0)
    assert p_h > p_a


def test_glicko2_volatility_remains_positive() -> None:
    g2 = Glicko2System()
    home = Glicko2Team(1500.0, 120.0, 0.06)
    away = Glicko2Team(1800.0, 60.0, 0.05)
    # Prekvapivá výhra outsidera by mala zostať matematicky stabilná
    new_h, new_a = g2.update(home, away, score_home=1.0)
    assert new_h.rating > 1500.0
    assert new_h.vol > 0.0
    assert new_a.vol > 0.0
