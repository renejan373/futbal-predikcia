"""Testy pre feature engineering a ochranu proti úniku budúcnosti."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MATCHES_CSV, SEASONS, TARGET_CLASSES
from src.features import (
    BASE_FEATURES,
    FEATURE_COLUMNS,
    build_features,
    build_team_states,
    load_matches,
    promoted_teams,
)

pytestmark = pytest.mark.skipif(
    not MATCHES_CSV.exists(),
    reason="Chyba data/interim/matches.csv – spusti python -m src.prepare",
)


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return load_matches()


@pytest.fixture(scope="module")
def features(matches: pd.DataFrame) -> pd.DataFrame:
    return build_features(matches)


def _points_for(matches: pd.DataFrame, team: str, limit: int = 5) -> int:
    """Body tímu z posledných `limit` zápasov pred koncom datasetu."""
    rows = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    rows = rows.sort_values(["date", "time"]).tail(limit)
    points = 0
    for row in rows.itertuples(index=False):
        if row.home_team == team:
            goals_for, goals_against = row.fthg, row.ftag
        else:
            goals_for, goals_against = row.ftag, row.fthg
        points += 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
    return points


def test_row_counts_and_target(features: pd.DataFrame) -> None:
    assert len(features) == 1900
    assert features["match_id"].is_unique
    assert set(features["target"].unique()) == set(range(len(TARGET_CLASSES)))

    per_season = features.groupby("season").size()
    assert list(per_season.index) == SEASONS
    assert (per_season == 380).all()


def test_no_missing_features(features: pd.DataFrame) -> None:
    missing = features[FEATURE_COLUMNS].isna().sum()
    assert not missing.any(), missing[missing > 0].to_dict()


def test_feature_ranges(features: pd.DataFrame) -> None:
    assert features["form_pts_5_H"].between(0, 15).all()
    assert features["form_pts_5_A"].between(0, 15).all()
    assert features["win_rate_5_H"].between(0, 1).all()
    assert features["rest_days_H"].between(0, 30).all()
    assert (features["gf_avg_5_H"] >= 0).all()
    assert (features["h2h_matches"] <= 5).all()


def test_promoted_teams(matches: pd.DataFrame) -> None:
    promoted = promoted_teams(matches)
    assert promoted["2021/22"] == set()
    assert promoted["2022/23"] == {"Bournemouth", "Fulham", "Nott'm Forest"}
    assert promoted["2025/26"] == {"Burnley", "Leeds", "Sunderland"}


def test_team_states(features: pd.DataFrame) -> None:
    states = build_team_states(features)
    assert len(states) == len(SEASONS) * 20 * 2
    for season in SEASONS:
        subset = states[states["season"] == season]
        assert subset["team"].nunique() == 20
        assert set(subset["venue"]) == {"home", "away"}
    assert not states[BASE_FEATURES].isna().any().any()
    assert states["matches_played"].between(0, 37).all()


def test_form_points_match_manual_calculation(
    matches: pd.DataFrame, features: pd.DataFrame
) -> None:
    for row in features.iloc[[600, 1200, 1800]].itertuples(index=False):
        history = matches[matches["date"] < row.date]
        expected_home = _points_for(history, row.home_team)
        expected_away = _points_for(history, row.away_team)
        assert row.form_pts_5_H == pytest.approx(expected_home)
        assert row.form_pts_5_A == pytest.approx(expected_away)


def test_h2h_counts_only_past_matches(
    matches: pd.DataFrame, features: pd.DataFrame
) -> None:
    row = features[features["h2h_matches"] > 0].iloc[0]
    prior = matches[
        (matches["date"] < row.date)
        & (
            ((matches["home_team"] == row.home_team) & (matches["away_team"] == row.away_team))
            | ((matches["home_team"] == row.away_team) & (matches["away_team"] == row.home_team))
        )
    ]
    assert row.h2h_matches == min(5, len(prior))


def test_removing_last_match_does_not_change_history(
    matches: pd.DataFrame, features: pd.DataFrame
) -> None:
    trimmed = matches.iloc[:-1].copy()
    trimmed_features = build_features(trimmed)

    common_ids = set(trimmed_features["match_id"])
    expected = features[features["match_id"].isin(common_ids)].reset_index(drop=True)
    actual = trimmed_features.reset_index(drop=True)

    pd.testing.assert_frame_equal(
        expected[FEATURE_COLUMNS],
        actual[FEATURE_COLUMNS],
        check_dtype=False,
    )


def test_changing_result_does_not_affect_past_or_current_match(
    matches: pd.DataFrame, features: pd.DataFrame
) -> None:
    """Zmena výsledku zápasu nesmie ovplyvniť minulosť ani vstupy daného zápasu."""
    index = 700
    modified = matches.copy()
    original_result = modified.loc[index, "ftr"]
    if original_result == "H":
        modified.loc[index, ["fthg", "ftag", "ftr"]] = [0, 9, "A"]
    else:
        modified.loc[index, ["fthg", "ftag", "ftr"]] = [9, 0, "H"]
    modified_features = build_features(modified)

    changed = features.iloc[index]
    same_match = modified_features[
        modified_features["match_id"] == changed["match_id"]
    ].iloc[0]
    for column in FEATURE_COLUMNS:
        assert same_match[column] == pytest.approx(changed[column]), column

    earlier = features["date"] < changed["date"]
    earlier_ids = set(features.loc[earlier, "match_id"])
    expected = features[features["match_id"].isin(earlier_ids)].reset_index(drop=True)
    actual = modified_features[
        modified_features["match_id"].isin(earlier_ids)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        expected[FEATURE_COLUMNS],
        actual[FEATURE_COLUMNS],
        check_dtype=False,
    )

    # Budúce zápasy zasiahnutých tímov už zmenené byť majú.
    future = modified_features["date"] > changed["date"]
    teams = {changed["home_team"], changed["away_team"]}
    involved = modified_features["home_team"].isin(teams) | modified_features[
        "away_team"
    ].isin(teams)
    assert future.sum() > 0 and involved.sum() > 0

    future_home_form = modified_features.loc[future & involved, "form_pts_5_H"]
    original_home_form = features.loc[future & involved, "form_pts_5_H"]
    assert not np.allclose(future_home_form, original_home_form)
