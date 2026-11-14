"""Tvorba predzápasových atribútov (feature engineering).

Kľúčové pravidlo: každý atribút zápasu i sa počíta VÝHRADNE zo zápasov,
ktoré sa odohrali pred zápasom i. Nedochádza teda k úniku informácií
z budúcnosti (data leakage).

Vstup:  data/interim/matches.csv
Výstup: data/processed/features.csv
        reports/tables/02_feature_summary.md

Použitie:
    python -m src.features
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import (
    ALL_SEASONS,
    ELO_HOME_ADVANTAGE,
    ELO_K,
    ELO_SEASON_REGRESSION,
    ELO_START,
    FEATURES_CSV,
    H2H_WINDOW,
    MATCHES_CSV,
    ROLLING_WINDOW,
    SEASONS,
    TABLES_DIR,
    TARGET_CLASSES,
    TEAM_STATES_CSV,
    ensure_directories,
)

# ---------------------------------------------------------------------------
# Zoznam atribútov v "long" formáte (jeden riadok = jeden tím v zápase)
# ---------------------------------------------------------------------------
BASE_FEATURES = [
    "form_pts_5",
    "form_pts_venue_5",
    "gf_avg_5",
    "ga_avg_5",
    "gf_venue_avg_5",
    "ga_venue_avg_5",
    "win_rate_5",
    "draw_rate_5",
    "sot_for_avg_5",
    "sot_against_avg_5",
    "shots_avg_5",
    "corners_avg_5",
    "rest_days",
    "season_pts_pg",
    "matches_before",
]

# Atribúty, ktoré existujú len raz na zápas (nie pre každý tím zvlášť).
H2H_FEATURES = ["h2h_pts_H", "h2h_gd_H", "h2h_matches"]

DIFF_FEATURES = [
    "elo_diff",
    "form_diff",
    "venue_form_diff",
    "pts_pg_diff",
    "rest_diff",
    "attack_defense_H",
    "attack_defense_A",
    "sot_adv_H",
    "sot_adv_A",
]

FEATURE_COLUMNS = (
    [f"{feature}_H" for feature in BASE_FEATURES]
    + [f"{feature}_A" for feature in BASE_FEATURES]
    + ["elo_H", "elo_A"]
    + ["is_promoted_H", "is_promoted_A"]
    + H2H_FEATURES
    + DIFF_FEATURES
)

# Hodnoty pre prvé zápasy tímu, keď ešte neexistuje história.
# Sú to rozumné futbalové konštanty (nie sú počítané z budúcich dát).
COLD_START_VALUES = {
    "form_pts_5": 6.75,
    "form_pts_venue_5": 6.75,
    "gf_avg_5": 1.35,
    "ga_avg_5": 1.35,
    "gf_venue_avg_5": 1.45,
    "ga_venue_avg_5": 1.15,
    "win_rate_5": 0.35,
    "draw_rate_5": 0.24,
    "sot_for_avg_5": 4.5,
    "sot_against_avg_5": 4.5,
    "shots_avg_5": 12.5,
    "corners_avg_5": 10.0,
    "rest_days": 7.0,
    "season_pts_pg": 1.35,
}

# Stĺpce, ktoré nesú kontext zápasu a nie sú vstupom modelu.
CONTEXT_COLUMNS = [
    "match_id",
    "season",
    "date",
    "time",
    "home_team",
    "away_team",
    "fthg",
    "ftag",
    "ftr",
    "target",
    "odds_h",
    "odds_d",
    "odds_a",
    "implied_h",
    "implied_d",
    "implied_a",
    "bookmaker_favorite",
    "has_odds",
    "source_file",
]


def load_matches() -> pd.DataFrame:
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(
            f"Chyba {MATCHES_CSV}. Najprv spusti: python -m src.prepare"
        )
    matches = pd.read_csv(MATCHES_CSV, parse_dates=["date"])
    matches["season"] = matches["season"].astype(str)
    matches["time"] = matches["time"].fillna("").astype(str)
    return matches


def to_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Prevedie zápasy na dva riadky (domáci a hosťujúci pohľad)."""
    common = ["match_id", "season", "date", "time"]
    home = matches[common + ["home_team", "away_team", "fthg", "ftag"]].copy()
    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "fthg": "goals_for",
            "ftag": "goals_against",
        }
    )
    home["is_home"] = True

    away = matches[common + ["away_team", "home_team", "ftag", "fthg"]].copy()
    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "ftag": "goals_for",
            "fthg": "goals_against",
        }
    )
    away["is_home"] = False

    stats = matches[
        common
        + [
            "home_shots",
            "away_shots",
            "home_sot",
            "away_sot",
            "home_corners",
            "away_corners",
        ]
    ]
    home = home.merge(
        stats.rename(
            columns={
                "home_shots": "shots_for",
                "away_shots": "shots_against",
                "home_sot": "sot_for",
                "away_sot": "sot_against",
                "home_corners": "corners_for",
                "away_corners": "corners_against",
            }
        ),
        on=common,
        how="left",
    )
    away = away.merge(
        stats.rename(
            columns={
                "away_shots": "shots_for",
                "home_shots": "shots_against",
                "away_sot": "sot_for",
                "home_sot": "sot_against",
                "away_corners": "corners_for",
                "home_corners": "corners_against",
            }
        ),
        on=common,
        how="left",
    )

    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.select(
        [long["goals_for"] > long["goals_against"], long["goals_for"] == long["goals_against"]],
        [3, 1],
        default=0,
    )
    long["win"] = (long["goals_for"] > long["goals_against"]).astype(int)
    long["draw"] = (long["goals_for"] == long["goals_against"]).astype(int)
    long["goal_diff"] = long["goals_for"] - long["goals_against"]

    long = long.sort_values(["team", "date", "time", "match_id"]).reset_index(drop=True)
    return long


def _mean_shifted(series: pd.Series) -> pd.Series:
    return series.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()


def _sum_shifted(series: pd.Series) -> pd.Series:
    return series.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum()


def add_rolling_features(long: pd.DataFrame) -> pd.DataFrame:
    """Kĺzavé štatistiky z posledných 5 zápasov (celkovo aj podľa ihriska)."""
    long = long.copy()
    grouped = long.groupby("team", group_keys=False)

    long["form_pts_5"] = grouped["points"].transform(_sum_shifted)
    long["gf_avg_5"] = grouped["goals_for"].transform(_mean_shifted)
    long["ga_avg_5"] = grouped["goals_against"].transform(_mean_shifted)
    long["win_rate_5"] = grouped["win"].transform(_mean_shifted)
    long["draw_rate_5"] = grouped["draw"].transform(_mean_shifted)
    long["sot_for_avg_5"] = grouped["sot_for"].transform(_mean_shifted)
    long["sot_against_avg_5"] = grouped["sot_against"].transform(_mean_shifted)
    long["shots_avg_5"] = grouped["shots_for"].transform(_mean_shifted)
    long["corners_avg_5"] = grouped["corners_for"].transform(_mean_shifted)

    # Forma len z domácich / len z vonkajších zápasov.
    for is_home, suffix in ((True, "home"), (False, "away")):
        subset = long[long["is_home"] == is_home].sort_values(
            ["team", "date", "time", "match_id"]
        )
        subset_grouped = subset.groupby("team", group_keys=False)

        venue_form = subset_grouped["points"].transform(_sum_shifted)
        venue_gf = subset_grouped["goals_for"].transform(_mean_shifted)
        venue_ga = subset_grouped["goals_against"].transform(_mean_shifted)

        long.loc[subset.index, "form_pts_venue_5"] = venue_form
        long.loc[subset.index, "gf_venue_avg_5"] = venue_gf
        long.loc[subset.index, "ga_venue_avg_5"] = venue_ga

    # Dni od posledného zápasu (bez shiftu – aktuálny dátum je známy pred výkopom).
    long["rest_days"] = (
        long.groupby("team")["date"].diff().dt.days.fillna(7).clip(lower=0, upper=30)
    )

    # Body a zápasy v aktuálnej sezóne pred týmto zápasom.
    long["season_points_before"] = (
        long.groupby(["team", "season"])["points"].cumsum() - long["points"]
    )
    long["matches_before"] = long.groupby(["team", "season"]).cumcount()
    long["season_pts_pg"] = np.where(
        long["matches_before"] > 0,
        long["season_points_before"] / long["matches_before"].clip(lower=1),
        np.nan,
    )

    return long


def add_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """Vzájomné zápasy – len tie, ktoré sa odohrali pred daným zápasom."""
    history: dict[tuple[str, str], list[tuple[str, str, int, int]]] = defaultdict(list)
    rows: list[dict[str, object]] = []

    for row in matches.sort_values(["date", "time", "match_id"]).itertuples(index=False):
        key = tuple(sorted((row.home_team, row.away_team)))
        past = history[key][-H2H_WINDOW:]

        points = 0
        goal_diff = 0
        for past_home, _past_away, past_hg, past_ag in past:
            if past_home == row.home_team:
                goals_for, goals_against = past_hg, past_ag
            else:
                goals_for, goals_against = past_ag, past_hg
            points += 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
            goal_diff += goals_for - goals_against

        rows.append(
            {
                "match_id": row.match_id,
                "h2h_pts_H": points,
                "h2h_gd_H": goal_diff,
                "h2h_matches": len(past),
            }
        )
        history[key].append((row.home_team, row.away_team, row.fthg, row.ftag))

    return pd.DataFrame(rows)


def add_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Elo ratingy aktualizované po každom zápase (pred zápasom sú známe)."""
    ratings: dict[str, float] = defaultdict(lambda: ELO_START)
    rows: list[dict[str, object]] = []
    current_season: str | None = None

    for row in matches.sort_values(["date", "time", "match_id"]).itertuples(index=False):
        # Regresia k priemeru na hranici sezóny – používa len minulé ratingy.
        if row.season != current_season:
            if current_season is not None:
                for team, rating in ratings.items():
                    ratings[team] = ELO_START + ELO_SEASON_REGRESSION * (rating - ELO_START)
            current_season = row.season

        home_rating = ratings[row.home_team]
        away_rating = ratings[row.away_team]

        rows.append(
            {
                "match_id": row.match_id,
                "elo_H": round(home_rating, 4),
                "elo_A": round(away_rating, 4),
            }
        )

        expected_home = 1.0 / (
            1.0 + 10.0 ** (-((home_rating + ELO_HOME_ADVANTAGE) - away_rating) / 400.0)
        )
        if row.fthg > row.ftag:
            score_home = 1.0
        elif row.fthg == row.ftag:
            score_home = 0.5
        else:
            score_home = 0.0

        ratings[row.home_team] = home_rating + ELO_K * (score_home - expected_home)
        ratings[row.away_team] = away_rating + ELO_K * ((1.0 - score_home) - (1.0 - expected_home))

    return pd.DataFrame(rows)


def promoted_teams(matches: pd.DataFrame) -> dict[str, set[str]]:
    """Tímy, ktoré v danej sezóne nie sú v predchádzajúcej sezóne."""
    teams_by_season = {
        season: set(group["home_team"]) | set(group["away_team"])
        for season, group in matches.groupby("season")
    }
    promoted: dict[str, set[str]] = {}
    previous: set[str] | None = None
    for season in ALL_SEASONS:
        if season not in teams_by_season:
            continue
        current = teams_by_season[season]
        promoted[season] = set() if previous is None else current - previous
        previous = current
    return promoted


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    long = to_long_format(matches)
    long = add_rolling_features(long)

    h2h = add_h2h(matches)
    elo = add_elo(matches)
    promoted = promoted_teams(matches)

    # Domáci a hosťujúci riadok long formátu.
    home_rows = long[long["is_home"]].set_index("match_id")
    away_rows = long[~long["is_home"]].set_index("match_id")

    features = matches.copy()
    features["match_id"] = features["match_id"].astype(str)

    for column in BASE_FEATURES:
        features[f"{column}_H"] = features["match_id"].map(home_rows[column])
        features[f"{column}_A"] = features["match_id"].map(away_rows[column])

    features["is_promoted_H"] = features.apply(
        lambda row: int(row["home_team"] in promoted.get(row["season"], set())), axis=1
    )
    features["is_promoted_A"] = features.apply(
        lambda row: int(row["away_team"] in promoted.get(row["season"], set())), axis=1
    )

    features = features.merge(h2h, on="match_id", how="left")
    features = features.merge(elo, on="match_id", how="left")

    # ------------------------------------------------------------------
    # Doplnenie chýbajúcich hodnôt pre prvé zápasy (studený štart)
    # ------------------------------------------------------------------
    for column in BASE_FEATURES:
        if column in COLD_START_VALUES:
            fill = COLD_START_VALUES[column]
            for side in ("H", "A"):
                features[f"{column}_{side}"] = features[f"{column}_{side}"].fillna(fill)

    for column in H2H_FEATURES:
        features[column] = features[column].fillna(0)

    # ------------------------------------------------------------------
    # Odvodené rozdiely (pomáhajú hlavne logistickej regresii)
    # ------------------------------------------------------------------
    features["elo_diff"] = features["elo_H"] - features["elo_A"]
    features["form_diff"] = features["form_pts_5_H"] - features["form_pts_5_A"]
    features["venue_form_diff"] = (
        features["form_pts_venue_5_H"] - features["form_pts_venue_5_A"]
    )
    features["pts_pg_diff"] = features["season_pts_pg_H"] - features["season_pts_pg_A"]
    features["rest_diff"] = features["rest_days_H"] - features["rest_days_A"]

    # Porovnanie útoku jedného tímu s obranou druhého.
    features["attack_defense_H"] = features["gf_avg_5_H"] - features["ga_avg_5_A"]
    features["attack_defense_A"] = features["gf_avg_5_A"] - features["ga_avg_5_H"]
    features["sot_adv_H"] = features["sot_for_avg_5_H"] - features["sot_against_avg_5_A"]
    features["sot_adv_A"] = features["sot_for_avg_5_A"] - features["sot_against_avg_5_H"]

    # ------------------------------------------------------------------
    # Cieľová premenná a kurzové pravdepodobnosti (len na vyhodnotenie)
    # ------------------------------------------------------------------
    features["target"] = features["ftr"].map({label: i for i, label in enumerate(TARGET_CLASSES)})

    def add_implied(frame: pd.DataFrame, columns: list[str], suffix: str) -> None:
        inverse = 1.0 / frame[columns]
        total = inverse.sum(axis=1)
        for label, column in zip(("h", "d", "a"), columns):
            frame[f"implied_{label}{suffix}"] = inverse[column] / total

    add_implied(features, ["odds_h", "odds_d", "odds_a"], "")
    add_implied(features, ["ps_h", "ps_d", "ps_a"], "_ps")

    implied = features[["implied_h", "implied_d", "implied_a"]]
    features["bookmaker_favorite"] = implied.idxmax(axis=1).map(
        {"implied_h": "H", "implied_d": "D", "implied_a": "A"}
    )
    features.loc[implied.isna().any(axis=1), "bookmaker_favorite"] = np.nan

    # Kontrola: všetky vstupné atribúty musia byť vyplnené.
    missing = features[FEATURE_COLUMNS].isna().sum()
    if missing.any():
        raise RuntimeError(
            "Chybajuce hodnoty v atributoch:\n"
            + missing[missing > 0].to_string()
        )

    features = features.sort_values(["season", "date", "time", "match_id"]).reset_index(drop=True)
    return features


def build_team_states(features: pd.DataFrame) -> pd.DataFrame:
    """Stav každého tímu pred jeho posledným zápasom v sezóne.

    Slúži aplikácii na predikciu ľubovoľnej dvojice tímov: domáci tím dodá
    svoje domáce atribúty, hosťujúci svoje vonkajšie.
    """
    order = ["season", "date", "time", "match_id"]
    last_home = (
        features.sort_values(order).groupby(["season", "home_team"], as_index=False).tail(1)
    )
    last_away = (
        features.sort_values(order).groupby(["season", "away_team"], as_index=False).tail(1)
    )

    def extract(frame: pd.DataFrame, side: str, team_column: str, venue: str) -> pd.DataFrame:
        suffix = f"_{side}"
        state = pd.DataFrame(
            {
                "season": frame["season"].values,
                "team": frame[team_column].values,
                "venue": venue,
                "as_of_date": frame["date"].values,
                "matches_played": frame[f"matches_before{suffix}"].values,
                "is_promoted": frame[f"is_promoted{suffix}"].values,
            }
        )
        for column in BASE_FEATURES:
            state[column] = frame[f"{column}{suffix}"].values
        state["elo"] = frame[f"elo{suffix}"].values
        return state

    states = pd.concat(
        [
            extract(last_home, "H", "home_team", "home"),
            extract(last_away, "A", "away_team", "away"),
        ],
        ignore_index=True,
    )
    states["as_of_date"] = pd.to_datetime(states["as_of_date"])
    return states.sort_values(["season", "team", "venue"]).reset_index(drop=True)


def build_feature_report(features: pd.DataFrame) -> str:
    """Markdown prehľad atribútov pre prácu (tabuľka do kapitoly 4.2)."""
    lines = [
        "# Prehľad vytvorených atribútov",
        "",
        f"- Počet zápasov: **{len(features)}**",
        f"- Počet vstupných atribútov modelu: **{len(FEATURE_COLUMNS)}**",
        f"- Sezóny: {features['season'].min()} – {features['season'].max()}",
        "",
        "## Zoznam atribútov",
        "",
        "| Atribút | Priemer | Smer. odchýlka | Min | Max |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for column in FEATURE_COLUMNS:
        series = features[column]
        lines.append(
            f"| `{column}` | {series.mean():.3f} | {series.std():.3f} | "
            f"{series.min():.3f} | {series.max():.3f} |"
        )

    lines += [
        "",
        "## Kontrola úniku budúcnosti",
        "",
        "Všetky kĺzavé atribúty sú počítané cez `shift(1)`, takže aktuálny zápas "
        "nikdy nevstupuje do svojich vlastných vstupov. Elo rating a vzájomné "
        "zápasy sa aktualizujú až po vyhodnotení zápasu. Testy v `tests/test_features.py` "
        "overujú, že odstránenie budúceho zápasu nezmení atribúty minulých zápasov.",
        "",
        "## Rozdelenie tried",
        "",
        "| Trieda | Počet | Podiel |",
        "| :--- | ---: | ---: |",
    ]
    distribution = features["ftr"].value_counts()
    for label in TARGET_CLASSES:
        count = int(distribution.get(label, 0))
        lines.append(f"| {label} | {count} | {count / len(features):.1%} |")

    lines += [
        "",
        "## Priemerné hodnoty podľa ihriska",
        "",
        "| Ukazovateľ | Domáci | Hostia |",
        "| :--- | ---: | ---: |",
    ]
    for column in ("form_pts_5", "gf_avg_5", "ga_avg_5", "elo", "rest_days"):
        lines.append(
            f"| {column} | {features[f'{column}_H'].mean():.2f} | "
            f"{features[f'{column}_A'].mean():.2f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    ensure_directories()
    matches = load_matches()
    features = build_features(matches)
    states = build_team_states(features)

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    features[CONTEXT_COLUMNS + FEATURE_COLUMNS].to_csv(
        FEATURES_CSV, index=False, encoding="utf-8"
    )
    states.to_csv(TEAM_STATES_CSV, index=False, encoding="utf-8")

    report = build_feature_report(features)
    report_path = TABLES_DIR / "02_feature_summary.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[ok] {len(features)} zapasov, {len(FEATURE_COLUMNS)} atributov")
    print(f"[ok] -> {FEATURES_CSV}")
    print(f"[ok] stavy timov -> {TEAM_STATES_CSV} ({len(states)} riadkov)")
    print(f"[ok] report -> {report_path}")
    print(f"[ok] triedy: {features['ftr'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
