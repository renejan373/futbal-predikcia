"""Výpočet a vyhodnotenie ratingových systémov Elo, Glicko a Glicko-2.

Prejde chronologicky všetky zápasy v `data/interim/matches.csv`, pred každým
zápasom zaznamená predikciu troch systémov a následne upraví ratingy podľa výsledku.

Výstupy:
    data/processed/rating_history.csv
    data/processed/current_ratings.csv
    reports/tables/rating_comparison.csv
    reports/tables/rating_results.md
    reports/figures/01_rating_accuracy.png .. 04_rating_confusion_matrix.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss

from src.config import (
    ELO_SEASON_REGRESSION,
    FIGURES_DIR,
    LEAGUES,
    MATCHES_CSV,
    PROCESSED_DIR,
    TABLES_DIR,
    TARGET_CLASSES,
    TEST_SEASON,
    ensure_directories,
    get_league_paths,
)
from src.ratings import (
    START_RATING,
    START_RD,
    EloSystem,
    Glicko2System,
    Glicko2Team,
    GlickoSystem,
    GlickoTeam,
)

RATING_HISTORY_CSV = PROCESSED_DIR / "rating_history.csv"
CURRENT_RATINGS_CSV = PROCESSED_DIR / "current_ratings.csv"
RATING_METADATA_JSON = PROCESSED_DIR / "rating_metadata.json"

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10})


def _score_from_ftr(ftr: str) -> float:
    return 1.0 if ftr == "H" else 0.5 if ftr == "D" else 0.0


def _target_index(ftr: str) -> int:
    return TARGET_CLASSES.index(ftr)


def compute_all_ratings(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologicky prepočíta Elo, Glicko a Glicko-2 pre všetky zápasy."""
    ordered = matches.sort_values(["season", "date", "time", "match_id"]).reset_index(drop=True)
    ordered["date"] = pd.to_datetime(ordered["date"])

    elo_sys = EloSystem()
    g1_sys = GlickoSystem()
    g2_sys = Glicko2System()

    elo_state: dict[str, float] = {}
    g1_state: dict[str, GlickoTeam] = {}
    g2_state: dict[str, Glicko2Team] = {}
    last_played: dict[str, pd.Timestamp] = {}
    matches_in_season: dict[str, int] = {}

    history_rows: list[dict] = []
    current_rows: list[dict] = []

    prev_season: str | None = None

    def flush_season_ratings(season_name: str) -> None:
        all_teams = sorted(set(matches_in_season.keys()))
        for t in all_teams:
            g1 = g1_state[t]
            g2 = g2_state[t]
            current_rows.append(
                {
                    "season": season_name,
                    "team": t,
                    "matches_played": matches_in_season[t],
                    "as_of_date": last_played[t],
                    "elo_rating": round(elo_state[t], 1),
                    "glicko_rating": round(g1.rating, 1),
                    "glicko_rd": round(g1.rd, 1),
                    "glicko2_rating": round(g2.rating, 1),
                    "glicko2_rd": round(g2.rd, 1),
                    "glicko2_vol": round(g2.vol, 4),
                }
            )

    for row in ordered.itertuples(index=False):
        if prev_season is not None and row.season != prev_season:
            # Uložíme koncové ratingy predchádzajúcej sezóny
            flush_season_ratings(prev_season)
            matches_in_season.clear()

            # Medzisezónna regresia k priemeru (1500) a nárast neistoty
            for t in list(elo_state.keys()):
                elo_state[t] = START_RATING + ELO_SEASON_REGRESSION * (elo_state[t] - START_RATING)
                g1_old = g1_state[t]
                g1_state[t] = GlickoTeam(
                    rating=START_RATING + ELO_SEASON_REGRESSION * (g1_old.rating - START_RATING),
                    rd=min(START_RD, math.sqrt(g1_old.rd ** 2 + 60.0 ** 2)) if hasattr(g1_old, "rd") else START_RD,
                )
                g2_old = g2_state[t]
                g2_state[t] = Glicko2Team(
                    rating=START_RATING + ELO_SEASON_REGRESSION * (g2_old.rating - START_RATING),
                    rd=min(START_RD, math.sqrt(g2_old.rd ** 2 + 60.0 ** 2)),
                    vol=g2_old.vol,
                )

        prev_season = row.season
        h, a = row.home_team, row.away_team

        # Inicializácia nových tímov
        for t in (h, a):
            if t not in elo_state:
                elo_state[t] = START_RATING
                g1_state[t] = GlickoTeam()
                g2_state[t] = Glicko2Team()
            matches_in_season[t] = matches_in_season.get(t, 0)

        # Inaktivita v dňoch pred týmto zápasom
        days_h = (row.date - last_played[h]).days if h in last_played else 0
        days_a = (row.date - last_played[a]).days if a in last_played else 0
        g1_h = g1_sys.apply_inactivity(g1_state[h], min(days_h, 60))
        g1_a = g1_sys.apply_inactivity(g1_state[a], min(days_a, 60))
        g2_h = g2_sys.apply_inactivity(g2_state[h], min(days_h, 60))
        g2_a = g2_sys.apply_inactivity(g2_state[a], min(days_a, 60))

        # Predikcie PRED zápasom
        elo_p = elo_sys.predict_proba(elo_state[h], elo_state[a])
        g1_p = g1_sys.predict_proba(g1_h, g1_a)
        g2_p = g2_sys.predict_proba(g2_h, g2_a)

        history_rows.append(
            {
                "match_id": row.match_id,
                "season": row.season,
                "date": row.date,
                "home_team": h,
                "away_team": a,
                "fthg": row.fthg,
                "ftag": row.ftag,
                "ftr": row.ftr,
                "target": _target_index(row.ftr),
                "odds_h": getattr(row, "odds_h", np.nan),
                "odds_d": getattr(row, "odds_d", np.nan),
                "odds_a": getattr(row, "odds_a", np.nan),
                "elo_H": elo_state[h],
                "elo_A": elo_state[a],
                "elo_pH": elo_p[0],
                "elo_pD": elo_p[1],
                "elo_pA": elo_p[2],
                "g1_H": g1_h.rating,
                "g1_A": g1_a.rating,
                "g1_rd_H": g1_h.rd,
                "g1_rd_A": g1_a.rd,
                "g1_pH": g1_p[0],
                "g1_pD": g1_p[1],
                "g1_pA": g1_p[2],
                "g2_H": g2_h.rating,
                "g2_A": g2_a.rating,
                "g2_rd_H": g2_h.rd,
                "g2_rd_A": g2_a.rd,
                "g2_vol_H": g2_h.vol,
                "g2_vol_A": g2_a.vol,
                "g2_pH": g2_p[0],
                "g2_pD": g2_p[1],
                "g2_pA": g2_p[2],
            }
        )

        # Aktualizácia PO zápase
        score_h = _score_from_ftr(row.ftr)
        elo_state[h], elo_state[a] = elo_sys.update(elo_state[h], elo_state[a], score_h)
        g1_state[h], g1_state[a] = g1_sys.update(g1_h, g1_a, score_h)
        g2_state[h], g2_state[a] = g2_sys.update(g2_h, g2_a, score_h)

        last_played[h] = row.date
        last_played[a] = row.date
        matches_in_season[h] += 1
        matches_in_season[a] += 1

    if prev_season is not None:
        flush_season_ratings(prev_season)

    return pd.DataFrame(history_rows), pd.DataFrame(current_rows)


def evaluate_models(history: pd.DataFrame, season: str = TEST_SEASON) -> pd.DataFrame:
    """Vyhodnotí Elo, Glicko, Glicko-2, baseline a kurzy na testovacej sezóne."""
    sub = history[history["season"] == season].copy()
    y_true = sub["target"].to_numpy()

    results = []

    def score_probs(name: str, model_type: str, probs: np.ndarray) -> None:
        preds = np.argmax(probs, axis=1)
        brier = float(
            np.mean(
                [
                    brier_score_loss((y_true == c).astype(int), probs[:, c])
                    for c in range(3)
                ]
            )
            * 3.0
        )
        results.append(
            {
                "model": name,
                "typ": model_type,
                "accuracy": round(float(accuracy_score(y_true, preds)), 4),
                "f1_macro": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 4),
                "log_loss": round(float(log_loss(y_true, probs, labels=[0, 1, 2])), 4),
                "brier": round(brier, 4),
            }
        )

    # 1. Baseline: Vždy výhra domácich
    base_probs = np.tile([0.998, 0.001, 0.001], (len(sub), 1))
    score_probs("Vždy výhra domácich", "baseline", base_probs)

    # 2. Ratingové systémy
    score_probs("Elo", "rating", sub[["elo_pH", "elo_pD", "elo_pA"]].to_numpy())
    score_probs("Glicko", "rating", sub[["g1_pH", "g1_pD", "g1_pA"]].to_numpy())
    score_probs("Glicko-2", "rating", sub[["g2_pH", "g2_pD", "g2_pA"]].to_numpy())

    # 3. Kurzy Bet365
    odds_valid = sub.dropna(subset=["odds_h", "odds_d", "odds_a"]).copy()
    if not odds_valid.empty:
        inv = 1.0 / odds_valid[["odds_h", "odds_d", "odds_a"]].to_numpy()
        market_probs = inv / inv.sum(axis=1, keepdims=True)
        y_odds = odds_valid["target"].to_numpy()
        preds_odds = np.argmax(market_probs, axis=1)
        brier_odds = float(
            np.mean(
                [
                    brier_score_loss((y_odds == c).astype(int), market_probs[:, c])
                    for c in range(3)
                ]
            )
            * 3.0
        )
        results.append(
            {
                "model": "Kurzy Bet365",
                "typ": "trh",
                "accuracy": round(float(accuracy_score(y_odds, preds_odds)), 4),
                "f1_macro": round(float(f1_score(y_odds, preds_odds, average="macro", zero_division=0)), 4),
                "log_loss": round(float(log_loss(y_odds, market_probs, labels=[0, 1, 2])), 4),
                "brier": round(brier_odds, 4),
            }
        )

    return pd.DataFrame(results)


def generate_charts(history: pd.DataFrame, comparison: pd.DataFrame) -> None:
    """Vygeneruje prehľadné grafy pre prácu SOČ."""
    # 1. Porovnanie presnosti a Log loss
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#95a5a6", "#2a9d8f", "#e9c46a", "#264653", "#e76f51"]
    axes[0].bar(comparison["model"], comparison["accuracy"] * 100, color=colors[: len(comparison)])
    axes[0].set_title("Presnosť predikcie na sezóne 2025/26 (%)")
    axes[0].set_ylabel("Presnosť (%)")
    axes[0].set_ylim(0, 65)
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(comparison["accuracy"] * 100):
        axes[0].text(i, v + 1.2, f"{v:.1f}%", ha="center", fontweight="bold")

    sub_ll = comparison[comparison["typ"] != "baseline"]
    axes[1].bar(sub_ll["model"], sub_ll["log_loss"], color=colors[1 : 1 + len(sub_ll)])
    axes[1].set_title("Kvalita pravdepodobností – Log loss (menej je lepšie)")
    axes[1].set_ylabel("Log loss")
    axes[1].set_ylim(0.9, 1.15)
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(sub_ll["log_loss"]):
        axes[1].text(i, v + 0.008, f"{v:.3f}", ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_rating_accuracy.png")
    plt.close(fig)

    # 2. Vývoj Glicko-2 ratingu top tímov v sezóne 2025/26
    sub = history[history["season"] == TEST_SEASON].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for team, color in [
        ("Arsenal", "#e63946"),
        ("Man City", "#457b9d"),
        ("Liverpool", "#2a9d8f"),
        ("Chelsea", "#1d3557"),
    ]:
        h_rows = sub[sub["home_team"] == team][["date", "g2_H"]].rename(columns={"g2_H": "rating"})
        a_rows = sub[sub["away_team"] == team][["date", "g2_A"]].rename(columns={"g2_A": "rating"})
        t_df = pd.concat([h_rows, a_rows]).sort_values("date")
        ax.plot(t_df["date"], t_df["rating"], label=team, linewidth=2, color=color)

    ax.set_title(f"Vývoj sily vybraných tímov (Glicko-2) v sezóne {TEST_SEASON}")
    ax.set_ylabel("Glicko-2 rating")
    ax.set_xlabel("Dátum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_rating_evolution.png")
    plt.close(fig)

    # 3. Vývoj neistoty (RD) nováčik vs. stabilný tím
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for team, label, color in [
        ("Burnley", "Burnley (nováčik)", "#e76f51"),
        ("Arsenal", "Arsenal (etablovaný tím)", "#264653"),
    ]:
        h_rows = sub[sub["home_team"] == team][["date", "g2_rd_H"]].rename(columns={"g2_rd_H": "rd"})
        a_rows = sub[sub["away_team"] == team][["date", "g2_rd_A"]].rename(columns={"g2_rd_A": "rd"})
        t_df = pd.concat([h_rows, a_rows]).sort_values("date").reset_index(drop=True)
        ax.plot(range(1, len(t_df) + 1), t_df["rd"], label=label, linewidth=2.2, color=color)

    ax.set_title(f"Pokles neistoty (RD) v systéme Glicko-2 počas sezóny {TEST_SEASON}")
    ax.set_xlabel("Poradie zápasu v sezóne")
    ax.set_ylabel("Neistota ratingu (RD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_glicko_uncertainty.png")
    plt.close(fig)

    # 4. Matica zámen pre Glicko-2
    y_true = sub["target"].to_numpy()
    y_pred = np.argmax(sub[["g2_pH", "g2_pD", "g2_pA"]].to_numpy(), axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Domáci", "Remíza", "Hostia"],
        yticklabels=["Domáci", "Remíza", "Hostia"],
        ax=ax,
    )
    ax.set_title("Matica zámen – Glicko-2 (sezóna 2025/26)")
    ax.set_xlabel("Predikovaný výsledok")
    ax.set_ylabel("Skutočný výsledok")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_rating_confusion_matrix.png")
    plt.close(fig)

    # 5. Graf prvej sezóny: Elo vs. Glicko vs. Glicko-2 (s využitím NumPy)
    generate_first_season_chart(history)


def generate_first_season_chart(history: pd.DataFrame) -> Path:
    """Vykreslí porovnanie Elo, Glicko a Glicko-2 pre najlepší tím prvej sezóny pomocou NumPy."""
    first_season = str(history["season"].min())
    sub = history[history["season"] == first_season].sort_values("date").copy()

    last_home = sub.groupby("home_team")["g2_H"].last()
    last_away = sub.groupby("away_team")["g2_A"].last()
    best_team = str(pd.concat([last_home, last_away]).groupby(level=0).max().idxmax())

    h_rows = sub[sub["home_team"] == best_team][
        ["date", "elo_H", "g1_H", "g2_H", "g2_rd_H"]
    ].rename(columns={"elo_H": "elo", "g1_H": "g1", "g2_H": "g2", "g2_rd_H": "rd"})

    a_rows = sub[sub["away_team"] == best_team][
        ["date", "elo_A", "g1_A", "g2_A", "g2_rd_A"]
    ].rename(columns={"elo_A": "elo", "g1_A": "g1", "g2_A": "g2", "g2_rd_A": "rd"})

    team_df = pd.concat([h_rows, a_rows]).sort_values("date").reset_index(drop=True)

    x = np.arange(1, len(team_df) + 1)
    elo_arr = team_df["elo"].to_numpy(dtype=float)
    g1_arr = team_df["g1"].to_numpy(dtype=float)
    g2_arr = team_df["g2"].to_numpy(dtype=float)
    rd_arr = team_df["rd"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.plot(x, elo_arr, label="Elo", color="#2a9d8f", linewidth=2.2, linestyle="--")
    ax.plot(x, g1_arr, label="Glicko", color="#e9c46a", linewidth=2.2)
    ax.plot(x, g2_arr, label="Glicko-2", color="#264653", linewidth=2.5)
    ax.fill_between(
        x,
        g2_arr - rd_arr,
        g2_arr + rd_arr,
        color="#264653",
        alpha=0.14,
        label="Glicko-2 neistota (± RD)",
    )
    ax.axhline(1500.0, color="#95a5a6", linestyle=":", linewidth=1.2, label="Štartovací rating (1500)")

    ax.set_title(f"Porovnanie Elo, Glicko a Glicko-2 v prvej sezóne ({first_season}) – {best_team}")
    ax.set_xlabel("Poradie zápasu v sezóne")
    ax.set_ylabel("Rating")
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = FIGURES_DIR / "05_first_season_elo_glicko.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_pipeline(league: str = "Premier League") -> pd.DataFrame:
    ensure_directories()
    paths = get_league_paths(league)
    matches = pd.read_csv(paths["matches_csv"], parse_dates=["date"])
    history, current = compute_all_ratings(matches)

    history.to_csv(paths["rating_history_csv"], index=False, encoding="utf-8")
    current.to_csv(paths["current_ratings_csv"], index=False, encoding="utf-8")

    comparison = evaluate_models(history, season=TEST_SEASON)
    suffix = "" if league == "Premier League" else f"_{LEAGUES[league]['slug']}"
    comparison.to_csv(TABLES_DIR / f"rating_comparison{suffix}.csv", index=False, encoding="utf-8")

    if league == "Premier League":
        generate_charts(history, comparison)

    # Uloženie metadát
    seasons_list = sorted(history["season"].unique())
    meta = {
        "league": league,
        "test_season": TEST_SEASON,
        "total_matches": len(history),
        "n_seasons": len(seasons_list),
        "first_season": seasons_list[0] if seasons_list else "",
        "last_season": seasons_list[-1] if seasons_list else "",
        "comparison": comparison.to_dict(orient="records"),
    }
    paths["rating_metadata_json"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    md_lines = [
        f"# Vyhodnotenie ratingových systémov – {league} (testovacia sezóna {TEST_SEASON})",
        "",
        "| Model | Typ | Accuracy | F1 (macro) | Log loss | Brier |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for r in comparison.itertuples(index=False):
        md_lines.append(
            f"| {r.model} | {r.typ} | {r.accuracy:.1%} | {r.f1_macro:.3f} | {r.log_loss:.3f} | {r.brier:.3f} |"
        )
    (TABLES_DIR / f"rating_results{suffix}.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[{league}]")
    print(comparison.to_string(index=False))
    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepočíta a vyhodnotí ratingy ligy")
    parser.add_argument(
        "--league",
        choices=list(LEAGUES),
        default="Premier League",
        help="liga na vyhodnotenie",
    )
    args = parser.parse_args()
    run_pipeline(league=args.league)
