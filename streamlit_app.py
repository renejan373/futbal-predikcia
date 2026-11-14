"""Streamlit aplikácia – predikcia výsledkov futbalových zápasov (Elo, Glicko, Glicko-2).

Spustenie:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from src.config import CURRENT_SEASON, MATCHES_CSV, TABLES_DIR
from src.evaluate_ratings import (
    CURRENT_RATINGS_CSV,
    RATING_HISTORY_CSV,
    RATING_METADATA_JSON,
)
from src.ratings import (
    EloSystem,
    Glicko2System,
    Glicko2Team,
    GlickoSystem,
    GlickoTeam,
)
from src.updater import update_live_season_pipeline

st.set_page_config(
    page_title="Predikcia futbalových zápasov",
    page_icon="⚽",
    layout="wide",
)

OUTCOME_LABELS = ["Výhra domácich", "Remíza", "Výhra hostí"]
OUTCOME_COLORS = ["#2a9d8f", "#e9c46a", "#264653"]
FORM_STYLE = {
    "V": ("#2a9d8f", "white"),
    "R": ("#e9c46a", "#1d1d1d"),
    "P": ("#e76f51", "white"),
}
SYSTEM_NAMES = {
    "glicko2": "Glicko-2 (sila + neistota + volatilita)",
    "glicko": "Glicko (sila + neistota)",
    "elo": "Elo (klasický rating)",
}


@st.cache_data(ttl=600)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not CURRENT_RATINGS_CSV.exists():
        st.error(
            "Chýbajú vypočítané ratingy. Spusti v termináli:\n\n"
            "```\npython -m src.evaluate_ratings\n```"
        )
        st.stop()
    ratings = pd.read_csv(CURRENT_RATINGS_CSV, parse_dates=["as_of_date"])
    ratings["season"] = ratings["season"].astype(str)
    history = pd.read_csv(RATING_HISTORY_CSV, parse_dates=["date"])
    history["season"] = history["season"].astype(str)
    matches = pd.read_csv(MATCHES_CSV, parse_dates=["date"])
    matches["season"] = matches["season"].astype(str)
    metadata = json.loads(RATING_METADATA_JSON.read_text(encoding="utf-8"))
    return ratings, history, matches, metadata


def recent_results(matches: pd.DataFrame, season: str, team: str, limit: int = 5) -> list[str]:
    rows = matches[
        (matches["season"] == season)
        & ((matches["home_team"] == team) | (matches["away_team"] == team))
    ].sort_values("date").tail(limit)

    results = []
    for row in rows.itertuples(index=False):
        if row.home_team == team:
            gf, ga = row.fthg, row.ftag
        else:
            gf, ga = row.ftag, row.fthg
        results.append("V" if gf > ga else "R" if gf == ga else "P")
    return results


def form_badges(results: list[str]) -> str:
    if not results:
        return "<span style='color:#888'>žiadne zápasy</span>"
    badges = []
    for r in results:
        bg, fg = FORM_STYLE[r]
        badges.append(
            f"<span style='display:inline-block;width:28px;height:28px;line-height:28px;"
            f"text-align:center;border-radius:6px;background:{bg};color:{fg};"
            f"font-weight:600;margin-right:4px'>{r}</span>"
        )
    return "".join(badges)


def probability_chart(probabilities: tuple[float, float, float]) -> alt.Chart:
    frame = pd.DataFrame(
        {"Výsledok": OUTCOME_LABELS, "Pravdepodobnosť": list(probabilities)}
    )
    base = alt.Chart(frame).encode(
        x=alt.X("Výsledok:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y(
            "Pravdepodobnosť:Q",
            axis=alt.Axis(format="%", title="Pravdepodobnosť"),
            scale=alt.Scale(domain=[0, 1]),
        ),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        color=alt.Color(
            "Výsledok:N",
            scale=alt.Scale(domain=OUTCOME_LABELS, range=OUTCOME_COLORS),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Výsledok:N"),
            alt.Tooltip("Pravdepodobnosť:Q", format=".1%"),
        ],
    )
    labels = base.mark_text(dy=-12, fontSize=14, fontWeight="bold").encode(
        text=alt.Text("Pravdepodobnosť:Q", format=".1%")
    )
    return (bars + labels).properties(height=300)


def compute_all_predictions(home_row: pd.Series, away_row: pd.Series) -> dict[str, tuple[float, float, float]]:
    elo_sys = EloSystem()
    g1_sys = GlickoSystem()
    g2_sys = Glicko2System()

    elo_p = elo_sys.predict_proba(float(home_row["elo_rating"]), float(away_row["elo_rating"]))
    g1_p = g1_sys.predict_proba(
        GlickoTeam(float(home_row["glicko_rating"]), float(home_row["glicko_rd"])),
        GlickoTeam(float(away_row["glicko_rating"]), float(away_row["glicko_rd"])),
    )
    g2_p = g2_sys.predict_proba(
        Glicko2Team(
            float(home_row["glicko2_rating"]),
            float(home_row["glicko2_rd"]),
            float(home_row["glicko2_vol"]),
        ),
        Glicko2Team(
            float(away_row["glicko2_rating"]),
            float(away_row["glicko2_rd"]),
            float(away_row["glicko2_vol"]),
        ),
    )
    return {"elo": elo_p, "glicko": g1_p, "glicko2": g2_p}


def main() -> None:
    ratings, history, matches, metadata = load_data()

    st.title("⚽ Predikcia výsledkov futbalových zápasov")
    st.caption(
        "Stredoškolská odborná činnosť · odbor 11 – Informatika · "
        "porovnanie ratingových systémov Elo, Glicko a Glicko-2"
    )

    available_seasons = sorted(ratings["season"].unique(), reverse=True)
    season = CURRENT_SEASON if CURRENT_SEASON in available_seasons else available_seasons[0]

    season_ratings = ratings[ratings["season"] == season].copy()
    teams = sorted(season_ratings["team"].unique())

    with st.sidebar:
        st.header("Nastavenia predikcie")
        st.markdown(f"**Aktuálna sezóna:** {season}")
        if season != CURRENT_SEASON:
            st.warning(
                f"Dáta pre sezónu {CURRENT_SEASON} zatiaľ nie sú stiahnuté. "
                "Klikni na tlačidlo nižšie (aktuálne sú zobrazené dáta zo sezóny "
                f"{season})."
            )

        default_home = "Arsenal" if "Arsenal" in teams else teams[0]
        home = st.selectbox("Domáci tím", teams, index=teams.index(default_home))

        away_options = [t for t in teams if t != home]
        default_away = "Chelsea" if "Chelsea" in away_options else away_options[0]
        away = st.selectbox("Hosťujúci tím", away_options, index=away_options.index(default_away))

        system_keys = list(SYSTEM_NAMES.keys())
        selected_system = st.selectbox(
            "Ratingový systém",
            system_keys,
            index=0,
            format_func=lambda k: SYSTEM_NAMES[k],
        )

        st.divider()
        st.subheader("🔄 Živá aktualizácia")
        st.caption("Stiahne najnovšie výsledky z internetu a prepočíta ratingy tímov.")
        if st.button(f"Stiahnuť výsledky ({CURRENT_SEASON})", use_container_width=True):
            with st.spinner(f"Sťahujem a prepočítavam sezónu {CURRENT_SEASON}..."):
                res = update_live_season_pipeline(CURRENT_SEASON)
                if res["success"]:
                    st.success(res["message"])
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(res["message"])

    home_row = season_ratings[season_ratings["team"] == home].iloc[0]
    away_row = season_ratings[season_ratings["team"] == away].iloc[0]

    all_preds = compute_all_predictions(home_row, away_row)
    probs = all_preds[selected_system]

    tab_pred, tab_rank, tab_comp = st.tabs(
        ["Predikcia zápasu", "Rebríček sily tímov", "Vývoj sily a porovnanie modelov"]
    )

    with tab_pred:
        st.subheader(f"{home} – {away}")
        st.caption(
            f"Predikcia na základe aktuálnych ratingov zo sezóny {season}: "
            f"**{home}** ({int(home_row['matches_played'])} zápasov k {home_row['as_of_date']:%d. %m. %Y}) vs. "
            f"**{away}** ({int(away_row['matches_played'])} zápasov k {away_row['as_of_date']:%d. %m. %Y})."
        )

        cols = st.columns(3)
        for col, label, p in zip(cols, OUTCOME_LABELS, probs):
            col.metric(label, f"{p:.1%}")

        st.altair_chart(probability_chart(probs), use_container_width=True)

        best_idx = max(range(3), key=lambda i: probs[i])
        st.success(
            f"Systém **{SYSTEM_NAMES[selected_system].split(' ')[0]}** tipuje "
            f"**{OUTCOME_LABELS[best_idx].lower()}** s pravdepodobnosťou **{probs[best_idx]:.1%}**."
        )

        st.divider()
        st.subheader("Porovnanie oboch tímov")
        left, right = st.columns(2)
        for col, team_name, row in ((left, home, home_row), (right, away, away_row)):
            col.markdown(f"### {team_name}")
            results = recent_results(matches, season, team_name)
            col.markdown(form_badges(results), unsafe_allow_html=True)
            col.write(
                f"- **Elo rating:** {row['elo_rating']:.1f}\n"
                f"- **Glicko rating:** {row['glicko_rating']:.1f} (± {row['glicko_rd']:.1f} RD)\n"
                f"- **Glicko-2 rating:** {row['glicko2_rating']:.1f} (± {row['glicko2_rd']:.1f} RD, "
                f"volatilita {row['glicko2_vol']:.4f})"
            )

        st.markdown("#### Čo tipujú jednotlivé systémy pre tento zápas?")
        sys_rows = []
        for k, p_tuple in all_preds.items():
            sys_rows.append(
                {
                    "Systém": SYSTEM_NAMES[k].split(" ")[0],
                    "Výhra domácich": f"{p_tuple[0]:.1%}",
                    "Remíza": f"{p_tuple[1]:.1%}",
                    "Výhra hostí": f"{p_tuple[2]:.1%}",
                }
            )
        st.dataframe(pd.DataFrame(sys_rows), hide_index=True, use_container_width=True)

    with tab_rank:
        st.subheader(f"Rebríček sily tímov Premier League ({season})")
        st.caption("Tímy zoradené podľa najnovšieho Glicko-2 ratingu.")
        ranked = season_ratings.sort_values("glicko2_rating", ascending=False).reset_index(drop=True)
        ranked.index = ranked.index + 1
        ranked = ranked.reset_index().rename(
            columns={
                "index": "Poradie",
                "team": "Tím",
                "matches_played": "Zápasy",
                "glicko2_rating": "Glicko-2",
                "glicko2_rd": "Neistota (RD)",
                "glicko2_vol": "Volatilita (σ)",
                "glicko_rating": "Glicko",
                "elo_rating": "Elo",
            }
        )
        st.dataframe(
            ranked[["Poradie", "Tím", "Zápasy", "Glicko-2", "Neistota (RD)", "Volatilita (σ)", "Glicko", "Elo"]],
            hide_index=True,
            use_container_width=True,
        )

    with tab_comp:
        st.subheader(f"Vývoj sily vybraných tímov počas sezóny {season}")
        sub_hist = history[history["season"] == season]
        plot_rows = []
        for team_name in (home, away):
            h_df = sub_hist[sub_hist["home_team"] == team_name][["date", "g2_H"]].rename(
                columns={"g2_H": "Rating"}
            )
            a_df = sub_hist[sub_hist["away_team"] == team_name][["date", "g2_A"]].rename(
                columns={"g2_A": "Rating"}
            )
            t_df = pd.concat([h_df, a_df]).sort_values("date")
            t_df["Tím"] = team_name
            plot_rows.append(t_df)

        if plot_rows:
            chart_df = pd.concat(plot_rows)
            line_chart = (
                alt.Chart(chart_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Dátum"),
                    y=alt.Y("Rating:Q", title="Glicko-2 rating", scale=alt.Scale(zero=False)),
                    color="Tím:N",
                    tooltip=["Tím:N", alt.Tooltip("date:T", format="%d. %m. %Y"), alt.Tooltip("Rating:Q", format=".1f")],
                )
                .properties(height=340)
            )
            st.altair_chart(line_chart, use_container_width=True)

        st.divider()
        st.subheader(f"Presnosť algoritmov na testovacej sezóne {metadata['test_season']}")
        comp_df = pd.DataFrame(metadata["comparison"]).rename(
            columns={
                "model": "Model",
                "typ": "Typ",
                "accuracy": "Presnosť (Accuracy)",
                "f1_macro": "F1 (macro)",
                "log_loss": "Log loss",
                "brier": "Brier score",
            }
        )
        comp_df["Presnosť (Accuracy)"] = comp_df["Presnosť (Accuracy)"].map(lambda v: f"{v:.1%}")
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.caption(
            "Poznámka: Log loss a Brier score hodnotia kvalitu odhadnutých pravdepodobností (menšie číslo je lepšie)."
        )


main()
