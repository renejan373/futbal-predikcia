"""Streamlit aplikácia – predikcia výsledkov futbalových zápasov (Elo, Glicko, Glicko-2).

Spustenie:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

alt.data_transformers.disable_max_rows()

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
    "combined": "Kombinácia (Elo + Glicko + Glicko-2)",
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


def build_rating_comparison_chart(
    ratings: pd.DataFrame,
    history: pd.DataFrame,
    selected_teams: list[str],
    seasons_in_range: list[str],
    selected_system: str = "Všetky",
) -> alt.Chart:
    """Vytvorí porovnávací čiarový graf (Elo, Glicko, Glicko-2 alebo jeden zvolený systém)."""
    all_systems = [
        ("Elo", "elo"),
        ("Glicko", "glicko"),
        ("Glicko-2", "glicko2"),
    ]
    systems = (
        [s for s in all_systems if s[0] == selected_system]
        if selected_system in ("Elo", "Glicko", "Glicko-2")
        else all_systems
    )

    if len(seasons_in_range) == 1:
        season = seasons_in_range[0]
        sub_hist = history[history["season"] == season].sort_values(["date", "match_id"])
        sub_curr = ratings[ratings["season"] == season].set_index("team")
        rows: list[dict] = []

        for team in selected_teams:
            h_df = sub_hist[sub_hist["home_team"] == team][
                ["date", "elo_H", "g1_H", "g2_H"]
            ].rename(columns={"elo_H": "elo", "g1_H": "glicko", "g2_H": "glicko2"})
            a_df = sub_hist[sub_hist["away_team"] == team][
                ["date", "elo_A", "g1_A", "g2_A"]
            ].rename(columns={"elo_A": "elo", "g1_A": "glicko", "g2_A": "glicko2"})
            t_df = pd.concat([h_df, a_df]).sort_values("date").reset_index(drop=True)

            if team in sub_curr.index:
                fin = sub_curr.loc[team]
                t_df = pd.concat(
                    [
                        t_df,
                        pd.DataFrame(
                            [
                                {
                                    "date": fin["as_of_date"],
                                    "elo": fin["elo_rating"],
                                    "glicko": fin["glicko_rating"],
                                    "glicko2": fin["glicko2_rating"],
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

            for idx, r in enumerate(t_df.itertuples(index=False), start=1):
                for sys_label, key in systems:
                    rows.append(
                        {
                            "Tím": team,
                            "Systém": sys_label,
                            "X": idx,
                            "Rating": float(getattr(r, key)),
                        }
                    )

        df_long = pd.DataFrame(rows)
        x_enc = alt.X("X:Q", title="Zápas v sezóne", axis=alt.Axis(tickMinStep=5))
        tooltip_x = alt.Tooltip("X:Q", title="Zápas")
    else:
        sub = ratings[
            ratings["season"].isin(seasons_in_range) & ratings["team"].isin(selected_teams)
        ].copy()
        full_idx = pd.MultiIndex.from_product(
            [seasons_in_range, selected_teams], names=["season", "team"]
        )
        sub = sub.set_index(["season", "team"]).reindex(full_idx).reset_index()

        col_map = {
            "Elo": "elo_rating",
            "Glicko": "glicko_rating",
            "Glicko-2": "glicko2_rating",
        }
        rows = []
        for r in sub.itertuples(index=False):
            for sys_label, _ in systems:
                val = getattr(r, col_map[sys_label])
                rows.append(
                    {
                        "Tím": r.team,
                        "Systém": sys_label,
                        "Sezóna": r.season,
                        "Rating": float(val) if pd.notna(val) else None,
                    }
                )

        df_long = pd.DataFrame(rows)
        x_enc = alt.X(
            "Sezóna:N",
            sort=seasons_in_range,
            title="Sezóna",
            axis=alt.Axis(labelAngle=-35),
        )
        tooltip_x = alt.Tooltip("Sezóna:N", title="Sezóna")

    y_min = float(df_long["Rating"].dropna().min()) if not df_long["Rating"].dropna().empty else 1400.0
    y_max = float(df_long["Rating"].dropna().max()) if not df_long["Rating"].dropna().empty else 1600.0
    pad = max(20.0, (y_max - y_min) * 0.06)
    y_domain = [round(y_min - pad, -1), round(y_max + pad, -1)]
    y_scale = alt.Scale(domain=y_domain, zero=False, nice=False)

    is_single = len(systems) == 1
    right_cols = 1 if len(selected_teams) <= 22 else (2 if len(selected_teams) <= 40 else 3)
    legend_cfg = (
        alt.Legend(
            orient="right",
            columns=right_cols,
            symbolLimit=0,
            title="Tímy",
            labelLimit=130,
        )
        if is_single
        else alt.Legend(orient="bottom", columns=6, symbolLimit=0, title=None)
    )

    sel = alt.selection_point(fields=["Tím"], bind="legend", on="click")

    lines = (
        alt.Chart(df_long)
        .mark_line(point=len(seasons_in_range) == 1 and len(selected_teams) <= 10)
        .encode(
            x=x_enc,
            y=alt.Y(
                "Rating:Q",
                title="Rating",
                scale=y_scale,
                axis=alt.Axis(tickCount=10),
            ),
            color=alt.Color("Tím:N", legend=legend_cfg),
            opacity=alt.condition(sel, alt.value(0.95), alt.value(0.18)),
            strokeWidth=alt.condition(sel, alt.value(3.0), alt.value(1.5)),
            tooltip=[
                alt.Tooltip("Tím:N"),
                alt.Tooltip("Systém:N"),
                tooltip_x,
                alt.Tooltip("Rating:Q", format=".1f"),
            ],
        )
        .add_params(sel)
    )

    baseline = (
        alt.Chart(df_long)
        .mark_rule(color="#888888", strokeDash=[4, 4], strokeWidth=1.0)
        .encode(y=alt.Y(datum=1500.0, scale=y_scale))
    )

    if is_single:
        return (
            alt.layer(baseline, lines)
            .properties(height=560, title=systems[0][0])
        )

    layered = (
        alt.layer(baseline, lines)
        .properties(width=310, height=430)
        .facet(
            column=alt.Column(
                "Systém:N",
                sort=["Elo", "Glicko", "Glicko-2"],
                title=None,
                header=alt.Header(labelFontSize=14, labelFontWeight="bold"),
            )
        )
        .resolve_scale(y="shared")
    )
    return layered


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
    combined_p = tuple((elo_p[i] + g1_p[i] + g2_p[i]) / 3.0 for i in range(3))
    return {"combined": combined_p, "elo": elo_p, "glicko": g1_p, "glicko2": g2_p}


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
        st.subheader("Porovnanie vývoja ratingov (Elo, Glicko, Glicko-2)")

        all_seasons_asc = sorted(ratings["season"].unique())
        start_season, end_season = st.select_slider(
            "Rozsah sezón",
            options=all_seasons_asc,
            value=(all_seasons_asc[0], all_seasons_asc[-1]),
        )
        i_start = all_seasons_asc.index(start_season)
        i_end = all_seasons_asc.index(end_season)
        seasons_in_range = all_seasons_asc[i_start : i_end + 1]

        teams_in_range = sorted(
            ratings[ratings["season"].isin(seasons_in_range)]["team"].unique()
        )
        big_clubs = [
            t
            for t in ["Man United", "Arsenal", "Liverpool", "Chelsea", "Man City", "Tottenham", "Everton"]
            if t in teams_in_range
        ]
        default_teams = big_clubs if big_clubs else teams_in_range[:3]

        if "comp_teams_widget" not in st.session_state:
            st.session_state["comp_teams_widget"] = default_teams

        preset_cols = st.columns([1, 1, 1, 3])
        if preset_cols[0].button("Big 6 + Everton", use_container_width=True):
            st.session_state["comp_teams_widget"] = default_teams
        if preset_cols[1].button("Aktuálny zápas", use_container_width=True):
            st.session_state["comp_teams_widget"] = [t for t in (home, away) if t in teams_in_range]
        if preset_cols[2].button("Všetky tímy", use_container_width=True):
            st.session_state["comp_teams_widget"] = teams_in_range

        # Keep only teams valid in the currently selected season range
        st.session_state["comp_teams_widget"] = [
            t for t in st.session_state["comp_teams_widget"] if t in teams_in_range
        ]

        selected_teams = st.multiselect(
            "Vyber tímy na zobrazenie",
            options=teams_in_range,
            key="comp_teams_widget",
        )

        chart_system = st.segmented_control(
            "Zobrazený ratingový systém",
            options=["Všetky", "Elo", "Glicko", "Glicko-2"],
            default="Všetky",
            key="comp_system_toggle",
        ) or "Všetky"

        if not selected_teams:
            st.info("Vyber aspoň jeden tím na zobrazenie grafu.")
        else:
            mode_note = (
                f"Zobrazený priebeh zápas po zápase v sezóne **{start_season}**."
                if len(seasons_in_range) == 1
                else f"Zobrazené koncové ratingy sezón **{start_season}** – **{end_season}**. "
                "Pri vypadnutých tímoch je čiara v sezóne mimo Premier League prerušená."
            )
            st.caption(mode_note)
            comp_chart = build_rating_comparison_chart(
                ratings, history, selected_teams, seasons_in_range, selected_system=chart_system
            )
            st.altair_chart(comp_chart, use_container_width=True)

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
