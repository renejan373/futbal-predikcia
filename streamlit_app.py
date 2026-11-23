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

import importlib

import src.config as _config_mod
import src.prepare as _prepare_mod
import src.evaluate_ratings as _eval_mod
import src.injuries as _injuries_mod
import src.updater as _updater_mod

for _mod in (_config_mod, _prepare_mod, _eval_mod, _injuries_mod, _updater_mod):
    importlib.reload(_mod)

CURRENT_SEASON = _config_mod.CURRENT_SEASON
LEAGUES = _config_mod.LEAGUES
get_league_paths = _config_mod.get_league_paths
CLUB_LOGOS = _injuries_mod.CLUB_LOGOS
get_team_logo_data_uri = _injuries_mod.get_team_logo_data_uri
get_player_photo_data_uri = _injuries_mod.get_player_photo_data_uri
load_injuries_by_league = _injuries_mod.load_injuries_by_league
update_live_season_pipeline = _updater_mod.update_live_season_pipeline

from src.ratings import (
    EloSystem,
    Glicko2System,
    Glicko2Team,
    GlickoSystem,
    GlickoTeam,
)

st.set_page_config(
    page_title="Predikcia futbalových zápasov",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
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
def load_data(league: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    paths = get_league_paths(league)
    if not paths["current_ratings_csv"].exists():
        st.error(
            f"Chýbajú vypočítané ratingy pre ligu {league}. Spusti v termináli:\n\n"
            f"```\npython -m src.evaluate_ratings --league \"{league}\"\n```"
        )
        st.stop()
    ratings = pd.read_csv(paths["current_ratings_csv"], parse_dates=["as_of_date"])
    ratings["season"] = ratings["season"].astype(str)
    history = pd.read_csv(paths["rating_history_csv"], parse_dates=["date"])
    history["season"] = history["season"].astype(str)
    matches = pd.read_csv(paths["matches_csv"], parse_dates=["date"])
    matches["season"] = matches["season"].astype(str)
    metadata = json.loads(paths["rating_metadata_json"].read_text(encoding="utf-8"))
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
        x_ticks = (
            [
                seasons_in_range[0],
                seasons_in_range[len(seasons_in_range) // 2],
                seasons_in_range[-1],
            ]
            if len(seasons_in_range) > 8
            else seasons_in_range
        )
        x_enc = alt.X(
            "Sezóna:N",
            sort=seasons_in_range,
            title=None,
            axis=alt.Axis(values=x_ticks, labelAngle=0, labelPadding=6),
        )
        tooltip_x = alt.Tooltip("Sezóna:N", title="Sezóna")

    y_min = float(df_long["Rating"].dropna().min()) if not df_long["Rating"].dropna().empty else 1400.0
    y_max = float(df_long["Rating"].dropna().max()) if not df_long["Rating"].dropna().empty else 1600.0
    pad = max(20.0, (y_max - y_min) * 0.06)
    y_domain = [round(y_min - pad, -1), round(y_max + pad, -1)]
    y_scale = alt.Scale(domain=y_domain, zero=False, nice=False)

    is_single = len(systems) == 1
    legend_cfg = alt.Legend(orient="bottom", columns=2, symbolLimit=0, title=None, labelLimit=130)
    chart_height = 360 if is_single else 300

    charts: list[alt.Chart] = []
    for sys_label, _ in systems:
        sub_df = df_long[df_long["Systém"] == sys_label].copy()
        sel = alt.selection_point(fields=["Tím"], bind="legend", on="click")

        lines = (
            alt.Chart(sub_df)
            .mark_line(point=len(seasons_in_range) == 1 and len(selected_teams) <= 8)
            .encode(
                x=x_enc,
                y=alt.Y(
                    "Rating:Q",
                    title=None,
                    scale=y_scale,
                    axis=alt.Axis(tickCount=6, labelPadding=6),
                ),
                color=alt.Color("Tím:N", legend=legend_cfg),
                opacity=alt.condition(sel, alt.value(0.95), alt.value(0.18)),
                strokeWidth=alt.condition(sel, alt.value(2.8), alt.value(1.5)),
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
            alt.Chart(sub_df)
            .mark_rule(color="#888888", strokeDash=[4, 4], strokeWidth=1.0)
            .encode(y=alt.Y(datum=1500.0, scale=y_scale))
        )

        charts.append(
            alt.layer(baseline, lines).properties(
                width="container",
                height=chart_height,
                title=sys_label,
                padding={"left": 8, "right": 8, "top": 8, "bottom": 4},
            )
        )

    return charts


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
    st.markdown(
        """
        <style>
            [data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'] {display: none;}
            .block-container {padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "startup_synced" not in st.session_state:
        st.session_state["startup_synced"] = True
        with st.spinner("Aktualizujem najnovšie štatistiky a ratingy..."):
            any_ok = False
            for lg in LEAGUES:
                res = update_live_season_pipeline(CURRENT_SEASON, league=lg)
                if res["success"]:
                    any_ok = True
            if any_ok:
                st.cache_data.clear()

    sections = [
        "⚽ Predikcia",
        "📊 Rebríček",
        "📈 Grafy",
        "🏥 Zranenia",
    ]
    section_titles = {
        "⚽ Predikcia": "⚽ Predikcia futbalových zápasov",
        "📊 Rebríček": "📊 Rebríček sily tímov",
        "📈 Grafy": "📈 Vývoj sily a porovnanie",
        "🏥 Zranenia": "🏥 Zranenia hráčov",
    }
    current_section = st.session_state.get("active_section_nav") or sections[0]
    if current_section not in sections:
        current_section = sections[0]

    st.title(section_titles[current_section])
    st.caption(
        "SOČ · odbor 11 – Informatika · Elo, Glicko a Glicko-2"
    )

    active_section = (
        st.segmented_control(
            "Sekcia",
            options=sections,
            default=sections[0],
            label_visibility="collapsed",
            key="active_section_nav",
        )
        or sections[0]
    )

    league_names = list(LEAGUES)
    if active_section != "🏥 Zranenia":
        active_league = (
            st.segmented_control(
                "Liga",
                options=league_names,
                default=league_names[0],
                label_visibility="collapsed",
                key="active_league_nav",
            )
            or league_names[0]
        )
    else:
        active_league = st.session_state.get("active_league_nav") or league_names[0]
        if active_league not in league_names:
            active_league = league_names[0]

    ratings, history, matches, metadata = load_data(active_league)

    available_seasons = sorted(ratings["season"].unique(), reverse=True)
    season = CURRENT_SEASON if CURRENT_SEASON in available_seasons else available_seasons[0]

    season_ratings = ratings[ratings["season"] == season].copy()
    teams = sorted(season_ratings["team"].unique())

    preferred_home = "Real Madrid" if active_league == "La Liga" else "Arsenal"
    preferred_away = "Barcelona" if active_league == "La Liga" else "Chelsea"
    default_home = preferred_home if preferred_home in teams else teams[0]
    if "pred_home" not in st.session_state or st.session_state["pred_home"] not in teams:
        st.session_state["pred_home"] = default_home

    system_keys = list(SYSTEM_NAMES.keys())
    if "pred_system" not in st.session_state or st.session_state["pred_system"] not in system_keys:
        st.session_state["pred_system"] = system_keys[0]

    if active_section == "⚽ Predikcia":
        with st.container(border=True):
            c_h, c_a, c_s = st.columns(3)
            home = c_h.selectbox("Domáci tím", teams, key="pred_home")
            away_options = [t for t in teams if t != home]
            default_away = preferred_away if preferred_away in away_options else away_options[0]
            if "pred_away" not in st.session_state or st.session_state["pred_away"] not in away_options:
                st.session_state["pred_away"] = default_away
            away = c_a.selectbox("Hosťujúci tím", away_options, key="pred_away")
            selected_system = c_s.selectbox(
                "Ratingový systém",
                system_keys,
                key="pred_system",
                format_func=lambda k: SYSTEM_NAMES[k],
            )
    else:
        home = st.session_state["pred_home"]
        away_options = [t for t in teams if t != home]
        default_away = preferred_away if preferred_away in away_options else away_options[0]
        away = st.session_state.get("pred_away", default_away)
        if away not in away_options:
            away = default_away
        selected_system = st.session_state["pred_system"]

    home_row = season_ratings[season_ratings["team"] == home].iloc[0]
    away_row = season_ratings[season_ratings["team"] == away].iloc[0]

    all_preds = compute_all_predictions(home_row, away_row)
    probs = all_preds[selected_system]

    if active_section == "⚽ Predikcia":
        st.subheader(f"{home} – {away}")
        st.caption(
            f"Predikcia na základe aktuálnych ratingov zo sezóny {season}: "
            f"**{home}** ({int(home_row['matches_played'])} zápasov k {home_row['as_of_date']:%d. %m. %Y}) vs. "
            f"**{away}** ({int(away_row['matches_played'])} zápasov k {away_row['as_of_date']:%d. %m. %Y})."
        )

        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 12px 0;">
              <div style="background:#f1f3f6;border-radius:10px;padding:10px 6px;text-align:center;">
                <div style="font-size:12px;color:#555;">Výhra domácich</div>
                <div style="font-size:22px;font-weight:700;color:#2a9d8f;">{probs[0]:.1%}</div>
              </div>
              <div style="background:#f1f3f6;border-radius:10px;padding:10px 6px;text-align:center;">
                <div style="font-size:12px;color:#555;">Remíza</div>
                <div style="font-size:22px;font-weight:700;color:#b8860b;">{probs[1]:.1%}</div>
              </div>
              <div style="background:#f1f3f6;border-radius:10px;padding:10px 6px;text-align:center;">
                <div style="font-size:12px;color:#555;">Výhra hostí</div>
                <div style="font-size:22px;font-weight:700;color:#264653;">{probs[2]:.1%}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

        st.markdown("#### Čo tipujú jednotlivé systémy?")
        rows_html = "".join(
            f"<tr style='border-bottom:1px solid #e6e8ec;'>"
            f"<td style='padding:8px 6px;font-weight:600;'>{SYSTEM_NAMES[k].split(' ')[0]}</td>"
            f"<td style='padding:8px 6px;text-align:center;'>{p_tuple[0]:.1%}</td>"
            f"<td style='padding:8px 6px;text-align:center;'>{p_tuple[1]:.1%}</td>"
            f"<td style='padding:8px 6px;text-align:center;'>{p_tuple[2]:.1%}</td>"
            f"</tr>"
            for k, p_tuple in all_preds.items()
        )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
            f"<thead><tr style='background:#f1f3f6;text-align:center;'>"
            f"<th style='padding:8px 6px;text-align:left;'>Systém</th>"
            f"<th style='padding:8px 6px;'>Domáci</th>"
            f"<th style='padding:8px 6px;'>Remíza</th>"
            f"<th style='padding:8px 6px;'>Hostia</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

    elif active_section == "📊 Rebríček":
        st.subheader(f"Rebríček sily tímov – {active_league} ({season})")
        rank_by = (
            st.segmented_control(
                "Zoradiť podľa",
                options=["Kombinácia", "Glicko-2", "Glicko", "Elo"],
                default="Kombinácia",
                key="ranking_sort_toggle",
            )
            or "Kombinácia"
        )
        sort_col_map = {
            "Kombinácia": "combined_rating",
            "Glicko-2": "glicko2_rating",
            "Glicko": "glicko_rating",
            "Elo": "elo_rating",
        }
        sort_col = sort_col_map[rank_by]

        ranked = season_ratings.copy()
        ranked["combined_rating"] = (
            ranked["elo_rating"] + ranked["glicko_rating"] + ranked["glicko2_rating"]
        ) / 3.0
        ranked = ranked.sort_values(sort_col, ascending=False).reset_index(drop=True)

        sort_desc = (
            "priemeru všetkých 3 systémov (Elo + Glicko + Glicko-2)"
            if rank_by == "Kombinácia"
            else f"systému {rank_by}"
        )
        st.caption(
            f"Zoradené podľa {sort_desc} · 🥇–🥉 Top 3 · 🟦 4. miesto · 🟥 Zostupová zóna"
        )

        n_teams = len(ranked)
        r_min = float(ranked[sort_col].min()) if n_teams else 1400.0
        r_max = float(ranked[sort_col].max()) if n_teams else 1800.0
        r_span = max(1.0, r_max - r_min)

        cards_html = []
        for idx, row in enumerate(ranked.itertuples(index=False), start=1):
            if idx == 1:
                accent, bg, badge = "#d4af37", "#fffdf4", "🥇 1."
            elif idx == 2:
                accent, bg, badge = "#9aa0a6", "#fafafa", "🥈 2."
            elif idx == 3:
                accent, bg, badge = "#cd7f32", "#fdfaf6", "🥉 3."
            elif idx == 4:
                accent, bg, badge = "#2a9d8f", "#ffffff", "4."
            elif idx > n_teams - 3:
                accent, bg, badge = "#e76f51", "#fff9f8", f"{idx}."
            else:
                accent, bg, badge = "#cfd6e0", "#ffffff", f"{idx}."

            active_val = float(getattr(row, sort_col))
            pct = max(6.0, min(100.0, ((active_val - r_min) / r_span) * 94.0 + 6.0))
            bar_color = "#2a9d8f" if idx <= 4 else ("#e76f51" if idx > n_teams - 3 else "#457b9d")

            form = recent_results(matches, season, row.team, limit=5)
            mini_form = "".join(
                f"<span style='display:inline-block;width:18px;height:18px;line-height:18px;"
                f"text-align:center;border-radius:4px;background:{FORM_STYLE[r][0]};"
                f"color:{FORM_STYLE[r][1]};font-size:10px;font-weight:700;margin-left:2px;'>{r}</span>"
                for r in form
            )

            logo_uri = get_team_logo_data_uri(row.team)
            logo_img = (
                f"<img src='{logo_uri}' style='width:24px;height:24px;object-fit:contain;vertical-align:middle;' />"
                if logo_uri
                else ""
            )

            def _cell(label: str, val: float, is_active: bool) -> str:
                style = "color:#1d3557;font-size:15px;" if is_active else ""
                return f"<div><span style='font-size:11px;color:#666;'>{label}</span><br><b style='{style}'>{val:.1f}</b></div>"

            rating_cells = (
                _cell("Kombi", row.combined_rating, rank_by == "Kombinácia")
                + _cell("Glicko-2", row.glicko2_rating, rank_by == "Glicko-2")
                + _cell("Glicko", row.glicko_rating, rank_by == "Glicko")
                + _cell("Elo", row.elo_rating, rank_by == "Elo")
            )

            cards_html.append(
                f"<div style='border:1px solid #e4e7ec;border-left:5px solid {accent};"
                f"border-radius:12px;padding:11px 12px;margin-bottom:9px;background:{bg};"
                f"box-shadow:0 1px 2px rgba(0,0,0,0.03);'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;gap:6px;'>"
                f"<span style='display:flex;align-items:center;gap:7px;font-size:16px;font-weight:700;color:#1d3557;'>{badge} {logo_img} {row.team}</span>"
                f"<span style='display:flex;align-items:center;gap:6px;'>{mini_form}"
                f"<span style='font-size:11px;color:#555;background:#eef1f5;padding:2px 7px;border-radius:10px;font-weight:600;'>{int(row.matches_played)} záp.</span>"
                f"</span></div>"
                f"<div style='background:#eef1f5;height:6px;border-radius:4px;margin:8px 0 9px 0;overflow:hidden;'>"
                f"<div style='width:{pct:.1f}%;height:100%;background:{bar_color};border-radius:4px;'></div>"
                f"</div>"
                f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:4px;font-size:13px;"
                f"background:#f7f8fa;border-radius:8px;padding:6px 6px;text-align:center;'>"
                f"{rating_cells}"
                f"</div>"
                f"<div style='margin-top:6px;font-size:11px;color:#666;display:flex;justify-content:space-between;'>"
                f"<span>Neistota (RD): <b>±{row.glicko2_rd:.1f}</b></span>"
                f"<span>Volatilita (σ): <b>{row.glicko2_vol:.4f}</b></span>"
                f"</div>"
                f"</div>"
            )
        st.markdown("".join(cards_html), unsafe_allow_html=True)

    elif active_section == "📈 Grafy":
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
        preset_candidates = (
            ["Real Madrid", "Barcelona", "Ath Madrid", "Ath Bilbao", "Valencia", "Sevilla", "Sociedad"]
            if active_league == "La Liga"
            else ["Man United", "Arsenal", "Liverpool", "Chelsea", "Man City", "Tottenham", "Everton"]
        )
        big_clubs = [t for t in preset_candidates if t in teams_in_range]
        default_teams = big_clubs if big_clubs else teams_in_range[:3]

        if (
            "comp_teams_widget" not in st.session_state
            or st.session_state.get("comp_teams_league") != active_league
        ):
            st.session_state["comp_teams_widget"] = default_teams
            st.session_state["comp_teams_league"] = active_league

        preset_label = "Top kluby La Ligy" if active_league == "La Liga" else "Big 6 + Everton"
        preset_cols = st.columns([1, 1, 4])
        if preset_cols[0].button(preset_label, use_container_width=True):
            st.session_state["comp_teams_widget"] = default_teams
        if preset_cols[1].button("Všetky tímy", use_container_width=True):
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
            options=["Glicko-2", "Glicko", "Elo", "Všetky"],
            default="Glicko-2",
            key="comp_system_toggle",
        ) or "Glicko-2"

        if not selected_teams:
            st.info("Vyber aspoň jeden tím na zobrazenie grafu.")
        else:
            mode_note = (
                f"Zobrazený priebeh zápas po zápase v sezóne **{start_season}**."
                if len(seasons_in_range) == 1
                else f"Zobrazené koncové ratingy sezón **{start_season}** – **{end_season}**. "
                f"Pri vypadnutých tímoch je čiara v sezóne mimo {active_league} prerušená."
            )
            st.caption(mode_note)
            comp_charts = build_rating_comparison_chart(
                ratings, history, selected_teams, seasons_in_range, selected_system=chart_system
            )
            for ch in comp_charts:
                st.altair_chart(ch, use_container_width=True)

        st.divider()
        st.subheader(f"Presnosť na sezóne {metadata['test_season']}")
        comp_rows_html = "".join(
            f"<tr style='border-bottom:1px solid #e6e8ec;'>"
            f"<td style='padding:8px 6px;font-weight:600;'>{r['model']}</td>"
            f"<td style='padding:8px 6px;text-align:center;'>{r['accuracy']:.1%}</td>"
            f"<td style='padding:8px 6px;text-align:center;'>{r['log_loss']:.3f}</td>"
            f"<td style='padding:8px 6px;text-align:center;'>{r['brier']:.3f}</td>"
            f"</tr>"
            for r in metadata["comparison"]
        )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
            f"<thead><tr style='background:#f1f3f6;text-align:center;'>"
            f"<th style='padding:8px 6px;text-align:left;'>Model</th>"
            f"<th style='padding:8px 6px;'>Presnosť</th>"
            f"<th style='padding:8px 6px;'>Log loss</th>"
            f"<th style='padding:8px 6px;'>Brier</th>"
            f"</tr></thead><tbody>{comp_rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Poznámka: Log loss a Brier score hodnotia kvalitu odhadnutých pravdepodobností (menšie číslo je lepšie)."
        )

    elif active_section == "🏥 Zranenia":
        st.subheader("Prehľad zranených hráčov (Top 5 líg)")
        updated_at, injuries_by_league = load_injuries_by_league()
        league_options = list(injuries_by_league.keys())

        if not league_options:
            st.info("Chýbajú dáta o zraneniach (`data/injuries.json`).")
        else:
            selected_league = (
                st.segmented_control(
                    "Vyber ligu",
                    options=league_options,
                    default="La Liga" if "La Liga" in league_options else league_options[0],
                    key="injuries_league_toggle",
                )
                or league_options[0]
            )

            injuries_by_club = injuries_by_league[selected_league]
            club_options = list(injuries_by_club.keys())

            if (
                "injuries_club" not in st.session_state
                or st.session_state["injuries_club"] not in club_options
            ):
                st.session_state["injuries_club"] = club_options[0]

            # Výber z 3 top klubov vybranej ligy s logom
            sel_cols = st.columns(len(club_options))
            for idx, club_name in enumerate(club_options):
                is_selected = st.session_state["injuries_club"] == club_name
                club_count = len(injuries_by_club.get(club_name, pd.DataFrame()))
                with sel_cols[idx].container(border=True):
                    c_logo, c_info = st.columns([1, 2.2], vertical_alignment="center")
                    logo_path = CLUB_LOGOS.get(club_name)
                    if logo_path and logo_path.exists():
                        c_logo.image(str(logo_path), width=52)
                    c_info.markdown(f"**{club_name}**  \nZranení hráči: **{club_count}**")
                    if st.button(
                        "Vybraný ✓" if is_selected else "Vybrať",
                        key=f"btn_club_{selected_league}_{club_name}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state["injuries_club"] = club_name
                        st.rerun()

            selected_club = st.session_state["injuries_club"]
            club_df = injuries_by_club.get(selected_club, pd.DataFrame())

            st.divider()
            hdr_logo, hdr_title = st.columns([0.8, 9.2], vertical_alignment="center")
            sel_logo = CLUB_LOGOS.get(selected_club)
            if sel_logo and sel_logo.exists():
                hdr_logo.image(str(sel_logo), width=48)
            hdr_title.markdown(f"### {selected_club} ({selected_league}) – aktuálne zranenia")

            if club_df.empty:
                st.info(f"Pre klub {selected_club} zatiaľ nie sú evidované žiadne zranenia.")
            else:
                in_match = int(club_df["Kde sa zranil (zápas / tréning)"].str.contains("zápas", case=False, na=False).sum())
                in_training = int(club_df["Kde sa zranil (zápas / tréning)"].str.contains("tréning", case=False, na=False).sum())

                m_cols = st.columns(3)
                m_cols[0].metric("Zranení hráči spolu", len(club_df))
                m_cols[1].metric("Zranení v zápase", in_match)
                m_cols[2].metric("Na tréningu / preťaženie", in_training)

                st.markdown("")
                card_cols = st.columns(2)
                for idx, (_, row) in enumerate(club_df.iterrows()):
                    col = card_cols[idx % 2]
                    photo_uri = get_player_photo_data_uri(row["Hráč"])
                    if photo_uri:
                        avatar_html = (
                            f"<img src='{photo_uri}' style='width:58px;height:58px;border-radius:50%;"
                            f"object-fit:cover;object-position:top center;border:2px solid #dfe3ea;flex-shrink:0;' />"
                        )
                    else:
                        initials = "".join(part[0] for part in row["Hráč"].split()[:2]).upper()
                        avatar_html = (
                            f"<div style='width:58px;height:58px;border-radius:50%;background:#1d3557;"
                            f"color:#fff;display:flex;align-items:center;justify-content:center;"
                            f"font-weight:700;font-size:18px;flex-shrink:0;'>{initials}</div>"
                        )

                    card_html = (
                        f"<div style='border:1px solid #e2e5ea;border-radius:12px;padding:12px 14px;"
                        f"margin-bottom:10px;background:#ffffff;box-shadow:0 1px 2px rgba(0,0,0,0.03);'>"
                        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>"
                        f"{avatar_html}"
                        f"<div>"
                        f"<div style='font-size:17px;font-weight:700;color:#1d3557;line-height:1.2;'>{row['Hráč']}</div>"
                        f"<div style='font-size:12px;color:#555;margin-top:3px;'>"
                        f"<span style='background:#f1f3f6;padding:2px 7px;border-radius:6px;font-weight:600;'>{row['Pozícia']}</span>"
                        f" &nbsp;·&nbsp; <b>{row['Stav']}</b></div>"
                        f"</div></div>"
                        f"<div style='font-size:13.5px;line-height:1.5;color:#222;'>"
                        f"<div><b>Zranenie:</b> {row['Zranenie']}</div>"
                        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:4px;'>"
                        f"<div><b>Kedy:</b> {row['Kedy sa zranil']}</div>"
                        f"<div><b>Na ako dlho:</b> {row['Na ako dlho']}</div>"
                        f"</div>"
                        f"<div style='margin-top:4px;'><b>Návrat:</b> {row['Predpokladaný návrat']}</div>"
                        f"<div style='margin-top:2px;'><b>Kde sa zranil:</b> {row['Kde sa zranil (zápas / tréning)']}</div>"
                        f"</div></div>"
                    )
                    col.markdown(card_html, unsafe_allow_html=True)

        if updated_at:
            st.caption(f"Údaje aktualizované k {updated_at} (zdroj: `data/injuries.json`).")


main()
