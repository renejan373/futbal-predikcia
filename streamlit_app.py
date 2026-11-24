"""Streamlit aplikácia – predikcia výsledkov futbalových zápasov (Elo, Glicko, Glicko-2).

Spustenie:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

try:
    from streamlit_local_storage import LocalStorage
except ImportError:
    LocalStorage = None

alt.data_transformers.disable_max_rows()

import importlib

import src.config as _config_mod
import src.prepare as _prepare_mod
import src.evaluate_ratings as _eval_mod
import src.injuries as _injuries_mod
import src.dennik as _dennik_mod
import src.updater as _updater_mod

for _mod in (_config_mod, _prepare_mod, _eval_mod, _injuries_mod, _dennik_mod, _updater_mod):
    importlib.reload(_mod)

CURRENT_SEASON = _config_mod.CURRENT_SEASON
LEAGUES = _config_mod.LEAGUES
get_league_paths = _config_mod.get_league_paths
CLUB_LOGOS = _injuries_mod.CLUB_LOGOS
get_team_logo_data_uri = _injuries_mod.get_team_logo_data_uri
get_player_photo_data_uri = _injuries_mod.get_player_photo_data_uri
fetch_live_injuries_by_league = _injuries_mod.fetch_live_injuries_by_league
load_injuries_by_league = _injuries_mod.load_injuries_by_league
update_live_season_pipeline = _updater_mod.update_live_season_pipeline
INITIAL_BALANCE = _dennik_mod.INITIAL_BALANCE
TIP_LABELS = _dennik_mod.TIP_LABELS
TIP_OPTIONS = _dennik_mod.TIP_OPTIONS
can_cancel_tip = _dennik_mod.can_cancel_tip
cancel_journal_tip = _dennik_mod.cancel_journal_tip
delete_journal_tip = _dennik_mod.delete_journal_tip
fetch_next_match_per_team = _dennik_mod.fetch_next_match_per_team
format_tip_text = _dennik_mod.format_tip_text
is_tip_hit = _dennik_mod.is_tip_hit
load_balance = _dennik_mod.load_balance
load_journal_tips = _dennik_mod.load_journal_tips
reset_wallet_and_tips = _dennik_mod.reset_wallet_and_tips
resolve_journal_tips = _dennik_mod.resolve_journal_tips
save_balance = _dennik_mod.save_balance
save_fixture_tip = _dennik_mod.save_fixture_tip
save_journal_tips = _dennik_mod.save_journal_tips
tip_fair_odds = _dennik_mod.tip_fair_odds

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
OUTCOME_COLORS = ["#2dd4bf", "#fbbf24", "#38bdf8"]
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


@st.cache_data(ttl=900)
def load_upcoming_fixtures(league: str) -> list[dict]:
    return fetch_next_match_per_team(league)


@st.cache_data(ttl=3600)
def load_cached_injuries() -> tuple[str, dict[str, dict[str, pd.DataFrame]]]:
    return fetch_live_injuries_by_league()


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


def head_to_head_matches(
    matches: pd.DataFrame, home: str, away: str, limit: int = 5
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Vráti posledných `limit` vzájomných zápasov a celkovú bilanciu medzi `home` a `away`."""
    h2h = matches[
        ((matches["home_team"] == home) & (matches["away_team"] == away))
        | ((matches["home_team"] == away) & (matches["away_team"] == home))
    ].sort_values("date", ascending=False)

    home_wins = int(
        (((h2h["home_team"] == home) & (h2h["ftr"] == "H")) | ((h2h["away_team"] == home) & (h2h["ftr"] == "A"))).sum()
    )
    away_wins = int(
        (((h2h["home_team"] == away) & (h2h["ftr"] == "H")) | ((h2h["away_team"] == away) & (h2h["ftr"] == "A"))).sum()
    )
    draws = int((h2h["ftr"] == "D").sum())
    summary = {
        "total": int(len(h2h)),
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
    }
    return h2h.head(limit), summary


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


def probability_chart(probabilities: tuple[float, float, float], is_dark: bool = True) -> alt.Chart:
    frame = pd.DataFrame(
        {"Výsledok": OUTCOME_LABELS, "Pravdepodobnosť": list(probabilities)}
    )
    label_color = "#f8fafc" if is_dark else "#0f172a"
    axis_color = "#cbd5e1" if is_dark else "#334155"
    grid_color = "rgba(148,163,184,0.18)" if is_dark else "#e2e8f0"
    base = alt.Chart(frame).encode(
        x=alt.X(
            "Výsledok:N",
            sort=None,
            axis=alt.Axis(labelAngle=0, title=None, labelColor=axis_color),
        ),
        y=alt.Y(
            "Pravdepodobnosť:Q",
            axis=alt.Axis(
                format="%",
                title="Pravdepodobnosť",
                labelColor=axis_color,
                titleColor=axis_color,
                gridColor=grid_color,
            ),
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
    labels = base.mark_text(dy=-12, fontSize=14, fontWeight="bold", color=label_color).encode(
        text=alt.Text("Pravdepodobnosť:Q", format=".1%")
    )
    chart_bg = "transparent" if is_dark else "#ffffff"
    return (
        (bars + labels)
        .properties(height=300)
        .configure(background=chart_bg)
        .configure_view(strokeWidth=0, fill=chart_bg)
    )


def build_rating_comparison_chart(
    ratings: pd.DataFrame,
    history: pd.DataFrame,
    selected_teams: list[str],
    seasons_in_range: list[str],
    selected_system: str = "Všetky",
    is_dark: bool = True,
) -> list[alt.Chart]:
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

    palette = [
        "#38bdf8", "#2dd4bf", "#fbbf24", "#fb7185", "#a78bfa",
        "#34d399", "#f472b6", "#60a5fa", "#facc15", "#f87171",
        "#818cf8", "#4ade80", "#fb923c", "#22d3ee", "#e879f9",
        "#a3e635", "#c084fc", "#f97316", "#14b8a6", "#cbd5e1",
    ]
    team_colors = [palette[i % len(palette)] for i in range(len(selected_teams))]

    is_single = len(systems) == 1
    chart_height = 380 if is_single else 320

    charts: list[alt.Chart] = []
    use_canvas = len(selected_teams) > 12
    axis_color = "#cbd5e1" if is_dark else "#334155"
    grid_color = "rgba(148,163,184,0.18)" if is_dark else "#e2e8f0"
    chart_bg = "transparent" if is_dark else "#ffffff"
    title_color = "#f8fafc" if is_dark else "#0f172a"

    legend_cols = 5 if len(selected_teams) > 12 else 2
    legend_cfg = alt.Legend(
        orient="bottom",
        columns=legend_cols,
        symbolLimit=0,
        title=None,
        labelLimit=130,
        labelColor=axis_color,
    )

    x_enc = x_enc.axis(
        **(
            {"tickMinStep": 5, "labelColor": axis_color, "titleColor": axis_color, "gridColor": grid_color}
            if len(seasons_in_range) == 1
            else {"values": x_ticks, "labelAngle": 0, "labelPadding": 6, "labelColor": axis_color, "gridColor": grid_color}
        )
    )

    for sys_label, _ in systems:
        sub_df = df_long[df_long["Systém"] == sys_label].copy()

        enc_kwargs: dict = dict(
            x=x_enc,
            y=alt.Y(
                "Rating:Q",
                title=None,
                scale=y_scale,
                axis=alt.Axis(
                    tickCount=6,
                    labelPadding=6,
                    labelColor=axis_color,
                    gridColor=grid_color,
                ),
            ),
            color=alt.Color(
                "Tím:N",
                scale=alt.Scale(domain=selected_teams, range=team_colors),
                legend=legend_cfg,
            ),
            tooltip=[
                alt.Tooltip("Tím:N"),
                alt.Tooltip("Systém:N"),
                tooltip_x,
                alt.Tooltip("Rating:Q", format=".1f"),
            ],
        )

        sel = alt.selection_point(fields=["Tím"], bind="legend", on="click", clear="dblclick")
        enc_kwargs["opacity"] = alt.condition(
            sel,
            alt.value(0.95),
            alt.value(0.10 if use_canvas else 0.18),
        )
        enc_kwargs["strokeWidth"] = alt.condition(sel, alt.value(2.8), alt.value(1.3 if use_canvas else 1.5))
        lines = (
            alt.Chart(sub_df)
            .mark_line(point=len(seasons_in_range) == 1 and len(selected_teams) <= 8)
            .encode(**enc_kwargs)
            .add_params(sel)
        )

        baseline = (
            alt.Chart(sub_df)
            .mark_rule(color="#888888", strokeDash=[4, 4], strokeWidth=1.0)
            .encode(y=alt.Y(datum=1500.0, scale=y_scale))
        )

        layered = (
            alt.layer(baseline, lines)
            .properties(
                width="container",
                height=chart_height,
                title=alt.TitleParams(sys_label, color=title_color),
                padding={"left": 8, "right": 8, "top": 8, "bottom": 4},
            )
            .configure(background=chart_bg)
            .configure_view(strokeWidth=0, fill=chart_bg)
        )
        if use_canvas:
            layered = layered.properties(usermeta={"embedOptions": {"renderer": "canvas"}})
        charts.append(layered)

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
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"
    is_dark = st.session_state["theme_mode"] == "dark"

    if is_dark:
        theme_css = """
            [data-testid='stAppViewContainer'] {
                background-color: #0b1120;
                background-image:
                    radial-gradient(at 15% 10%, rgba(56, 189, 248, 0.14) 0px, transparent 45%),
                    radial-gradient(at 85% 20%, rgba(45, 212, 191, 0.11) 0px, transparent 45%),
                    radial-gradient(at 50% 90%, rgba(99, 102, 241, 0.10) 0px, transparent 50%),
                    linear-gradient(155deg, #090d16 0%, #0f172a 52%, #0c192c 100%);
                background-attachment: fixed;
            }
            [data-testid='stVerticalBlockBorderWrapper'] {
                background: rgba(21, 31, 50, 0.92);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 14px;
            }
        """
    else:
        theme_css = """
            :root, .stApp {
                --text-color: #0f172a;
                --background-color: #ffffff;
                --secondary-background-color: #f1f5f9;
            }
            [data-testid='stAppViewContainer'], .stApp {
                background: #ffffff !important;
                color: #0f172a !important;
            }
            [data-testid='stAppViewContainer'] h1,
            [data-testid='stAppViewContainer'] h2,
            [data-testid='stAppViewContainer'] h3,
            [data-testid='stAppViewContainer'] h4,
            [data-testid='stAppViewContainer'] p,
            [data-testid='stAppViewContainer'] label,
            [data-testid='stAppViewContainer'] [data-testid='stCaptionContainer'],
            [data-testid='stAppViewContainer'] [data-testid='stMarkdownContainer'] {
                color: #0f172a !important;
            }
            [data-testid='stVerticalBlockBorderWrapper'] {
                background: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 14px;
            }
            /* Segmented control / button group (Streamlit 1.67+ selectors) */
            [data-testid='stButtonGroup'],
            [data-baseweb='button-group'] {
                background-color: #f1f5f9 !important;
                border-radius: 10px !important;
            }
            button[data-testid='stBaseButton-segmented_control'],
            button[kind='segmented_control'],
            [data-testid='stButtonGroup'] button,
            [data-baseweb='button-group'] button {
                background-color: #f1f5f9 !important;
                color: #0f172a !important;
                border-color: #cbd5e1 !important;
            }
            button[data-testid='stBaseButton-segmented_control'] *,
            button[kind='segmented_control'] *,
            [data-testid='stButtonGroup'] button *,
            [data-baseweb='button-group'] button * {
                color: #0f172a !important;
            }
            button[data-testid='stBaseButton-segmented_control']:hover,
            button[kind='segmented_control']:hover,
            [data-testid='stButtonGroup'] button:hover {
                background-color: #e2e8f0 !important;
                color: #0f172a !important;
            }
            button[data-testid='stBaseButton-segmented_controlActive'],
            button[kind='segmented_controlActive'],
            [data-testid='stButtonGroup'] button[aria-pressed='true'],
            [data-testid='stButtonGroup'] button[data-active='true'],
            [data-baseweb='button-group'] button[aria-pressed='true'] {
                background-color: #1d3557 !important;
                color: #ffffff !important;
                border-color: #1d3557 !important;
            }
            button[data-testid='stBaseButton-segmented_controlActive'] *,
            button[kind='segmented_controlActive'] *,
            [data-testid='stButtonGroup'] button[aria-pressed='true'] *,
            [data-testid='stButtonGroup'] button[data-active='true'] *,
            [data-baseweb='button-group'] button[aria-pressed='true'] * {
                color: #ffffff !important;
            }
            /* Selectbox, Multiselect & Dropdown Popovers */
            [data-baseweb='select'] > div {
                background-color: #ffffff !important;
                border-color: #cbd5e1 !important;
                color: #0f172a !important;
            }
            [data-baseweb='select'] * {
                color: #0f172a !important;
            }
            [data-baseweb='popover'],
            [data-baseweb='popover'] ul,
            [data-baseweb='menu'] {
                background-color: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
            }
            [data-baseweb='menu'] li,
            [data-baseweb='menu'] li * {
                color: #0f172a !important;
            }
            [data-baseweb='menu'] li:hover {
                background-color: #f1f5f9 !important;
            }
            [data-baseweb='tag'] {
                background-color: #e0f2fe !important;
                color: #0369a1 !important;
            }
            [data-testid='stNumberInput'] > div {
                background-color: #ffffff !important;
                border-color: #cbd5e1 !important;
            }
            [data-testid='stNumberInput'] input {
                background-color: #ffffff !important;
                color: #0f172a !important;
            }
            [data-testid='stNumberInput'] button {
                background-color: #f1f5f9 !important;
                color: #0f172a !important;
            }
            button[data-testid='stBaseButton-secondary'],
            button[kind='secondary'],
            [data-testid='stButton'] button,
            .stButton > button {
                background-color: #f1f5f9 !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
            }
            button[data-testid='stBaseButton-secondary'] *,
            button[kind='secondary'] *,
            [data-testid='stButton'] button *,
            .stButton > button * {
                color: #0f172a !important;
            }
            button[data-testid='stBaseButton-secondary']:hover,
            button[kind='secondary']:hover,
            [data-testid='stButton'] button:hover,
            .stButton > button:hover {
                background-color: #e2e8f0 !important;
                color: #0f172a !important;
            }
            button[data-testid='stBaseButton-primary'],
            button[kind='primary'],
            [data-testid='stButton'] button[kind='primary'],
            .stButton > button[kind='primary'] {
                background-color: #1d3557 !important;
                color: #ffffff !important;
                border-color: #1d3557 !important;
            }
            button[data-testid='stBaseButton-primary'] *,
            button[kind='primary'] *,
            [data-testid='stButton'] button[kind='primary'] *,
            .stButton > button[kind='primary'] * {
                color: #ffffff !important;
            }
            [data-testid='stAlert'] {
                background-color: #ecfdf5 !important;
                color: #065f46 !important;
                border: 1px solid #a7f3d0 !important;
            }
            [data-testid='stAlert'] * {
                color: #065f46 !important;
            }
        """

    st.markdown(
        f"""
        <style>
            [data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'] {{display: none;}}
            [data-testid='stHeader'] {{background: transparent;}}
            .block-container {{padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem;}}
            {theme_css}
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

    local_storage = None
    if LocalStorage is not None:
        try:
            local_storage = LocalStorage(key="dennik_ls")
            if not st.session_state.get("dennik_ls_hydrated"):
                raw_ls = local_storage.getItem("futbal_dennik_state")
                if raw_ls is not None:
                    parsed_ls = json.loads(raw_ls) if isinstance(raw_ls, str) else raw_ls
                    if isinstance(parsed_ls, dict):
                        if "balance" in parsed_ls:
                            save_balance(float(parsed_ls["balance"]))
                        if isinstance(parsed_ls.get("tips"), list):
                            save_journal_tips(parsed_ls["tips"])
                st.session_state["dennik_ls_hydrated"] = True
        except Exception:
            local_storage = None

    sections = [
        "⚽ Predikcia",
        "📊 Rebríček",
        "📈 Grafy",
        "🏥 Zranenia",
        "📝 Tipérsky denník",
    ]
    section_titles = {
        "⚽ Predikcia": "⚽ Predikcia futbalových zápasov",
        "📊 Rebríček": "📊 Rebríček sily tímov",
        "📈 Grafy": "📈 Vývoj sily a porovnanie",
        "🏥 Zranenia": "🏥 Zranenia hráčov",
        "📝 Tipérsky denník": "📝 Tipérsky denník",
    }
    current_section = st.session_state.get("active_section_nav") or sections[0]
    if current_section not in sections:
        current_section = sections[0]

    hdr_left, hdr_right = st.columns([5, 1.3], vertical_alignment="center")
    hdr_left.title(section_titles[current_section])
    if hdr_right.button(
        "☀️ Light" if is_dark else "🌙 Dark",
        key="theme_toggle_btn",
        use_container_width=True,
    ):
        st.session_state["theme_mode"] = "light" if is_dark else "dark"
        st.rerun()

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

    preferred_matchups = {
        "Premier League": ("Arsenal", "Chelsea"),
        "La Liga": ("Real Madrid", "Barcelona"),
        "Serie A": ("Inter", "Juventus"),
        "Bundesliga": ("Bayern Munich", "Dortmund"),
        "Ligue 1": ("Paris SG", "Marseille"),
    }
    preferred_home, preferred_away = preferred_matchups.get(active_league, (teams[0], teams[min(1, len(teams) - 1)]))
    default_home = preferred_home if preferred_home in teams else teams[0]
    if "pred_home" not in st.session_state or st.session_state["pred_home"] not in teams:
        st.session_state["pred_home"] = default_home

    system_keys = list(SYSTEM_NAMES.keys())
    if "pred_system" not in st.session_state or st.session_state["pred_system"] not in system_keys:
        st.session_state["pred_system"] = system_keys[0]

    def _swap_teams() -> None:
        h_val = st.session_state.get("pred_home")
        a_val = st.session_state.get("pred_away")
        if h_val and a_val and h_val != a_val:
            st.session_state["pred_home"] = a_val
            st.session_state["pred_away"] = h_val

    if active_section == "⚽ Predikcia":
        with st.container(border=True):
            c_h, c_swap, c_a, c_s = st.columns([3, 0.7, 3, 3.3], vertical_alignment="bottom")
            home = c_h.selectbox("Domáci tím", teams, key="pred_home")
            away_options = [t for t in teams if t != home]
            default_away = preferred_away if preferred_away in away_options else away_options[0]
            if "pred_away" not in st.session_state or st.session_state["pred_away"] not in away_options:
                st.session_state["pred_away"] = default_away
            c_swap.button(
                "⇄",
                key="swap_teams_btn",
                help="Vymeniť domáci a hosťujúci tím",
                on_click=_swap_teams,
                use_container_width=True,
            )
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
        home_logo_uri = get_team_logo_data_uri(home)
        away_logo_uri = get_team_logo_data_uri(away)
        home_logo_html = (
            f"<img src='{home_logo_uri}' style='width:32px;height:32px;object-fit:contain;vertical-align:middle;' />"
            if home_logo_uri
            else ""
        )
        away_logo_html = (
            f"<img src='{away_logo_uri}' style='width:32px;height:32px;object-fit:contain;vertical-align:middle;' />"
            if away_logo_uri
            else ""
        )
        hdr_title_color = "#f8fafc" if is_dark else "#0f172a"
        hdr_vs_color = "#94a3b8" if is_dark else "#64748b"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:6px 0 4px 0;'>"
            f"{home_logo_html}"
            f"<span style='font-size:24px;font-weight:700;color:{hdr_title_color};'>{home}</span>"
            f"<span style='font-size:18px;font-weight:600;color:{hdr_vs_color};'>–</span>"
            f"{away_logo_html}"
            f"<span style='font-size:24px;font-weight:700;color:{hdr_title_color};'>{away}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Predikcia na základe aktuálnych ratingov zo sezóny {season}: "
            f"**{home}** ({int(home_row['matches_played'])} zápasov k {home_row['as_of_date']:%d. %m. %Y}) vs. "
            f"**{away}** ({int(away_row['matches_played'])} zápasov k {away_row['as_of_date']:%d. %m. %Y})."
        )

        c_card_bg = "rgba(21,31,50,0.9)" if is_dark else "#f8fafc"
        c_card_bd = "rgba(148,163,184,0.18)" if is_dark else "#e2e8f0"
        c_lbl = "#94a3b8" if is_dark else "#475569"
        c_draw = "#fbbf24" if is_dark else "#b45309"
        c_home = "#2dd4bf" if is_dark else "#0f766e"
        c_away = "#38bdf8" if is_dark else "#1d4ed8"
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 12px 0;">
              <div style="background:{c_card_bg};border:1px solid {c_card_bd};border-radius:12px;padding:10px 6px;text-align:center;">
                <div style="font-size:12px;color:{c_lbl};">Výhra domácich</div>
                <div style="font-size:22px;font-weight:700;color:{c_home};">{probs[0]:.1%}</div>
              </div>
              <div style="background:{c_card_bg};border:1px solid {c_card_bd};border-radius:12px;padding:10px 6px;text-align:center;">
                <div style="font-size:12px;color:{c_lbl};">Remíza</div>
                <div style="font-size:22px;font-weight:700;color:{c_draw};">{probs[1]:.1%}</div>
              </div>
              <div style="background:{c_card_bg};border:1px solid {c_card_bd};border-radius:12px;padding:10px 6px;text-align:center;">
                <div style="font-size:12px;color:{c_lbl};">Výhra hostí</div>
                <div style="font-size:22px;font-weight:700;color:{c_away};">{probs[2]:.1%}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.altair_chart(probability_chart(probs, is_dark=is_dark), use_container_width=True, theme=None)

        best_idx = max(range(3), key=lambda i: probs[i])
        st.success(
            f"Systém **{SYSTEM_NAMES[selected_system].split(' ')[0]}** tipuje "
            f"**{OUTCOME_LABELS[best_idx].lower()}** s pravdepodobnosťou **{probs[best_idx]:.1%}**."
        )

        st.divider()
        st.subheader("Porovnanie oboch tímov")
        left, right = st.columns(2)
        for col, team_name, row in ((left, home, home_row), (right, away, away_row)):
            t_logo = get_team_logo_data_uri(team_name)
            t_logo_html = (
                f"<img src='{t_logo}' style='width:24px;height:24px;object-fit:contain;vertical-align:middle;' />"
                if t_logo
                else ""
            )
            t_col_txt = "#f8fafc" if is_dark else "#0f172a"
            col.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
                f"{t_logo_html}"
                f"<span style='font-size:20px;font-weight:700;color:{t_col_txt};'>{team_name}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            results = recent_results(matches, season, team_name)
            col.markdown(form_badges(results), unsafe_allow_html=True)
            col.write(
                f"- **Elo rating:** {row['elo_rating']:.1f}\n"
                f"- **Glicko rating:** {row['glicko_rating']:.1f} (± {row['glicko_rd']:.1f} RD)\n"
                f"- **Glicko-2 rating:** {row['glicko2_rating']:.1f} (± {row['glicko2_rd']:.1f} RD, "
                f"volatilita {row['glicko2_vol']:.4f})"
            )

        st.markdown("#### Čo tipujú jednotlivé systémy?")
        tbl_bg = "rgba(21,31,50,0.85)" if is_dark else "#ffffff"
        tbl_hdr = "rgba(30,41,59,0.95)" if is_dark else "#f1f5f9"
        tbl_hdr_txt = "#94a3b8" if is_dark else "#475569"
        tbl_txt = "#f1f5f9" if is_dark else "#0f172a"
        tbl_sub = "#e2e8f0" if is_dark else "#334155"
        tbl_bd = "rgba(148,163,184,0.14)" if is_dark else "#e2e8f0"
        rows_html = "".join(
            f"<tr style='border-bottom:1px solid {tbl_bd};'>"
            f"<td style='padding:8px 6px;font-weight:600;color:{tbl_txt};'>{SYSTEM_NAMES[k].split(' ')[0]}</td>"
            f"<td style='padding:8px 6px;text-align:center;color:{tbl_sub};'>{p_tuple[0]:.1%}</td>"
            f"<td style='padding:8px 6px;text-align:center;color:{tbl_sub};'>{p_tuple[1]:.1%}</td>"
            f"<td style='padding:8px 6px;text-align:center;color:{tbl_sub};'>{p_tuple[2]:.1%}</td>"
            f"</tr>"
            for k, p_tuple in all_preds.items()
        )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:14px;background:{tbl_bg};border:1px solid {tbl_bd};border-radius:10px;overflow:hidden;'>"
            f"<thead><tr style='background:{tbl_hdr};text-align:center;color:{tbl_hdr_txt};'>"
            f"<th style='padding:8px 6px;text-align:left;'>Systém</th>"
            f"<th style='padding:8px 6px;'>Domáci</th>"
            f"<th style='padding:8px 6px;'>Remíza</th>"
            f"<th style='padding:8px 6px;'>Hostia</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Vzájomné zápasy (H2H)")
        h2h_recent, h2h_sum = head_to_head_matches(matches, home, away, limit=5)
        if h2h_sum["total"] == 0:
            st.info(f"Tímy {home} a {away} sa v dostupných ligových dátach zatiaľ nestretli.")
        else:
            st.caption(
                f"Celková ligová bilancia ({h2h_sum['total']} zápasov): "
                f"**{home}** {h2h_sum['home_wins']}× výhra · "
                f"Remízy: {h2h_sum['draws']}× · "
                f"**{away}** {h2h_sum['away_wins']}× výhra"
            )
            h2h_rows = []
            for r in h2h_recent.itertuples(index=False):
                date_str = pd.Timestamp(r.date).strftime("%d. %m. %Y")
                score_str = f"{int(r.fthg)} : {int(r.ftag)}"
                if r.ftr == "D":
                    res_badge = (
                        f"<span style='background:{FORM_STYLE['R'][0]};color:{FORM_STYLE['R'][1]};"
                        f"padding:2px 8px;border-radius:6px;font-size:12px;font-weight:700;'>Remíza</span>"
                    )
                else:
                    winner = r.home_team if r.ftr == "H" else r.away_team
                    badge_bg = "#2a9d8f" if winner == home else "#38bdf8"
                    badge_fg = "#ffffff" if winner == home else "#0f172a"
                    res_badge = (
                        f"<span style='background:{badge_bg};color:{badge_fg};"
                        f"padding:2px 8px;border-radius:6px;font-size:12px;font-weight:700;'>{winner}</span>"
                    )
                h2h_rows.append(
                    f"<tr style='border-bottom:1px solid {tbl_bd};'>"
                    f"<td style='padding:8px 6px;color:{tbl_hdr_txt};white-space:nowrap;'>{date_str}</td>"
                    f"<td style='padding:8px 6px;color:{tbl_hdr_txt};white-space:nowrap;'>{r.season}</td>"
                    f"<td style='padding:8px 6px;color:{tbl_txt};font-weight:600;'>{r.home_team} – {r.away_team}</td>"
                    f"<td style='padding:8px 6px;text-align:center;font-weight:700;color:{tbl_txt};white-space:nowrap;'>{score_str}</td>"
                    f"<td style='padding:8px 6px;text-align:center;'>{res_badge}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;font-size:14px;background:{tbl_bg};border:1px solid {tbl_bd};border-radius:10px;overflow:hidden;'>"
                f"<thead><tr style='background:{tbl_hdr};text-align:left;color:{tbl_hdr_txt};'>"
                f"<th style='padding:8px 6px;'>Dátum</th>"
                f"<th style='padding:8px 6px;'>Sezóna</th>"
                f"<th style='padding:8px 6px;'>Zápas</th>"
                f"<th style='padding:8px 6px;text-align:center;'>Skóre</th>"
                f"<th style='padding:8px 6px;text-align:center;'>Víťaz</th>"
                f"</tr></thead><tbody>{''.join(h2h_rows)}</tbody></table>",
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
                accent, bg, badge = (
                    ("#facc15", "rgba(250, 204, 21, 0.08)", "🥇 1.")
                    if is_dark
                    else ("#d4af37", "#fffdf4", "🥇 1.")
                )
            elif idx == 2:
                accent, bg, badge = (
                    ("#cbd5e1", "rgba(148, 163, 184, 0.08)", "🥈 2.")
                    if is_dark
                    else ("#9aa0a6", "#fafafa", "🥈 2.")
                )
            elif idx == 3:
                accent, bg, badge = (
                    ("#fb923c", "rgba(251, 146, 60, 0.08)", "🥉 3.")
                    if is_dark
                    else ("#cd7f32", "#fdfaf6", "🥉 3.")
                )
            elif idx == 4:
                accent, bg, badge = (
                    ("#2dd4bf", "rgba(21, 31, 50, 0.90)", "4.")
                    if is_dark
                    else ("#2a9d8f", "#ffffff", "4.")
                )
            elif idx > n_teams - 3:
                accent, bg, badge = (
                    ("#fb7185", "rgba(251, 113, 133, 0.08)", f"{idx}.")
                    if is_dark
                    else ("#e76f51", "#fff9f8", f"{idx}.")
                )
            else:
                accent, bg, badge = (
                    ("#475569", "rgba(21, 31, 50, 0.85)", f"{idx}.")
                    if is_dark
                    else ("#cbd5e1", "#ffffff", f"{idx}.")
                )

            active_val = float(getattr(row, sort_col))
            pct = max(6.0, min(100.0, ((active_val - r_min) / r_span) * 94.0 + 6.0))
            bar_color = "#2dd4bf" if idx <= 4 else ("#fb7185" if idx > n_teams - 3 else "#38bdf8")

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

            c_title = "#f8fafc" if is_dark else "#1d3557"
            c_pill_bg = "rgba(30,41,59,0.9)" if is_dark else "#eef1f5"
            c_pill_txt = "#cbd5e1" if is_dark else "#475569"
            c_box_bg = "rgba(15,23,42,0.65)" if is_dark else "#f1f5f9"
            c_muted = "#94a3b8" if is_dark else "#64748b"
            c_val = "#e2e8f0" if is_dark else "#0f172a"
            c_active = "#38bdf8" if is_dark else "#1d4ed8"
            c_bd = "rgba(148,163,184,0.18)" if is_dark else "#e2e8f0"

            def _cell(label: str, val: float, is_active: bool) -> str:
                style = f"color:{c_active};font-size:15px;" if is_active else f"color:{c_val};"
                return f"<div><span style='font-size:11px;color:{c_muted};'>{label}</span><br><b style='{style}'>{val:.1f}</b></div>"

            rating_cells = (
                _cell("Kombi", row.combined_rating, rank_by == "Kombinácia")
                + _cell("Glicko-2", row.glicko2_rating, rank_by == "Glicko-2")
                + _cell("Glicko", row.glicko_rating, rank_by == "Glicko")
                + _cell("Elo", row.elo_rating, rank_by == "Elo")
            )

            cards_html.append(
                f"<div style='border:1px solid {c_bd};border-left:5px solid {accent};"
                f"border-radius:12px;padding:11px 12px;margin-bottom:9px;background:{bg};'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;gap:6px;'>"
                f"<span style='display:flex;align-items:center;gap:7px;font-size:16px;font-weight:700;color:{c_title};'>{badge} {logo_img} {row.team}</span>"
                f"<span style='display:flex;align-items:center;gap:6px;'>{mini_form}"
                f"<span style='font-size:11px;color:{c_pill_txt};background:{c_pill_bg};padding:2px 7px;border-radius:10px;font-weight:600;'>{int(row.matches_played)} záp.</span>"
                f"</span></div>"
                f"<div style='background:rgba(148,163,184,0.18);height:6px;border-radius:4px;margin:8px 0 9px 0;overflow:hidden;'>"
                f"<div style='width:{pct:.1f}%;height:100%;background:{bar_color};border-radius:4px;'></div>"
                f"</div>"
                f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:4px;font-size:13px;"
                f"background:{c_box_bg};border-radius:8px;padding:6px 6px;text-align:center;'>"
                f"{rating_cells}"
                f"</div>"
                f"<div style='margin-top:6px;font-size:11px;color:{c_muted};display:flex;justify-content:space-between;'>"
                f"<span>Neistota (RD): <b style='color:{c_pill_txt};'>±{row.glicko2_rd:.1f}</b></span>"
                f"<span>Volatilita (σ): <b style='color:{c_pill_txt};'>{row.glicko2_vol:.4f}</b></span>"
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
        preset_map = {
            "Premier League": (
                "Big 6 + Everton",
                ["Man United", "Arsenal", "Liverpool", "Chelsea", "Man City", "Tottenham", "Everton"],
            ),
            "La Liga": (
                "Top kluby La Ligy",
                ["Real Madrid", "Barcelona", "Ath Madrid", "Ath Bilbao", "Valencia", "Sevilla", "Sociedad"],
            ),
            "Serie A": (
                "Top kluby Serie A",
                ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Atalanta", "Lazio"],
            ),
            "Bundesliga": (
                "Top kluby Bundesligy",
                ["Bayern Munich", "Dortmund", "Leverkusen", "RB Leipzig", "Stuttgart", "Ein Frankfurt"],
            ),
            "Ligue 1": (
                "Top kluby Ligue 1",
                ["Paris SG", "Marseille", "Monaco", "Lyon", "Lille", "Nice", "Lens"],
            ),
        }
        preset_label, preset_candidates = preset_map.get(
            active_league, ("Top kluby", teams_in_range[:6])
        )
        big_clubs = [t for t in preset_candidates if t in teams_in_range]
        default_teams = big_clubs if big_clubs else teams_in_range[:3]

        if (
            "comp_teams_widget" not in st.session_state
            or st.session_state.get("comp_teams_league") != active_league
        ):
            st.session_state["comp_teams_widget"] = default_teams
            st.session_state["comp_teams_league"] = active_league

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
                ratings, history, selected_teams, seasons_in_range, selected_system=chart_system, is_dark=is_dark
            )
            for ch in comp_charts:
                st.altair_chart(ch, use_container_width=True, theme=None)

        st.divider()
        st.subheader(f"Presnosť na sezóne {metadata['test_season']}")
        c_bg = "rgba(21,31,50,0.85)" if is_dark else "#ffffff"
        c_hdr = "rgba(30,41,59,0.95)" if is_dark else "#f1f5f9"
        c_hdr_txt = "#94a3b8" if is_dark else "#475569"
        c_txt = "#f1f5f9" if is_dark else "#0f172a"
        c_sub = "#e2e8f0" if is_dark else "#334155"
        c_bd = "rgba(148,163,184,0.14)" if is_dark else "#e2e8f0"
        comp_rows_html = "".join(
            f"<tr style='border-bottom:1px solid {c_bd};'>"
            f"<td style='padding:8px 6px;font-weight:600;color:{c_txt};'>{r['model']}</td>"
            f"<td style='padding:8px 6px;text-align:center;color:{c_sub};'>{r['accuracy']:.1%}</td>"
            f"<td style='padding:8px 6px;text-align:center;color:{c_sub};'>{r['log_loss']:.3f}</td>"
            f"<td style='padding:8px 6px;text-align:center;color:{c_sub};'>{r['brier']:.3f}</td>"
            f"</tr>"
            for r in metadata["comparison"]
        )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:14px;background:{c_bg};border:1px solid {c_bd};border-radius:10px;overflow:hidden;'>"
            f"<thead><tr style='background:{c_hdr};text-align:center;color:{c_hdr_txt};'>"
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
        updated_at, injuries_by_league = load_cached_injuries()
        league_options = list(LEAGUES)

        selected_league = (
            st.segmented_control(
                "Vyber ligu",
                options=league_options,
                default=active_league if active_league in league_options else league_options[0],
                key="injuries_league_toggle",
            )
            or league_options[0]
        )

        lg_ratings, _, _, _ = load_data(selected_league)
        lg_seasons = sorted(lg_ratings["season"].unique(), reverse=True)
        lg_season = CURRENT_SEASON if CURRENT_SEASON in lg_seasons else lg_seasons[0]
        all_league_clubs = sorted(lg_ratings[lg_ratings["season"] == lg_season]["team"].unique())

        injuries_by_club = injuries_by_league.get(selected_league, {})
        featured_clubs = sorted(
            [c for c in all_league_clubs if not injuries_by_club.get(c, pd.DataFrame()).empty],
            key=lambda c: len(injuries_by_club.get(c, pd.DataFrame())),
            reverse=True,
        )[:3]

        club_key = f"injuries_club_select_{selected_league}"
        if club_key not in st.session_state or st.session_state[club_key] not in all_league_clubs:
            st.session_state[club_key] = featured_clubs[0] if featured_clubs else all_league_clubs[0]

        if featured_clubs:
            sel_cols = st.columns(len(featured_clubs))
            for idx, club_name in enumerate(featured_clubs):
                is_selected = st.session_state[club_key] == club_name
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
                        st.session_state[club_key] = club_name
                        st.rerun()

        selected_club = st.selectbox(
            f"Vyber ľubovoľný klub ({selected_league} – {len(all_league_clubs)} klubov)",
            options=all_league_clubs,
            key=club_key,
            format_func=lambda c: f"{c} ({len(injuries_by_club[c])} zran.)" if c in injuries_by_club and not injuries_by_club[c].empty else f"{c} (0 zran.)",
        )
        club_df = injuries_by_club.get(selected_club, pd.DataFrame())

        st.divider()
        hdr_logo, hdr_title = st.columns([0.8, 9.2], vertical_alignment="center")
        sel_logo = CLUB_LOGOS.get(selected_club)
        if sel_logo and sel_logo.exists():
            hdr_logo.image(str(sel_logo), width=48)
        hdr_title.markdown(f"### {selected_club} ({selected_league}) – aktuálne zranenia")

        if club_df.empty:
            st.info(f"Pre klub **{selected_club}** momentálne nie je nikto zranený.")
        else:
            n_inj = int((club_df["Stav"] != "Suspendovaný").sum())
            n_susp = int((club_df["Stav"] == "Suspendovaný").sum())

            m_cols = st.columns(3)
            m_cols[0].metric("Absencie spolu", len(club_df))
            m_cols[1].metric("Zranení hráči", n_inj)
            m_cols[2].metric("Suspendovaní / trest", n_susp)

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

                inj_bg = "rgba(21,31,50,0.92)" if is_dark else "#ffffff"
                inj_bd = "rgba(148,163,184,0.18)" if is_dark else "#e2e8f0"
                inj_title = "#f8fafc" if is_dark else "#1d3557"
                inj_txt = "#cbd5e1" if is_dark else "#334155"
                inj_bold = "#f1f5f9" if is_dark else "#0f172a"
                inj_muted = "#94a3b8" if is_dark else "#475569"
                inj_pill_bg = "rgba(56,189,248,0.16)" if is_dark else "#e0f2fe"
                inj_pill_txt = "#38bdf8" if is_dark else "#0369a1"
                extra_dates = ""
                if str(row.get("Kedy sa zranil", "")).strip() or str(row.get("Na ako dlho", "")).strip():
                    extra_dates = (
                        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:4px;'>"
                        f"<div><b style='color:{inj_bold};'>Kedy:</b> {row['Kedy sa zranil']}</div>"
                        f"<div><b style='color:{inj_bold};'>Na ako dlho:</b> {row['Na ako dlho']}</div>"
                        f"</div>"
                    )
                card_html = (
                    f"<div style='border:1px solid {inj_bd};border-radius:12px;padding:12px 14px;"
                    f"margin-bottom:10px;background:{inj_bg};'>"
                    f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>"
                    f"{avatar_html}"
                    f"<div>"
                    f"<div style='font-size:17px;font-weight:700;color:{inj_title};line-height:1.2;'>{row['Hráč']}</div>"
                    f"<div style='font-size:12px;color:{inj_muted};margin-top:3px;'>"
                    f"<span style='background:{inj_pill_bg};color:{inj_pill_txt};padding:2px 7px;border-radius:6px;font-weight:600;'>{row['Pozícia']}</span>"
                    f" &nbsp;·&nbsp; <b style='color:{inj_bold};'>{row['Stav']}</b></div>"
                    f"</div></div>"
                    f"<div style='font-size:13.5px;line-height:1.5;color:{inj_txt};'>"
                    f"<div><b style='color:{inj_bold};'>Zranenie / dôvod:</b> {row['Zranenie']}</div>"
                    f"{extra_dates}"
                    f"<div style='margin-top:4px;'><b style='color:{inj_bold};'>Predpokladaný návrat:</b> {row['Predpokladaný návrat']}</div>"
                    f"</div></div>"
                )
                col.markdown(card_html, unsafe_allow_html=True)

        if updated_at:
            st.caption(f"Údaje aktualizované k {updated_at} (živé dáta + záloha `data/injuries.json`).")

    elif active_section == "📝 Tipérsky denník":
        st.subheader(f"Najbližšie neodohrané zápasy – {active_league}")
        st.caption(
            "Pre každý tím ligy je zobrazený 1 najbližší neodohraný zápas. "
            "Zadaj vklad (1–100 €) a vyber tip. Pred začiatkom zápasu môžeš tip stornovať a vklad sa ti vráti."
        )

        tips = load_journal_tips()
        if tips:
            matches_by_lg: dict[str, pd.DataFrame] = {}
            for lg in {t.get("league") for t in tips if t.get("league") in LEAGUES and t.get("actual_ftr") is None}:
                _, _, lg_matches, _ = load_data(lg)
                matches_by_lg[lg] = lg_matches
            tips, changed = resolve_journal_tips(tips, matches_by_lg)
            if changed:
                save_journal_tips(tips)
                st.session_state["dennik_ls_dirty"] = True

        balance = load_balance()
        in_play = round(sum(float(t.get("stake", 0.0)) for t in tips if t.get("actual_ftr") is None), 2)
        resolved_tips = [t for t in tips if t.get("actual_ftr") in ("H", "D", "A")]
        hits = sum(1 for t in resolved_tips if is_tip_hit(t.get("tip_code", ""), t.get("actual_ftr", "")))
        acc = (hits / len(resolved_tips)) if resolved_tips else 0.0

        w_bg = "rgba(21,31,50,0.92)" if is_dark else "#f8fafc"
        w_bd = "rgba(148,163,184,0.18)" if is_dark else "#e2e8f0"
        w_lbl = "#94a3b8" if is_dark else "#475569"
        w_bal = "#2dd4bf" if is_dark else "#0f766e"
        w_play = "#38bdf8" if is_dark else "#1d4ed8"
        w_txt = "#f8fafc" if is_dark else "#0f172a"

        w_c1, w_c2 = st.columns([5.2, 1.3], vertical_alignment="center")
        w_c1.markdown(
            f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:6px;">
              <div style="background:{w_bg};border:1px solid {w_bd};border-radius:12px;padding:10px 8px;text-align:center;">
                <div style="font-size:12px;color:{w_lbl};">💶 Zostatok na účte</div>
                <div style="font-size:21px;font-weight:700;color:{w_bal};">{balance:.2f} €</div>
              </div>
              <div style="background:{w_bg};border:1px solid {w_bd};border-radius:12px;padding:10px 8px;text-align:center;">
                <div style="font-size:12px;color:{w_lbl};">🎟️ V hre (aktívne tipy)</div>
                <div style="font-size:21px;font-weight:700;color:{w_play};">{in_play:.2f} €</div>
              </div>
              <div style="background:{w_bg};border:1px solid {w_bd};border-radius:12px;padding:10px 8px;text-align:center;">
                <div style="font-size:12px;color:{w_lbl};">📊 Úspešnosť</div>
                <div style="font-size:21px;font-weight:700;color:{w_txt};">{f"{hits}/{len(resolved_tips)} ({acc:.0%})" if resolved_tips else "–"}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if w_c2.button("🔄 Reset (100 €)", key="reset_wallet_btn", help="Obnoviť zostatok na 100.00 € a vymazať tipy", use_container_width=True):
            reset_wallet_and_tips()
            st.session_state["dennik_ls_dirty"] = True
            st.toast("Účet obnovený na 100.00 €", icon="💶")
            st.rerun()

        saved_lookup: dict[tuple[str, str, str, str], dict] = {
            (t.get("league", ""), t.get("home", ""), t.get("away", ""), t.get("match_date", "")): t
            for t in tips
            if t.get("actual_ftr") is None
        }

        with st.spinner(f"Načítavam najbližšie zápasy pre {active_league}..."):
            fixtures = load_upcoming_fixtures(active_league)

        if not fixtures:
            st.info(f"Pre ligu {active_league} sa momentálne nenašli žiadne naplánované zápasy v najbližších 14 dňoch.")
        else:
            ratings_by_team = {
                row["team"]: row
                for _, row in season_ratings.iterrows()
            }
            c_muted = "#94a3b8" if is_dark else "#475569"
            c_title = "#f8fafc" if is_dark else "#0f172a"

            for fx in fixtures:
                f_home = fx["home"]
                f_away = fx["away"]
                h_logo = get_team_logo_data_uri(f_home)
                a_logo = get_team_logo_data_uri(f_away)
                h_img = f"<img src='{h_logo}' style='width:22px;height:22px;object-fit:contain;vertical-align:middle;' /> " if h_logo else ""
                a_img = f" <img src='{a_logo}' style='width:22px;height:22px;object-fit:contain;vertical-align:middle;' />" if a_logo else ""

                p_comb: tuple[float, float, float] = (0.34, 0.33, 0.33)
                prob_line = ""
                if f_home in ratings_by_team and f_away in ratings_by_team:
                    fx_preds = compute_all_predictions(ratings_by_team[f_home], ratings_by_team[f_away])
                    p_comb = fx_preds["combined"]
                    prob_line = f" &nbsp;·&nbsp; Model: 1 ({p_comb[0]:.0%}) · X ({p_comb[1]:.0%}) · 2 ({p_comb[2]:.0%})"

                existing_tip = saved_lookup.get((active_league, f_home, f_away, fx["match_date"]))
                active_code = existing_tip.get("tip_code") if existing_tip else None
                cancellable = can_cancel_tip(existing_tip) if existing_tip else True
                badge_bg = "rgba(56,189,248,0.16)" if is_dark else "#e0f2fe"
                badge_fg = "#38bdf8" if is_dark else "#0369a1"
                if existing_tip and active_code:
                    e_stake = float(existing_tip.get("stake", 0.0))
                    e_odds = float(existing_tip.get("odds", 1.0))
                    e_win = float(existing_tip.get("potential_win", round(e_stake * e_odds, 2)))
                    lock_lbl = "" if cancellable else " · 🔒 Zápas už začal"
                    tip_badge_html = (
                        f"<div style='margin-bottom:6px;'>"
                        f"<span style='display:inline-block;background:{badge_bg};color:{badge_fg};"
                        f"padding:3px 10px;border-radius:8px;font-size:12.5px;font-weight:700;'>"
                        f"🎯 Tvoj tip: {format_tip_text(active_code, f_home, f_away)} · "
                        f"Vklad {e_stake:.2f} € (kurz {e_odds:.2f} → výhra {e_win:.2f} €){lock_lbl}</span></div>"
                    )
                else:
                    tip_badge_html = ""

                with st.container(border=True):
                    top_l, top_r = st.columns([4.8, 1.3], vertical_alignment="center")
                    top_l.markdown(
                        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:4px;'>"
                        f"<span style='font-size:16px;font-weight:700;color:{c_title};'>{h_img}{f_home} – {f_away}{a_img}</span>"
                        f"<span style='font-size:12px;color:{c_muted};'>🕒 {fx['kickoff_str']}{prob_line}</span>"
                        f"</div>{tip_badge_html}",
                        unsafe_allow_html=True,
                    )
                    default_stake = float(existing_tip.get("stake", 10.0)) if existing_tip else min(10.0, max(1.0, balance))
                    default_stake = max(1.0, min(100.0, default_stake))
                    stake_val = top_r.number_input(
                        "Vklad (€)",
                        min_value=1.0,
                        max_value=100.0,
                        value=default_stake,
                        step=1.0,
                        format="%.2f",
                        key=f"stake_{active_league}_{fx['event_id']}",
                        disabled=bool(existing_tip and not cancellable),
                    )

                    btn_specs = [
                        ("1", f"1 · {f_home}", f"Výhra {f_home}"),
                        ("1X", f"1X · {f_home}/X", f"Výhra alebo remíza {f_home}"),
                        ("X", "X · Remíza", "Remíza"),
                        ("X2", f"X2 · {f_away}/X", f"Výhra alebo remíza {f_away}"),
                        ("2", f"2 · {f_away}", f"Výhra {f_away}"),
                    ]
                    if existing_tip and cancellable:
                        b_cols = st.columns([1.15, 1.25, 0.9, 1.25, 1.15, 0.75])
                    else:
                        b_cols = st.columns([1.15, 1.25, 0.9, 1.25, 1.15])

                    for idx_b, (t_code, t_label, t_help) in enumerate(btn_specs):
                        is_sel = active_code == t_code
                        f_odds = tip_fair_odds(t_code, p_comb)
                        btn_text = f"✓ {t_label}" if is_sel else f"{t_label} ({f_odds:.2f})"
                        if b_cols[idx_b].button(
                            btn_text,
                            key=f"dennik_{active_league}_{fx['event_id']}_{t_code}",
                            type="primary" if is_sel else "secondary",
                            help=f"{t_help} · kurz {f_odds:.2f}" + (" (kliknutím stornuješ)" if is_sel and cancellable else ""),
                            disabled=bool(existing_tip and not cancellable),
                            use_container_width=True,
                        ):
                            if is_sel and existing_tip:
                                ok, msg = cancel_journal_tip(existing_tip["id"])
                                if ok:
                                    st.session_state["dennik_ls_dirty"] = True
                                st.toast(msg, icon="💶" if ok else "⚠️")
                            else:
                                ok, msg, _ = save_fixture_tip(
                                    league=active_league,
                                    season=season,
                                    home=f_home,
                                    away=f_away,
                                    kickoff_str=fx["kickoff_str"],
                                    match_date=fx["match_date"],
                                    tip_code=t_code,
                                    stake=stake_val,
                                    odds=f_odds,
                                    kickoff_utc_iso=fx.get("kickoff_utc_iso", ""),
                                )
                                if ok:
                                    st.session_state["dennik_ls_dirty"] = True
                                    st.toast(
                                        f"Tip uložený: {format_tip_text(t_code, f_home, f_away)} za {stake_val:.2f} €",
                                        icon="📝",
                                    )
                                else:
                                    st.toast(msg, icon="⚠️")
                            st.rerun()

                    if existing_tip and cancellable:
                        if b_cols[5].button(
                            "✕ Storno",
                            key=f"dennik_cancel_{active_league}_{fx['event_id']}",
                            help="Stornovať tip pred začiatkom zápasu a vrátiť vklad",
                            use_container_width=True,
                        ):
                            ok, msg = cancel_journal_tip(existing_tip["id"])
                            if ok:
                                st.session_state["dennik_ls_dirty"] = True
                            st.toast(msg, icon="💶" if ok else "⚠️")
                            st.rerun()

        st.divider()
        st.subheader("Moje uložené tipy")
        if not tips:
            st.info("Zatiaľ nemáš uložené žiadne tipy. Zadaj vklad (1–100 €) a klikni na možnosť pri zápase vyššie.")
        else:
            for tip in tips:
                tid = tip["id"]
                t_code = tip.get("tip_code", "")
                t_stake = float(tip.get("stake", 0.0))
                t_odds = float(tip.get("odds", 1.0))
                t_pot = float(tip.get("potential_win", round(t_stake * t_odds, 2)))
                actual = tip.get("actual_ftr")
                if actual in ("H", "D", "A"):
                    hit = is_tip_hit(t_code, actual)
                    badge = f"✅ Výhra {tip.get('payout', t_pot):.2f} €" if hit else "❌ Prehra (0.00 €)"
                    res_info = f"Skóre {tip.get('actual_score', '')} · {badge}"
                else:
                    res_info = f"⏳ Čaká sa · Možná výhra {t_pot:.2f} €"

                cancellable_row = can_cancel_tip(tip)
                with st.container(border=True):
                    c_info, c_pick, c_del = st.columns([4.6, 3.6, 1.4], vertical_alignment="center")
                    c_info.markdown(
                        f"**{tip['home']} – {tip['away']}**  \n"
                        f"<span style='font-size:12px;color:#94a3b8;'>{tip['league']} · {tip.get('kickoff_str', '')}</span>",
                        unsafe_allow_html=True,
                    )
                    c_pick.markdown(
                        f"Tip: **{format_tip_text(t_code, tip['home'], tip['away'])}** · **{t_stake:.2f} €** (kurz {t_odds:.2f})  \n"
                        f"<span style='font-size:12px;color:#94a3b8;'>{res_info}</span>",
                        unsafe_allow_html=True,
                    )
                    if actual is None:
                        if cancellable_row:
                            if c_del.button("✕ Stornovať", key=f"cancel_journal_{tid}", help="Vráti vklad na účet", use_container_width=True):
                                ok, msg = cancel_journal_tip(tid)
                                if ok:
                                    st.session_state["dennik_ls_dirty"] = True
                                st.toast(msg, icon="💶" if ok else "⚠️")
                                st.rerun()
                        else:
                            c_del.caption("🔒 Prebieha")
                    else:
                        if c_del.button("🗑️ Zmazať", key=f"del_journal_{tid}", help="Odstrániť z histórie", use_container_width=True):
                            delete_journal_tip(tid)
                            st.session_state["dennik_ls_dirty"] = True
                            st.rerun()

    if local_storage is not None and st.session_state.pop("dennik_ls_dirty", False):
        seq = st.session_state.get("dennik_ls_seq", 0) + 1
        st.session_state["dennik_ls_seq"] = seq
        payload = json.dumps(
            {"balance": load_balance(), "tips": load_journal_tips()},
            ensure_ascii=False,
        )
        local_storage.setItem("futbal_dennik_state", payload, key=f"ls_save_{seq}")


main()
