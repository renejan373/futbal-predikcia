"""Streamlit aplikácia – predikcia výsledku futbalového zápasu.

Spustenie:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.config import (
    FEATURES_CSV,
    MATCHES_CSV,
    MODELS_DIR,
    MODEL_METADATA_JSON,
    TARGET_NAMES_SK,
    TEAM_STATES_CSV,
)
from src.features import BASE_FEATURES, FEATURE_COLUMNS
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


@st.cache_data(ttl=600)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not FEATURES_CSV.exists():
        st.error(
            "Chýbajú spracované dáta. Spusti v termináli:\n\n"
            "```\npython -m src.download\npython -m src.prepare\npython -m src.features\n```"
        )
        st.stop()
    features = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    features["season"] = features["season"].astype(str)
    matches = pd.read_csv(MATCHES_CSV, parse_dates=["date"])
    matches["season"] = matches["season"].astype(str)
    states = pd.read_csv(TEAM_STATES_CSV, parse_dates=["as_of_date"])
    states["season"] = states["season"].astype(str)
    metadata = json.loads(MODEL_METADATA_JSON.read_text(encoding="utf-8"))
    return features, matches, states, metadata


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def team_state(states: pd.DataFrame, season: str, team: str, venue: str) -> pd.Series:
    row = states[
        (states["season"] == season)
        & (states["team"] == team)
        & (states["venue"] == venue)
    ]
    if row.empty:
        st.error(f"Pre tím {team} chýba stav pre sezónu {season}.")
        st.stop()
    return row.iloc[0]


def h2h_history(
    features: pd.DataFrame, home: str, away: str, until: pd.Timestamp
) -> tuple[int, int, int, pd.DataFrame]:
    """Body a gólový rozdiel domácich z posledných vzájomných zápasov."""
    pair = features[
        (
            ((features["home_team"] == home) & (features["away_team"] == away))
            | ((features["home_team"] == away) & (features["away_team"] == home))
        )
        & (features["date"] <= until)
    ].sort_values("date")
    last = pair.tail(5)

    points = 0
    goal_diff = 0
    for row in last.itertuples(index=False):
        if row.home_team == home:
            goals_for, goals_against = row.fthg, row.ftag
        else:
            goals_for, goals_against = row.ftag, row.fthg
        points += 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
        goal_diff += goals_for - goals_against

    return points, goal_diff, len(last), last


def build_feature_row(
    home_state: pd.Series,
    away_state: pd.Series,
    h2h_points: int,
    h2h_goal_diff: int,
    h2h_matches: int,
) -> pd.DataFrame:
    row: dict[str, float] = {}
    for column in BASE_FEATURES:
        row[f"{column}_H"] = float(home_state[column])
        row[f"{column}_A"] = float(away_state[column])

    row["elo_H"] = float(home_state["elo"])
    row["elo_A"] = float(away_state["elo"])

    row["is_promoted_H"] = int(home_state["is_promoted"])
    row["is_promoted_A"] = int(away_state["is_promoted"])
    row["h2h_pts_H"] = float(h2h_points)
    row["h2h_gd_H"] = float(h2h_goal_diff)
    row["h2h_matches"] = float(h2h_matches)

    row["elo_diff"] = row["elo_H"] - row["elo_A"]
    row["form_diff"] = row["form_pts_5_H"] - row["form_pts_5_A"]
    row["venue_form_diff"] = row["form_pts_venue_5_H"] - row["form_pts_venue_5_A"]
    row["pts_pg_diff"] = row["season_pts_pg_H"] - row["season_pts_pg_A"]
    row["rest_diff"] = row["rest_days_H"] - row["rest_days_A"]
    row["attack_defense_H"] = row["gf_avg_5_H"] - row["ga_avg_5_A"]
    row["attack_defense_A"] = row["gf_avg_5_A"] - row["ga_avg_5_H"]
    row["sot_adv_H"] = row["sot_for_avg_5_H"] - row["sot_against_avg_5_A"]
    row["sot_adv_A"] = row["sot_for_avg_5_A"] - row["sot_against_avg_5_H"]

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def recent_results(
    matches: pd.DataFrame, team: str, before: pd.Timestamp, limit: int = 5
) -> list[str]:
    rows = matches[
        ((matches["home_team"] == team) | (matches["away_team"] == team))
        & (matches["date"] < before)
    ].sort_values("date").tail(limit)

    results = []
    for row in rows.itertuples(index=False):
        if row.home_team == team:
            goals_for, goals_against = row.fthg, row.ftag
        else:
            goals_for, goals_against = row.ftag, row.fthg
        results.append("V" if goals_for > goals_against else "R" if goals_for == goals_against else "P")
    return results


def form_badges(results: list[str]) -> str:
    if not results:
        return "<span style='color:#888'>žiadne zápasy</span>"
    badges = []
    for result in results:
        background, color = FORM_STYLE[result]
        badges.append(
            f"<span style='display:inline-block;width:28px;height:28px;line-height:28px;"
            f"text-align:center;border-radius:6px;background:{background};color:{color};"
            f"font-weight:600;margin-right:4px'>{result}</span>"
        )
    return "".join(badges)


def probability_chart(probabilities) -> alt.Chart:
    frame = pd.DataFrame(
        {"Výsledok": OUTCOME_LABELS, "Pravdepodobnosť": probabilities}
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
    return (bars + labels).properties(height=320)


def render_prediction_tab(
    probabilities,
    home: str,
    away: str,
    home_state: pd.Series,
    away_state: pd.Series,
) -> None:
    best_index = int(probabilities.argmax())
    best_probability = float(probabilities[best_index])

    h_matches = int(home_state.get("matches_played", 0))
    a_matches = int(away_state.get("matches_played", 0))
    st.subheader(f"{home} – {away}")
    st.caption(
        f"Predikcia vychádza z odohratých zápasov v sezóne: "
        f"**{home}** ({h_matches} zápasov k {home_state['as_of_date']:%d. %m. %Y}) vs. "
        f"**{away}** ({a_matches} zápasov k {away_state['as_of_date']:%d. %m. %Y})."
    )

    columns = st.columns(3)
    for column, label, probability in zip(columns, OUTCOME_LABELS, probabilities):
        column.metric(label, f"{probability:.1%}")

    st.altair_chart(probability_chart(probabilities), width="stretch")
    st.success(
        f"Model tipuje **{OUTCOME_LABELS[best_index].lower()}** "
        f"s pravdepodobnosťou **{best_probability:.1%}**."
    )
    st.info(
        "Predikcia je odhad na základe historických dát a nemá slúžiť ako "
        "stávková rada. Futbal je ovplyvnený aj náhodnými udalosťami "
        "(zranenia, červené karty, rozhodcovské verdikty)."
    )


def render_comparison_tab(
    matches: pd.DataFrame,
    home: str,
    away: str,
    home_state: pd.Series,
    away_state: pd.Series,
    h2h_points: int,
    h2h_goal_diff: int,
    h2h_matches: int,
    h2h_frame: pd.DataFrame,
) -> None:
    st.subheader("Porovnanie aktuálnej formy")

    left, right = st.columns(2)
    for column, team, state in (
        (left, home, home_state),
        (right, away, away_state),
    ):
        results = recent_results(matches, team, state["as_of_date"])
        column.markdown(f"### {team}")
        column.markdown(form_badges(results), unsafe_allow_html=True)
        column.caption(
            f"Body z posledných 5 zápasov: **{state['form_pts_5']:.0f}** · "
            f"priemer gólov: **{state['gf_avg_5']:.2f}** strelených / "
            f"**{state['ga_avg_5']:.2f}** inkasovaných"
        )

    table = pd.DataFrame(
        {
            "Ukazovateľ": [
                "Body z posledných 5 zápasov",
                "Body na zápas v sezóne",
                "Priemer strelených gólov",
                "Priemer inkasovaných gólov",
                "Priemer striel na bránu",
                "Elo rating",
                "Dni od posledného zápasu",
                "Odohrané zápasy v sezóne",
            ],
            home: [
                f"{home_state['form_pts_5']:.0f}",
                f"{home_state['season_pts_pg']:.2f}",
                f"{home_state['gf_avg_5']:.2f}",
                f"{home_state['ga_avg_5']:.2f}",
                f"{home_state['sot_for_avg_5']:.2f}",
                f"{home_state['elo']:.0f}",
                f"{home_state['rest_days']:.0f}",
                f"{home_state['matches_played']:.0f}",
            ],
            away: [
                f"{away_state['form_pts_5']:.0f}",
                f"{away_state['season_pts_pg']:.2f}",
                f"{away_state['gf_avg_5']:.2f}",
                f"{away_state['ga_avg_5']:.2f}",
                f"{away_state['sot_for_avg_5']:.2f}",
                f"{away_state['elo']:.0f}",
                f"{away_state['rest_days']:.0f}",
                f"{away_state['matches_played']:.0f}",
            ],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Vzájomná bilancia (posledných 5 zápasov)")
    if h2h_matches == 0:
        st.write("Tímy proti sebe v dostupných dátach ešte nehrali.")
    else:
        summary = st.columns(3)
        summary[0].metric("Body domácich vo vzájomných zápasoch", f"{h2h_points:.0f}")
        summary[1].metric("Gólový rozdiel domácich", f"{h2h_goal_diff:+.0f}")
        summary[2].metric("Odohraných vzájomných zápasov", f"{h2h_matches}")

        history = h2h_frame.assign(
            Dátum=h2h_frame["date"].dt.strftime("%d. %m. %Y"),
            Zápas=h2h_frame["home_team"] + " – " + h2h_frame["away_team"],
            Výsledok=h2h_frame["fthg"].astype(str) + " : " + h2h_frame["ftag"].astype(str),
        )[["Dátum", "Zápas", "Výsledok"]].iloc[::-1]
        st.dataframe(history, hide_index=True, width="stretch")


def render_model_tab(metadata: dict, selected_key: str) -> None:
    st.subheader("O použitom modeli")
    model_info = metadata["models"][selected_key]
    metrics = model_info["test_metrics"]

    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    columns[1].metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
    columns[2].metric("Log loss", f"{metrics['log_loss']:.3f}")
    columns[3].metric("Brier score", f"{metrics['brier']:.3f}")

    st.caption(
        f"Metriky sú z testovacej sezóny {metadata['test_season']}, ktorú model "
        "počas trénovania nevidel."
    )

    st.markdown("#### Porovnanie všetkých modelov")
    rows = []
    for key, info in metadata["models"].items():
        test = info["test_metrics"]
        rows.append(
            {
                "Model": info["name_sk"],
                "Accuracy": f"{test['accuracy']:.1%}",
                "F1 (macro)": f"{test['f1_macro']:.3f}",
                "Log loss": f"{test['log_loss']:.3f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("#### Nastavenie modelu")
    st.json(model_info.get("best_params", {}))

    st.markdown("#### Dátová základňa")
    st.write(
        f"- Tréningové sezóny: {', '.join(metadata['train_seasons'])}\n"
        f"- Validačná sezóna: {metadata['valid_season']}\n"
        f"- Testovacia sezóna: {metadata['test_season']}\n"
        f"- Počet zápasov: {metadata['n_train'] + metadata['n_valid'] + metadata['n_test']}\n"
        f"- Počet atribútov: {len(FEATURE_COLUMNS)}\n"
        f"- Knižnica scikit-learn: {metadata['sklearn_version']}"
    )


def main() -> None:
    features, matches, states, metadata = load_data()

    st.title("⚽ Predikcia výsledkov futbalových zápasov")
    st.caption(
        "Stredoškolská odborná činnosť · odbor 11 – Informatika · "
        "historické dáta Premier League (football-data.co.uk)"
    )

    seasons = sorted(states["season"].unique(), reverse=True)
    model_keys = list(metadata["models"].keys())
    best_key = metadata.get("best_model", {}).get("key", model_keys[0])

    with st.sidebar:
        st.header("Nastavenia predikcie")
        season = st.selectbox("Sezóna", seasons)

        teams = sorted(states[states["season"] == season]["team"].unique())
        default_home = "Arsenal" if "Arsenal" in teams else teams[0]
        home = st.selectbox("Domáci tím", teams, index=teams.index(default_home))

        away_options = [team for team in teams if team != home]
        default_away = "Chelsea" if "Chelsea" in away_options else away_options[0]
        away = st.selectbox(
            "Hosťujúci tím",
            away_options,
            index=away_options.index(default_away),
        )

        selected_model = st.selectbox(
            "Model",
            model_keys,
            index=model_keys.index(best_key),
            format_func=lambda key: metadata["models"][key]["name_sk"],
        )
        st.caption(
            f"Predvolený model je vybraný podľa log loss na sezóne "
            f"{metadata['valid_season']}."
        )
        st.divider()
        st.subheader("🔄 Živá aktualizácia")
        st.caption(
            "Stiahne najnovšie odohraté zápasy z internetu a prepočíta formu tímov."
        )
        if st.button("Stiahnuť najnovšie výsledky", use_container_width=True):
            with st.spinner(f"Sťahujem a prepočítavam sezónu {season}..."):
                res = update_live_season_pipeline(season)
                if res["success"]:
                    st.success(res["message"])
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(res["message"])

        st.caption(
            "Dáta: football-data.co.uk · "
            f"testovacia sezóna {metadata['test_season']}"
        )

    home_state = team_state(states, season, home, "home")
    away_state = team_state(states, season, away, "away")

    season_end = features[features["season"] == season]["date"].max()
    h2h_points, h2h_goal_diff, h2h_matches, h2h_frame = h2h_history(
        features, home, away, season_end
    )

    features_row = build_feature_row(
        home_state,
        away_state,
        h2h_points,
        h2h_goal_diff,
        h2h_matches,
    )

    model_path = MODELS_DIR / f"{selected_model}.pkl"
    if not model_path.exists():
        st.error(
            f"Chýba súbor modelu {model_path.name}. Spusti `python -m src.train`."
        )
        st.stop()

    model = load_model(str(model_path))
    probabilities = model.predict_proba(features_row)[0]

    prediction_tab, comparison_tab, model_tab = st.tabs(
        ["Predikcia", "Porovnanie tímov", "O modeli"]
    )
    with prediction_tab:
        render_prediction_tab(
            probabilities, home, away, home_state, away_state
        )
    with comparison_tab:
        render_comparison_tab(
            matches,
            home,
            away,
            home_state,
            away_state,
            h2h_points,
            h2h_goal_diff,
            h2h_matches,
            h2h_frame,
        )
    with model_tab:
        render_model_tab(metadata, selected_model)


main()