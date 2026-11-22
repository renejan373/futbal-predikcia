"""Načítanie prehľadu zranených hráčov pre Top 5 európskych líg a vybrané kluby."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, PROJECT_DIR

INJURIES_JSON: Path = DATA_DIR / "injuries.json"
LOGOS_DIR: Path = PROJECT_DIR / "assets" / "logos"

CLUB_LOGOS: dict[str, Path] = {
    # Premier League
    "Arsenal": LOGOS_DIR / "arsenal.png",
    "Liverpool": LOGOS_DIR / "liverpool.png",
    "Man City": LOGOS_DIR / "man_city.png",
    # La Liga
    "Real Madrid": LOGOS_DIR / "real_madrid.png",
    "FC Barcelona": LOGOS_DIR / "fc_barcelona.png",
    "Atlético Madrid": LOGOS_DIR / "atletico_madrid.png",
    # Serie A
    "Inter Miláno": LOGOS_DIR / "inter_milano.png",
    "AC Miláno": LOGOS_DIR / "ac_milano.png",
    "Juventus": LOGOS_DIR / "juventus.png",
    # Bundesliga
    "Bayern Mníchov": LOGOS_DIR / "bayern_mnichov.png",
    "Borussia Dortmund": LOGOS_DIR / "borussia_dortmund.png",
    "Bayer Leverkusen": LOGOS_DIR / "bayer_leverkusen.png",
    # Ligue 1
    "Paris Saint-Germain": LOGOS_DIR / "psg.png",
    "Olympique Marseille": LOGOS_DIR / "marseille.png",
    "AS Monaco": LOGOS_DIR / "monaco.png",
}

COLUMN_RENAME = {
    "player": "Hráč",
    "position": "Pozícia",
    "injury": "Zranenie",
    "injured_on": "Kedy sa zranil",
    "duration": "Na ako dlho",
    "expected_return": "Predpokladaný návrat",
    "context": "Kde sa zranil (zápas / tréning)",
    "status": "Stav",
}


def load_injuries_by_league() -> tuple[str, dict[str, dict[str, pd.DataFrame]]]:
    """Vráti dátum aktualizácie a slovník {liga: {klub: DataFrame zranených hráčov}}."""
    if not INJURIES_JSON.exists():
        return "", {}

    payload = json.loads(INJURIES_JSON.read_text(encoding="utf-8"))
    updated_at = str(payload.get("updated_at", ""))
    leagues_raw = payload.get("leagues", {})

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for league, clubs_raw in leagues_raw.items():
        league_clubs: dict[str, pd.DataFrame] = {}
        for club, records in clubs_raw.items():
            league_clubs[club] = pd.DataFrame(records).rename(columns=COLUMN_RENAME)
        result[league] = league_clubs
    return updated_at, result
