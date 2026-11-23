"""Načítanie prehľadu zranených hráčov pre Top 5 európskych líg a vybrané kluby."""

from __future__ import annotations

import base64
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, PROJECT_DIR

INJURIES_JSON: Path = DATA_DIR / "injuries.json"
LOGOS_DIR: Path = PROJECT_DIR / "assets" / "logos"
PLAYERS_DIR: Path = PROJECT_DIR / "assets" / "players"

CLUB_LOGOS: dict[str, Path] = {
    # Premier League
    "Arsenal": LOGOS_DIR / "arsenal.png",
    "Aston Villa": LOGOS_DIR / "aston_villa.png",
    "Bournemouth": LOGOS_DIR / "bournemouth.png",
    "Brentford": LOGOS_DIR / "brentford.png",
    "Brighton": LOGOS_DIR / "brighton.png",
    "Chelsea": LOGOS_DIR / "chelsea.png",
    "Coventry": LOGOS_DIR / "coventry.png",
    "Crystal Palace": LOGOS_DIR / "crystal_palace.png",
    "Everton": LOGOS_DIR / "everton.png",
    "Fulham": LOGOS_DIR / "fulham.png",
    "Hull": LOGOS_DIR / "hull.png",
    "Ipswich": LOGOS_DIR / "ipswich.png",
    "Leeds": LOGOS_DIR / "leeds.png",
    "Liverpool": LOGOS_DIR / "liverpool.png",
    "Man City": LOGOS_DIR / "man_city.png",
    "Man United": LOGOS_DIR / "man_united.png",
    "Newcastle": LOGOS_DIR / "newcastle.png",
    "Nott'm Forest": LOGOS_DIR / "nottm_forest.png",
    "Sunderland": LOGOS_DIR / "sunderland.png",
    "Tottenham": LOGOS_DIR / "tottenham.png",
    # La Liga
    "Alaves": LOGOS_DIR / "alaves.png",
    "Ath Bilbao": LOGOS_DIR / "ath_bilbao.png",
    "Ath Madrid": LOGOS_DIR / "atletico_madrid.png",
    "Atlético Madrid": LOGOS_DIR / "atletico_madrid.png",
    "Barcelona": LOGOS_DIR / "fc_barcelona.png",
    "FC Barcelona": LOGOS_DIR / "fc_barcelona.png",
    "Betis": LOGOS_DIR / "betis.png",
    "Celta": LOGOS_DIR / "celta.png",
    "Elche": LOGOS_DIR / "elche.png",
    "Espanol": LOGOS_DIR / "espanol.png",
    "Getafe": LOGOS_DIR / "getafe.png",
    "La Coruna": LOGOS_DIR / "la_coruna.png",
    "Levante": LOGOS_DIR / "levante.png",
    "Malaga": LOGOS_DIR / "malaga.png",
    "Osasuna": LOGOS_DIR / "osasuna.png",
    "Real Madrid": LOGOS_DIR / "real_madrid.png",
    "Santander": LOGOS_DIR / "santander.png",
    "Sevilla": LOGOS_DIR / "sevilla.png",
    "Sociedad": LOGOS_DIR / "sociedad.png",
    "Valencia": LOGOS_DIR / "valencia.png",
    "Vallecano": LOGOS_DIR / "vallecano.png",
    "Villarreal": LOGOS_DIR / "villarreal.png",
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


@lru_cache(maxsize=64)
def get_team_logo_data_uri(team_name: str) -> str:
    """Vráti base64 data URI loga tímu pre vloženie do HTML karty."""
    path = CLUB_LOGOS.get(team_name)
    if not path or not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _player_slug(player_name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", player_name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")


@lru_cache(maxsize=128)
def get_player_photo_data_uri(player_name: str) -> str:
    """Vráti base64 data URI fotky hráča z assets/players/ (alebo prázdny reťazec)."""
    slug = _player_slug(player_name)
    for ext, mime in ((".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".png", "image/png")):
        path = PLAYERS_DIR / f"{slug}{ext}"
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    return ""

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
