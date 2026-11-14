"""Modul pre živú aktualizáciu dát z football-data.co.uk v aplikácii Streamlit."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import (
    ALL_SEASON_CODES,
    ALL_SEASONS,
    BASE_URL,
    CURRENT_SEASON,
    FEATURES_CSV,
    MATCHES_CSV,
    RAW_DIR,
    REQUEST_TIMEOUT_SECONDS,
    TEAM_STATES_CSV,
    USER_AGENT,
    ensure_directories,
)
from src.features import build_features, build_team_states
from src.prepare import OUTPUT_COLUMNS, clean_matches, load_season


def fetch_season_data(season: str = CURRENT_SEASON) -> bytes:
    """Stiahne čerstvý CSV súbor pre zvolenú sezónu."""
    if season not in ALL_SEASON_CODES:
        raise ValueError(f"Neznáma sezóna: {season}")

    code = ALL_SEASON_CODES[season]
    url = BASE_URL.format(code=code)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        content = response.read()

    if len(content) < 1_000:
        raise RuntimeError(f"Príliš malá odpoveď ({len(content)} B) zo servera.")

    return content


def update_live_season_pipeline(season: str = CURRENT_SEASON) -> dict[str, any]:
    """Stiahne najnovšie výsledky, očistí ich a prepočíta priebežné atribúty a stav tímov.

    Vracia slovník s výsledkom:
        success: bool
        message: str
        matches_count: int
        last_date: str
    """
    ensure_directories()
    code = ALL_SEASON_CODES.get(season)
    if not code:
        return {
            "success": False,
            "message": f"Sezóna '{season}' nemá priradený kód v konfigurácii.",
        }

    raw_file = RAW_DIR / f"E0_{code}.csv"

    # 1. Stiahnutie z internetu
    try:
        content = fetch_season_data(season)
        # Overenie platnosti CSV
        test_df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
        if not {"HomeTeam", "AwayTeam", "FTR"}.issubset(test_df.columns):
            return {
                "success": False,
                "message": "Stiahnutý súbor neobsahuje očakávanú štruktúru futbalových dát.",
            }
        raw_file.write_bytes(content)
    except urllib.error.URLError as e:
        return {
            "success": False,
            "message": f"Chyba pripojenia k football-data.co.uk: {e.reason}",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Nepodarilo sa stiahnuť dáta: {e}",
        }

    # 2. Spracovanie a zjednotenie všetkých sezón
    try:
        frames: list[pd.DataFrame] = []
        for s in ALL_SEASONS:
            path = RAW_DIR / f"E0_{ALL_SEASON_CODES[s]}.csv"
            if path.exists():
                frames.append(load_season(s))

        if not frames:
            return {"success": False, "message": "Nenašli sa žiadne raw CSV súbory."}

        merged, _ = clean_matches(pd.concat(frames, ignore_index=True))
        merged = merged[OUTPUT_COLUMNS]
        merged.to_csv(MATCHES_CSV, index=False, encoding="utf-8")

        # 3. Prepočet atribútov a stavu tímov
        features = build_features(merged)
        features.to_csv(FEATURES_CSV, index=False, encoding="utf-8")

        states = build_team_states(features)
        states.to_csv(TEAM_STATES_CSV, index=False, encoding="utf-8")

        # Štatistiky k aktuálnej sezóne
        cur_matches = merged[merged["season"] == season]
        matches_count = len(cur_matches)
        if matches_count > 0:
            last_date = cur_matches["date"].max().strftime("%d. %m. %Y")
        else:
            last_date = "neznámy"

        return {
            "success": True,
            "message": (
                f"Dáta boli úspešne stiahnuté a aktualizované! "
                f"V sezóne {season} je k {last_date} odohratých {matches_count} zápasov."
            ),
            "matches_count": matches_count,
            "last_date": last_date,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Chyba pri spracovaní dát: {e}",
        }
