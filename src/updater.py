"""Modul pre živú aktualizáciu dát z football-data.co.uk v aplikácii Streamlit."""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import pandas as pd

from src.config import (
    ALL_SEASON_CODES,
    ALL_SEASONS,
    BASE_URL,
    CURRENT_SEASON,
    MATCHES_CSV,
    RAW_DIR,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
    ensure_directories,
)
from src.evaluate_ratings import CURRENT_RATINGS_CSV, RATING_HISTORY_CSV, compute_all_ratings
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


def update_live_season_pipeline(season: str = CURRENT_SEASON) -> dict[str, object]:
    """Stiahne najnovšie výsledky, očistí ich a prepočíta Elo, Glicko a Glicko-2."""
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

    # 2. Spracovanie a prepočet ratingov
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

        history, current = compute_all_ratings(merged)
        history.to_csv(RATING_HISTORY_CSV, index=False, encoding="utf-8")
        current.to_csv(CURRENT_RATINGS_CSV, index=False, encoding="utf-8")

        cur_matches = merged[merged["season"] == season]
        matches_count = len(cur_matches)
        last_date = (
            cur_matches["date"].max().strftime("%d. %m. %Y")
            if matches_count > 0
            else "neznámy"
        )

        return {
            "success": True,
            "message": (
                f"Dáta boli úspešne stiahnuté a prepočítané! "
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
