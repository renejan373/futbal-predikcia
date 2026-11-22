"""Modul pre živú aktualizáciu dát z football-data.co.uk v aplikácii Streamlit."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pandas as pd

from src.config import (
    ALL_SEASON_CODES,
    ALL_SEASONS,
    BASE_URL,
    CURRENT_SEASON,
    LEAGUES,
    RAW_DIR,
    REQUEST_TIMEOUT_SECONDS,
    TEST_SEASON,
    USER_AGENT,
    ensure_directories,
    get_league_paths,
)
from src.evaluate_ratings import compute_all_ratings, evaluate_models
from src.prepare import OUTPUT_COLUMNS, clean_matches, load_season


def fetch_season_data(season: str = CURRENT_SEASON, league: str = "Premier League") -> bytes:
    """Stiahne čerstvý CSV súbor pre zvolenú sezónu a ligu."""
    if season not in ALL_SEASON_CODES:
        raise ValueError(f"Neznáma sezóna: {season}")

    code = ALL_SEASON_CODES[season]
    div = LEAGUES[league]["div"]
    url = BASE_URL.format(code=code, div=div)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        content = response.read()

    if len(content) < 1_000:
        raise RuntimeError(f"Príliš malá odpoveď ({len(content)} B) zo servera.")

    return content


def update_live_season_pipeline(
    season: str = CURRENT_SEASON, league: str = "Premier League"
) -> dict[str, object]:
    """Stiahne najnovšie výsledky, očistí ich a prepočíta Elo, Glicko a Glicko-2 pre zvolenú ligu."""
    ensure_directories()
    code = ALL_SEASON_CODES.get(season)
    if not code:
        return {
            "success": False,
            "message": f"Sezóna '{season}' nemá priradený kód v konfigurácii.",
        }

    div = LEAGUES[league]["div"]
    paths = get_league_paths(league)
    raw_file = RAW_DIR / f"{div}_{code}.csv"

    # 1. Stiahnutie z internetu
    try:
        content = fetch_season_data(season, league=league)
        test_df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
        if not {"HomeTeam", "AwayTeam", "FTR"}.issubset(test_df.columns):
            return {
                "success": False,
                "message": "Stiahnutý súbor neobsahuje očakávanú štruktúru futbalových dát.",
            }
        if raw_file.exists():
            existing_df = pd.read_csv(raw_file, encoding="latin-1")
            if len(test_df) >= len(existing_df):
                raw_file.write_bytes(content)
        else:
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
            path = RAW_DIR / f"{div}_{ALL_SEASON_CODES[s]}.csv"
            if path.exists():
                frames.append(load_season(s, league=league))

        if not frames:
            return {"success": False, "message": "Nenašli sa žiadne raw CSV súbory."}

        merged, _ = clean_matches(pd.concat(frames, ignore_index=True))
        merged = merged[OUTPUT_COLUMNS]
        merged.to_csv(paths["matches_csv"], index=False, encoding="utf-8")

        history, current = compute_all_ratings(merged)
        history.to_csv(paths["rating_history_csv"], index=False, encoding="utf-8")
        current.to_csv(paths["current_ratings_csv"], index=False, encoding="utf-8")

        comparison = evaluate_models(history, season=TEST_SEASON)
        seasons_list = sorted(history["season"].unique())
        meta = {
            "league": league,
            "test_season": TEST_SEASON,
            "total_matches": int(len(history)),
            "n_seasons": int(len(seasons_list)),
            "first_season": seasons_list[0] if seasons_list else "",
            "last_season": seasons_list[-1] if seasons_list else "",
            "comparison": comparison.to_dict(orient="records"),
        }
        paths["rating_metadata_json"].write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

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
