"""Stiahnutie historických dát Premier League z football-data.co.uk.

Použitie:
    python -m src.download            # stiahne všetky sezóny, existujúce preskočí
    python -m src.download --force    # stiahne znova
    python -m src.download --season 2025/26
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    ALL_SEASON_CODES,
    BASE_URL,
    LEAGUES,
    RAW_DIR,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
    ensure_directories,
)


def raw_path(season: str, league: str = "Premier League") -> Path:
    """Cesta k raw CSV pre danú sezónu a ligu."""
    div = LEAGUES[league]["div"]
    return RAW_DIR / f"{div}_{ALL_SEASON_CODES[season]}.csv"


def fetch(url: str) -> bytes:
    """Stiahne URL s rozumným User-Agentom a timeoutom."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


def download_season(season: str, force: bool = False, league: str = "Premier League") -> Path:
    """Stiahne jednu sezónu pre zvolenú ligu a overí základnú štruktúru CSV."""
    path = raw_path(season, league=league)
    if path.exists() and not force:
        print(f"[skip] {league} {season}: {path.name} uz existuje")
        return path

    div = LEAGUES[league]["div"]
    url = BASE_URL.format(code=ALL_SEASON_CODES[season], div=div)
    print(f"[get ] {season}: {url}")
    content = fetch(url)

    if len(content) < 1_000:
        raise RuntimeError(f"Prilis mala odpoved ({len(content)} B) pre {season}")

    # latin-1 je bezpecne cistit len na validaciu; ak ma CSV viac stlpcov na konci riadku (napr. 2003/04, 2004/05),
    # orezeme prebytocne koncove stlpce na dlzku hlavicky, aby sa nestratil ziadny zapas.
    try:
        frame = pd.read_csv(io.BytesIO(content), encoding="latin-1")
    except pd.errors.ParserError:
        import csv

        text = content.decode("latin-1")
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            raise
        n_cols = len(rows[0])
        cleaned_rows = [rows[0]] + [r[:n_cols] for r in rows[1:] if any(cell.strip() for cell in r)]
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(cleaned_rows)
        content = buf.getvalue().encode("latin-1")
        frame = pd.read_csv(io.BytesIO(content), encoding="latin-1")
    expected = {"HomeTeam", "AwayTeam", "FTR"}
    if not expected.issubset(frame.columns):
        raise RuntimeError(
            f"Neocakavana struktura CSV pre {season}: {sorted(frame.columns)[:10]}"
        )

    path.write_bytes(content)
    print(f"[ok  ] {season}: {len(frame)} zapasov -> {path.name}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stiahne CSV dat z football-data.co.uk")
    parser.add_argument("--force", action="store_true", help="stiahni aj existujuce subory")
    parser.add_argument(
        "--league",
        choices=list(LEAGUES),
        default="Premier League",
        help="liga na stiahnutie (predvolene Premier League)",
    )
    parser.add_argument(
        "--season",
        choices=ALL_SEASON_CODES,
        action="append",
        help="konkretna sezona (mozno zadat viackrat); predvolene vsetky",
    )
    args = parser.parse_args()

    ensure_directories()
    seasons = args.season or list(ALL_SEASON_CODES)

    failures: list[tuple[str, str]] = []
    for season in seasons:
        try:
            download_season(season, force=args.force, league=args.league)
        except (urllib.error.URLError, TimeoutError, RuntimeError, pd.errors.ParserError) as exc:
            failures.append((season, str(exc)))
            print(f"[err ] {args.league} {season}: {exc}", file=sys.stderr)

    if failures:
        print(f"\nZlyhalo {len(failures)} z {len(seasons)} sezon.", file=sys.stderr)
        return 1

    print(f"\nHotovo: {len(seasons)} sezon v {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
