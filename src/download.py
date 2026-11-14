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
    ALL_SEASON_CODES as SEASON_CODES,
    BASE_URL,
    RAW_DIR,
    REQUEST_TIMEOUT_SECONDS,
    SEASON_CODES,
    USER_AGENT,
    ensure_directories,
)


def raw_path(season: str) -> Path:
    """Cesta k raw CSV pre danú sezónu."""
    return RAW_DIR / f"E0_{SEASON_CODES[season]}.csv"


def fetch(url: str) -> bytes:
    """Stiahne URL s rozumným User-Agentom a timeoutom."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


def download_season(season: str, force: bool = False) -> Path:
    """Stiahne jednu sezónu a overí základnú štruktúru CSV."""
    path = raw_path(season)
    if path.exists() and not force:
        print(f"[skip] {season}: {path.name} uz existuje")
        return path

    url = BASE_URL.format(code=SEASON_CODES[season])
    print(f"[get ] {season}: {url}")
    content = fetch(url)

    if len(content) < 1_000:
        raise RuntimeError(f"Prilis mala odpoved ({len(content)} B) pre {season}")

    # latin-1 je bezpecne cistit len na validaciu; subor ukladame tak, ako prisiel.
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
        "--season",
        choices=SEASON_CODES,
        action="append",
        help="konkretna sezona (mozno zadat viackrat); predvolene vsetky",
    )
    args = parser.parse_args()

    ensure_directories()
    seasons = args.season or list(SEASON_CODES)

    failures: list[tuple[str, str]] = []
    for season in seasons:
        try:
            download_season(season, force=args.force)
        except (urllib.error.URLError, TimeoutError, RuntimeError, pd.errors.ParserError) as exc:
            failures.append((season, str(exc)))
            print(f"[err ] {season}: {exc}", file=sys.stderr)

    if failures:
        print(f"\nZlyhalo {len(failures)} z {len(seasons)} sezon.", file=sys.stderr)
        return 1

    print(f"\nHotovo: {len(seasons)} sezon v {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
