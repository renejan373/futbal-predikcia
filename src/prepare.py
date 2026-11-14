"""Cistenie a zlucenie raw CSV do jedneho konzistentneho datasetu.

Vstup:  data/raw/E0_<kod>.csv
Vystup: data/interim/matches.csv
        reports/tables/01_data_quality.md

Použitie:
    python -m src.prepare
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import (
    ALL_SEASON_CODES,
    ALL_SEASONS,
    INTERIM_DIR,
    MATCHES_CSV,
    NUMERIC_COLUMNS,
    RAW_COLUMNS,
    RAW_DIR,
    SEASON_CODES,
    SEASONS,
    TABLES_DIR,
    TARGET_CLASSES,
    TEAM_ALIASES,
    ensure_directories,
)

# Stlpce, ktore musia po vycisteni existovat a nesmu byt prazdne.
KEY_COLUMNS = ["date", "home_team", "away_team", "fthg", "ftag", "ftr"]

OUTPUT_COLUMNS = [
    "match_id",
    "season",
    "date",
    "time",
    "home_team",
    "away_team",
    "fthg",
    "ftag",
    "ftr",
    "hthg",
    "htag",
    "htr",
    "home_shots",
    "away_shots",
    "home_sot",
    "away_sot",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow",
    "away_yellow",
    "home_red",
    "away_red",
    "odds_h",
    "odds_d",
    "odds_a",
    "ps_h",
    "ps_d",
    "ps_a",
    "has_odds",
    "has_ps_odds",
    "source_file",
]


def load_season(season: str) -> pd.DataFrame:
    """Nacita jeden raw CSV a vrati ho s jednotnymi nazvami stlpcov."""
    path = RAW_DIR / f"E0_{ALL_SEASON_CODES[season]}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Chyba {path}. Najprv spusti: python -m src.download"
        )

    raw = pd.read_csv(path, encoding="latin-1")
    renamed = raw.rename(columns={k: v for k, v in RAW_COLUMNS.items() if k in raw.columns})

    # Nie kazda sezona musi obsahovat vsetky kurzy - chybajuce stlpce doplnime.
    for column in RAW_COLUMNS.values():
        if column not in renamed.columns:
            renamed[column] = pd.NA

    renamed = renamed[list(RAW_COLUMNS.values())].copy()
    renamed["season"] = season
    renamed["source_file"] = path.name
    return renamed


def clean_matches(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Zjednoti typy, nazvy timov a odstrani nepouzitelne riadky."""
    stats: dict[str, int] = {"raw_rows": len(frame)}

    frame["date"] = pd.to_datetime(
        frame["date"], dayfirst=True, format="mixed", errors="coerce"
    )
    stats["invalid_dates"] = int(frame["date"].isna().sum())
    frame = frame[frame["date"].notna()].copy()

    for column in ("home_team", "away_team"):
        frame[column] = (
            frame[column]
            .astype("string")
            .str.strip()
            .map(lambda value: TEAM_ALIASES.get(str(value), str(value)))
        )

    frame["time"] = frame["time"].fillna("").astype("string").str.strip()
    frame["ftr"] = frame["ftr"].astype("string").str.strip().str.upper()
    frame["htr"] = frame["htr"].astype("string").str.strip().str.upper()

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    stats["missing_result_rows"] = int(frame["ftr"].isna().sum())
    frame = frame.dropna(subset=["ftr", "fthg", "ftag"]).copy()

    stats["invalid_result_rows"] = int((~frame["ftr"].isin(TARGET_CLASSES)).sum())
    frame = frame[frame["ftr"].isin(TARGET_CLASSES)].copy()

    # Goly musia byt nezáporné celé cisla.
    for column in ("fthg", "ftag"):
        frame = frame[frame[column] >= 0]
        frame[column] = frame[column].astype("int64")
    for column in ("hthg", "htag"):
        frame[column] = frame[column].round().astype("Int64")

    frame["home_team"] = frame["home_team"].astype(str)
    frame["away_team"] = frame["away_team"].astype(str)

    frame["has_odds"] = frame[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)
    frame["has_ps_odds"] = frame[["ps_h", "ps_d", "ps_a"]].notna().all(axis=1)

    frame["match_id"] = frame.apply(_match_id, axis=1)

    stats["duplicate_ids"] = int(frame.duplicated("match_id").sum())
    frame = frame.drop_duplicates("match_id", keep="last")

    frame["season"] = pd.Categorical(frame["season"], categories=ALL_SEASONS, ordered=True)
    frame = frame.sort_values(["season", "date", "time"]).reset_index(drop=True)
    frame["season"] = frame["season"].astype(str)

    stats["final_rows"] = len(frame)
    return frame, stats


def _match_id(row: pd.Series) -> str:
    """Stabilny identifikator zapasu nezavisly od poradia riadkov."""

    def slug(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", value)

    return (
        f"{str(row['season']).replace('/', '_')}_"
        f"{row['date']:%Y%m%d}_"
        f"{slug(row['home_team'])}_{slug(row['away_team'])}"
    )


def validate(frame: pd.DataFrame) -> list[str]:
    """Vrati zoznam varovani; prazdny zoznam znamena ciste data."""
    warnings: list[str] = []

    for column in KEY_COLUMNS:
        if column not in frame.columns:
            warnings.append(f"Chyba klucovy stlpec: {column}")
        elif frame[column].isna().any():
            warnings.append(f"Stlpec {column} obsahuje chybajuce hodnoty")

    per_season = frame.groupby("season").size()
    for season, count in per_season.items():
        if count != 380:
            warnings.append(f"Sezona {season}: ocakavanych 380 zapasov, najdenych {count}")

    for season, group in frame.groupby("season"):
        teams = set(group["home_team"]) | set(group["away_team"])
        if len(teams) != 20:
            warnings.append(f"Sezona {season}: ocakavanych 20 timov, najdenych {len(teams)}")

    if (frame["fthg"] < 0).any() or (frame["ftag"] < 0).any():
        warnings.append("Zaporne hodnoty v pocte golov")

    if frame.duplicated("match_id").any():
        warnings.append("Duplicitne match_id")

    return warnings


def build_quality_report(
    frame: pd.DataFrame,
    stats: dict[str, int],
    warnings: list[str],
) -> str:
    """Zostavi Markdown report o kvalite dat pre pracu a kontrolu."""
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    lines: list[str] = [
        "# Kvalita dát – zber a čistenie",
        "",
        f"*Generované: {now}*",
        "",
        "## Prehľad",
        "",
        f"- Počet riadkov v raw súboroch: **{stats['raw_rows']}**",
        f"- Počet použiteľných zápasov: **{stats['final_rows']}**",
        f"- Neplatné dátumy: {stats['invalid_dates']}",
        f"- Riadky bez výsledku: {stats['missing_result_rows']}",
        f"- Neplatné hodnoty FTR: {stats['invalid_result_rows']}",
        f"- Duplicitné zápasy: {stats['duplicate_ids']}",
        "",
        "## Zápasy podľa sezóny",
        "",
        "| Sezóna | Zápasy | Tímy | S kurzami B365 | S kurzami Pinnacle |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]

    for season, group in frame.groupby("season"):
        teams = len(set(group["home_team"]) | set(group["away_team"]))
        lines.append(
            f"| {season} | {len(group)} | {teams} | "
            f"{int(group['has_odds'].sum())} | {int(group['has_ps_odds'].sum())} |"
        )

    lines += [
        "",
        "## Rozdelenie výsledkov",
        "",
        "| Výsledok | Počet | Podiel |",
        "| :--- | ---: | ---: |",
    ]
    distribution = frame["ftr"].value_counts()
    for result in TARGET_CLASSES:
        count = int(distribution.get(result, 0))
        share = count / len(frame) if len(frame) else 0
        lines.append(f"| {result} | {count} | {share:.1%} |")

    lines += [
        "",
        "## Chýbajúce hodnoty (vybrané stĺpce)",
        "",
        "| Stĺpec | Chýba | Podiel |",
        "| :--- | ---: | ---: |",
    ]
    for column in (
        "time",
        "home_shots",
        "away_shots",
        "home_sot",
        "away_sot",
        "odds_h",
        "odds_d",
        "odds_a",
    ):
        missing = int(frame[column].isna().sum())
        lines.append(f"| {column} | {missing} | {missing / len(frame):.1%} |")

    lines += ["", "## Kontroly", ""]
    if warnings:
        lines += [f"- ⚠️ {warning}" for warning in warnings]
    else:
        lines += [
            "- ✅ 380 zápasov v každej sezóne",
            "- ✅ 20 tímov v každej sezóne",
            "- ✅ Žiadne duplicitné zápasy",
            "- ✅ Výsledky len H / D / A",
            "- ✅ Nezáporné góly",
        ]

    lines += [
        "",
        "## Mapovanie pôvodných stĺpcov",
        "",
        "| Pôvodný (football-data) | Nový názov |",
        "| :--- | :--- |",
    ]
    for original, new in RAW_COLUMNS.items():
        lines.append(f"| `{original}` | `{new}` |")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Vycisti a zluc CSV data Premier League")
    parser.add_argument("--strict", action="store_true", help="varovania ber ako chybu")
    args = parser.parse_args()

    ensure_directories()

    frames: list[pd.DataFrame] = []
    for season in SEASONS:
        path = RAW_DIR / f"E0_{SEASON_CODES[season]}.csv"
        if not path.exists():
            print(f"[skip] {season}: chyba {path.name} (spusti download)")
            continue
        frames.append(load_season(season))
        print(f"[load] {season}: {len(frames[-1])} riadkov")

    if not frames:
        print("Nemam ziadne data. Spusti najprv: python -m src.download", file=sys.stderr)
        return 1

    frame, stats = clean_matches(pd.concat(frames, ignore_index=True))
    frame = frame[OUTPUT_COLUMNS]
    warnings = validate(frame)

    MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MATCHES_CSV, index=False, encoding="utf-8")

    report = build_quality_report(frame, stats, warnings)
    report_path = TABLES_DIR / "01_data_quality.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n[ok] {len(frame)} zapasov -> {MATCHES_CSV}")
    print(f"[ok] report -> {report_path}")

    if warnings:
        print("\nVarovania:")
        for warning in warnings:
            print(f"  - {warning}")
        if args.strict:
            return 1
    else:
        print("Vsetky kontroly presli.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
