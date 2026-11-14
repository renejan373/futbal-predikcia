"""Centrálna konfigurácia projektu.

Všetky cesty, konštanty a rozhodnutia (sezóny, split, seed) sú na jednom mieste,
aby sa dali ľahko zmeniť a aby ich videla aj komisia v prílohe práce.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Cesty
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

MATCHES_CSV = INTERIM_DIR / "matches.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
TEAM_STATES_CSV = PROCESSED_DIR / "team_states.csv"
MODEL_METADATA_JSON = MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Dáta a sezóny
# ---------------------------------------------------------------------------
def _build_season_codes(start_year: int = 1993, end_year: int = 2025) -> dict[str, str]:
    """Vygeneruje kódy sezón pre football-data.co.uk (od 1993/94 po 2025/26)."""
    codes: dict[str, str] = {}
    for year in range(start_year, end_year + 1):
        y1 = year % 100
        y2 = (year + 1) % 100
        codes[f"{year}/{y2:02d}"] = f"{y1:02d}{y2:02d}"
    return codes


SEASON_CODES: dict[str, str] = _build_season_codes(1993, 2025)
SEASONS: list[str] = list(SEASON_CODES)

# Aktuálna prebiehajúca sezóna pre živú predikciu v aplikácii
CURRENT_SEASON: str = "2026/27"
CURRENT_SEASON_CODE: str = "2627"
ALL_SEASON_CODES: dict[str, str] = {
    **SEASON_CODES,
    CURRENT_SEASON: CURRENT_SEASON_CODE,
}
ALL_SEASONS: list[str] = list(ALL_SEASON_CODES)

# Časovo podmienené rozdelenie: staršie sezóny trénujú, posledná ukončená testuje.
TRAIN_SEASONS: list[str] = SEASONS[:-2]
VALID_SEASON: str = SEASONS[-2]
TEST_SEASON: str = SEASONS[-1]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
REQUEST_TIMEOUT_SECONDS = 90
USER_AGENT = "Mozilla/5.0 (SOS-project; educational use)"
EXPECTED_MATCHES_PER_SEASON = 380

# ---------------------------------------------------------------------------
# Feature engineering a modely
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
ROLLING_WINDOW = 5
H2H_WINDOW = 5

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
# Na začiatku sezóny sa ratingy mierne vracajú k priemeru, aby sa zabránilo
# dlhodobému "utekaniu" hodnôt (regresia k priemeru je bežná prax).
ELO_SEASON_REGRESSION = 0.75

TARGET_CLASSES = ["H", "D", "A"]
TARGET_NAMES_SK = {
    "H": "Výhra domácich",
    "D": "Remíza",
    "A": "Výhra hostí",
}

# ---------------------------------------------------------------------------
# Normalizácia názvov tímov
# ---------------------------------------------------------------------------
# football-data.co.uk používa skrátené názvy; slovník je poistka pre prípad,
# že sa v niektorej sezóne objaví iný variant.
TEAM_ALIASES: dict[str, str] = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Watford FC": "Watford",
    "West Ham United": "West Ham",
    "Crystal Palace FC": "Crystal Palace",
    "AFC Bournemouth": "Bournemouth",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Hull City": "Hull",
    "Derby County": "Derby",
    "Aston Villa FC": "Aston Villa",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    "Sunderland AFC": "Sunderland",
}

# Stĺpce, ktoré z raw CSV naozaj používame (zvyšok zahadzujeme).
RAW_COLUMNS: dict[str, str] = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "fthg",
    "FTAG": "ftag",
    "FTR": "ftr",
    "HTHG": "hthg",
    "HTAG": "htag",
    "HTR": "htr",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "B365H": "odds_h",
    "B365D": "odds_d",
    "B365A": "odds_a",
    "PSH": "ps_h",
    "PSD": "ps_d",
    "PSA": "ps_a",
}

# Numerické stĺpce, ktoré musia zostať číslami (inak by ich pandas vtiahol ako text).
NUMERIC_COLUMNS = [
    "fthg", "ftag", "hthg", "htag",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellow", "away_yellow", "home_red", "away_red",
    "odds_h", "odds_d", "odds_a", "ps_h", "ps_d", "ps_a",
]


def ensure_directories() -> None:
    """Vytvorí všetky pracovné priečinky, ak ešte neexistujú."""
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
