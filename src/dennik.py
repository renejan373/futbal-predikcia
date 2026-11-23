"""Načítanie najbližších neodohraných zápasov pre Top 5 líg a ukladanie tipov do Tipérskeho denníka."""

from __future__ import annotations

import concurrent.futures
import json
import urllib.request
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import DATA_DIR, TEAM_ALIASES

TIPS_JOURNAL_JSON: Path = DATA_DIR / "tips_journal.json"

ESPN_LEAGUE_CODES: dict[str, str] = {
    "Premier League": "eng.1",
    "La Liga": "esp.1",
    "Serie A": "ita.1",
    "Bundesliga": "ger.1",
    "Ligue 1": "fra.1",
}

ESPN_TEAM_ALIASES: dict[str, str] = {
    **TEAM_ALIASES,
    # Premier League
    "Brighton & Hove Albion": "Brighton",
    "Coventry City": "Coventry",
    # La Liga
    "Alavés": "Alaves",
    "Athletic Club": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid",
    "Celta Vigo": "Celta",
    "Deportivo": "La Coruna",
    "Deportivo La Coruña": "La Coruna",
    "Espanyol": "Espanol",
    "Málaga": "Malaga",
    "Racing Santander": "Santander",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    # Serie A
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Internazionale": "Inter",
    # Bundesliga
    "1. FC Union Berlin": "Union Berlin",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Augsburg": "Augsburg",
    "FC Cologne": "FC Koln",
    "1. FC Köln": "FC Koln",
    "Hamburg SV": "Hamburg",
    "Hamburger SV": "Hamburg",
    "SC Freiburg": "Freiburg",
    "SC Paderborn 07": "Paderborn",
    "SV Elversberg": "Elversberg",
    "TSG Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
    # Ligue 1
    "AJ Auxerre": "Auxerre",
    "AS Monaco": "Monaco",
    "Le Havre AC": "Le Havre",
    "Paris Saint-Germain": "Paris SG",
    "Stade Rennais": "Rennes",
}

TIP_OPTIONS: list[tuple[str, str, str]] = [
    ("1", "Výhra domácich", "Výhra domáceho tímu"),
    ("1X", "Domáci neprehrajú", "Výhra domácich alebo remíza"),
    ("X", "Remíza", "Remíza"),
    ("X2", "Hostia neprehrajú", "Remíza alebo výhra hosťujúceho tímu"),
    ("2", "Výhra hostí", "Výhra hosťujúceho tímu"),
]

TIP_LABELS: dict[str, str] = {code: label for code, label, _ in TIP_OPTIONS}


def format_tip_text(tip_code: str, home: str, away: str) -> str:
    """Vráti čitateľný popis tipu s konkrétnym menom tímu."""
    if tip_code == "1":
        return f"{home} – výhra (1)"
    if tip_code == "1X":
        return f"{home} – výhra / remíza (1X)"
    if tip_code == "X":
        return "Remíza (X)"
    if tip_code == "X2":
        return f"{away} – výhra / remíza (X2)"
    if tip_code == "2":
        return f"{away} – výhra (2)"
    return tip_code


def normalize_espn_team(name: str) -> str:
    """Prevedie názov tímu z ESPN na názov používaný v našich ratingoch."""
    return ESPN_TEAM_ALIASES.get(name, name)


def _fetch_espn_day(code: str, day_str: str) -> list[dict]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={day_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("events", [])
    except Exception:
        return []


def fetch_next_match_per_team(league: str, days_ahead: int = 14) -> list[dict]:
    """Vráti 1 najbližší neodohraný zápas pre každý tím zvolenej ligy."""
    code = ESPN_LEAGUE_CODES.get(league)
    if not code:
        return []

    today = date.today()
    day_strs = [(today + timedelta(days=i)).strftime("%Y%m%d") for i in range(days_ahead)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
        batches = list(ex.map(lambda d: _fetch_espn_day(code, d), day_strs))

    now_utc = datetime.now(timezone.utc)
    local_tz = ZoneInfo("Europe/Bratislava")
    raw_fixtures: list[dict] = []
    seen_event_ids: set[str] = set()

    for events in batches:
        for ev in events:
            ev_id = str(ev.get("id", ""))
            if not ev_id or ev_id in seen_event_ids:
                continue
            status_name = ev.get("status", {}).get("type", {}).get("name", "")
            if status_name != "STATUS_SCHEDULED":
                continue

            iso_date = ev.get("date", "")
            try:
                dt_utc = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt_utc <= now_utc:
                continue

            comps = ev.get("competitions", [])
            if not comps:
                continue
            competitors = comps[0].get("competitors", [])
            home_raw = next((c["team"]["displayName"] for c in competitors if c.get("homeAway") == "home"), None)
            away_raw = next((c["team"]["displayName"] for c in competitors if c.get("homeAway") == "away"), None)
            if not home_raw or not away_raw:
                continue

            seen_event_ids.add(ev_id)
            dt_local = dt_utc.astimezone(local_tz)
            raw_fixtures.append(
                {
                    "event_id": ev_id,
                    "league": league,
                    "kickoff_utc": dt_utc,
                    "kickoff_str": dt_local.strftime("%d. %m. %Y · %H:%M"),
                    "match_date": dt_local.strftime("%Y-%m-%d"),
                    "home": normalize_espn_team(home_raw),
                    "away": normalize_espn_team(away_raw),
                }
            )

    raw_fixtures.sort(key=lambda x: x["kickoff_utc"])

    # Vyberieme presne 1 najbližší zápas pre každý tím
    assigned_teams: set[str] = set()
    next_per_team: list[dict] = []
    for fx in raw_fixtures:
        if fx["home"] in assigned_teams or fx["away"] in assigned_teams:
            continue
        assigned_teams.add(fx["home"])
        assigned_teams.add(fx["away"])
        next_per_team.append(fx)

    return next_per_team


def load_journal_tips(path: Path = TIPS_JOURNAL_JSON) -> list[dict]:
    """Načíta uložené tipy z Tipérskeho denníka."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_journal_tips(tips: list[dict], path: Path = TIPS_JOURNAL_JSON) -> None:
    """Zapíše tipy Tipérskeho denníka do JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tips, ensure_ascii=False, indent=2), encoding="utf-8")


def save_fixture_tip(
    *,
    league: str,
    season: str,
    home: str,
    away: str,
    kickoff_str: str,
    match_date: str,
    tip_code: str,
    path: Path = TIPS_JOURNAL_JSON,
) -> dict:
    """Uloží alebo aktualizuje tip pre konkrétny budúci zápas."""
    if tip_code not in TIP_LABELS:
        raise ValueError(f"Neplatný tip: {tip_code}")

    tips = load_journal_tips(path)
    now_str = datetime.now().strftime("%d. %m. %Y %H:%M")

    for tip in tips:
        if (
            tip.get("actual_ftr") is None
            and tip.get("league") == league
            and tip.get("home") == home
            and tip.get("away") == away
            and tip.get("match_date") == match_date
        ):
            tip["tip_code"] = tip_code
            tip["updated_at"] = now_str
            save_journal_tips(tips, path)
            return tip

    entry = {
        "id": uuid.uuid4().hex[:10],
        "created_at": now_str,
        "league": league,
        "season": season,
        "home": home,
        "away": away,
        "kickoff_str": kickoff_str,
        "match_date": match_date,
        "tip_code": tip_code,
        "actual_ftr": None,
        "actual_score": None,
    }
    tips.insert(0, entry)
    save_journal_tips(tips, path)
    return entry


def delete_journal_tip(tip_id: str, path: Path = TIPS_JOURNAL_JSON) -> None:
    """Odstráni tip z denníka."""
    tips = [t for t in load_journal_tips(path) if t.get("id") != tip_id]
    save_journal_tips(tips, path)


def clear_journal_tips(path: Path = TIPS_JOURNAL_JSON) -> None:
    """Vymaže všetky tipy z denníka."""
    save_journal_tips([], path)


def is_tip_hit(tip_code: str, actual_ftr: str) -> bool:
    """Overí, či tip (1, 1X, X, X2, 2) vyšiel pri skutočnom výsledku (H, D, A)."""
    if tip_code == "1":
        return actual_ftr == "H"
    if tip_code == "1X":
        return actual_ftr in ("H", "D")
    if tip_code == "X":
        return actual_ftr == "D"
    if tip_code == "X2":
        return actual_ftr in ("D", "A")
    if tip_code == "2":
        return actual_ftr == "A"
    return False


def resolve_journal_tips(
    tips: list[dict],
    matches_by_league: dict[str, pd.DataFrame],
) -> tuple[list[dict], bool]:
    """Automaticky vyhodnotí tipy podľa odohraných zápasov v `matches`."""
    changed = False
    for tip in tips:
        if tip.get("actual_ftr") is not None:
            continue
        df = matches_by_league.get(tip.get("league", ""))
        if df is None or df.empty:
            continue

        cand = df[
            (df["season"].astype(str) == str(tip.get("season")))
            & (df["home_team"] == tip.get("home"))
            & (df["away_team"] == tip.get("away"))
        ]
        if tip.get("match_date"):
            target_dt = pd.to_datetime(tip["match_date"], errors="coerce")
            if pd.notna(target_dt):
                cand = cand[pd.to_datetime(cand["date"]) >= target_dt - pd.Timedelta(days=2)]

        if not cand.empty:
            row = cand.sort_values("date").iloc[0]
            tip["actual_ftr"] = str(row["ftr"])
            tip["actual_score"] = f"{int(row['fthg'])}:{int(row['ftag'])}"
            changed = True

    return tips, changed
