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
WALLET_JSON: Path = DATA_DIR / "wallet.json"
INITIAL_BALANCE: float = 100.0

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
                    "kickoff_utc_iso": dt_utc.isoformat(),
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


def load_balance(path: Path = WALLET_JSON) -> float:
    """Načíta aktuálny zostatok na účte v EUR (predvolene 100.00 €)."""
    if not path.exists():
        return INITIAL_BALANCE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        val = float(data.get("balance", INITIAL_BALANCE))
        return round(max(0.0, val), 2)
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return INITIAL_BALANCE


def save_balance(balance: float, path: Path = WALLET_JSON) -> None:
    """Zapíše aktuálny zostatok na účte v EUR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"balance": round(max(0.0, float(balance)), 2)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def tip_fair_odds(tip_code: str, probs: tuple[float, float, float]) -> float:
    """Vypočíta férový kurz z pravdepodobností (H, D, A) pre daný tip."""
    p_h, p_d, p_a = probs
    prob_map = {
        "1": p_h,
        "1X": p_h + p_d,
        "X": p_d,
        "X2": p_d + p_a,
        "2": p_a,
    }
    p = max(0.01, min(0.99, float(prob_map.get(tip_code, 0.33))))
    return round(1.0 / p, 2)


def can_cancel_tip(tip: dict, now_utc: datetime | None = None) -> bool:
    """Tip je možné stornovať iba ak ešte nebol vyhodnotený a zápas ešte nezačal."""
    if tip.get("actual_ftr") is not None:
        return False
    iso = tip.get("kickoff_utc_iso")
    if not iso:
        return True
    try:
        kickoff = datetime.fromisoformat(str(iso))
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        ref_now = now_utc or datetime.now(timezone.utc)
        return ref_now < kickoff
    except ValueError:
        return True


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
    stake: float = 10.0,
    odds: float = 2.0,
    kickoff_utc_iso: str = "",
    path: Path = TIPS_JOURNAL_JSON,
    wallet_path: Path = WALLET_JSON,
) -> tuple[bool, str, dict | None]:
    """Uloží alebo aktualizuje tip pre budúci zápas a odráta vklad zo zostatku."""
    if tip_code not in TIP_LABELS:
        return False, f"Neplatný tip: {tip_code}", None

    stake_val = round(float(stake), 2)
    if stake_val < 1.0 or stake_val > 100.0:
        return False, "Vklad musí byť od 1.00 € do 100.00 €.", None

    tips = load_journal_tips(path)
    balance = load_balance(wallet_path)
    now_str = datetime.now().strftime("%d. %m. %Y %H:%M")

    for tip in tips:
        if (
            tip.get("actual_ftr") is None
            and tip.get("league") == league
            and tip.get("home") == home
            and tip.get("away") == away
            and tip.get("match_date") == match_date
        ):
            if not can_cancel_tip(tip):
                return False, "Zápas už začal, tip sa už nedá zmeniť.", None
            old_stake = round(float(tip.get("stake", 0.0)), 2)
            available = round(balance + old_stake, 2)
            if stake_val > available:
                return False, f"Nedostatok prostriedkov (k dispozícii {available:.2f} €).", None
            new_balance = round(available - stake_val, 2)
            tip["tip_code"] = tip_code
            tip["stake"] = stake_val
            tip["odds"] = round(float(odds), 2)
            tip["potential_win"] = round(stake_val * float(odds), 2)
            tip["updated_at"] = now_str
            save_journal_tips(tips, path)
            save_balance(new_balance, wallet_path)
            return True, "Tip aktualizovaný.", tip

    if stake_val > balance:
        return False, f"Nedostatok prostriedkov na účte (zostatok {balance:.2f} €).", None

    new_balance = round(balance - stake_val, 2)
    entry = {
        "id": uuid.uuid4().hex[:10],
        "created_at": now_str,
        "league": league,
        "season": season,
        "home": home,
        "away": away,
        "kickoff_str": kickoff_str,
        "kickoff_utc_iso": kickoff_utc_iso,
        "match_date": match_date,
        "tip_code": tip_code,
        "stake": stake_val,
        "odds": round(float(odds), 2),
        "potential_win": round(stake_val * float(odds), 2),
        "payout": None,
        "actual_ftr": None,
        "actual_score": None,
    }
    tips.insert(0, entry)
    save_journal_tips(tips, path)
    save_balance(new_balance, wallet_path)
    return True, "Tip uložený.", entry


def cancel_journal_tip(
    tip_id: str,
    path: Path = TIPS_JOURNAL_JSON,
    wallet_path: Path = WALLET_JSON,
    now_utc: datetime | None = None,
) -> tuple[bool, str]:
    """Stornuje tip pred začiatkom zápasu a vráti vklad na účet."""
    tips = load_journal_tips(path)
    target = next((t for t in tips if t.get("id") == tip_id), None)
    if target is None:
        return False, "Tip sa nenašiel."
    if not can_cancel_tip(target, now_utc=now_utc):
        return False, "Zápas už začal alebo bol vyhodnotený – tip už nie je možné stornovať."

    refund = round(float(target.get("stake", 0.0)), 2)
    balance = load_balance(wallet_path)
    save_balance(round(balance + refund, 2), wallet_path)
    save_journal_tips([t for t in tips if t.get("id") != tip_id], path)
    return True, f"Tip stornovaný, vrátených {refund:.2f} €."


def delete_journal_tip(tip_id: str, path: Path = TIPS_JOURNAL_JSON) -> None:
    """Odstráni už vyhodnotený tip z histórie denníka (bez zmeny zostatku)."""
    tips = [t for t in load_journal_tips(path) if t.get("id") != tip_id]
    save_journal_tips(tips, path)


def reset_wallet_and_tips(
    path: Path = TIPS_JOURNAL_JSON,
    wallet_path: Path = WALLET_JSON,
) -> None:
    """Obnoví zostatok na 100.00 € a vymaže všetky tipy."""
    save_balance(INITIAL_BALANCE, wallet_path)
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
    wallet_path: Path = WALLET_JSON,
) -> tuple[list[dict], bool]:
    """Automaticky vyhodnotí tipy podľa odohraných zápasov v `matches` a pripíše výhru."""
    changed = False
    balance = load_balance(wallet_path)
    balance_changed = False

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
            ftr = str(row["ftr"])
            tip["actual_ftr"] = ftr
            tip["actual_score"] = f"{int(row['fthg'])}:{int(row['ftag'])}"
            stake_val = float(tip.get("stake", 0.0))
            odds_val = float(tip.get("odds", 1.0))
            if stake_val > 0 and is_tip_hit(tip.get("tip_code", ""), ftr):
                payout = round(stake_val * odds_val, 2)
                tip["payout"] = payout
                balance = round(balance + payout, 2)
                balance_changed = True
            else:
                tip["payout"] = 0.0
            changed = True

    if balance_changed:
        save_balance(balance, wallet_path)
    return tips, changed
