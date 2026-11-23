"""Ukladanie a vyhodnocovanie osobných tipov používateľa (Súboj Ty vs. Model)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

USER_TIPS_JSON: Path = DATA_DIR / "user_tips.json"
PICK_CODES = ("H", "D", "A")
PICK_SHORT = {"H": "1", "D": "X", "A": "2"}


def load_tips(path: Path = USER_TIPS_JSON) -> list[dict]:
    """Načíta uložené tipy zo súboru JSON."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_tips(tips: list[dict], path: Path = USER_TIPS_JSON) -> None:
    """Zapíše zoznam tipov do súboru JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tips, ensure_ascii=False, indent=2), encoding="utf-8")


def add_tip(
    *,
    league: str,
    season: str,
    home: str,
    away: str,
    user_pick: str,
    model_system: str,
    model_pick: str,
    probs: tuple[float, float, float],
    ref_date: str,
    path: Path = USER_TIPS_JSON,
) -> dict:
    """Pridá alebo aktualizuje nevyhodnotený tip pre daný zápas."""
    if user_pick not in PICK_CODES:
        raise ValueError(f"Neplatný tip: {user_pick}")

    tips = load_tips(path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Ak už existuje nevyhodnotený tip na rovnaký zápas, aktualizujeme ho
    for tip in tips:
        if (
            tip.get("actual_ftr") is None
            and tip.get("league") == league
            and tip.get("season") == season
            and tip.get("home") == home
            and tip.get("away") == away
        ):
            tip["user_pick"] = user_pick
            tip["model_system"] = model_system
            tip["model_pick"] = model_pick
            tip["probs"] = [round(float(p), 4) for p in probs]
            tip["updated_at"] = now_str
            save_tips(tips, path)
            return tip

    entry = {
        "id": uuid.uuid4().hex[:10],
        "created_at": now_str,
        "league": league,
        "season": season,
        "home": home,
        "away": away,
        "user_pick": user_pick,
        "model_system": model_system,
        "model_pick": model_pick,
        "probs": [round(float(p), 4) for p in probs],
        "ref_date": ref_date,
        "actual_ftr": None,
        "actual_score": None,
        "resolved_at": None,
    }
    tips.insert(0, entry)
    save_tips(tips, path)
    return entry


def delete_tip(tip_id: str, path: Path = USER_TIPS_JSON) -> None:
    """Odstráni tip podľa ID."""
    tips = [t for t in load_tips(path) if t.get("id") != tip_id]
    save_tips(tips, path)


def clear_tips(path: Path = USER_TIPS_JSON) -> None:
    """Vymaže všetky uložené tipy."""
    save_tips([], path)


def set_manual_result(
    tip_id: str,
    result_ftr: str,
    score: str | None = None,
    path: Path = USER_TIPS_JSON,
) -> None:
    """Manuálne nastaví výsledok zápasu pre tip."""
    if result_ftr not in PICK_CODES:
        raise ValueError(f"Neplatný výsledok: {result_ftr}")
    tips = load_tips(path)
    for tip in tips:
        if tip.get("id") == tip_id:
            tip["actual_ftr"] = result_ftr
            tip["actual_score"] = score or f"({PICK_SHORT[result_ftr]})"
            tip["resolved_at"] = datetime.now().strftime("%Y-%m-%d")
            break
    save_tips(tips, path)


def resolve_tips_from_matches(
    tips: list[dict],
    matches_by_league: dict[str, pd.DataFrame],
) -> tuple[list[dict], bool]:
    """Automaticky vyhodnotí nevyhodnotené tipy podľa odohraných zápasov v `matches`."""
    changed = False
    for tip in tips:
        if tip.get("actual_ftr") is not None:
            continue
        lg = tip.get("league")
        df = matches_by_league.get(lg)
        if df is None or df.empty:
            continue

        ref_ts = pd.to_datetime(tip.get("ref_date"), errors="coerce")
        cand = df[
            (df["season"].astype(str) == str(tip.get("season")))
            & (df["home_team"] == tip.get("home"))
            & (df["away_team"] == tip.get("away"))
        ]
        if pd.notna(ref_ts):
            cand = cand[pd.to_datetime(cand["date"]) > ref_ts]

        if not cand.empty:
            row = cand.sort_values("date").iloc[0]
            tip["actual_ftr"] = str(row["ftr"])
            tip["actual_score"] = f"{int(row['fthg'])}:{int(row['ftag'])}"
            tip["resolved_at"] = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            changed = True

    return tips, changed


def compute_score_summary(tips: list[dict]) -> dict[str, int | float]:
    """Vypočíta súhrnné skóre Ty vs. Model z vyhodnotených tipov."""
    resolved = [t for t in tips if t.get("actual_ftr") in PICK_CODES]
    total_all = len(tips)
    total_res = len(resolved)
    user_hits = sum(1 for t in resolved if t.get("user_pick") == t.get("actual_ftr"))
    model_hits = sum(1 for t in resolved if t.get("model_pick") == t.get("actual_ftr"))
    return {
        "total_all": total_all,
        "total_resolved": total_res,
        "pending": total_all - total_res,
        "user_hits": user_hits,
        "model_hits": model_hits,
        "user_acc": (user_hits / total_res) if total_res else 0.0,
        "model_acc": (model_hits / total_res) if total_res else 0.0,
    }
