"""Načítanie prehľadu zranených hráčov pre Top 5 európskych líg a vybrané kluby."""

from __future__ import annotations

import base64
import concurrent.futures
import html as html_lib
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, PROJECT_DIR, TEAM_ALIASES

INJURIES_JSON: Path = DATA_DIR / "injuries.json"
LOGOS_DIR: Path = PROJECT_DIR / "assets" / "logos"
PLAYERS_DIR: Path = PROJECT_DIR / "assets" / "players"

CLUB_LOGOS: dict[str, Path] = {
    # Premier League
    "Arsenal": LOGOS_DIR / "arsenal.png",
    "Aston Villa": LOGOS_DIR / "aston_villa.png",
    "Bournemouth": LOGOS_DIR / "bournemouth.png",
    "Brentford": LOGOS_DIR / "brentford.png",
    "Brighton": LOGOS_DIR / "brighton.png",
    "Chelsea": LOGOS_DIR / "chelsea.png",
    "Coventry": LOGOS_DIR / "coventry.png",
    "Crystal Palace": LOGOS_DIR / "crystal_palace.png",
    "Everton": LOGOS_DIR / "everton.png",
    "Fulham": LOGOS_DIR / "fulham.png",
    "Hull": LOGOS_DIR / "hull.png",
    "Ipswich": LOGOS_DIR / "ipswich.png",
    "Leeds": LOGOS_DIR / "leeds.png",
    "Liverpool": LOGOS_DIR / "liverpool.png",
    "Man City": LOGOS_DIR / "man_city.png",
    "Man United": LOGOS_DIR / "man_united.png",
    "Newcastle": LOGOS_DIR / "newcastle.png",
    "Nott'm Forest": LOGOS_DIR / "nottm_forest.png",
    "Sunderland": LOGOS_DIR / "sunderland.png",
    "Tottenham": LOGOS_DIR / "tottenham.png",
    # La Liga
    "Alaves": LOGOS_DIR / "alaves.png",
    "Ath Bilbao": LOGOS_DIR / "ath_bilbao.png",
    "Ath Madrid": LOGOS_DIR / "atletico_madrid.png",
    "Atlético Madrid": LOGOS_DIR / "atletico_madrid.png",
    "Barcelona": LOGOS_DIR / "fc_barcelona.png",
    "FC Barcelona": LOGOS_DIR / "fc_barcelona.png",
    "Betis": LOGOS_DIR / "betis.png",
    "Celta": LOGOS_DIR / "celta.png",
    "Elche": LOGOS_DIR / "elche.png",
    "Espanol": LOGOS_DIR / "espanol.png",
    "Getafe": LOGOS_DIR / "getafe.png",
    "La Coruna": LOGOS_DIR / "la_coruna.png",
    "Levante": LOGOS_DIR / "levante.png",
    "Malaga": LOGOS_DIR / "malaga.png",
    "Osasuna": LOGOS_DIR / "osasuna.png",
    "Real Madrid": LOGOS_DIR / "real_madrid.png",
    "Santander": LOGOS_DIR / "santander.png",
    "Sevilla": LOGOS_DIR / "sevilla.png",
    "Sociedad": LOGOS_DIR / "sociedad.png",
    "Valencia": LOGOS_DIR / "valencia.png",
    "Vallecano": LOGOS_DIR / "vallecano.png",
    "Villarreal": LOGOS_DIR / "villarreal.png",
    # Serie A
    "Inter Miláno": LOGOS_DIR / "inter_milano.png",
    "Inter": LOGOS_DIR / "inter_milano.png",
    "AC Miláno": LOGOS_DIR / "ac_milano.png",
    "Milan": LOGOS_DIR / "ac_milano.png",
    "Juventus": LOGOS_DIR / "juventus.png",
    "Atalanta": LOGOS_DIR / "atalanta.png",
    "Bologna": LOGOS_DIR / "bologna.png",
    "Cagliari": LOGOS_DIR / "cagliari.png",
    "Como": LOGOS_DIR / "como.png",
    "Fiorentina": LOGOS_DIR / "fiorentina.png",
    "Frosinone": LOGOS_DIR / "frosinone.png",
    "Genoa": LOGOS_DIR / "genoa.png",
    "Lazio": LOGOS_DIR / "lazio.png",
    "Lecce": LOGOS_DIR / "lecce.png",
    "Monza": LOGOS_DIR / "monza.png",
    "Napoli": LOGOS_DIR / "napoli.png",
    "Parma": LOGOS_DIR / "parma.png",
    "Roma": LOGOS_DIR / "roma.png",
    "Sassuolo": LOGOS_DIR / "sassuolo.png",
    "Torino": LOGOS_DIR / "torino.png",
    "Udinese": LOGOS_DIR / "udinese.png",
    "Venezia": LOGOS_DIR / "venezia.png",
    # Bundesliga
    "Bayern Mníchov": LOGOS_DIR / "bayern_mnichov.png",
    "Bayern Munich": LOGOS_DIR / "bayern_mnichov.png",
    "Borussia Dortmund": LOGOS_DIR / "borussia_dortmund.png",
    "Dortmund": LOGOS_DIR / "borussia_dortmund.png",
    "Bayer Leverkusen": LOGOS_DIR / "bayer_leverkusen.png",
    "Leverkusen": LOGOS_DIR / "bayer_leverkusen.png",
    "Augsburg": LOGOS_DIR / "augsburg.png",
    "Ein Frankfurt": LOGOS_DIR / "ein_frankfurt.png",
    "Elversberg": LOGOS_DIR / "elversberg.png",
    "FC Koln": LOGOS_DIR / "fc_koln.png",
    "Freiburg": LOGOS_DIR / "freiburg.png",
    "Hamburg": LOGOS_DIR / "hamburg.png",
    "Hoffenheim": LOGOS_DIR / "hoffenheim.png",
    "M'gladbach": LOGOS_DIR / "m_gladbach.png",
    "Mainz": LOGOS_DIR / "mainz.png",
    "Paderborn": LOGOS_DIR / "paderborn.png",
    "RB Leipzig": LOGOS_DIR / "rb_leipzig.png",
    "Schalke 04": LOGOS_DIR / "schalke_04.png",
    "Stuttgart": LOGOS_DIR / "stuttgart.png",
    "Union Berlin": LOGOS_DIR / "union_berlin.png",
    "Werder Bremen": LOGOS_DIR / "werder_bremen.png",
    # Ligue 1
    "Paris Saint-Germain": LOGOS_DIR / "psg.png",
    "Paris SG": LOGOS_DIR / "psg.png",
    "Olympique Marseille": LOGOS_DIR / "marseille.png",
    "Marseille": LOGOS_DIR / "marseille.png",
    "AS Monaco": LOGOS_DIR / "monaco.png",
    "Monaco": LOGOS_DIR / "monaco.png",
    "Angers": LOGOS_DIR / "angers.png",
    "Auxerre": LOGOS_DIR / "auxerre.png",
    "Brest": LOGOS_DIR / "brest.png",
    "Le Havre": LOGOS_DIR / "le_havre.png",
    "Le Mans": LOGOS_DIR / "le_mans.png",
    "Lens": LOGOS_DIR / "lens.png",
    "Lille": LOGOS_DIR / "lille.png",
    "Lorient": LOGOS_DIR / "lorient.png",
    "Lyon": LOGOS_DIR / "lyon.png",
    "Nice": LOGOS_DIR / "nice.png",
    "Paris FC": LOGOS_DIR / "paris_fc.png",
    "Rennes": LOGOS_DIR / "rennes.png",
    "Strasbourg": LOGOS_DIR / "strasbourg.png",
    "Toulouse": LOGOS_DIR / "toulouse.png",
    "Troyes": LOGOS_DIR / "troyes.png",
}


@lru_cache(maxsize=64)
def get_team_logo_data_uri(team_name: str) -> str:
    """Vráti base64 data URI loga tímu pre vloženie do HTML karty."""
    path = CLUB_LOGOS.get(team_name)
    if not path or not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _player_slug(player_name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", player_name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")


@lru_cache(maxsize=512)
def get_player_photo_data_uri(player_name: str) -> str:
    """Vráti base64 data URI fotky hráča z assets/players/ alebo z Wikipédie."""
    slug = _player_slug(player_name)
    for ext, mime in ((".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".png", "image/png")):
        path = PLAYERS_DIR / f"{slug}{ext}"
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"

    # Živý fallback cez Wikipédia API pre novo zranených hráčov
    headers = {"User-Agent": "FutbalPredikciaApp/1.0 (educational project)"}
    try:
        q = urllib.parse.quote(f"{player_name} footballer")
        surl = (
            f"https://en.wikipedia.org/w/api.php?action=query&generator=search"
            f"&gsrsearch={q}&gsrlimit=1&prop=pageimages&pithumbsize=240&format=json"
        )
        req = urllib.request.Request(surl, headers=headers)
        data = json.loads(urllib.request.urlopen(req, timeout=4).read().decode("utf-8"))
        for pg in data.get("query", {}).get("pages", {}).values():
            thumb = pg.get("thumbnail", {}).get("source")
            if thumb:
                img_bytes = urllib.request.urlopen(urllib.request.Request(thumb, headers=headers), timeout=4).read()
                encoded = base64.b64encode(img_bytes).decode("ascii")
                return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        pass
    return ""

COLUMN_RENAME = {
    "player": "Hráč",
    "position": "Pozícia",
    "injury": "Zranenie",
    "injured_on": "Kedy sa zranil",
    "duration": "Na ako dlho",
    "expected_return": "Predpokladaný návrat",
    "context": "Kde sa zranil (zápas / tréning)",
    "status": "Stav",
}


INJURY_CLUB_ALIASES: dict[str, str] = {
    **TEAM_ALIASES,
    "Atlético Madrid": "Ath Madrid",
    "FC Barcelona": "Barcelona",
    "Inter Miláno": "Inter",
    "AC Miláno": "Milan",
    "Bayern Mníchov": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen",
    "Paris Saint-Germain": "Paris SG",
    "Olympique Marseille": "Marseille",
    "AS Monaco": "Monaco",
    # SportsGambler -> canonical names
    "Nott’m Forest": "Nott'm Forest",
    "Brighton & Hove Albion": "Brighton",
    "Coventry City": "Coventry",
    "Alavés": "Alaves",
    "Athletic Club": "Ath Bilbao",
    "Celta Vigo": "Celta",
    "Deportivo": "La Coruna",
    "Deportivo La Coruña": "La Coruna",
    "Espanyol": "Espanol",
    "Málaga": "Malaga",
    "Racing Club": "Santander",
    "Racing Santander": "Santander",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "1. FC Union Berlin": "Union Berlin",
    "B. M'gladbach": "M'gladbach",
    "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Frankfurt": "Ein Frankfurt",
    "FC Augsburg": "Augsburg",
    "FC Cologne": "FC Koln",
    "1. FC Köln": "FC Koln",
    "Hamburg SV": "Hamburg",
    "Hamburger SV": "Hamburg",
    "Mainz 05": "Mainz",
    "SC Freiburg": "Freiburg",
    "SC Paderborn 07": "Paderborn",
    "SV Elversberg": "Elversberg",
    "TSG Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
    "AJ Auxerre": "Auxerre",
    "Le Havre AC": "Le Havre",
    "PSG": "Paris SG",
    "Stade Rennais": "Rennes",
}

SPORTSGAMBLER_SLUGS: dict[str, str] = {
    "Premier League": "england-premier-league",
    "La Liga": "spain-la-liga",
    "Serie A": "italy-serie-a",
    "Bundesliga": "germany-bundesliga",
    "Ligue 1": "france-ligue-1",
}

_POS_MAP_SK: dict[str, str] = {
    "G": "Brankár",
    "D": "Obranca",
    "M": "Stredopoliar",
    "F": "Útočník",
}

_INJURY_TRANSLATIONS: list[tuple[str, str]] = [
    ("cruciate ligament", "Poranenie skríženého väzu (ACL)"),
    ("acl", "Poranenie predného skríženého väzu (ACL)"),
    ("hamstring", "Svalové zranenie zadného stehna (hamstring)"),
    ("muscle", "Svalové zranenie"),
    ("thigh", "Zranenie stehna"),
    ("calf", "Zranenie lýtka"),
    ("knee", "Zranenie kolena"),
    ("ankle", "Zranenie členka"),
    ("groin", "Zranenie slabín"),
    ("achilles", "Poranenie Achillovej šľachy"),
    ("shoulder", "Zranenie ramena"),
    ("back", "Problémy s chrbtom"),
    ("foot", "Zranenie chodidla"),
    ("hip", "Zranenie bedra"),
    ("illness", "Choroba / viróza"),
    ("concussion", "Otras mozgu"),
    ("red card", "Červená karta (stopka)"),
    ("yellow card", "Trest za žlté karty"),
    ("suspended", "Suspendovaný (trest)"),
    ("knock", "Menší úder / narazenie"),
]


def _translate_injury_sk(info: str, is_susp: bool) -> str:
    if not info:
        return "Trest / suspendovaný" if is_susp else "Zranenie (bližšie nešpecifikované)"
    low = info.lower()
    for key, sk in _INJURY_TRANSLATIONS:
        if key in low:
            return f"{sk} ({info})" if info.lower() not in sk.lower() else sk
    return info


def _format_return_and_duration(ret: str, is_susp: bool) -> tuple[str, str]:
    if not ret or ret in ("-", "?"):
        return ("Neurčený", "Neurčená" if not is_susp else "1–3 zápasy")
    try:
        dt = datetime.strptime(ret[:10], "%Y-%m-%d").date()
        ret_sk = dt.strftime("%d. %m. %Y")
        days = (dt - date.today()).days
        if days <= 0:
            dur = "Krátkodobá (čoskoro späť)"
        elif days <= 7:
            dur = f"Cca {days} dní"
        elif days <= 45:
            weeks = max(1, round(days / 7))
            dur = f"Cca {weeks} týždňov" if weeks >= 5 else ("1 týždeň" if weeks == 1 else f"Cca {weeks} týždne")
        else:
            months = max(1, round(days / 30))
            dur = f"Cca {months} mesiacov" if months >= 5 else ("1 mesiac" if months == 1 else f"Cca {months} mesiace")
        return ret_sk, dur
    except ValueError:
        return ret, "Neurčená"


def _scrape_sportsgambler_league(league: str, slug: str) -> tuple[str, dict[str, list[dict]]]:
    url = f"https://www.sportsgambler.com/injuries/football/{slug}/"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    )
    raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")

    blocks = raw.split('<div class="injury-block">')[1:]
    clubs: dict[str, list[dict]] = {}
    for blk in blocks:
        m_team = re.search(r'<h3 class="injuries-title"[^>]*>.*?<a[^>]*>(.*?)</a>', blk, re.S)
        if not m_team:
            m_team = re.search(r'<h3 class="injuries-title"[^>]*>(.*?)</h3>', blk, re.S)
        if not m_team:
            continue
        raw_team = html_lib.unescape(re.sub(r"<[^>]+>", "", m_team.group(1))).strip()
        canonical_team = INJURY_CLUB_ALIASES.get(raw_team, raw_team)

        rows = blk.split('<div class="inj-row">')[1:]
        players: list[dict] = []
        for r in rows:
            def _grab(cls_pat: str) -> str:
                m = re.search(rf'<span class="{cls_pat}"[^>]*>(.*?)</span>', r, re.S)
                return html_lib.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip() if m else ""

            m_type = re.search(r'<span class="inj-type\s+([^"]+)"', r)
            inj_type = m_type.group(1).strip() if m_type else "injury"
            name = _grab("inj-player")
            if not name:
                continue
            pos_raw = _grab(r'inj-position[^"]*')
            info = _grab("inj-info")
            ret = _grab(r'inj-return[^"]*')
            games = _grab(r'inj-game[^"]*')
            is_susp = "redcard" in inj_type or "susp" in inj_type
            ret_sk, dur_sk = _format_return_and_duration(ret, is_susp)
            ctx = (
                f"V zápase ({league} – disciplinárny trest)"
                if is_susp
                else (f"V zápase / na tréningu ({games} zápasov v sezóne)" if games and games != "-" else f"V zápase / na tréningu ({league})")
            )
            players.append(
                {
                    "player": name,
                    "position": _POS_MAP_SK.get(pos_raw, pos_raw or "Hráč"),
                    "injury": _translate_injury_sk(info, is_susp),
                    "injured_on": "Sezóna 2026/27 (aktuálne)",
                    "duration": dur_sk,
                    "expected_return": ret_sk,
                    "context": ctx,
                    "status": "Suspendovaný" if is_susp else "Mimo hry",
                }
            )
        clubs[canonical_team] = players
    return league, clubs


def fetch_live_injuries_by_league(save_snapshot: bool = False) -> tuple[str, dict[str, dict[str, pd.DataFrame]]]:
    """Stiahne aktuálne zranenia pre všetky kluby Top 5 líg; pri chybe použije `data/injuries.json`."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = list(
                ex.map(
                    lambda item: _scrape_sportsgambler_league(item[0], item[1]),
                    SPORTSGAMBLER_SLUGS.items(),
                )
            )
        today_str = date.today().isoformat()
        raw_leagues = {lg: clubs for lg, clubs in results if clubs}
        if not raw_leagues:
            return load_injuries_by_league()

        if save_snapshot:
            INJURIES_JSON.parent.mkdir(parents=True, exist_ok=True)
            INJURIES_JSON.write_text(
                json.dumps({"updated_at": today_str, "leagues": raw_leagues}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        df_leagues: dict[str, dict[str, pd.DataFrame]] = {
            lg: {
                club: pd.DataFrame(records).rename(columns=COLUMN_RENAME)
                for club, records in clubs.items()
            }
            for lg, clubs in raw_leagues.items()
        }
        return today_str, df_leagues
    except Exception:
        return load_injuries_by_league()


def load_injuries_by_league() -> tuple[str, dict[str, dict[str, pd.DataFrame]]]:
    """Vráti dátum aktualizácie a slovník {liga: {klub: DataFrame zranených hráčov}}."""
    if not INJURIES_JSON.exists():
        return "", {}

    payload = json.loads(INJURIES_JSON.read_text(encoding="utf-8"))
    updated_at = str(payload.get("updated_at", ""))
    leagues_raw = payload.get("leagues", {})

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for league, clubs_raw in leagues_raw.items():
        league_clubs: dict[str, pd.DataFrame] = {}
        for club, records in clubs_raw.items():
            canonical = INJURY_CLUB_ALIASES.get(club, club)
            league_clubs[canonical] = pd.DataFrame(records).rename(columns=COLUMN_RENAME)
        result[league] = league_clubs
    return updated_at, result
