"""Implementácia troch ratingových systémov pre futbal: Elo, Glicko a Glicko-2.

Všetky tri algoritmy sú napísané čisto v Pythone bez knižníc strojového učenia,
aby bol výpočet priehľadný, rýchly a ľahko vysvetliteľný pri obhajobe SOČ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Základné konštanty systémov
# ---------------------------------------------------------------------------
START_RATING = 1500.0
START_RD = 350.0
MIN_RD = 30.0
START_VOLATILITY = 0.06
GLICKO2_SCALE = 173.7178
GLICKO_Q = math.log(10.0) / 400.0

# Výhoda domáceho ihriska vyjadrená v ratingových bodoch
HOME_ADVANTAGE = 60.0
# K-faktor pre klasické Elo
ELO_K = 20.0
# Rýchlosť rastu neistoty (RD) pri prestávke medzi sezónami
SEASON_RD_INCREASE = 60.0
# Parameter systému Glicko-2 (obmedzuje príliš rýchlu zmenu volatility)
GLICKO2_TAU = 0.5
# Parameter remíz v Davidsonovom modeli (zodpovedá ~26,5 % remíz pri vyrovnaných tímoch)
DAVIDSON_NU = 0.72


def davidson_probabilities(
    rating_home: float,
    rating_away: float,
    home_advantage: float = HOME_ADVANTAGE,
    nu: float = DAVIDSON_NU,
    uncertainty_factor: float = 1.0,
) -> tuple[float, float, float]:
    """Prepočíta rozdiel ratingov na pravdepodobnosti (Výhra H, Remíza D, Výhra A).

    Používa Davidsonovo rozšírenie Bradley-Terryho modelu pre zápasy s remízou:
        r = 10 ^ ((R_H + domaca_vyhoda - R_A) * faktor / 800)
        P(H) = r / (r + 1/r + nu)
        P(D) = nu / (r + 1/r + nu)
        P(A) = (1/r) / (r + 1/r + nu)
    """
    diff = (rating_home + home_advantage - rating_away) * uncertainty_factor
    # Obmedzenie exponentu proti pretečeniu pri extrémnych rozdieloch
    diff = max(-1200.0, min(1200.0, diff))
    r = 10.0 ** (diff / 800.0)
    denom = r + (1.0 / r) + nu
    p_home = r / denom
    p_draw = nu / denom
    p_away = (1.0 / r) / denom
    return p_home, p_draw, p_away


# ---------------------------------------------------------------------------
# 1. Klasické Elo
# ---------------------------------------------------------------------------
@dataclass
class EloSystem:
    """Klasický Elo ratingový systém."""

    k_factor: float = ELO_K
    home_advantage: float = HOME_ADVANTAGE

    def expected_score(self, rating_a: float, rating_b: float, bias_a: float = 0.0) -> float:
        diff = (rating_a + bias_a) - rating_b
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def predict_proba(self, rating_home: float, rating_away: float) -> tuple[float, float, float]:
        return davidson_probabilities(
            rating_home, rating_away, home_advantage=self.home_advantage
        )

    def update(
        self, rating_home: float, rating_away: float, score_home: float
    ) -> tuple[float, float]:
        """Aktualizuje ratingy po zápase (score_home: 1.0=výhra, 0.5=remíza, 0.0=prehra)."""
        exp_home = self.expected_score(rating_home, rating_away, bias_a=self.home_advantage)
        exp_away = 1.0 - exp_home
        score_away = 1.0 - score_home

        new_home = rating_home + self.k_factor * (score_home - exp_home)
        new_away = rating_away + self.k_factor * (score_away - exp_away)
        return new_home, new_away


# ---------------------------------------------------------------------------
# 2. Glicko (Glicko-1)
# ---------------------------------------------------------------------------
@dataclass
class GlickoTeam:
    rating: float = START_RATING
    rd: float = START_RD


class GlickoSystem:
    """Glicko ratingový systém s mierou neistoty (RD - Rating Deviation)."""

    def __init__(
        self,
        home_advantage: float = HOME_ADVANTAGE,
        c_per_day: float = 1.5,
    ) -> None:
        self.home_advantage = home_advantage
        self.c_per_day = c_per_day

    @staticmethod
    def _g(rd: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * (GLICKO_Q ** 2) * (rd ** 2) / (math.pi ** 2))

    def _expected(self, rating: float, opp_rating: float, opp_rd: float, bias: float) -> float:
        g_opp = self._g(opp_rd)
        exponent = -g_opp * ((rating + bias) - opp_rating) / 400.0
        exponent = max(-20.0, min(20.0, exponent))
        return 1.0 / (1.0 + 10.0 ** exponent)

    def apply_inactivity(self, team: GlickoTeam, days: int) -> GlickoTeam:
        """Neistota (RD) mierne narastá s počtom dní bez zápasu."""
        if days <= 0:
            return team
        new_rd = math.sqrt(team.rd ** 2 + (self.c_per_day ** 2) * days)
        return GlickoTeam(rating=team.rating, rd=min(START_RD, new_rd))

    def predict_proba(self, home: GlickoTeam, away: GlickoTeam) -> tuple[float, float, float]:
        # Vyššia kombinovaná neistota mierne tlmí extrémne predikcie
        combined_rd = math.sqrt(home.rd ** 2 + away.rd ** 2)
        uncertainty_factor = self._g(combined_rd)
        return davidson_probabilities(
            home.rating,
            away.rating,
            home_advantage=self.home_advantage,
            uncertainty_factor=uncertainty_factor,
        )

    def _update_one(
        self, team: GlickoTeam, opp: GlickoTeam, score: float, bias: float
    ) -> GlickoTeam:
        g_opp = self._g(opp.rd)
        e = self._expected(team.rating, opp.rating, opp.rd, bias=bias)
        d2_inv = (GLICKO_Q ** 2) * (g_opp ** 2) * e * (1.0 - e)
        rd2_inv = 1.0 / (team.rd ** 2)

        denom = rd2_inv + d2_inv
        new_rating = team.rating + (GLICKO_Q / denom) * g_opp * (score - e)
        new_rd = max(MIN_RD, min(START_RD, math.sqrt(1.0 / denom)))
        return GlickoTeam(rating=new_rating, rd=new_rd)

    def update(
        self, home: GlickoTeam, away: GlickoTeam, score_home: float
    ) -> tuple[GlickoTeam, GlickoTeam]:
        new_home = self._update_one(home, away, score_home, bias=self.home_advantage)
        new_away = self._update_one(away, home, 1.0 - score_home, bias=-self.home_advantage)
        return new_home, new_away


# ---------------------------------------------------------------------------
# 3. Glicko-2
# ---------------------------------------------------------------------------
@dataclass
class Glicko2Team:
    rating: float = START_RATING
    rd: float = START_RD
    vol: float = START_VOLATILITY


class Glicko2System:
    """Glicko-2 ratingový systém s neistotou (RD) a kolísavosťou formy (volatilitou)."""

    def __init__(
        self,
        home_advantage: float = HOME_ADVANTAGE,
        tau: float = GLICKO2_TAU,
        c_per_day: float = 1.5,
    ) -> None:
        self.home_advantage = home_advantage
        self.tau = tau
        self.c_per_day = c_per_day

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * (phi ** 2) / (math.pi ** 2))

    @staticmethod
    def _expected(mu: float, opp_mu: float, opp_phi: float, bias_mu: float) -> float:
        g_opp = Glicko2System._g(opp_phi)
        exponent = -g_opp * ((mu + bias_mu) - opp_mu)
        exponent = max(-20.0, min(20.0, exponent))
        return 1.0 / (1.0 + math.exp(exponent))

    def apply_inactivity(self, team: Glicko2Team, days: int) -> Glicko2Team:
        if days <= 0:
            return team
        new_rd = math.sqrt(team.rd ** 2 + (self.c_per_day ** 2) * days)
        return Glicko2Team(rating=team.rating, rd=min(START_RD, new_rd), vol=team.vol)

    def predict_proba(self, home: Glicko2Team, away: Glicko2Team) -> tuple[float, float, float]:
        combined_phi = math.sqrt((home.rd / GLICKO2_SCALE) ** 2 + (away.rd / GLICKO2_SCALE) ** 2)
        uncertainty_factor = self._g(combined_phi)
        return davidson_probabilities(
            home.rating,
            away.rating,
            home_advantage=self.home_advantage,
            uncertainty_factor=uncertainty_factor,
        )

    def _new_volatility(self, phi: float, sigma: float, delta: float, v: float) -> float:
        """Výpočet novej volatility pomocou Illinoisovho algoritmu (Glickman)."""
        a = math.log(sigma ** 2)
        tau2 = self.tau ** 2
        phi2 = phi ** 2
        delta2 = delta ** 2

        def f(x: float) -> float:
            ex = math.exp(x)
            num1 = ex * (delta2 - phi2 - v - ex)
            den1 = 2.0 * ((phi2 + v + ex) ** 2)
            return (num1 / den1) - ((x - a) / tau2)

        A = a
        if delta2 > phi2 + v:
            B = math.log(delta2 - phi2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        fA = f(A)
        fB = f(B)
        epsilon = 1e-6
        while abs(B - A) > epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA = fA / 2.0
            B = C
            fB = fC

        return math.exp(A / 2.0)

    def _update_one(
        self, team: Glicko2Team, opp: Glicko2Team, score: float, bias_rating: float
    ) -> Glicko2Team:
        mu = (team.rating - START_RATING) / GLICKO2_SCALE
        phi = team.rd / GLICKO2_SCALE
        opp_mu = (opp.rating - START_RATING) / GLICKO2_SCALE
        opp_phi = opp.rd / GLICKO2_SCALE
        bias_mu = bias_rating / GLICKO2_SCALE

        g_opp = self._g(opp_phi)
        e = self._expected(mu, opp_mu, opp_phi, bias_mu=bias_mu)
        v = 1.0 / ((g_opp ** 2) * e * (1.0 - e))
        delta = v * g_opp * (score - e)

        new_sigma = self._new_volatility(phi, team.vol, delta, v)
        phi_star = math.sqrt(phi ** 2 + new_sigma ** 2)
        new_phi = 1.0 / math.sqrt((1.0 / (phi_star ** 2)) + (1.0 / v))
        new_mu = mu + (new_phi ** 2) * g_opp * (score - e)

        new_rating = START_RATING + GLICKO2_SCALE * new_mu
        new_rd = max(MIN_RD, min(START_RD, GLICKO2_SCALE * new_phi))
        return Glicko2Team(rating=new_rating, rd=new_rd, vol=new_sigma)

    def update(
        self, home: Glicko2Team, away: Glicko2Team, score_home: float
    ) -> tuple[Glicko2Team, Glicko2Team]:
        new_home = self._update_one(home, away, score_home, bias_rating=self.home_advantage)
        new_away = self._update_one(away, home, 1.0 - score_home, bias_rating=-self.home_advantage)
        return new_home, new_away
