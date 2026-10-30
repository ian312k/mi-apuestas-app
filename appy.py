# app.py — Predictor de fútbol v2
# =============================================================================
# Cambios principales respecto a la v1
#   * Sin fugas de información: los tiros a puerta del propio partido ya no se
#     usan; todas las features (DC, Elo, forma, descanso, SOT) son pre-partido
#     y las de entrenamiento se calculan con ventana expansiva.
#   * Dixon-Coles real (máxima verosimilitud, ponderado por tiempo, con
#     regularización/shrinkage y rho estimado) y vectorizado.
#   * ML: regresión logística (por defecto) o árboles calibrados, entrenable
#     con las 7 ligas juntas; se compara siempre contra el mercado.
#   * Probabilidad ajustada = mezcla modelo/mercado; Kelly fraccionado con tope.
#   * Mejor cuota entre casas, partidos en vivo excluidos, créditos de API.
#   * Resultados persistentes (session_state), fragmentos, caches, SQLite.
# =============================================================================
import io
import os
import re
import sqlite3
import unicodedata
from collections import defaultdict, deque
from contextlib import closing
from datetime import datetime
from difflib import get_close_matches
from html import escape as html_escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# st.fragment con fallback para versiones anteriores
fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)

# ======================================================
# 1. CONFIGURACIÓN
# ======================================================
st.set_page_config(page_title="Análisis predictivo de resultados de fútbol", layout="wide", page_icon="🛡️")

DB_FILE = "mis_apuestas_pro.db"
LEGACY_CSV = "mis_apuestas_pro.csv"
CACHE_DIR = "data_cache"
N_SEASONS = 4
UA = {"User-Agent": "Mozilla/5.0 (compatible; predictor-futbol/2.0)"}

LEAGUES = {
    "SP1": "🇪🇸 La Liga",
    "E0": "🏴 Premier League",
    "I1": "🇮🇹 Serie A",
    "D1": "🇩🇪 Bundesliga",
    "F1": "🇫🇷 Ligue 1",
    "N1": "🇳🇱 Eredivisie",
    "P1": "🇵🇹 Primeira Liga",
}
API_LEAGUE_MAP = {
    "SP1": "soccer_spain_la_liga",
    "E0": "soccer_epl",
    "I1": "soccer_italy_serie_a",
    "D1": "soccer_germany_bundesliga",
    "F1": "soccer_france_ligue_one",
    "N1": "soccer_netherlands_eredivisie",
    "P1": "soccer_portugal_primeira_liga",
}

# --- TRADUCTOR DE EQUIPOS (API -> CSV histórico) ---
# Los nombres de football-data.co.uk cambian por temporada: si algún equipo no
# empareja, aparecerá en "Equipos sin emparejar" (pestaña Escáner) para que lo
# agregues aquí.
TEAM_MAP = {
    # 🏴 Premier League
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest", "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton", "Brighton and Hove Albion": "Brighton",
    "Leeds United": "Leeds", "West Ham United": "West Ham",
    "Newcastle United": "Newcastle", "Tottenham Hotspur": "Tottenham",
    "Leicester City": "Leicester", "AFC Bournemouth": "Bournemouth",
    "Ipswich Town": "Ipswich", "Luton Town": "Luton",
    # 🇪🇸 La Liga
    "Athletic Club": "Ath Bilbao", "Athletic Bilbao": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid", "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis", "Celta Vigo": "Celta", "RC Celta de Vigo": "Celta",
    "RCD Espanyol": "Espanol", "Espanyol": "Espanol",
    "Real Sociedad": "Sociedad", "Rayo Vallecano": "Vallecano",
    "Deportivo Alavés": "Alaves", "Alavés": "Alaves", "Real Oviedo": "Oviedo",
    "Levante UD": "Levante", "Elche CF": "Elche", "Real Valladolid": "Valladolid",
    "UD Las Palmas": "Las Palmas", "CD Leganés": "Leganes",
    # 🇮🇹 Serie A
    "Internazionale": "Inter", "Inter Milan": "Inter", "AC Milan": "Milan",
    "AS Roma": "Roma", "Hellas Verona": "Verona", "Parma Calcio 1913": "Parma",
    # 🇩🇪 Bundesliga
    "Borussia Dortmund": "Dortmund", "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt", "Borussia Monchengladbach": "M'gladbach",
    "Borussia Mönchengladbach": "M'gladbach", "VfL Wolfsburg": "Wolfsburg",
    "VfB Stuttgart": "Stuttgart", "TSG Hoffenheim": "Hoffenheim",
    "1899 Hoffenheim": "Hoffenheim", "SC Freiburg": "Freiburg",
    "FSV Mainz 05": "Mainz", "FC Augsburg": "Augsburg",
    "1. FC Heidenheim": "Heidenheim", "VfL Bochum": "Bochum",
    "1. FC Köln": "FC Koln", "FC St. Pauli": "St Pauli", "Hamburger SV": "Hamburg",
    "1. FC Union Berlin": "Union Berlin", "SV Werder Bremen": "Werder Bremen",
    # 🇫🇷 Ligue 1
    "Paris Saint Germain": "Paris SG", "Paris Saint-Germain": "Paris SG",
    "Olympique Marseille": "Marseille", "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco", "OGC Nice": "Nice", "RC Lens": "Lens",
    "Stade Rennais": "Rennes", "Stade Brestois 29": "Brest",
    "RC Strasbourg": "Strasbourg", "Stade de Reims": "Reims",
    "Montpellier HSC": "Montpellier", "Le Havre AC": "Le Havre",
    "AJ Auxerre": "Auxerre", "Angers SCO": "Angers",
    "AS Saint-Étienne": "St Etienne", "FC Lorient": "Lorient", "FC Metz": "Metz",
    # 🇳🇱 Eredivisie
    "Ajax Amsterdam": "Ajax", "AFC Ajax": "Ajax", "FC Twente": "Twente",
    "FC Utrecht": "Utrecht", "SC Heerenveen": "Heerenveen", "FC Groningen": "Groningen",
    "PEC Zwolle": "Zwolle", "Fortuna Sittard": "For Sittard",
    "Heracles Almelo": "Heracles", "RKC Waalwijk": "Waalwijk",
    "NEC Nijmegen": "Nijmegen", "Sparta Rotterdam": "Sparta Rotterdam",
    # 🇵🇹 Primeira Liga
    "Sporting CP": "Sp Lisbon", "Sporting Lisbon": "Sp Lisbon",
    "SL Benfica": "Benfica", "FC Porto": "Porto", "Sporting Braga": "Sp Braga",
    "SC Braga": "Sp Braga", "Vitoria Guimaraes": "Guimaraes",
    "Vitória SC": "Guimaraes", "FC Famalicão": "Famalicao",
    "Gil Vicente FC": "Gil Vicente", "FC Arouca": "Arouca",
}

_STOP_TOKENS = {"fc", "cf", "afc", "ac", "sc", "ud", "rcd", "ca", "cd", "sd", "ssc",
                "as", "us", "vfl", "vfb", "tsg", "fsv", "sv", "bv", "1", "de", "club"}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in _STOP_TOKENS)


def match_team(api_name, candidates, cutoff=0.8):
    """Empareja el nombre de la API con un nombre del histórico (o None)."""
    if not api_name or not candidates:
        return None
    mapped = TEAM_MAP.get(api_name, api_name)
    if mapped in candidates:
        return mapped
    norm_map = {_norm(c): c for c in candidates}
    n = _norm(mapped)
    if n in norm_map:
        return norm_map[n]
    close = get_close_matches(n, list(norm_map), n=1, cutoff=cutoff)
    return norm_map[close[0]] if close else None


def log_unmatched(league, name):
    st.session_state.setdefault("unmatched", set()).add((league, str(name)))


# ======================================================
# 2. UTILIDADES DE CUOTAS / APUESTAS
# ======================================================
def odds_to_probs(oh, od, oa):
    inv = np.array([1.0 / max(float(oh), 1.01), 1.0 / max(float(od), 1.01), 1.0 / max(float(oa), 1.01)])
    return inv / inv.sum()


def overround(oh, od, oa):
    return 1.0 / max(float(oh), 1.01) + 1.0 / max(float(od), 1.01) + 1.0 / max(float(oa), 1.01) - 1.0


def safe_fair_odds(p, eps=1e-12):
    return 1.0 / float(np.clip(p, eps, 1.0))


def blend_probs(p_model, p_market, w_model):
    """Mezcla modelo/mercado: w_model=0.4 -> 40% modelo, 60% mercado."""
    p = w_model * np.asarray(p_model, float) + (1.0 - w_model) * np.asarray(p_market, float)
    return p / p.sum()


def kelly_pct(prob, odd, fraction=0.25, cap_pct=3.0, min_edge=0.0):
    """% del banco a apostar: Kelly fraccionado con tope por apuesta. 'min_edge' es la ventaja
    mínima en puntos de probabilidad (prob - 1/cuota); con 0 equivale a exigir EV > 0."""
    if prob <= 0 or odd <= 1.0:
        return 0.0
    if prob - 1.0 / odd <= min_edge:
        return 0.0
    b = odd - 1.0
    f = (b * prob - (1.0 - prob)) / b
    return float(min(max(0.0, f * fraction) * 100.0, cap_pct))


def prob_edges(adj, odds):
    """Ventaja en puntos de probabilidad: prob. ajustada - probabilidad implícita de la cuota (1/cuota)."""
    return np.asarray(adj, float) - 1.0 / np.maximum(np.asarray(odds, float), 1.01)


def pick_from_probs(adj, odds, P):
    """Mejor selección 1X2. Se elige y se filtra por VENTAJA EN PROBABILIDAD (no por EV bruto): el EV crece
    con la cuota, así que un umbral de EV favorece sistemáticamente empates y cuotas largas, donde el
    ruido del modelo se amplifica. Devuelve (k, ev, kelly%) o None."""
    adj = np.asarray(adj, float)
    odds = np.asarray(odds, float)
    edge = prob_edges(adj, odds)
    ok = (odds <= P["odd_max"]) & (odds > 1.01) & np.isin(np.arange(3), P.get("allowed") or [0, 1, 2])
    score = np.where(ok, edge, -np.inf)
    k = int(np.argmax(score))
    best = float(score[k])
    if not np.isfinite(best) or best <= 0 or best < P["edge_min"]:
        return None
    return k, float(adj[k] * odds[k] - 1.0), kelly_pct(adj[k], odds[k], P["kelly_frac"], P["cap_pct"], P["edge_min"])


# ======================================================
# 3. DATOS (football-data.co.uk) con respaldo local
# ======================================================
def _season_codes(n):
    today = datetime.now()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{(start - i) % 100:02d}{(start - i + 1) % 100:02d}" for i in range(n)]


def _pick_odds(raw):
    """Cuota de referencia: promedio de mercado > Pinnacle > Bet365 (todas pre-partido)."""
    out = pd.DataFrame(np.nan, index=raw.index, columns=["odd_h", "odd_d", "odd_a"])
    for trip in (("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A")):
        if all(c in raw.columns for c in trip):
            cand = raw[list(trip)].apply(pd.to_numeric, errors="coerce")
            cand.columns = ["odd_h", "odd_d", "odd_a"]
            ok = (cand > 1.01).all(axis=1)
            fill = out["odd_h"].isna() & ok
            if fill.any():
                out.loc[fill, ["odd_h", "odd_d", "odd_a"]] = cand.loc[fill].to_numpy()
    return out


def _clean_season(raw, season):
    tmp = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
        "home": raw["HomeTeam"].astype(str).str.strip(),
        "away": raw["AwayTeam"].astype(str).str.strip(),
        "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce"),
        "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce"),
    })
    for new, old in (("sot_h", "HST"), ("sot_a", "AST")):
        tmp[new] = pd.to_numeric(raw[old], errors="coerce") if old in raw.columns else np.nan
    tmp = pd.concat([tmp, _pick_odds(raw)], axis=1)
    tmp = tmp.dropna(subset=["date", "home_goals", "away_goals"])
    tmp = tmp[(tmp["home"] != "nan") & (tmp["away"] != "nan")]
    tmp["season"] = season
    return tmp


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_soccer_data(league_code, n_seasons=N_SEASONS):
    """Descarga las últimas temporadas. Si falla una descarga usa la copia local."""
    frames, msgs = [], []
    os.makedirs(CACHE_DIR, exist_ok=True)
    for s in _season_codes(n_seasons):
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{league_code}.csv"
        path = os.path.join(CACHE_DIR, f"{league_code}_{s}.csv")
        raw = None
        try:
            r = requests.get(url, headers=UA, timeout=25)
            r.raise_for_status()
            raw = pd.read_csv(io.BytesIO(r.content), encoding="latin1", on_bad_lines="skip")
            raw.to_csv(path, index=False)
        except Exception as e:
            msgs.append(f"{league_code} {s}: descarga fallida ({type(e).__name__})")
            if os.path.exists(path):
                try:
                    raw = pd.read_csv(path)
                    msgs.append(f"{league_code} {s}: usando copia local")
                except Exception:
                    raw = None
        if raw is None or raw.empty:
            continue
        try:
            frames.append(_clean_season(raw, s))
        except Exception as e:
            msgs.append(f"{league_code} {s}: formato inesperado ({e})")
    if not frames:
        return pd.DataFrame(), msgs
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    return df, msgs


# ======================================================
# 4. CONTEXTO PRE-PARTIDO (Elo, forma, descanso, tiros a puerta)
# ======================================================
CTX_COLS = ["elo_diff", "form_h", "form_a", "rest_h", "rest_a", "sot_h_pre", "sot_a_pre"]


class Tracker:
    """Estado secuencial por equipo. Las features se leen ANTES de actualizar con el partido,
    por lo que nunca contienen información del propio encuentro."""
    DEF_SOT = 4.2

    def __init__(self, k=20.0, home_adv=60.0, form_n=5, sot_alpha=0.12):
        self.k, self.ha, self.alpha = k, home_adv, sot_alpha
        self.elo, self.last, self.sot = {}, {}, {}
        self.form = defaultdict(lambda: deque(maxlen=form_n))
        self.sot_mean = self.DEF_SOT

    def _form(self, t):
        f = self.form.get(t)
        if not f:
            return 1.35
        w = np.arange(1, len(f) + 1, dtype=float)
        return float(np.dot(w, np.array(f, float)) / w.sum())

    def features(self, home, away, date):
        def rest(t):
            if t not in self.last:
                return 7.0
            return float(np.clip((date - self.last[t]).days, 2, 14))
        eh, ea = self.elo.get(home, 1500.0), self.elo.get(away, 1500.0)
        return np.array([
            eh + self.ha - ea, self._form(home), self._form(away), rest(home), rest(away),
            self.sot.get(home, self.sot_mean), self.sot.get(away, self.sot_mean),
        ], dtype=float)

    def update(self, r):
        h, a, hg, ag = r.home, r.away, r.home_goals, r.away_goals
        eh, ea = self.elo.get(h, 1500.0), self.elo.get(a, 1500.0)
        exp_h = 1.0 / (1.0 + 10 ** (-(eh + self.ha - ea) / 400.0))
        s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = self.k * (1.0 + np.log1p(abs(hg - ag))) * (s - exp_h)
        self.elo[h], self.elo[a] = eh + delta, ea - delta
        self.form[h].append(3 if hg > ag else (1 if hg == ag else 0))
        self.form[a].append(3 if ag > hg else (1 if hg == ag else 0))
        self.last[h] = self.last[a] = r.date
        for t, v in ((h, r.sot_h), (a, r.sot_a)):
            if pd.notna(v):
                prev = self.sot.get(t)
                self.sot[t] = float(v) if prev is None else self.alpha * float(v) + (1 - self.alpha) * prev
                self.sot_mean = 0.995 * self.sot_mean + 0.005 * float(v)


def compute_context(df):
    """Añade columnas CTX_COLS (pre-partido) y devuelve también el estado final del Tracker."""
    d = df.sort_values("date").reset_index(drop=True)
    tr = Tracker()
    rows = []
    for r in d.itertuples(index=False):
        rows.append(tr.features(r.home, r.away, r.date))
        tr.update(r)
    ctx = np.vstack(rows) if rows else np.zeros((0, len(CTX_COLS)))
    out = d.copy()
    for j, name in enumerate(CTX_COLS):
        out[name] = ctx[:, j]
    return out, tr

# ======================================================
# 5. DIXON-COLES POR MÁXIMA VEROSIMILITUD (ponderado por tiempo + ridge)
# ======================================================
def fit_dixon_coles(df, ref_date=None, xi=0.004, reg=3.0, window_matches=None):
    """Ajusta log(lambda)=mu0+gamma+att[h]+dfn[a], log(mu)=mu0+att[a]+dfn[h] con la corrección
    tau de Dixon-Coles (rho estimado). 'reg' es una penalización ridge sobre att/dfn
    (shrinkage hacia el equipo promedio). Usa solo los partidos de 'df'."""
    d = df.dropna(subset=["date", "home", "away", "home_goals", "away_goals"]).sort_values("date")
    if window_matches is not None and len(d) > window_matches:
        d = d.tail(window_matches)
    if len(d) < 60:
        return None
    ref = pd.Timestamp(ref_date) if ref_date is not None else d["date"].max()
    days = np.clip((ref - d["date"]).dt.days.to_numpy(float), 0, None)
    w = np.exp(-xi * days)

    teams = sorted(set(d["home"]) | set(d["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h = d["home"].map(idx).to_numpy()
    a = d["away"].map(idx).to_numpy()
    x = d["home_goals"].to_numpy(float)
    y = d["away_goals"].to_numpy(float)
    m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
    m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)

    def nll(p):
        mu0, gam, rho = p[0], p[1], p[2]
        al, be = p[3:3 + n], p[3 + n:]
        e1 = mu0 + gam + al[h] + be[a]
        e2 = mu0 + al[a] + be[h]
        lam, mu = np.exp(e1), np.exp(e2)
        tau = np.ones_like(lam)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        tau = np.maximum(tau, 1e-6)
        ll = w * (np.log(tau) + x * e1 - lam + y * e2 - mu)
        val = -ll.sum() + reg * (al @ al + be @ be)
        # gradiente analítico
        dt1, dt2, dtr = np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
        dt1[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
        dt2[m00] = dt1[m00]
        dt1[m01] = lam[m01] * rho / tau[m01]
        dt2[m10] = mu[m10] * rho / tau[m10]
        dtr[m00] = -lam[m00] * mu[m00] / tau[m00]
        dtr[m01] = lam[m01] / tau[m01]
        dtr[m10] = mu[m10] / tau[m10]
        dtr[m11] = -1.0 / tau[m11]
        g1 = w * (x - lam + dt1)
        g2 = w * (y - mu + dt2)
        grad = np.zeros_like(p)
        grad[0] = -(g1.sum() + g2.sum())
        grad[1] = -g1.sum()
        grad[2] = -(w * dtr).sum()
        grad[3:3 + n] = -(np.bincount(h, g1, n) + np.bincount(a, g2, n)) + 2 * reg * al
        grad[3 + n:] = -(np.bincount(a, g1, n) + np.bincount(h, g2, n)) + 2 * reg * be
        return val, grad

    p0 = np.zeros(3 + 2 * n)
    p0[0] = np.log(max(float(np.average((x + y) / 2.0, weights=w)), 0.5))
    p0[1], p0[2] = 0.25, -0.05
    bounds = [(None, None), (None, None), (-0.25, 0.10)] + [(-2.5, 2.5)] * (2 * n)
    res = minimize(nll, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400})
    p = res.x
    counts = pd.concat([d["home"], d["away"]]).value_counts()
    return {
        "teams": teams, "idx": idx, "mu0": float(p[0]), "gam": float(p[1]), "rho": float(p[2]),
        "al": p[3:3 + n].copy(), "be": p[3 + n:].copy(), "n": int(len(d)), "ref": ref,
        "team_n": counts.to_dict(), "xi": xi, "reg": reg,
        "avg_h": float(np.average(x, weights=w)), "avg_a": float(np.average(y, weights=w)),
    }


def dc_expected(model, home, away):
    ih, ia = model["idx"].get(home), model["idx"].get(away)
    al_h = model["al"][ih] if ih is not None else 0.0
    be_h = model["be"][ih] if ih is not None else 0.0
    al_a = model["al"][ia] if ia is not None else 0.0
    be_a = model["be"][ia] if ia is not None else 0.0
    lam = float(np.exp(model["mu0"] + model["gam"] + al_h + be_a))
    mu = float(np.exp(model["mu0"] + al_a + be_h))
    return lam, mu


def dc_matrix(lam, mu, rho, max_goals=10):
    g = np.arange(max_goals)
    probs = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    probs[0, 0] *= 1 - lam * mu * rho
    probs[0, 1] *= 1 + lam * rho
    probs[1, 0] *= 1 + mu * rho
    probs[1, 1] *= 1 - rho
    probs = np.maximum(probs, 0.0)
    return probs / probs.sum()


def predict_dc(model, home, away, max_goals=10):
    """Predicción Dixon-Coles. Equipos sin historial se tratan como equipo promedio (known=False)."""
    known = home in model["idx"] and away in model["idx"]
    lam, mu = dc_expected(model, home, away)
    probs = dc_matrix(lam, mu, model["rho"], max_goals)
    g = np.arange(max_goals)
    tot = np.add.outer(g, g)
    order = np.argsort(probs.ravel())[::-1][:3]
    top = []
    for k in order:
        i, j = np.unravel_index(k, probs.shape)
        top.append((f"{i}-{j}", float(probs[i, j])))
    return {
        "lam": lam, "mu": mu,
        "ph": float(np.tril(probs, -1).sum()), "pd": float(np.trace(probs)), "pa": float(np.triu(probs, 1).sum()),
        "po15": float(probs[tot > 1].sum()), "po25": float(probs[tot > 2].sum()),
        "pbtts": float(probs[1:, 1:].sum()), "top": top, "probs": probs, "known": known,
    }


# ======================================================
# 6. FEATURES, DATASET SIN FUGAS (ventana expansiva) Y MODELOS ML
# ======================================================
FEATURE_NAMES = [
    "Mkt H", "Mkt D", "Mkt A", "DC H", "DC D", "DC A", "λ Local", "λ Visita", "Δλ",
    "Elo Δ", "Forma L", "Forma V", "Descanso L", "Descanso V", "SOT L", "SOT V",
]


def outcome_1x2_label(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def make_features(mk, dc, ctx):
    return np.array([
        mk[0], mk[1], mk[2], dc["ph"], dc["pd"], dc["pa"], dc["lam"], dc["mu"], dc["lam"] - dc["mu"], *ctx,
    ], dtype=float)


def iter_blocks(dts, start_pos, block_days):
    """Bloques de 'block_days' días: (inicio, i0, i1). La historia de cada bloque son las filas < i0
    (estrictamente anteriores a la fecha de inicio del bloque)."""
    n, pos = len(dts), start_pos
    step = np.timedelta64(block_days, "D")
    while pos < n:
        bs = dts[pos]
        i0 = int(np.searchsorted(dts, bs, side="left"))
        i1 = max(int(np.searchsorted(dts, bs + step, side="left")), pos + 1)
        yield bs, i0, i1
        pos = i1


def build_dataset(df_ctx, xi, reg, league="", min_hist=150, block_days=14, window_matches=1500):
    """Una fila por partido con features 100% pre-partido. Las features de Dixon-Coles de cada
    partido salen de un modelo ajustado SOLO con partidos anteriores al bloque (sin fuga)."""
    d = df_ctx.sort_values("date").reset_index(drop=True)
    if len(d) <= min_hist + 20:
        return pd.DataFrame()
    dts = d["date"].to_numpy()
    recs = []
    for bs, i0, i1 in iter_blocks(dts, min_hist, block_days):
        model = fit_dixon_coles(d.iloc[:i0], ref_date=pd.Timestamp(bs), xi=xi, reg=reg, window_matches=window_matches)
        if model is None:
            continue
        for r in d.iloc[i0:i1].itertuples(index=False):
            if r.home not in model["idx"] or r.away not in model["idx"]:
                continue
            if not (r.odd_h > 1.01 and r.odd_d > 1.01 and r.odd_a > 1.01):
                continue
            mk = odds_to_probs(r.odd_h, r.odd_d, r.odd_a)
            dc = predict_dc(model, r.home, r.away)
            x = make_features(mk, dc, [getattr(r, c) for c in CTX_COLS])
            rec = {
                "date": r.date, "league": league, "home": r.home, "away": r.away,
                "y": outcome_1x2_label(r.home_goals, r.away_goals),
                "odd_h": r.odd_h, "odd_d": r.odd_d, "odd_a": r.odd_a,
            }
            rec.update(dict(zip(FEATURE_NAMES, x)))
            recs.append(rec)
    return pd.DataFrame(recs)


Z_NAMES = ["DC vs mercado L", "DC vs mercado E", "DC vs mercado V", "Elo Δ", "Forma L−V", "Descanso L−V", "SOT L−V"]


class MLModel:
    """kind='offset' (recomendado): multinomial cuyo punto de partida es el propio mercado (log de las
    probabilidades sin margen) más una corrección con ridge sobre señales propias (Dixon-Coles vs mercado,
    Elo, forma, descanso, tiros). Sin evidencia fuerte no se aparta del mercado, lo que evita probabilidades
    'ruidosas' (sobre todo en el empate, que es casi impredecible).
    kind='trees': XGBoost/RandomForest poco profundo + calibración (regresión logística sobre log-probs,
    ajustada en el 20% cronológicamente más reciente)."""

    def __init__(self, kind="offset", seed=42, alpha=0.03):
        self.kind, self.seed, self.alpha = kind, seed, alpha
        self.base = self.cal = None
        self.W = self.b = self.mu = self.sd = None

    # --- corrección sobre el mercado ---
    @staticmethod
    def _z(X):
        X = np.asarray(X, float)
        mk, dc = np.clip(X[:, 0:3], 1e-6, 1.0), np.clip(X[:, 3:6], 1e-6, 1.0)
        return np.column_stack([np.log(dc / mk), X[:, 9], X[:, 10] - X[:, 11], X[:, 12] - X[:, 13], X[:, 14] - X[:, 15]])

    def _fit_offset(self, X, y):
        X = np.asarray(X, float)
        Z = self._z(X)
        self.mu, self.sd = Z.mean(axis=0), Z.std(axis=0) + 1e-9
        Z = (Z - self.mu) / self.sd
        base = np.log(np.clip(X[:, 0:3], 1e-6, 1.0))
        n, d = Z.shape
        Y = np.eye(3)[np.asarray(y, int)]
        lam = self.alpha * n

        def f(p):
            W, b = p[:d * 3].reshape(d, 3), p[d * 3:]
            L = base + b + Z @ W
            L = L - L.max(axis=1, keepdims=True)
            E = np.exp(L)
            P = E / E.sum(axis=1, keepdims=True)
            nll = -np.log(np.clip((P * Y).sum(axis=1), 1e-12, None)).sum() + 0.5 * lam * (W ** 2).sum()
            G = P - Y
            return nll, np.concatenate([(Z.T @ G + lam * W).ravel(), G.sum(axis=0)])

        r = minimize(f, np.zeros(d * 3 + 3), jac=True, method="L-BFGS-B")
        self.W, self.b = r.x[:d * 3].reshape(d, 3), r.x[d * 3:]

    def _predict_offset(self, X):
        X = np.asarray(X, float)
        L = np.log(np.clip(X[:, 0:3], 1e-6, 1.0)) + self.b + ((self._z(X) - self.mu) / self.sd) @ self.W
        L = L - L.max(axis=1, keepdims=True)
        E = np.exp(L)
        return E / E.sum(axis=1, keepdims=True)

    # --- árboles calibrados ---
    def _make_tree(self):
        if HAS_XGB:
            return xgb.XGBClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_lambda=3.0, objective="multi:softprob", eval_metric="mlogloss",
                tree_method="hist", random_state=self.seed, n_jobs=-1,
            )
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=8, random_state=self.seed, n_jobs=-1)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        if self.kind == "offset":
            self._fit_offset(X, y)
        else:
            split = int(len(X) * 0.8)
            self.base = self._make_tree()
            self.base.fit(X[:split], y[:split])
            if len(np.unique(y[split:])) == 3:
                P = np.clip(self.base.predict_proba(X[split:]), 1e-6, 1.0)
                self.cal = LogisticRegression(C=1.0, max_iter=2000).fit(np.log(P), y[split:])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if self.kind == "offset":
            return self._predict_offset(X)
        P = self.base.predict_proba(X)
        if self.cal is not None:
            P = self.cal.predict_proba(np.log(np.clip(P, 1e-6, 1.0)))
        return P

    def explain(self, x):
        """Tabla Feature / Importancia / Valor para un partido (x = vector de FEATURE_NAMES)."""
        x = np.asarray(x, float).reshape(1, -1)
        if self.kind == "offset":
            return pd.DataFrame({"Feature": Z_NAMES, "Importancia": np.abs(self.W).mean(axis=1), "Valor": self._z(x)[0]})
        return pd.DataFrame({"Feature": FEATURE_NAMES, "Importancia": np.asarray(self.base.feature_importances_, float), "Valor": x[0]})

# ======================================================
# 7. EVALUACIÓN WALK-FORWARD, ESTRATEGIA Y AFINACIÓN
# ======================================================
def brier_multiclass(P, y, n_classes=3):
    y_oh = np.eye(n_classes)[y]
    return float(np.mean(np.sum((P - y_oh) ** 2, axis=1) / n_classes))


def _ll(P, y):
    return float(log_loss(y, np.clip(P, 1e-12, 1.0), labels=[0, 1, 2]))


def evaluate_walkforward(ds, kind="offset", n_test=400, min_train=500, retrain_every=30, seed=42):
    """Evalúa ML vs Dixon-Coles vs Mercado sobre los últimos 'n_test' partidos del dataset.
    Las features ya son pre-partido; el ML se reentrena cada 'retrain_every' partidos usando solo
    filas con fecha estrictamente anterior."""
    ds = ds.sort_values("date").reset_index(drop=True)
    if len(ds) < min_train + 30:
        return None
    n_test = int(min(n_test, len(ds) - min_train))
    start = len(ds) - n_test
    X = ds[FEATURE_NAMES].to_numpy(float)
    y = ds["y"].to_numpy(int)
    dts = ds["date"].to_numpy()
    P_ml = np.zeros((n_test, 3))
    filled = np.zeros(n_test, dtype=bool)
    model, s = None, start
    while s < len(ds):
        e = min(s + retrain_every, len(ds))
        tr_end = int(np.searchsorted(dts, dts[s], side="left"))
        if tr_end >= min_train and len(np.unique(y[:tr_end])) == 3:
            model = MLModel(kind, seed).fit(X[:tr_end], y[:tr_end])
        if model is not None:
            P_ml[s - start:e - start] = model.predict_proba(X[s:e])
            filled[s - start:e - start] = True
        s = e
    if not filled.any():
        return None
    P_ml = P_ml[filled]
    P_dc = X[start:, 3:6][filled]
    P_mk = X[start:, 0:3][filled]
    y_t = y[start:][filled]
    meta = ds.iloc[start:][filled].reset_index(drop=True)
    P = {"Mercado": P_mk, "Dixon-Coles": P_dc, "ML": P_ml}
    return {
        "n": int(len(y_t)), "P": P, "y": y_t, "meta": meta,
        "ll": {k: _ll(v, y_t) for k, v in P.items()},
        "brier": {k: brier_multiclass(v, y_t) for k, v in P.items()},
    }


def simulate_strategy(meta, P_model, P_mk, y, w_model, edge_min, odd_max, kelly_frac, cap_pct,
                      allowed=(0, 1, 2), bank0=100.0, seed=0):
    """Simula apostar a la selección con mayor VENTAJA EN PROBABILIDAD (prob. ajustada - 1/cuota) cuando la
    ventaja >= edge_min, la cuota <= odd_max y el resultado está permitido. Devuelve apuestas, resumen con
    IC95% (bootstrap), desglose por tipo de apuesta y curvas de capital (stake plano y Kelly)."""
    odds = meta[["odd_h", "odd_d", "odd_a"]].to_numpy(float)
    P_adj = w_model * np.asarray(P_model) + (1 - w_model) * np.asarray(P_mk)
    P_adj = P_adj / P_adj.sum(axis=1, keepdims=True)
    edge = P_adj - 1.0 / odds
    ok = (odds <= odd_max) & np.isin(np.arange(3), list(allowed) or [0, 1, 2])[None, :]
    score = np.where(ok, edge, -np.inf)
    k = score.argmax(axis=1)
    rows = np.arange(len(k))
    best = score[rows, k]
    sel = np.isfinite(best) & (best > 0) & (best >= edge_min)
    if not sel.any():
        return {"bets": pd.DataFrame(), "n_bets": 0}
    ids = rows[sel]
    odd = odds[ids, k[ids]]
    p = P_adj[ids, k[ids]]
    won = (y[ids] == k[ids])
    profit = np.where(won, odd - 1.0, -1.0)
    bets = pd.DataFrame({
        "Fecha": meta["date"].dt.strftime("%Y-%m-%d").to_numpy()[ids],
        "Partido": (meta["home"] + " vs " + meta["away"]).to_numpy()[ids],
        "Pick": np.array(["Local", "Empate", "Visita"])[k[ids]], "Cuota": odd, "P ajust.": p,
        "Ventaja (pp)": best[sel] * 100, "EV": p * odd - 1.0, "Acierto": won, "P/L (U)": profit,
    })
    n_b = len(bets)
    by_pick = bets.groupby("Pick").agg(Apuestas=("Cuota", "size"), Esperados=("P ajust.", "sum"),
                                       Reales=("Acierto", "sum"), Profit=("P/L (U)", "sum"))
    by_pick = by_pick.reindex(["Local", "Empate", "Visita"]).dropna(how="all")
    by_pick["% apuestas"] = by_pick["Apuestas"] / n_b * 100
    by_pick["Yield %"] = by_pick["Profit"] / by_pick["Apuestas"] * 100
    rng = np.random.default_rng(seed)
    boots = profit[rng.integers(0, n_b, size=(2000, n_b))].mean(axis=1) * 100
    equity = np.cumsum(profit)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    bank, kelly_curve = bank0, []
    for pi, oi, wi in zip(p, odd, won):
        stake = bank * kelly_pct(pi, oi, kelly_frac, cap_pct, 0.0) / 100.0
        bank += stake * (oi - 1.0) if wi else -stake
        kelly_curve.append(bank)
    return {
        "bets": bets, "n_bets": n_b, "hit": float(won.mean()), "profit": float(profit.sum()), "by_pick": by_pick.reset_index(),
        "yield": float(profit.mean() * 100), "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "max_dd": float((equity - peak).min()), "equity": equity, "kelly_curve": np.array(kelly_curve), "bank0": bank0,
    }


def dc_walkforward_logloss(df, xi, reg, n_test=300, min_hist=150, block_days=14, window_matches=1500):
    """LogLoss de Dixon-Coles (y del mercado en las mismas filas) con reajuste por bloques."""
    d = df.sort_values("date").reset_index(drop=True)
    if len(d) < min_hist + 40:
        return None
    dts = d["date"].to_numpy()
    start = max(min_hist, len(d) - n_test)
    P_dc, P_mk, ys = [], [], []
    for bs, i0, i1 in iter_blocks(dts, start, block_days):
        model = fit_dixon_coles(d.iloc[:i0], ref_date=pd.Timestamp(bs), xi=xi, reg=reg, window_matches=window_matches)
        if model is None:
            continue
        for r in d.iloc[i0:i1].itertuples(index=False):
            if r.home not in model["idx"] or r.away not in model["idx"]:
                continue
            if not (r.odd_h > 1.01 and r.odd_d > 1.01 and r.odd_a > 1.01):
                continue
            dc = predict_dc(model, r.home, r.away)
            P_dc.append([dc["ph"], dc["pd"], dc["pa"]])
            P_mk.append(odds_to_probs(r.odd_h, r.odd_d, r.odd_a))
            ys.append(outcome_1x2_label(r.home_goals, r.away_goals))
    if len(ys) < 30:
        return None
    P_dc, P_mk, ys = np.array(P_dc), np.array(P_mk), np.array(ys)
    return {"n": len(ys), "dc": _ll(P_dc, ys), "mk": _ll(P_mk, ys)}


# ======================================================
# 8. BASE DE DATOS DE APUESTAS (SQLite) + migración desde el CSV anterior
# ======================================================
BET_COLS = ["ID", "Fecha", "Tipo", "Liga", "Partido", "Pick", "Cuota", "Stake", "Prob", "EV", "Estado", "Ganancia", "Cuota_Cierre"]


def _db():
    return closing(sqlite3.connect(DB_FILE))


def _normalize_bets_df(df):
    df = df.copy()
    text_cols = ["ID", "Fecha", "Tipo", "Liga", "Partido", "Pick", "Estado"]
    for c in BET_COLS:
        if c not in df.columns:
            df[c] = pd.Series([None] * len(df), index=df.index, dtype=object) if c in text_cols else np.nan
    for c in text_cols:
        df[c] = df[c].astype(object)
    legacy = df["Tipo"].isna() & df["Liga"].isin(["Simple", "Parlay"])
    df.loc[legacy, "Tipo"] = df.loc[legacy, "Liga"]
    df.loc[legacy, "Liga"] = "—"
    df["Tipo"] = df["Tipo"].fillna("Simple")
    ev_missing = df["EV"].isna() & df["Prob"].notna() & df["Cuota"].notna()
    df.loc[ev_missing, "EV"] = df.loc[ev_missing, "Prob"] * df.loc[ev_missing, "Cuota"] - 1
    df["ID"] = df["ID"].astype(str)
    df["Ganancia"] = pd.to_numeric(df["Ganancia"], errors="coerce").fillna(0.0)
    df["Estado"] = df["Estado"].fillna("Pendiente")
    return df[BET_COLS]


def _insert_rows(con, df):
    con.executemany(
        f"INSERT OR IGNORE INTO bets ({','.join(BET_COLS)}) VALUES ({','.join('?' * len(BET_COLS))})",
        [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in row) for row in df.itertuples(index=False, name=None)],
    )


def init_db():
    with _db() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS bets (ID TEXT PRIMARY KEY, Fecha TEXT, Tipo TEXT, Liga TEXT, Partido TEXT, "
            "Pick TEXT, Cuota REAL, Stake REAL, Prob REAL, EV REAL, Estado TEXT, Ganancia REAL, Cuota_Cierre REAL)"
        )
        con.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        empty = con.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0
        if empty and os.path.exists(LEGACY_CSV):
            try:
                _insert_rows(con, _normalize_bets_df(pd.read_csv(LEGACY_CSV)))
            except Exception:
                pass
        con.commit()


def load_bets():
    with _db() as con:
        out = pd.read_sql_query("SELECT * FROM bets ORDER BY ID DESC", con)[BET_COLS]
    for c in ("Cuota", "Stake", "Prob", "EV", "Ganancia", "Cuota_Cierre"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def save_bet(rec):
    with _db() as con:
        _insert_rows(con, _normalize_bets_df(pd.DataFrame([rec])))
        con.commit()


def import_bets_df(df):
    with _db() as con:
        _insert_rows(con, _normalize_bets_df(df))
        con.commit()


def update_bet_status(bet_id, status):
    with _db() as con:
        row = con.execute("SELECT Stake, Cuota FROM bets WHERE ID=?", (str(bet_id),)).fetchone()
        if row is None:
            return
        stake = float(row[0]) if row[0] is not None else 0.0
        odd = float(row[1]) if row[1] is not None else 1.0
        profit = stake * (odd - 1) if status == "Ganada" else (-stake if status == "Perdida" else 0.0)
        con.execute("UPDATE bets SET Estado=?, Ganancia=? WHERE ID=?", (status, profit, str(bet_id)))
        con.commit()


def update_bet_closing(bet_id, closing_odd):
    with _db() as con:
        con.execute("UPDATE bets SET Cuota_Cierre=? WHERE ID=?", (float(closing_odd), str(bet_id)))
        con.commit()


def delete_bet(bet_id):
    with _db() as con:
        con.execute("DELETE FROM bets WHERE ID=?", (str(bet_id),))
        con.commit()


def get_setting(key, default):
    with _db() as con:
        row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_setting(key, value):
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        con.commit()


# ======================================================
# 9. API DE CUOTAS (The Odds API)
# ======================================================
def _secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def _scrub(msg, api_key):
    msg = str(msg)
    if api_key:
        msg = msg.replace(api_key, "***")
    return re.sub(r"apiKey=[^&\s)]+", "apiKey=***", msg)


def _to_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def call_odds_api(sport_key, api_key):
    """Una llamada = 1 crédito. Cacheada 10 min (los errores NO se cachean porque lanzan excepción)."""
    r = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/",
        params={"regions": "eu", "markets": "h2h", "oddsFormat": "decimal", "apiKey": api_key},
        timeout=20,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return {
        "data": r.json(), "ts": datetime.now(),
        "used": _to_int(r.headers.get("x-requests-used")), "remaining": _to_int(r.headers.get("x-requests-remaining")),
    }


def parse_odds_item(item):
    """Cuota promedio (consenso, para la prob. de mercado) y MEJOR cuota por resultado entre todas las casas."""
    hn, an = item.get("home_team"), item.get("away_team")
    prices = {"H": [], "D": [], "A": []}
    best = {"H": (np.nan, ""), "D": (np.nan, ""), "A": (np.nan, "")}
    for book in item.get("bookmakers") or []:
        for m in book.get("markets") or []:
            if m.get("key") != "h2h":
                continue
            for o in m.get("outcomes") or []:
                pr = o.get("price")
                if pr is None or pr <= 1.01:
                    continue
                nm = o.get("name")
                k = "H" if nm == hn else ("A" if nm == an else "D")
                prices[k].append(float(pr))
                if np.isnan(best[k][0]) or pr > best[k][0]:
                    best[k] = (float(pr), book.get("title", ""))
    if any(len(v) == 0 for v in prices.values()):
        return None
    return {
        "avg": tuple(float(np.mean(prices[k])) for k in "HDA"),
        "best": tuple(best[k][0] for k in "HDA"),
        "books": tuple(best[k][1] for k in "HDA"),
        "n_books": int(max(len(v) for v in prices.values())),
    }


# ======================================================
# 10. GRÁFICOS
# ======================================================
def plot_score_heatmap(probs, home_team, away_team):
    limit = 6
    cut = probs[:limit, :limit]
    fig = go.Figure(go.Heatmap(
        z=cut, x=[f"{away_team} {i}" for i in range(limit)], y=[f"{home_team} {i}" for i in range(limit)],
        colorscale="Viridis", text=np.round(cut * 100, 1), texttemplate="%{text}%", hoverongaps=False,
    ))
    fig.update_layout(title="🔥 Probabilidad de marcador exacto", xaxis_title=f"Goles {away_team}",
                      yaxis_title=f"Goles {home_team}", height=450, margin=dict(l=40, r=40, t=40, b=40))
    return fig


def calibration_figure(P_dict, y, bins=10, min_n=15):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                             line=dict(dash="dash", color="#888")))
    Y = np.eye(3)[y].ravel()
    edges = np.linspace(0, 1, bins + 1)
    for name, P in P_dict.items():
        p = np.asarray(P).ravel()
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        xs, ys, ns = [], [], []
        for b in range(bins):
            m = idx == b
            if m.sum() >= min_n:
                xs.append(float(p[m].mean()))
                ys.append(float(Y[m].mean()))
                ns.append(int(m.sum()))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name, text=ns,
                                 hovertemplate="Predicho %{x:.2f}<br>Real %{y:.2f}<br>n=%{text}<extra></extra>"))
    fig.update_layout(title="Curva de calibración (todas las clases 1X2)", xaxis_title="Probabilidad predicha",
                      yaxis_title="Frecuencia observada", height=420, margin=dict(l=40, r=20, t=40, b=40),
                      template="plotly_dark")
    return fig


def equity_figure(series, title, ytitle):
    fig = go.Figure()
    for name, ys in series.items():
        fig.add_trace(go.Scatter(y=list(ys), mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="Apuesta #", yaxis_title=ytitle, height=380,
                      margin=dict(l=40, r=20, t=40, b=40), template="plotly_dark")
    return fig


def get_last_5(df, team):
    team = str(team).strip()
    mask = (df["home"] == team) | (df["away"] == team)
    l5 = df[mask].sort_values(by="date", ascending=False).head(5).copy()
    if l5.empty:
        return pd.DataFrame(columns=["Sede", "Rival", "Score", "Tiros a puerta"])
    is_home = l5["home"] == team
    l5["Rival"] = np.where(is_home, l5["away"], l5["home"])
    l5["Score"] = l5["home_goals"].astype(int).astype(str) + "-" + l5["away_goals"].astype(int).astype(str)
    l5["Tiros a puerta"] = pd.Series(np.where(is_home, l5["sot_h"], l5["sot_a"]), index=l5.index).astype("Float64")
    l5["Sede"] = np.where(is_home, "🏠", "✈️")
    return l5[["Sede", "Rival", "Score", "Tiros a puerta"]]

# ======================================================
# 11. BUNDLES CACHEADOS (liga, dataset, ML) Y ESCÁNER
# ======================================================
ss = st.session_state


@st.cache_resource(ttl=3600, show_spinner=False)
def get_league_bundle(code, xi, reg):
    """Histórico + contexto pre-partido + Dixon-Coles ajustado con TODO el histórico (para predecir)."""
    df, msgs = fetch_live_soccer_data(code, N_SEASONS)
    if df.empty or len(df) < 120:
        return {"ok": False, "msgs": msgs}
    df_ctx, tracker = compute_context(df)
    model = fit_dixon_coles(df_ctx, ref_date=df_ctx["date"].max(), xi=xi, reg=reg)
    if model is None:
        return {"ok": False, "msgs": msgs + ["Datos insuficientes para ajustar Dixon-Coles"]}
    recent = df_ctx[df_ctx["date"] >= df_ctx["date"].max() - pd.Timedelta(days=400)]
    teams = sorted((set(recent["home"]) | set(recent["away"])) & set(model["idx"]))
    return {"ok": True, "df": df_ctx, "tracker": tracker, "model": model, "teams": teams, "msgs": msgs}


@st.cache_resource(ttl=3600, show_spinner=False)
def get_dataset(code, xi, reg):
    b = get_league_bundle(code, xi, reg)
    if not b["ok"]:
        return pd.DataFrame()
    return build_dataset(b["df"], xi, reg, league=code)


@st.cache_resource(ttl=3600, show_spinner=False)
def get_ml_bundle(kind, scope, code, xi, reg):
    """Modelo ML final + dataset usado. scope='pooled' entrena con las 7 ligas juntas."""
    codes = list(LEAGUES) if scope == "pooled" else [code]
    parts = [ds for ds in (get_dataset(c, xi, reg) for c in codes) if not ds.empty]
    if not parts:
        return None
    ds = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    if len(ds) < 300:
        return None
    model = MLModel(kind).fit(ds[FEATURE_NAMES].to_numpy(float), ds["y"].to_numpy(int))
    return {"model": model, "ds": ds, "codes": codes}


def get_ml_model(P, code):
    with st.spinner("Entrenando modelo ML (solo la primera vez; 'pooled' puede tardar ~30 s)..."):
        mlb = get_ml_bundle(P["ml_kind"], P["ml_scope"], code, P["xi"], P["reg"])
    return mlb["model"] if mlb else None


def lookup_scanner_odds(code, home, away):
    """Cuotas del escáner para un partido futuro (ignora partidos ya iniciados: sus cuotas son en vivo)."""
    stored = ss.market_storage.get(code)
    if not stored:
        return None
    now = pd.Timestamp.now(tz="UTC")
    for item in stored["data"]:
        t = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(t) or t <= now:
            continue
        if match_team(item.get("home_team", ""), [home]) and match_team(item.get("away_team", ""), [away]):
            o = parse_odds_item(item)
            if o:
                return o
    return None


def market_probs_for(code, home, away):
    found = lookup_scanner_odds(code, home, away)
    if found:
        return odds_to_probs(*found["avg"]), f"consenso de {found['n_books']} casas (escáner)"
    oi = ss.odds_inputs.get(f"{code}|{home}|{away}")
    if oi:
        return odds_to_probs(oi["oh"], oi["od"], oi["oa"]), "cuotas capturadas en 💰 Valor"
    return None, ""


def predict_fixture(code, b, ml, home, away, mdate, odds, P, hora):
    dc = predict_dc(b["model"], home, away)
    mk = odds_to_probs(*odds["avg"])
    p_dc = np.array([dc["ph"], dc["pd"], dc["pa"]])
    if ml is not None:
        x = make_features(mk, dc, b["tracker"].features(home, away, mdate))
        p_ml = ml.predict_proba(x.reshape(1, -1))[0]
    else:
        p_ml = p_dc
    adj = blend_probs(p_ml, mk, P["w_model"])
    tn = b["model"]["team_n"]
    return {
        "league": LEAGUES[code], "hora": hora, "partido": f"{home} vs {away}", "home": home, "away": away,
        "odd_h": odds["best"][0], "odd_d": odds["best"][1], "odd_a": odds["best"][2],
        "book_h": odds["books"][0], "book_d": odds["books"][1], "book_a": odds["books"][2],
        "mk_h": mk[0], "mk_d": mk[1], "mk_a": mk[2], "dc_h": p_dc[0], "dc_d": p_dc[1], "dc_a": p_dc[2],
        "ml_h": p_ml[0], "ml_d": p_ml[1], "ml_a": p_ml[2], "adj_h": adj[0], "adj_d": adj[1], "adj_a": adj[2],
        "n_books": odds["n_books"], "aviso": "⚠️ pocos partidos" if min(tn.get(home, 0), tn.get(away, 0)) < 20 else "",
    }


def scan_league(code, items, P, ml):
    """Predice todos los partidos PRE-PARTIDO del horizonte. Devuelve un DataFrame 'crudo':
    el pick, el EV y el stake se calculan al mostrar (así reaccionan a los sliders sin recalcular)."""
    b = get_league_bundle(code, P["xi"], P["reg"])
    if not b["ok"]:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for item in items:
        t = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(t):
            continue
        hours = (t - now).total_seconds() / 3600
        if hours <= 0 or hours > P["horizon_h"]:
            continue  # partidos iniciados traen cuotas en vivo -> EV falso
        h = match_team(item.get("home_team", ""), b["teams"])
        a = match_team(item.get("away_team", ""), b["teams"])
        if not h:
            log_unmatched(code, item.get("home_team", ""))
        if not a:
            log_unmatched(code, item.get("away_team", ""))
        if not h or not a:
            continue
        odds = parse_odds_item(item)
        if odds is None:
            continue
        rows.append(predict_fixture(code, b, ml, h, a, t.tz_localize(None), odds, P, t.strftime("%d/%m %H:%M")))
    return pd.DataFrame(rows)


def decorate_scan(raw, P, bank):
    out = []
    for r in raw.to_dict("records"):
        adj = [r["adj_h"], r["adj_d"], r["adj_a"]]
        odds = [r["odd_h"], r["odd_d"], r["odd_a"]]
        pk = pick_from_probs(adj, odds, P)
        edges = prob_edges(adj, odds)
        if pk:
            k, best_ev, kp = pk
            pick = [f"Gana {r['home']}", "Empate", f"Gana {r['away']}"][k]
            odd, book, stake, edge_pp = odds[k], r[f"book_{'hda'[k]}"], bank * kp / 100.0, float(edges[k] * 100)
        else:
            best_ev = float(np.max(np.array(adj) * np.array(odds) - 1.0))
            pick, odd, book, kp, stake, edge_pp = "No Bet", np.nan, "", 0.0, 0.0, float(np.max(edges) * 100)
        out.append({
            "Liga": r["league"], "Hora (UTC)": r["hora"], "Partido": r["partido"],
            "Mejores cuotas": f"H:{odds[0]:.2f} D:{odds[1]:.2f} A:{odds[2]:.2f}",
            "Mercado": f"{r['mk_h']:.2f}|{r['mk_d']:.2f}|{r['mk_a']:.2f}",
            "DC": f"{r['dc_h']:.2f}|{r['dc_d']:.2f}|{r['dc_a']:.2f}",
            "ML": f"{r['ml_h']:.2f}|{r['ml_d']:.2f}|{r['ml_a']:.2f}",
            "Ajustada": f"{adj[0]:.2f}|{adj[1]:.2f}|{adj[2]:.2f}",
            "Ventaja pp": edge_pp, "Mejor EV": best_ev, "Pick": pick, "Cuota": odd, "Casa": book, "Kelly %": kp, "Stake $": stake, "Aviso": r["aviso"],
        })
    return pd.DataFrame(out)


def safest_picks(raw, P, bank):
    """Por partido, el resultado 1X2 con MAYOR probabilidad ajustada (el más 'seguro'), con su mejor cuota.
    Seguro no es rentable: se marca si además tiene ventaja sobre la cuota (valor)."""
    out = []
    for r in raw.to_dict("records"):
        adj = np.array([r["adj_h"], r["adj_d"], r["adj_a"]])
        mk = np.array([r["mk_h"], r["mk_d"], r["mk_a"]])
        odds = np.array([r["odd_h"], r["odd_d"], r["odd_a"]])
        k = int(np.argmax(adj))
        edge = float(prob_edges(adj, odds)[k])
        value = edge > 0 and edge >= P["edge_min"]
        kp = kelly_pct(adj[k], odds[k], P["kelly_frac"], P["cap_pct"], P["edge_min"]) if value else 0.0
        out.append({
            "Liga": r["league"], "Hora (UTC)": r["hora"], "Partido": r["partido"],
            "Pick": [f"Gana {r['home']}", "Empate", f"Gana {r['away']}"][k],
            "Prob. ajustada %": adj[k] * 100, "Prob. mercado %": mk[k] * 100, "Mejor cuota": odds[k],
            "Casa": r[f"book_{'hda'[k]}"], "Ventaja pp": edge * 100, "EV": adj[k] * odds[k] - 1.0,
            "¿Valor?": "✅" if value else "—", "Kelly %": kp, "Stake $": bank * kp / 100.0, "Aviso": r["aviso"],
        })
    return pd.DataFrame(out)


def render_scan_table(raw, P, bank, key):
    dec = decorate_scan(raw, P, bank)
    only = st.checkbox("Mostrar solo picks (ventaja ≥ mínima y cuota ≤ máxima)", value=False, key=f"only_{key}")
    if only:
        dec = dec[dec["Pick"] != "No Bet"]
    dec = dec.sort_values("Ventaja pp", ascending=False).reset_index(drop=True)
    st.dataframe(
        dec.style.format({"Ventaja pp": "{:+.1f}", "Mejor EV": "{:.3f}", "Cuota": "{:.2f}", "Kelly %": "{:.2f}", "Stake $": "{:.2f}"}, na_rep="—"),
        width="stretch", hide_index=True,
    )
    return dec

# ======================================================
# 12. ESTADO, ESTILOS Y SIDEBAR
# ======================================================
ss.setdefault("ticket", [])
ss.setdefault("api_key", "")  # solo la key tecleada; la de secrets/entorno nunca se envía al navegador
ss.setdefault("api_usage", {"used": None, "remaining": None})
ss.setdefault("market_storage", {})
ss.setdefault("odds_inputs", {})
ss.setdefault("results", {})
ss.setdefault("unmatched", set())
init_db()

st.markdown("""
<style>
    div[data-testid="stMetric"] { background-color: #262730; border: 1px solid #464b5c; padding: 15px; border-radius: 10px; }
    .ticket-box { background-color: #1e1e1e; border: 1px solid #ffd700; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

if ss.get("flash"):
    st.success(ss.pop("flash"))

TREE_LABEL = "XGBoost + calibración" if HAS_XGB else "RandomForest + calibración (XGBoost no instalado)"

with st.sidebar:
    st.header("⚙️ Configuración")
    cb1, cb2 = st.columns(2)
    if cb1.button("🔄 Datos", key="update_btn", help="Vuelve a descargar los históricos y reajusta los modelos"):
        fetch_live_soccer_data.clear()
        get_league_bundle.clear()
        get_dataset.clear()
        get_ml_bundle.clear()
        ss.results = {}
        st.rerun()
    if cb2.button("🧹 Odds", key="clear_odds_btn", help="Borra las cuotas del escáner guardadas en memoria"):
        ss.market_storage = {}
        call_odds_api.clear()
        st.rerun()

    code = st.selectbox("Liga", list(LEAGUES), format_func=lambda x: LEAGUES[x], key="league_code")

    with st.expander("🧠 Modelo y bankroll", expanded=False):
        xi = st.slider("Decaimiento temporal ξ (por día)", 0.000, 0.012, 0.004, 0.001, format="%.3f",
                       help="Mayor ξ = el modelo olvida más rápido los partidos antiguos. Afínalo en 🧪 Laboratorio.")
        reg = st.slider("Regularización (shrinkage)", 0.0, 12.0, 3.0, 0.5,
                        help="Encoge ataque/defensa hacia el equipo promedio; ayuda con equipos de pocos partidos.")
        ml_kind = st.selectbox("Modelo ML", ["offset", "trees"], index=0,
                               format_func=lambda k: "Corrección regularizada sobre el mercado (recomendada)" if k == "offset" else TREE_LABEL,
                               help="'Sobre el mercado' parte de las cuotas y solo se aparta con evidencia: menos ruido, sobre todo en el empate.")
        ml_scope = st.selectbox("Entrenamiento del ML", ["pooled", "league"], index=0,
                                format_func=lambda k: "Las 7 ligas juntas (más datos)" if k == "pooled" else "Solo la liga seleccionada")
        w_model = st.slider("Peso del modelo vs mercado (%)", 0, 100, 50, 5,
                            help="Probabilidad ajustada = peso·modelo + (1-peso)·mercado. Más peso al mercado = apuestas más prudentes. "
                                 "Si desactivas el ML y usas solo Dixon-Coles, conviene un peso bajo.") / 100
        kelly_frac = st.slider("Fracción de Kelly", 0.05, 1.0, 0.25, 0.05)
        cap_pct = st.slider("Tope por apuesta (% del banco)", 0.5, 10.0, 3.0, 0.5)
        edge_min = st.slider("Ventaja mínima (puntos de probabilidad)", 0.0, 6.0, 2.0, 0.5,
                             help="Prob. ajustada − 1/cuota. Se usa en lugar de un EV mínimo porque el EV crece con la cuota y "
                                  "favorecería empates y cuotas largas, donde el ruido del modelo pesa más.") / 100
        allowed_lbl = st.multiselect("Resultados 1X2 permitidos", ["Local", "Empate", "Visita"], default=["Local", "Visita"],
                                     help="El empate empieza desactivado: es casi impredecible y el modelo tiende a inventar ventaja en él. "
                                          "Actívalo solo si en 🧪 Laboratorio → «Por tipo de apuesta» sus aciertos reales superan a los esperados.")
        allowed = [["Local", "Empate", "Visita"].index(a) for a in allowed_lbl] or [0, 1, 2]
        odd_max = st.slider("Cuota máxima", 2.0, 15.0, 8.0, 0.5, help="Evita cuotas largas, donde el error del modelo pesa más.")

    P = dict(xi=xi, reg=reg, ml_kind=ml_kind, ml_scope=ml_scope, w_model=w_model, kelly_frac=kelly_frac,
             cap_pct=cap_pct, edge_min=edge_min, allowed=allowed, odd_max=odd_max, horizon_h=168)

    with st.spinner("Cargando datos y ajustando Dixon-Coles..."):
        bundle = get_league_bundle(code, xi, reg)
    if not bundle["ok"]:
        st.error("Error cargando datos.")
        for m in bundle["msgs"]:
            st.caption(m)
        st.stop()

    df, model, teams = bundle["df"], bundle["model"], bundle["teams"]
    st.success(f"✅ {len(df)} partidos cargados")
    if bundle["msgs"]:
        with st.expander("ℹ️ Estado de la descarga"):
            for m in bundle["msgs"]:
                st.caption(m)

    st.divider()
    st.markdown("### 🗓️ Estado de la liga")
    st.write(f"**Última actualización:** {df['date'].max().strftime('%d/%m/%Y')}")
    with st.expander("🔎 Ver últimos 5 partidos globales"):
        latest = df.sort_values("date", ascending=False).head(5).copy()
        latest["Score"] = latest["home_goals"].astype(int).astype(str) + "-" + latest["away_goals"].astype(int).astype(str)
        latest["date"] = latest["date"].dt.strftime("%d/%m")
        st.dataframe(latest[["date", "home", "Score", "away"]], hide_index=True, width="stretch")

    st.divider()
    if "bank" not in ss:  # el banco se guarda en la BD para que no vuelva a 1000 al recargar la página
        ss.bank = float(get_setting("bank", 1000.0))
    bank = st.number_input("💰 Tu banco ($)", min_value=0.0, step=50.0, key="bank",
                           on_change=lambda: set_setting("bank", ss.bank))
    if ss.api_usage.get("remaining") is not None:
        st.metric("Créditos API restantes", ss.api_usage["remaining"])
    if ss.ticket:
        st.divider()
        st.markdown(f"**Ticket:** {len(ss.ticket)} selecciones")
        if st.button("🗑️ Limpiar ticket", key="clear_ticket_sidebar"):
            ss.ticket = []
            st.rerun()
    st.caption("Herramienta de análisis: el mercado suele ser muy eficiente y las apuestas implican riesgo de pérdida.")

st.title(f"Análisis predictivo de resultados de fútbol: {LEAGUES[code]}")

c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams, key="sel_home")
away = c2.selectbox("Visitante", [t for t in teams if t != home], key="sel_away")
dc = predict_dc(model, home, away)
match_id = f"{code}|{home}|{away}"

t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
    ["📊 Análisis", "💰 Valor", "📜 Historial", "💎 Escáner", "🧪 Laboratorio", "📈 Rendimiento (Risk)", "🤖 ML 1X2", "🌎 Multi-Liga"]
)


# ======================================================
# 13. TAB 2 (como fragmento: teclear cuotas no re-ejecuta toda la app)
# ======================================================
@fragment
def render_value_tab(code, home, away, bank, P):
    b = get_league_bundle(code, P["xi"], P["reg"])
    dcp = predict_dc(b["model"], home, away)
    mid = f"{code}|{home}|{away}"
    col_a, col_t = st.columns([2, 1])

    with col_a:
        st.markdown("### 🏦 Comparador inteligente")
        found = lookup_scanner_odds(code, home, away)
        if found:
            d_h, d_d, d_a = found["best"]
            st.success(f"✅ Mejores cuotas cargadas del escáner ({found['n_books']} casas). La prob. de mercado usa el consenso.")
        else:
            d_h, d_d, d_a = 2.0, 3.2, 3.5
            st.info("ℹ️ Cuotas por defecto (no hay datos del escáner para este partido). Captura las tuyas.")
        sfx = f"{mid}_{d_h:.2f}_{d_d:.2f}_{d_a:.2f}"
        co1, co2, co3 = st.columns(3)
        oh = co1.number_input("Cuota Local", 1.01, 100.0, float(d_h), key=f"oh_{sfx}")
        od = co2.number_input("Cuota Empate", 1.01, 100.0, float(d_d), key=f"od_{sfx}")
        oa = co3.number_input("Cuota Visita", 1.01, 100.0, float(d_a), key=f"oa_{sfx}")
        st.caption("👇 Cuotas manuales para mercados alternativos:")
        cx1, cx2 = st.columns(2)
        odd_o25 = cx1.number_input("Cuota Over 2.5", 1.01, 100.0, 1.90, key=f"o25_{mid}")
        odd_btts = cx2.number_input("Cuota BTTS (Sí)", 1.01, 100.0, 1.90, key=f"btts_{mid}")
        ss.odds_inputs[mid] = {"oh": float(oh), "od": float(od), "oa": float(oa), "o25": float(odd_o25), "btts": float(odd_btts)}

        mk = odds_to_probs(*found["avg"]) if found else odds_to_probs(oh, od, oa)
        st.caption(f"Margen implícito de las cuotas 1X2 ingresadas: {overround(oh, od, oa) * 100:.1f}%")

        use_ml = st.checkbox("Incluir modelo ML en la probabilidad (la primera vez entrena)", value=False, key="value_use_ml")
        p_dc = np.array([dcp["ph"], dcp["pd"], dcp["pa"]])
        p_model, src = p_dc, "DC"
        if use_ml:
            ml = get_ml_model(P, code)
            if ml is not None:
                x = make_features(mk, dcp, b["tracker"].features(home, away, pd.Timestamp.now().normalize()))
                p_model, src = ml.predict_proba(x.reshape(1, -1))[0], "ML"
            else:
                st.warning("No se pudo entrenar el ML (historial insuficiente); se usa Dixon-Coles.")
        p_adj = blend_probs(p_model, mk, P["w_model"])
        odds = np.array([oh, od, oa])

        st.markdown("#### 🧠 Valor y Kelly (1X2)")
        tbl = pd.DataFrame({
            "Selección": [f"Gana {home}", "Empate", f"Gana {away}"], "Cuota": odds, "P. mercado": mk,
            f"P. {src}": p_model, "P. ajustada": p_adj, "Ventaja (pp)": prob_edges(p_adj, odds) * 100,
            "EV": p_adj * odds - 1.0, "EV (modelo puro)": p_model * odds - 1.0,
            "Kelly %": [kelly_pct(p, o, P["kelly_frac"], P["cap_pct"], P["edge_min"]) for p, o in zip(p_adj, odds)],
        })
        st.dataframe(tbl.style.format({
            "Cuota": "{:.2f}", "P. mercado": "{:.3f}", f"P. {src}": "{:.3f}", "P. ajustada": "{:.3f}", "Ventaja (pp)": "{:+.1f}",
            "EV": "{:+.3f}", "EV (modelo puro)": "{:+.3f}", "Kelly %": "{:.2f}",
        }), hide_index=True, width="stretch")
        st.caption(f"P. ajustada = {P['w_model'] * 100:.0f}% modelo + {(1 - P['w_model']) * 100:.0f}% mercado. "
                   f"Se apuesta solo con ventaja ≥ {P['edge_min'] * 100:.1f} pp (prob. ajustada − 1/cuota) y cuota ≤ {P['odd_max']:.1f}.")
        pk = pick_from_probs(p_adj, odds, P)
        if pk:
            k, ev_k, kp = pk
            sel_name = [f"Gana {home}", "Empate", f"Gana {away}"][k]
            st.success(f"💎 **Recomendación:** {sel_name} @ {odds[k]:.2f} | EV {ev_k * 100:+.1f}% | "
                       f"Stake: ${bank * kp / 100:.2f} ({kp:.2f}% del banco)")
        else:
            st.warning("📉 Kelly sugiere: **No apostar** (sin ventaja suficiente tras ajustar por el mercado).")

        st.markdown("#### ⚽ Mercados de goles")
        w = P["w_model"]
        rows, goal_adj = [], {}
        for lbl, key_, p_raw, o in (("Over 2.5 Goles", "O25", dcp["po25"], odd_o25), ("BTTS (Ambos anotan)", "BTTS", dcp["pbtts"], odd_btts)):
            p_a = w * p_raw + (1 - w) * (1.0 / o)
            goal_adj[lbl] = (key_, o, p_a)
            rows.append({"Mercado": lbl, "Cuota": o, "P. modelo": p_raw, "P. ajustada": p_a, "Ventaja (pp)": (p_a - 1.0 / o) * 100,
                         "EV": p_a * o - 1.0, "EV (modelo puro)": p_raw * o - 1.0,
                         "Kelly %": kelly_pct(p_a, o, P["kelly_frac"], P["cap_pct"], P["edge_min"])})
        st.dataframe(pd.DataFrame(rows).style.format({
            "Cuota": "{:.2f}", "P. modelo": "{:.3f}", "P. ajustada": "{:.3f}", "Ventaja (pp)": "{:+.1f}", "EV": "{:+.3f}",
            "EV (modelo puro)": "{:+.3f}", "Kelly %": "{:.2f}"}), hide_index=True, width="stretch")
        st.caption("Sin probabilidad de mercado para estos mercados: se usa 1/cuota como referencia (incluye el margen).")

        st.divider()
        st.markdown("### ➕ Agregar al ticket")
        options = {
            f"Gana {home}": ("H", oh, p_adj[0]), "Empate": ("D", od, p_adj[1]), f"Gana {away}": ("A", oa, p_adj[2]),
            **goal_adj,
        }
        with st.form(f"add_ticket_{mid}"):
            sel_pick = st.selectbox("Selección", list(options))
            submitted = st.form_submit_button("Añadir selección")
        if submitted:
            key_, odd_, prob_ = options[sel_pick]
            if any(i["match_id"] == mid for i in ss.ticket):
                st.warning("Ya hay una selección de este partido en el ticket: están correlacionadas y multiplicar "
                           "sus probabilidades sobreestimaría el valor.")
            else:
                ss.ticket.append({"match_id": mid, "match": f"{home} vs {away}", "pick": sel_pick, "key": key_,
                                  "odd": float(odd_), "prob": float(prob_), "league": LEAGUES[code]})
                ss.flash = "Selección añadida al ticket"
                st.rerun()

    with col_t:
        st.markdown("### 🎫 Ticket")
        if not ss.ticket:
            st.info("Vacío")
        else:
            tot_odd, tot_prob = 1.0, 1.0
            for idx, item in enumerate(list(ss.ticket)):
                st.markdown(
                    f"<div class='ticket-box'><small>{html_escape(item['league'])}</small><br>"
                    f"<strong>{html_escape(item['match'])}</strong><br>{html_escape(item['pick'])} @ {item['odd']:.2f}</div>",
                    unsafe_allow_html=True)
                if st.button("❌", key=f"del_{idx}"):
                    ss.ticket.pop(idx)
                    st.rerun()
                tot_odd *= item["odd"]
                tot_prob *= item["prob"]
            st.divider()
            ev_t = tot_prob * tot_odd - 1.0
            m1, m2 = st.columns(2)
            m1.metric("Cuota total", f"{tot_odd:.2f}")
            m2.metric("EV combinado", f"{ev_t * 100:+.1f}%")
            if len(ss.ticket) > 1:
                st.caption("Parlay: se asume independencia entre partidos distintos.")
            kp_t = kelly_pct(tot_prob, tot_odd, P["kelly_frac"], P["cap_pct"], 0.0)
            if ev_t > 0:
                st.success(f"Kelly sugiere ${bank * kp_t / 100:.2f} ({kp_t:.2f}% del banco)")
            else:
                st.warning("EV combinado ≤ 0: Kelly sugiere no apostar.")
            stake_parlay = st.number_input("Stake ($)", 1.0, 5000.0, 50.0, key="ticket_stake")
            st.info(f"Ganancia potencial: ${stake_parlay * tot_odd - stake_parlay:.2f}")
            if st.button("💾 Guardar", key="save_ticket"):
                leagues_used = " + ".join(dict.fromkeys(i["league"] for i in ss.ticket))
                save_bet({
                    "ID": pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f"),
                    "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "Tipo": "Simple" if len(ss.ticket) == 1 else "Parlay",
                    "Liga": leagues_used,
                    "Partido": ss.ticket[0]["match"] if len(ss.ticket) == 1 else f"Combinada ({len(ss.ticket)})",
                    "Pick": " + ".join(i["pick"] for i in ss.ticket),
                    "Cuota": round(tot_odd, 2), "Stake": float(stake_parlay), "Prob": round(tot_prob, 4),
                    "EV": round(ev_t, 4), "Estado": "Pendiente", "Ganancia": 0.0,
                })
                ss.ticket = []
                ss.flash = "💾 Apuesta guardada en el historial"
                st.rerun()


# ======================================================
# 14. TABS 1 A 4
# ======================================================
# --- TAB 1: ANÁLISIS ---
with t1:
    low = [t for t in (home, away) if model["team_n"].get(t, 0) < 20]
    if low:
        st.warning(f"⚠️ Pocos partidos en el histórico para: {', '.join(low)}. Sus fuerzas están encogidas hacia el promedio.")
    st.markdown("### 🥅 Goles esperados (λ del modelo)")
    a, b_, c = st.columns(3)
    a.metric(home, f"{dc['lam']:.2f}")
    b_.metric("Total", f"{dc['lam'] + dc['mu']:.2f}")
    c.metric(away, f"{dc['mu']:.2f}")
    st.caption(f"Dixon-Coles por máxima verosimilitud · ventaja local γ={model['gam']:.2f} · ρ={model['rho']:.3f} · "
               f"{model['n']} partidos · ξ={xi:.3f} · regularización={reg:.1f}")

    st.divider()
    st.markdown("### 🏁 ¿Quién gana? (1X2) + mercados de goles")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"🏠 {home}", f"{dc['ph'] * 100:.1f}%")
    m2.metric("🤝 Empate", f"{dc['pd'] * 100:.1f}%")
    m3.metric(f"✈️ {away}", f"{dc['pa'] * 100:.1f}%")
    best_1x2 = max(dc["ph"], dc["pd"], dc["pa"])
    pick_1x2 = f"Gana {home}" if best_1x2 == dc["ph"] else (f"Gana {away}" if best_1x2 == dc["pa"] else "Empate")
    st.info(f"**Pick modelo (1X2):** {pick_1x2}  |  **Cuotas justas (DC, sin margen):** "
            f"H={safe_fair_odds(dc['ph']):.2f}  D={safe_fair_odds(dc['pd']):.2f}  A={safe_fair_odds(dc['pa']):.2f}")
    g1, g2, g3 = st.columns(3)
    g1.metric("Over 1.5", f"{dc['po15'] * 100:.1f}%")
    g2.metric("Over 2.5", f"{dc['po25'] * 100:.1f}%")
    g3.metric("BTTS (Ambos anotan)", f"{dc['pbtts'] * 100:.1f}%")
    st.plotly_chart(plot_score_heatmap(dc["probs"], home, away), width="stretch")

    st.markdown("### ⭐ Top 3 marcadores más probables")
    top_df = pd.DataFrame([{"Marcador": s, "Prob (%)": p * 100} for s, p in dc["top"]])
    st.dataframe(top_df.style.format({"Prob (%)": "{:.2f}"}), width="stretch", hide_index=True)

    st.divider()
    st.markdown("### 📉 Últimos 5")
    cf1, cf2 = st.columns(2)
    with cf1:
        st.write(f"**{home}**")
        st.dataframe(get_last_5(df, home), width="stretch", hide_index=True)
    with cf2:
        st.write(f"**{away}**")
        st.dataframe(get_last_5(df, away), width="stretch", hide_index=True)

# --- TAB 2: VALOR + TICKET ---
with t2:
    render_value_tab(code, home, away, bank, P)

# --- TAB 3: HISTORIAL ---
with t3:
    st.markdown("### 📜 Historial de apuestas")
    db = load_bets()
    if db.empty:
        st.warning("Aún no hay historial.")
    else:
        st.dataframe(db, width="stretch", hide_index=True)
        st.divider()
        st.markdown("### 🛠️ Administrar apuestas")
        opts = {f"{r.ID} | {r.Fecha} | {r.Partido} | {r.Pick}": r.ID for r in db.itertuples()}
        sel = st.selectbox("Selecciona la apuesta a editar/borrar:", ["-- Seleccionar --"] + list(opts), key="hist_sel")
        if sel != "-- Seleccionar --":
            bet_id = opts[sel]
            fila = db[db["ID"] == bet_id].iloc[0]
            st.info(f"**Seleccionado:** {fila['Partido']} - {fila['Pick']} (Cuota: {fila['Cuota']})")
            ce1, ce2, ce3 = st.columns(3)
            with ce1:
                estados = ["Pendiente", "Ganada", "Perdida", "Push"]
                nuevo = st.selectbox("Estado:", estados, index=estados.index(fila["Estado"]) if fila["Estado"] in estados else 0,
                                     key=f"estado_{bet_id}")
                if st.button("💾 Actualizar estado", key=f"upd_{bet_id}"):
                    update_bet_status(bet_id, nuevo)
                    ss.flash = f"Apuesta {bet_id} actualizada a {nuevo}"
                    st.rerun()
            with ce2:
                cierre_def = float(fila["Cuota_Cierre"]) if pd.notna(fila["Cuota_Cierre"]) else float(fila["Cuota"])
                cierre = st.number_input("Cuota de cierre (para CLV)", 1.01, 100.0, cierre_def, key=f"cierre_{bet_id}")
                if st.button("📌 Guardar cuota de cierre", key=f"clv_{bet_id}"):
                    update_bet_closing(bet_id, cierre)
                    ss.flash = "Cuota de cierre guardada"
                    st.rerun()
            with ce3:
                st.write("Zona de peligro")
                confirm = st.checkbox("Confirmo eliminar", key=f"confirm_{bet_id}")
                if st.button("🗑️ Eliminar apuesta", type="primary", key=f"del_bet_{bet_id}", disabled=not confirm):
                    delete_bet(bet_id)
                    ss.flash = f"Apuesta {bet_id} eliminada"
                    st.rerun()

    st.divider()
    st.markdown("### 💾 Respaldo")
    st.caption("Si despliegas en la nube, el disco es efímero: descarga respaldos periódicos y restáuralos cuando haga falta.")
    cbk1, cbk2 = st.columns(2)
    cbk1.download_button("📥 Descargar respaldo (CSV)", data=load_bets().to_csv(index=False).encode("utf-8"),
                         file_name="respaldo_apuestas.csv", mime="text/csv", key="dl_backup")
    up = cbk2.file_uploader("Restaurar desde respaldo CSV", type="csv", key="restore_up")
    if up is not None and cbk2.button("Importar respaldo", key="restore_btn"):
        try:
            import_bets_df(pd.read_csv(up))
            ss.flash = "Respaldo importado (las apuestas existentes no se duplican)."
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo importar: {e}")

# --- TAB 4: ESCÁNER ---
with t4:
    st.markdown("## 💎 Escáner de valor")
    server_key = os.environ.get("ODDS_API_KEY") or _secret("ODDS_API_KEY") or ""
    api_in = st.text_input("🔑 API Key (opcional si está en st.secrets['ODDS_API_KEY'] o en la variable ODDS_API_KEY):",
                           value=ss.api_key, type="password", key="api_key_input",
                           placeholder="Usando la key configurada en el servidor" if server_key else "")
    if api_in != ss.api_key:
        ss.api_key = api_in
    api_key = ss.api_key or server_key

    if not api_key:
        st.info("Pon tu API key para usar el escáner.")
    else:
        stored = ss.market_storage.get(code)
        if stored:
            st.info(f"📂 Datos en memoria: {len(stored['data'])} partidos. Descargados a las {stored['timestamp'].strftime('%H:%M:%S')}.")
        else:
            st.warning("⚠️ Sin datos descargados.")
        cdl1, cdl2 = st.columns([2, 1])
        force = cdl2.checkbox("Ignorar caché de 10 min", key="force_api")
        if cdl1.button("⬇️ Descargar/Actualizar odds (1 crédito)", key="dl_odds"):
            try:
                if force:
                    call_odds_api.clear()
                with st.spinner("Conectando..."):
                    resp = call_odds_api(API_LEAGUE_MAP[code], api_key)
                ss.market_storage[code] = {"timestamp": resp["ts"], "data": resp["data"]}
                if resp["remaining"] is not None:
                    ss.api_usage = {"used": resp["used"], "remaining": resp["remaining"]}
                ss.flash = "✅ Odds descargadas"
                st.rerun()
            except Exception as e:
                st.error(f"Error API: {_scrub(e, api_key)}")

        st.divider()
        st.markdown("### 🎯 Pronósticos de la jornada")
        st.caption("Compara mercado, Dixon-Coles y ML sobre la MEJOR cuota entre casas. Solo partidos que aún no empiezan.")
        sc1, sc2 = st.columns(2)
        use_ml_scan = sc1.checkbox("Usar modelo ML (si no, solo Dixon-Coles)", value=True, key="scan_use_ml")
        horizon_days = sc2.slider("Ventana (días)", 1, 14, 7, key="scan_horizon")
        if st.button("📌 Generar pronósticos", key="gen_scan"):
            if not stored or not stored["data"]:
                st.warning("Primero descarga las odds de esta liga.")
            else:
                ml = get_ml_model(P, code) if use_ml_scan else None
                if use_ml_scan and ml is None:
                    st.warning("No se pudo entrenar el ML; se usa solo Dixon-Coles.")
                raw = scan_league(code, stored["data"], {**P, "horizon_h": horizon_days * 24}, ml)
                ss.results[f"scan_{code}"] = {"raw": raw, "ml": ml is not None, "ts": datetime.now()}

        res = ss.results.get(f"scan_{code}")
        if res is not None:
            if res["raw"].empty:
                st.info("No se encontraron partidos válidos (nombres, cuotas o ventana). Revisa «Equipos sin emparejar».")
            else:
                st.success(f"✅ {len(res['raw'])} partidos analizados ({'ML + Dixon-Coles' if res['ml'] else 'Dixon-Coles'}).")
                dec = render_scan_table(res["raw"], P, bank, key=f"scan_{code}")
                st.download_button("📥 Descargar jornada (CSV)", data=dec.to_csv(index=False).encode("utf-8"),
                                   file_name=f"jornada_{code}.csv", mime="text/csv", key=f"dl_scan_{code}")

    with st.expander("🧩 Equipos sin emparejar"):
        if ss.unmatched:
            st.caption("Estos nombres de la API no coinciden con el histórico (o el equipo no tiene historial reciente). "
                       "Agrégalos a TEAM_MAP: nombre de la API → nombre en football-data.")
            st.dataframe(pd.DataFrame(sorted(ss.unmatched), columns=["Liga", "Nombre en la API"]), hide_index=True,
                         width="stretch")
        else:
            st.write("Ninguno por ahora.")

# ======================================================
# 15. TABS 5 A 8
# ======================================================
# --- TAB 5: LABORATORIO ---
with t5:
    st.markdown("## 🧪 Laboratorio")

    st.markdown("### 🎛️ Afinar Dixon-Coles (LogLoss walk-forward)")
    st.caption("Reajusta el modelo por bloques de 14 días usando solo el pasado y mide el LogLoss sobre los últimos partidos. "
               "Menor es mejor; compáralo con el del mercado.")
    lc1, lc2, lc3 = st.columns(3)
    n_tune = lc1.slider("Partidos de prueba", 100, 600, 300, 50, key="tune_n")
    xi_grid = lc2.multiselect("ξ (decaimiento)", [0.0, 0.002, 0.004, 0.008, 0.012], default=[0.002, 0.004, 0.008], key="tune_xi")
    reg_grid = lc3.multiselect("Regularización", [0.5, 1.0, 3.0, 6.0, 10.0], default=[1.0, 3.0, 6.0], key="tune_reg")
    if st.button("▶️ Ejecutar búsqueda", key="tune_btn"):
        combos = [(x_, r_) for x_ in xi_grid for r_ in reg_grid]
        if not combos:
            st.warning("Elige al menos un valor de ξ y uno de regularización.")
        else:
            rows, prog = [], st.progress(0.0)
            for i, (x_, r_) in enumerate(combos, 1):
                out = dc_walkforward_logloss(df, x_, r_, n_test=n_tune)
                if out:
                    rows.append({"ξ": x_, "Regularización": r_, "LogLoss DC": out["dc"], "LogLoss mercado": out["mk"], "Partidos": out["n"]})
                prog.progress(i / len(combos))
            ss.results[f"tune_{code}"] = pd.DataFrame(rows)
    tune = ss.results.get(f"tune_{code}")
    if tune is not None:
        if tune.empty:
            st.warning("Historial insuficiente para la búsqueda.")
        else:
            tune = tune.sort_values("LogLoss DC").reset_index(drop=True)
            best_row = tune.iloc[0]
            st.success(f"Mejor combinación: ξ={best_row['ξ']:.3f} y regularización={best_row['Regularización']:.1f} "
                       f"(LogLoss {best_row['LogLoss DC']:.4f} vs mercado {best_row['LogLoss mercado']:.4f}). "
                       "Aplícala en la barra lateral → «Modelo y bankroll».")
            st.dataframe(tune.style.format({"ξ": "{:.3f}", "Regularización": "{:.1f}", "LogLoss DC": "{:.4f}", "LogLoss mercado": "{:.4f}"}),
                         hide_index=True, width="stretch")

    st.divider()
    st.markdown("### 📈 Backtest de la estrategia (EV con probabilidad ajustada)")
    st.caption("Simula lo que recomienda la app: apostar a la selección con mayor ventaja en probabilidad cuando ventaja ≥ mínima y "
               "cuota ≤ máxima, usando la cuota de mercado promedio de cada partido. Los parámetros salen de la barra lateral y "
               "se recalculan al instante.")
    bc1, bc2, bc3 = st.columns(3)
    n_bt = bc1.slider("Partidos de prueba", 200, 3000, 800, 100, key="bt_n")
    rt_bt = bc2.slider("Reentrenar ML cada (partidos)", 10, 100, 30, 10, key="bt_rt")
    src_bt = bc3.radio("Modelo a evaluar", ["ML", "Dixon-Coles"], horizontal=True, key="bt_src")
    lab_cfg = (ml_kind, ml_scope, code if ml_scope == "league" else "pooled", xi, reg, n_bt, rt_bt)
    if st.button("▶️ Ejecutar backtest", key="bt_btn"):
        with st.spinner("Construyendo dataset sin fugas y evaluando (la primera vez puede tardar)..."):
            mlb = get_ml_bundle(ml_kind, ml_scope, code, xi, reg)
            ev_bt = evaluate_walkforward(mlb["ds"], kind=ml_kind, n_test=n_bt, min_train=500, retrain_every=rt_bt) if mlb else None
        if ev_bt is None:
            st.warning("Historial insuficiente para el backtest (prueba con alcance 'Las 7 ligas juntas').")
        else:
            ss.results["lab_eval"] = {**ev_bt, "cfg": lab_cfg}
    ev_bt = ss.results.get("lab_eval")
    if ev_bt is not None and ev_bt["cfg"] != lab_cfg:
        st.info("Los resultados guardados son de otra liga o configuración del modelo: vuelve a ejecutar el backtest.")
        ev_bt = None
    if ev_bt is not None:
        P_model = ev_bt["P"]["ML"] if src_bt == "ML" else ev_bt["P"]["Dixon-Coles"]
        sim = simulate_strategy(ev_bt["meta"], P_model, ev_bt["P"]["Mercado"], ev_bt["y"], w_model, edge_min, odd_max, kelly_frac, cap_pct, allowed)
        st.caption(f"Muestra: {ev_bt['n']} partidos"
                   f"{' de varias ligas' if ev_bt['meta']['league'].nunique() > 1 else ''} · modelo: {src_bt}.")
        if sim["n_bets"] == 0:
            st.info("Con estos parámetros no habría habido apuestas (ventaja mínima o peso del modelo muy exigentes). Prueba bajar la ventaja mínima o subir el peso del modelo.")
        else:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Apuestas", f"{sim['n_bets']}")
            a2.metric("Aciertos", f"{sim['hit'] * 100:.0f}%")
            a3.metric("Profit", f"{sim['profit']:+.2f} U")
            a4.metric("Yield (ROI)", f"{sim['yield']:+.2f}%")
            lo, hi = sim["ci"]
            if lo <= 0 <= hi:
                st.warning(f"IC 95% del yield: [{lo:+.1f}%, {hi:+.1f}%] — incluye 0: **sin evidencia estadística de ventaja** con esta muestra.")
            elif lo > 0:
                st.success(f"IC 95% del yield: [{lo:+.1f}%, {hi:+.1f}%] — positivo en esta muestra; confírmalo con más temporadas y con CLV real.")
            else:
                st.error(f"IC 95% del yield: [{lo:+.1f}%, {hi:+.1f}%] — la estrategia pierde con estos parámetros.")
            st.caption(f"Máximo drawdown (stake plano): {sim['max_dd']:.2f} U · Banco simulado con Kelly "
                       f"(inicio {sim['bank0']:.0f}): {sim['kelly_curve'][-1]:.1f}")
            bp = sim["by_pick"]
            st.markdown("**Por tipo de apuesta** — ¿se concentra en un resultado? Compara aciertos esperados vs reales.")
            st.dataframe(bp.style.format({"Esperados": "{:.1f}", "Reales": "{:.0f}", "Profit": "{:+.2f}", "% apuestas": "{:.0f}", "Yield %": "{:+.1f}"}),
                         hide_index=True, width="stretch")
            top = bp.loc[bp["% apuestas"].idxmax()]
            if top["% apuestas"] > 55 and sim["n_bets"] >= 15:
                st.warning(f"«{top['Pick']}» concentra el {top['% apuestas']:.0f}% de las apuestas. Suele ser ruido del modelo (el empate es "
                           "casi impredecible y su cuota amplifica cualquier error), no una ventaja real. Si los aciertos reales no superan "
                           "a los esperados, excluye ese resultado en la barra lateral o sube la ventaja mínima.")
            cg1, cg2 = st.columns(2)
            cg1.plotly_chart(equity_figure({"Stake plano": sim["equity"]}, "Capital acumulado (unidades)", "Unidades"), width="stretch")
            cg2.plotly_chart(equity_figure({"Kelly fraccionado": sim["kelly_curve"]}, "Banco simulado", "Banco"), width="stretch")
            with st.expander("Ver apuestas simuladas"):
                st.dataframe(sim["bets"].style.format({"Cuota": "{:.2f}", "P ajust.": "{:.3f}", "EV": "{:.3f}", "P/L (U)": "{:+.2f}"}),
                             hide_index=True, width="stretch")

# --- TAB 6: RENDIMIENTO ---
with t6:
    st.markdown("## 📈 Rendimiento (Risk)")
    hist = load_bets()
    fin = hist[hist["Estado"].isin(["Ganada", "Perdida", "Push"])].copy()
    if fin.empty:
        st.info("No hay apuestas finalizadas.")
    else:
        fin = fin.sort_values("ID").reset_index(drop=True)
        turnover = fin.loc[fin["Estado"] != "Push", "Stake"].sum()
        tot_prof = fin["Ganancia"].sum()
        roi = (tot_prof / turnover * 100) if turnover > 0 else 0.0
        fin["Equity"] = fin["Ganancia"].cumsum()
        fin["Peak"] = fin["Equity"].cummax().clip(lower=0)  # el capital inicial (0) cuenta como pico
        fin["Drawdown"] = fin["Equity"] - fin["Peak"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Beneficio neto", f"${tot_prof:,.2f}")
        r2.metric("ROI", f"{roi:.2f}%")
        r3.metric("Max drawdown", f"${fin['Drawdown'].min():,.2f}")
        r4.metric("Apuestas", len(fin))

        decided = fin[fin["Estado"] != "Push"].dropna(subset=["Prob"])
        q1, q2, q3 = st.columns(3)
        if not decided.empty:
            won_flag = (decided["Estado"] == "Ganada").astype(float)
            q1.metric("Aciertos esperados vs reales", f"{decided['Prob'].sum():.1f} vs {int(won_flag.sum())}",
                      help="Suma de las probabilidades que guardaste vs apuestas ganadas. Si lo real queda muy por debajo, el modelo está sobreconfiado.")
            q2.metric("Brier de tus picks", f"{np.mean((decided['Prob'] - won_flag) ** 2):.3f}")
        clv_df = fin.dropna(subset=["Cuota_Cierre"])
        if not clv_df.empty:
            q3.metric("CLV promedio", f"{((clv_df['Cuota'] / clv_df['Cuota_Cierre'] - 1).mean()) * 100:+.2f}%",
                      help="Cuota tomada vs cuota de cierre. Un CLV positivo sostenido es la mejor señal de ventaja real.")
        else:
            q3.caption("Registra la cuota de cierre en 📜 Historial para medir CLV.")

        st.plotly_chart(equity_figure({"Capital acumulado": fin["Equity"]}, "Curva de capital ($)", "$"), width="stretch")

        def breakdown(frame, col, bins=None):
            f = frame.copy()
            if bins is not None:
                f[col] = pd.cut(f["EV"], bins=bins[0], labels=bins[1]).astype(object).fillna("sin EV")
            g = f[f["Estado"] != "Push"].groupby(col).agg(Apuestas=("ID", "count"), Stake=("Stake", "sum"), Profit=("Ganancia", "sum"))
            g["ROI %"] = np.where(g["Stake"] > 0, g["Profit"] / g["Stake"] * 100, 0.0)
            return g.reset_index()

        tabs_fmt = {"Stake": "{:.2f}", "Profit": "{:+.2f}", "ROI %": "{:+.1f}"}
        bk1, bk2, bk3 = st.columns(3)
        bk1.markdown("**Por liga**")
        bk1.dataframe(breakdown(fin, "Liga").style.format(tabs_fmt), hide_index=True, width="stretch")
        bk2.markdown("**Por tipo**")
        bk2.dataframe(breakdown(fin, "Tipo").style.format(tabs_fmt), hide_index=True, width="stretch")
        fin["Rango EV"] = ""
        bk3.markdown("**Por rango de EV**")
        bk3.dataframe(breakdown(fin, "Rango EV", bins=([-np.inf, 0.0, 0.05, 0.10, np.inf], ["≤0", "0–5%", "5–10%", ">10%"])).style.format(tabs_fmt),
                      hide_index=True, width="stretch")
        with st.expander("Ver apuestas finalizadas"):
            st.dataframe(fin.drop(columns=["Rango EV"]), width="stretch", hide_index=True)

# --- TAB 7: ML 1X2 ---
with t7:
    st.markdown("## 🤖 ML 1X2: mercado + Dixon-Coles + contexto")
    kind_lbl = "Corrección regularizada sobre el mercado" if ml_kind == "offset" else TREE_LABEL
    scope_lbl = "las 7 ligas juntas" if ml_scope == "pooled" else f"solo {LEAGUES[code]}"
    st.caption(f"Modelo: {kind_lbl} · entrenado con {scope_lbl}. Todas las features son pre-partido (sin fugas): "
               "cuotas sin margen, Dixon-Coles ajustado solo con el pasado, Elo, forma, días de descanso y tiros a puerta previos.")

    ec1, ec2, ec3 = st.columns(3)
    n_test_ml = ec1.slider("Partidos de prueba", 50, 1000, 300, 25, key="ml_n_test")
    min_train_ml = ec2.slider("Mínimo para entrenar", 200, 2000, 500, 50, key="ml_min_train")
    retrain_ml = ec3.slider("Reentrenar cada (partidos)", 10, 100, 30, 10, key="ml_retrain")
    ml_cfg = (ml_kind, ml_scope, code if ml_scope == "league" else "pooled", xi, reg, n_test_ml, min_train_ml, retrain_ml)
    if st.button("▶️ Evaluar (walk-forward) vs mercado", key="ml_eval_btn"):
        with st.spinner("Construyendo dataset sin fugas y evaluando (la primera vez puede tardar)..."):
            mlb = get_ml_bundle(ml_kind, ml_scope, code, xi, reg)
            ev_ml = evaluate_walkforward(mlb["ds"], kind=ml_kind, n_test=n_test_ml, min_train=min_train_ml, retrain_every=retrain_ml) if mlb else None
        if ev_ml is None:
            st.warning("No se pudo evaluar (historial insuficiente). Prueba con 'Las 7 ligas juntas' o menos partidos de prueba.")
        else:
            ss.results["ml_eval"] = {**ev_ml, "cfg": ml_cfg}
    ev_ml = ss.results.get("ml_eval")
    if ev_ml is not None and ev_ml["cfg"] != ml_cfg:
        st.info("La evaluación guardada es de otra liga o configuración del modelo: vuelve a evaluar.")
        ev_ml = None
    if ev_ml is not None:
        ll_mk = ev_ml["ll"]["Mercado"]
        cmp_df = pd.DataFrame({
            "Modelo": list(ev_ml["ll"]), "LogLoss": list(ev_ml["ll"].values()), "Brier": list(ev_ml["brier"].values()),
            "Δ LogLoss vs mercado": [v - ll_mk for v in ev_ml["ll"].values()],
        })
        st.caption(f"Partidos evaluados: {ev_ml['n']} (LogLoss/Brier: menor es mejor).")
        st.dataframe(cmp_df.style.format({"LogLoss": "{:.4f}", "Brier": "{:.4f}", "Δ LogLoss vs mercado": "{:+.4f}"}),
                     hide_index=True, width="stretch")
        if ev_ml["ll"]["ML"] < ll_mk:
            st.success("El ML supera al mercado en esta muestra. Confírmalo con más partidos antes de subir stakes.")
        else:
            st.warning("El ML no supera al mercado en esta muestra: es lo habitual. Úsalo como complemento (peso moderado) y vigila el CLV.")
        bias_rows = [{"Modelo": nm, "P(Local)": P_[:, 0].mean(), "P(Empate)": P_[:, 1].mean(), "P(Visita)": P_[:, 2].mean(),
                      "σ P(Empate)": P_[:, 1].std()} for nm, P_ in ev_ml["P"].items()]
        freq = np.bincount(ev_ml["y"], minlength=3) / len(ev_ml["y"])
        bias_rows.append({"Modelo": "Frecuencia real", "P(Local)": freq[0], "P(Empate)": freq[1], "P(Visita)": freq[2], "σ P(Empate)": np.nan})
        st.markdown("**Sesgo por resultado**")
        st.dataframe(pd.DataFrame(bias_rows).style.format({"P(Local)": "{:.3f}", "P(Empate)": "{:.3f}", "P(Visita)": "{:.3f}", "σ P(Empate)": "{:.3f}"}, na_rep="—"),
                     hide_index=True, width="stretch")
        st.caption("Si σ P(Empate) del ML es mucho mayor que la del mercado, el modelo está inventando señal en el empate "
                   "(casi impredecible) y generará 'valor' falso en ese resultado.")
        st.plotly_chart(calibration_figure(ev_ml["P"], ev_ml["y"]), width="stretch")

    st.divider()
    st.markdown("### 🎯 Predicción ML para el partido seleccionado")
    st.caption("La probabilidad de mercado sale del consenso del escáner o de las cuotas capturadas en 💰 Valor.")
    if st.button("🧠 Predecir con ML", key="ml_pred_btn"):
        mk_now, mk_src = market_probs_for(code, home, away)
        if mk_now is None:
            st.warning("Primero descarga las odds en 💎 Escáner o captura cuotas en 💰 Valor para este partido.")
        else:
            ml = get_ml_model(P, code)
            if ml is None:
                st.warning("No se pudo entrenar el ML (historial insuficiente).")
            else:
                x_now = make_features(mk_now, dc, bundle["tracker"].features(home, away, pd.Timestamp.now().normalize()))
                p_ml = ml.predict_proba(x_now.reshape(1, -1))[0]
                ss.results["ml_pred"] = {"match_id": match_id, "mk": mk_now, "mk_src": mk_src, "p_ml": p_ml, "table": ml.explain(x_now)}
    pred = ss.results.get("ml_pred")
    if pred and pred["match_id"] == match_id:
        pa_, pb_, pc_ = st.columns(3)
        pa_.metric(f"Gana {home}", f"{pred['p_ml'][0] * 100:.1f}%")
        pb_.metric("Empate", f"{pred['p_ml'][1] * 100:.1f}%")
        pc_.metric(f"Gana {away}", f"{pred['p_ml'][2] * 100:.1f}%")
        adj_now = blend_probs(pred["p_ml"], pred["mk"], w_model)
        comp = pd.DataFrame({
            "Modelo": ["Mercado (sin margen)", "Dixon-Coles", "ML", f"Ajustada ({w_model * 100:.0f}% modelo)"],
            "H": [pred["mk"][0], dc["ph"], pred["p_ml"][0], adj_now[0]],
            "D": [pred["mk"][1], dc["pd"], pred["p_ml"][1], adj_now[1]],
            "A": [pred["mk"][2], dc["pa"], pred["p_ml"][2], adj_now[2]],
        })
        st.caption(f"Mercado: {pred['mk_src']}.")
        st.dataframe(comp.style.format({"H": "{:.3f}", "D": "{:.3f}", "A": "{:.3f}"}), hide_index=True, width="stretch")
        imp_df = pred["table"].sort_values("Importancia")
        st.markdown("### 🔍 Importancia de variables")
        st.caption("Corrección sobre el mercado: |coeficiente| medio de cada señal (el mercado es el punto de partida). Árboles: importancia del modelo base.")
        fig_imp = go.Figure(go.Bar(x=imp_df["Importancia"], y=imp_df["Feature"], orientation="h", marker_color="#76b900"))
        fig_imp.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
        st.plotly_chart(fig_imp, width="stretch")
        with st.expander("Valores de las features de este partido"):
            st.dataframe(imp_df.sort_values("Importancia", ascending=False).style.format({"Importancia": "{:.3f}", "Valor": "{:.3f}"}),
                         hide_index=True, width="stretch")

# --- TAB 8: MULTI-LIGA ---
with t8:
    st.markdown("## 🌎 Super escáner: todas las ligas")
    st.caption("Analiza las 7 ligas con las odds que ya hayas descargado en 💎 Escáner (no gasta créditos). "
               "Con entrenamiento 'pooled' el ML se entrena una sola vez para todas.")
    mc1, mc2 = st.columns(2)
    use_ml_multi = mc1.checkbox("Usar modelo ML (si no, solo Dixon-Coles)", value=True, key="multi_use_ml")
    horizon_multi = mc2.slider("Ventana (días)", 1, 14, 7, key="multi_horizon")
    if st.button(f"⬇️ Descargar odds de las 7 ligas ({len(LEAGUES)} créditos)", key="dl_all_odds", disabled=not api_key,
                 help="Necesita la API key (💎 Escáner o st.secrets). Usa la caché de 10 min si ya las descargaste."):
        errs = []
        with st.spinner("Descargando odds de todas las ligas..."):
            for lc, ln in LEAGUES.items():
                try:
                    resp = call_odds_api(API_LEAGUE_MAP[lc], api_key)
                    ss.market_storage[lc] = {"timestamp": resp["ts"], "data": resp["data"]}
                    if resp["remaining"] is not None:
                        ss.api_usage = {"used": resp["used"], "remaining": resp["remaining"]}
                except Exception as e:
                    errs.append(f"{ln}: {_scrub(e, api_key)}")
        for e in errs:
            st.error(f"Error API · {e}")
        if not errs:
            ss.flash = "✅ Odds de las 7 ligas descargadas. Ahora ejecuta el análisis masivo."
            st.rerun()
    if st.button("🚀 Ejecutar análisis masivo (7 ligas)", key="run_multi"):
        results, skipped = {}, []
        prog, status = st.progress(0.0), st.empty()
        for i, (lc, ln) in enumerate(LEAGUES.items(), 1):
            status.text(f"Analizando {ln} ({lc})...")
            stored_m = ss.market_storage.get(lc)
            if stored_m and stored_m["data"]:
                ml_m = get_ml_model(P, lc) if use_ml_multi else None
                results[lc] = scan_league(lc, stored_m["data"], {**P, "horizon_h": horizon_multi * 24}, ml_m)
            else:
                skipped.append(ln)
            prog.progress(i / len(LEAGUES))
        status.text("✅ Análisis completo finalizado.")
        ss.results["multi"] = {"results": results, "skipped": skipped}

    multi = ss.results.get("multi")
    if multi is None:
        st.info("Presiona el botón para iniciar el escaneo de todas las ligas con odds descargadas.")
    else:
        if multi["skipped"]:
            st.caption(f"Sin odds descargadas (omitidas): {', '.join(multi['skipped'])}")
        all_dec = []
        for lc, raw in multi["results"].items():
            if raw.empty:
                with st.expander(f"⚽ {LEAGUES[lc]} (sin partidos válidos)", expanded=False):
                    st.write("No se encontraron partidos pre-partido con nombres y cuotas válidos en la ventana.")
                continue
            dec_l = decorate_scan(raw, P, bank).sort_values("Ventaja pp", ascending=False).reset_index(drop=True)
            n_picks = int((dec_l["Pick"] != "No Bet").sum())
            all_dec.append(dec_l)
            with st.expander(f"⚽ {LEAGUES[lc]} ({n_picks} picks de {len(dec_l)} partidos)", expanded=n_picks > 0):
                st.dataframe(dec_l.style.format({"Ventaja pp": "{:+.1f}", "Mejor EV": "{:.3f}", "Cuota": "{:.2f}", "Kelly %": "{:.2f}", "Stake $": "{:.2f}"}, na_rep="—"),
                             hide_index=True, width="stretch")
        if all_dec:
            st.divider()
            st.markdown("### 📥 Descargar todo")
            df_master = pd.concat(all_dec, ignore_index=True).sort_values("Ventaja pp", ascending=False)
            st.download_button("Descargar CSV combinado (todas las ligas)", data=df_master.to_csv(index=False).encode("utf-8"),
                               file_name="super_jornada_ml.csv", mime="text/csv", key="dl_multi")

        # --- APUESTAS MÁS SEGURAS (todas las ligas) ---
        raws = [r for r in multi["results"].values() if not r.empty]
        if raws:
            st.divider()
            st.markdown("### 🛡️ Apuestas más seguras de la jornada")
            st.caption("Para cada partido, el resultado 1X2 con mayor probabilidad ajustada (modelo + mercado). "
                       "Ojo: «seguro» no significa rentable. Las cuotas bajas suelen ser justas o incluso peores que justas; "
                       "solo las marcadas con ✅ tienen además ventaja sobre la cuota (valor). Una combinada de varias "
                       "«seguras» multiplica el riesgo: 4 picks al 75% aciertan juntos solo el 32% de las veces.")
            sf1, sf2, sf3 = st.columns(3)
            min_prob = sf1.slider("Probabilidad mínima (%)", 40, 90, 60, 5, key="safe_min_prob")
            top_n = sf2.slider("Mostrar los mejores", 5, 50, 15, 5, key="safe_top_n")
            only_value = sf3.checkbox("Solo con valor (✅)", value=False, key="safe_only_value")
            safe = safest_picks(pd.concat(raws, ignore_index=True), P, bank)
            safe = safe[safe["Prob. ajustada %"] >= min_prob]
            if only_value:
                safe = safe[safe["¿Valor?"] == "✅"]
            safe = safe.sort_values("Prob. ajustada %", ascending=False).head(top_n).reset_index(drop=True)
            if safe.empty:
                st.info("Ningún partido cumple los filtros. Baja la probabilidad mínima o desmarca «Solo con valor».")
            else:
                s1, s2, s3 = st.columns(3)
                s1.metric("Picks", len(safe))
                s2.metric("Con valor ✅", int((safe["¿Valor?"] == "✅").sum()))
                s3.metric("Aciertos esperados", f"{safe['Prob. ajustada %'].sum() / 100:.1f} de {len(safe)}")
                st.dataframe(safe.style.format({"Prob. ajustada %": "{:.1f}", "Prob. mercado %": "{:.1f}", "Mejor cuota": "{:.2f}",
                                                "Ventaja pp": "{:+.1f}", "EV": "{:+.3f}", "Kelly %": "{:.2f}", "Stake $": "{:.2f}"}),
                             hide_index=True, width="stretch")
                st.download_button("📥 Descargar apuestas seguras (CSV)", data=safe.to_csv(index=False).encode("utf-8"),
                                   file_name="apuestas_seguras.csv", mime="text/csv", key="dl_safe")
