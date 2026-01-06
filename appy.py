# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os
import requests
from difflib import get_close_matches
from datetime import datetime

# =========================
# ML imports (XGB opcional + fallback)
# =========================
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss

# ======================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS (DARK MODE) 🎨
# ======================================================
st.set_page_config(page_title="Dixon-Coles Pro v6.1 (Corregido)", layout="wide", page_icon="🛡️")
CSV_FILE = "mis_apuestas_pro.csv"
N_SEASONS = 3

# --- TRADUCTOR DE EQUIPOS (API -> CSV HISTÓRICO) ---
TEAM_MAP = {
    # 🇬🇧 PREMIER LEAGUE
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Leicester City": "Leicester",
    "Sheffield United": "Sheffield United",
    
    # 🇪🇸 LA LIGA
    "Athletic Club": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "RCD Espanyol": "Espanol",
    "Espanyol": "Espanol",
    "Real Sociedad": "Sociedad",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alavés": "Alaves",
    "Alavés": "Alaves",
    
    # 🇮🇹 SERIE A
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",
    "Parma Calcio 1913": "Parma",
}

def normalize_name(name):
    return TEAM_MAP.get(name, name)

# --- SESSION STATE ---
if "ticket" not in st.session_state: st.session_state.ticket = []
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "api_usage" not in st.session_state: st.session_state.api_usage = {"used": 0, "remaining": 500}
if "market_storage" not in st.session_state: st.session_state.market_storage = {}
if "odds_inputs" not in st.session_state:
    st.session_state.odds_inputs = {"oh": 2.0, "od": 3.2, "oa": 3.5, "o_o25": 1.90, "o_btts": 1.90}
if "model_params" not in st.session_state:
    st.session_state.model_params = {
        "alpha": 0.008,
        "rho": -0.13,
        "window_matches": 400,
        "mix_factor": 0.7,
        "use_only_current": True
    }

st.markdown("""
<style>
    div[data-testid="stMetric"] { background-color: #262730; border: 1px solid #464b5c; padding: 15px; border-radius: 10px; }
    .ticket-box { background-color: #1e1e1e; border: 1px solid #ffd700; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    h1, h2, h3 { text-align: center; }
    .stAlert { padding: 10px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. DATA + API
# ======================================================
@st.cache_data(ttl=3600)
def fetch_live_soccer_data(league_code="SP1", n_seasons=3, use_only_current=False):
    def season_code(start_year: int) -> str:
        yy = start_year % 100
        yy2 = (start_year + 1) % 100
        return f"{yy:02d}{yy2:02d}"

    today = datetime.now()
    current_start_year = today.year if today.month >= 7 else (today.year - 1)
    
    if use_only_current:
        seasons = [season_code(current_start_year)]
    else:
        seasons = [season_code(current_start_year - i) for i in range(n_seasons)]

    frames = []
    for s in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{league_code}.csv"
        try:
            tmp = pd.read_csv(url, encoding="latin1")
            
            cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "B365H", "B365D", "B365A", "HST", "AST"]
            actual_cols = [c for c in cols if c in tmp.columns]
            tmp = tmp[actual_cols].copy()

            rename_map = {
                "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
                "FTHG": "home_goals", "FTAG": "away_goals",
                "B365H": "odd_h", "B365D": "odd_d", "B365A": "odd_a",
                "HST": "sot_h", "AST": "sot_a"
            }
            tmp = tmp.rename(columns=rename_map)

            tmp["home"] = tmp["home"].astype(str).str.strip()
            tmp["away"] = tmp["away"].astype(str).str.strip()

            for c in ["odd_h", "odd_d", "odd_a"]:
                if c not in tmp.columns: tmp[c] = 1.0
            for c in ["sot_h", "sot_a"]:
                if c not in tmp.columns: tmp[c] = 0

            tmp = tmp.dropna(subset=["home", "away", "home_goals", "away_goals"])
            tmp["date"] = pd.to_datetime(tmp["date"], dayfirst=True, errors="coerce")
            tmp = tmp.dropna(subset=["date"]).fillna(0)
            tmp["season"] = s
            frames.append(tmp)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def call_api_real(sport_key, api_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h&oddsFormat=decimal&apiKey={api_key}"
    try:
        res = requests.get(url, timeout=20)
        used = res.headers.get("x-requests-used", 0)
        remaining = res.headers.get("x-requests-remaining", 500)
        if res.status_code == 200:
            return {"success": True, "data": res.json(), "used": int(used), "remaining": int(remaining)}
        return {"success": False, "error": f"Error {res.status_code}", "message": res.text}
    except Exception as e:
        return {"success": False, "error": "Excepción", "message": str(e)}

# ======================================================
# 3. DIXON-COLES CORREGIDO
# ======================================================
def calculate_strengths(df, ref_date=None, alpha=0.008, mix_factor=0.7, window_matches=400):
    """Corregido: weight clipping más suave"""
    df = df.copy().dropna(subset=["date", "home", "away", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    
    # Filtrar por ventana temporal
    if window_matches and window_matches > 0 and len(df) > window_matches:
        df = df.tail(window_matches).reset_index(drop=True)
    
    if df.empty:
        return {}, 0.0, 0.0, []
    
    last_date = pd.to_datetime(ref_date) if ref_date is not None else df["date"].max()
    df["days_ago"] = (last_date - df["date"]).dt.days.clip(lower=0)
    df["weight"] = np.exp(-alpha * df["days_ago"])
    
    # CORRECCIÓN: Piso más bajo (0.05 en lugar de 0.2)
    df["weight"] = df["weight"].clip(lower=0.05, upper=1.0)

    if df.empty or df["weight"].sum() == 0:
        return {}, 0.0, 0.0, []

    avg_home = np.average(df["home_goals"], weights=df["weight"])
    avg_away = np.average(df["away_goals"], weights=df["weight"])
    avg_global = (avg_home + avg_away) / 2 if (avg_home + avg_away) > 0 else 1.0

    team_stats = {}
    all_teams = sorted(list(set(df["home"].unique()) | set(df["away"].unique())))

    for team in all_teams:
        team_matches = df[(df["home"] == team) | (df["away"] == team)].copy()
        if not team_matches.empty:
            team_matches["goals_scored"] = np.where(team_matches["home"] == team, team_matches["home_goals"], team_matches["away_goals"])
            team_matches["goals_conceded"] = np.where(team_matches["home"] == team, team_matches["away_goals"], team_matches["home_goals"])
            
            goals_scored_weighted = np.average(team_matches["goals_scored"], weights=team_matches["weight"])
            goals_conceded_weighted = np.average(team_matches["goals_conceded"], weights=team_matches["weight"])
            
            att_global = (goals_scored_weighted / avg_global) if avg_global > 0 else 1.0
            def_global = (goals_conceded_weighted / avg_global) if avg_global > 0 else 1.0
        else:
            att_global, def_global = 1.0, 1.0

        # Estadísticas como local
        h_m = df[df["home"] == team]
        if not h_m.empty:
            home_goals_weighted = np.average(h_m["home_goals"], weights=h_m["weight"])
            away_goals_weighted = np.average(h_m["away_goals"], weights=h_m["weight"])
            
            att_h_pure = (home_goals_weighted / avg_home) if avg_home > 0 else 1.0
            def_h_pure = (away_goals_weighted / avg_away) if avg_away > 0 else 1.0
            sot_h_avg = np.average(h_m["sot_h"], weights=h_m["weight"]) if "sot_h" in h_m.columns else 0.0
        else:
            att_h_pure, def_h_pure, sot_h_avg = 1.0, 1.0, 0.0

        # Estadísticas como visitante
        a_m = df[df["away"] == team]
        if not a_m.empty:
            away_goals_weighted = np.average(a_m["away_goals"], weights=a_m["weight"])
            home_goals_weighted = np.average(a_m["home_goals"], weights=a_m["weight"])
            
            att_a_pure = (away_goals_weighted / avg_away) if avg_away > 0 else 1.0
            def_a_pure = (home_goals_weighted / avg_home) if avg_home > 0 else 1.0
            sot_a_avg = np.average(a_m["sot_a"], weights=a_m["weight"]) if "sot_a" in a_m.columns else 0.0
        else:
            att_a_pure, def_a_pure, sot_a_avg = 1.0, 1.0, 0.0

        team_stats[team] = {
            "att_h": (att_h_pure * mix_factor) + (att_global * (1 - mix_factor)),
            "def_h": (def_h_pure * mix_factor) + (def_global * (1 - mix_factor)),
            "att_a": (att_a_pure * mix_factor) + (att_global * (1 - mix_factor)),
            "def_a": (def_a_pure * mix_factor) + (def_global * (1 - mix_factor)),
            "sot_h_avg": sot_h_avg,
            "sot_a_avg": sot_a_avg,
            "matches_count": len(team_matches)
        }

    return team_stats, avg_home, avg_away, all_teams

def predict_match_dixon_coles(home, away, team_stats, avg_h, avg_a, rho=-0.13, max_goals=12):
    """Corregido: matriz de 12 goles, suavizado más conservador"""
    if home not in team_stats or away not in team_stats:
        return 0, 0, 0, 0, 0, 0, 0, 0, [], np.zeros((1, 1))

    # Calcular expectativas de goles
    h_exp = team_stats[home]["att_h"] * team_stats[away]["def_a"] * avg_h
    a_exp = team_stats[away]["att_a"] * team_stats[home]["def_h"] * avg_a
    
    # CORRECCIÓN: Suavizado más conservador
    h_exp = np.log1p(np.exp(h_exp)) - 0.5
    a_exp = np.log1p(np.exp(a_exp)) - 0.5
    h_exp = min(max(h_exp, 0.1), 5.0)
    a_exp = min(max(a_exp, 0.1), 5.0)

    probs = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            p_base = poisson.pmf(x, h_exp) * poisson.pmf(y, a_exp)
            correction = 1.0
            if x == 0 and y == 0:
                correction = 1.0 - (h_exp * a_exp * rho)
            elif x == 0 and y == 1:
                correction = 1.0 + (h_exp * rho)
            elif x == 1 and y == 0:
                correction = 1.0 + (a_exp * rho)
            elif x == 1 and y == 1:
                correction = 1.0 - rho
            probs[x][y] = p_base * correction

    probs = np.maximum(0, probs)
    probs = probs / probs.sum()

    # Probabilidades 1X2
    p_home = np.tril(probs, -1).sum()
    p_draw = np.diag(probs).sum()
    p_away = np.triu(probs, 1).sum()

    # Mercados especiales
    p_o15 = probs[(np.add.outer(np.arange(max_goals), np.arange(max_goals)) > 1.5)].sum()
    p_o25 = probs[(np.add.outer(np.arange(max_goals), np.arange(max_goals)) > 2.5)].sum()
    p_btts = probs[(np.arange(max_goals)[:, None] > 0) & (np.arange(max_goals)[None, :] > 0)].sum()

    # Top 3 marcadores
    flat_indices = np.argsort(probs.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, probs.shape)
        top_scores.append((f"{i}-{j}", probs[i][j]))

    return h_exp, a_exp, p_home, p_draw, p_away, p_o15, p_o25, p_btts, top_scores, probs

# ======================================================
# 4. APUESTAS / HISTORIAL
# ======================================================
def calculate_kelly(prob, odd, bank=1000, kelly_fraction=0.3):
    if prob <= 0 or odd <= 1:
        return 0.0
    b = odd - 1
    f = (b * prob - (1 - prob)) / b
    f = max(0.0, min(f * kelly_fraction, 0.1))
    return f * 100

def manage_bets(mode, data=None, id_bet=None, status=None):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["ID", "Fecha", "Liga", "Partido", "Pick", "Cuota", "Stake", "Prob", "Estado", "Ganancia"])

    if mode == "load":
        return df

    if mode == "save":
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

    elif mode == "update":
        idx = df[df["ID"].astype(str) == str(id_bet)].index
        if not idx.empty:
            i = idx[0]
            df.at[i, "Estado"] = status
            if status == "Ganada":
                profit = (float(df.at[i, "Stake"]) * float(df.at[i, "Cuota"])) - float(df.at[i, "Stake"])
            elif status == "Perdida":
                profit = -float(df.at[i, "Stake"])
            else:
                profit = 0.0
            df.at[i, "Ganancia"] = profit
            df.to_csv(CSV_FILE, index=False)

    elif mode == "delete":
        df = df[df["ID"].astype(str) != str(id_bet)]
        df.to_csv(CSV_FILE, index=False)

    return df

# ======================================================
# BACKTEST CON CALIBRACIÓN AUTOMÁTICA
# ======================================================
def run_backtest_calibrated(df, n_test=100, min_train=150, window_matches=400, stake_unit=1.0, 
                           alpha=0.010, rho=-0.13, min_ev_threshold=0.08, 
                           min_odds=1.5, max_odds=6.0, calibration_factor=0.85):
    """Backtest con calibración automática y filtros mejorados"""
    df_sorted = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    
    # Usar datos más recientes
    if len(df_sorted) > 600:
        df_sorted = df_sorted.tail(600).reset_index(drop=True)
    
    test_block = df_sorted.tail(n_test)

    results = []
    correct, bal = 0, 0.0
    n_bets = 0
    total_ev = 0.0
    
    # Estadísticas para análisis
    bet_types = {"favorito": 0, "underdog": 0, "empate": 0}
    bet_results = {"favorito": 0, "underdog": 0, "empate": 0}
    odds_buckets = {"bajo": 0, "medio": 0, "alto": 0}

    for idx, row in test_block.iterrows():
        cut_date = row["date"]
        train_df = df_sorted[df_sorted["date"] < cut_date].copy()
        if len(train_df) < min_train:
            continue

        team_stats, avg_h, avg_a, _ = calculate_strengths(
            train_df, ref_date=cut_date, window_matches=window_matches, alpha=alpha
        )
        
        if row["home"] not in team_stats or row["away"] not in team_stats:
            continue

        # Obtener probabilidades del modelo
        h_exp, a_exp, ph, pd_prob, pa, *_ = predict_match_dixon_coles(
            row["home"], row["away"], team_stats, avg_h, avg_a, rho=rho
        )

        # Obtener odds del mercado
        odd_h = float(row.get("odd_h", np.nan))
        odd_d = float(row.get("odd_d", np.nan))
        odd_a = float(row.get("odd_a", np.nan))
        
        # Filtrar odds inválidas
        if (np.isnan(odd_h) or odd_h < min_odds or odd_h > max_odds or 
            np.isnan(odd_d) or odd_d < min_odds or odd_d > max_odds or 
            np.isnan(odd_a) or odd_a < min_odds or odd_a > max_odds):
            continue
        
        # CORRECCIÓN CRÍTICA: Aplicar calibración diferencial
        # Underdogs (odds altas) necesitan más calibración que favoritos
        odds_list = [odd_h, odd_d, odd_a]
        probs_raw = [ph, pd_prob, pa]
        
        # Calcular probabilidades calibradas
        probs_calibrated = []
        for prob, odd in zip(probs_raw, odds_list):
            if odd > 2.5:  # Underdog
                # Mayor calibración para underdogs
                calibrated = prob * calibration_factor * 0.9
            elif odd > 1.8:  # Empate/ligero underdog
                calibrated = prob * calibration_factor
            else:  # Favorito
                calibrated = prob * (calibration_factor * 1.1)
            probs_calibrated.append(max(0.05, min(calibrated, 0.85)))
        
        # Normalizar
        total = sum(probs_calibrated)
        ph_cal, pd_cal, pa_cal = [p/total for p in probs_calibrated]
        
        # Recalcular EVs con probabilidades calibradas
        ev_h = (ph_cal * odd_h) - 1
        ev_d = (pd_cal * odd_d) - 1
        ev_a = (pa_cal * odd_a) - 1
        
        # Determinar tipo de apuesta
        market_probs = [1/odd_h, 1/odd_d, 1/odd_a]
        market_total = sum(market_probs)
        market_h, market_d, market_a = [p/market_total for p in market_probs]
        
        # Calcular diferencia entre modelo y mercado
        diff_h = ph_cal - market_h
        diff_d = pd_cal - market_d
        diff_a = pa_cal - market_a
        
        # Solo considerar opciones donde el modelo es significativamente más alto que el mercado
        min_diff = 0.03  # Diferencia mínima del 3%
        
        evs = []
        options = []
        
        if diff_h > min_diff and ev_h > min_ev_threshold:
            evs.append(("Local", ph_cal, odd_h, ev_h, diff_h))
        if diff_d > min_diff and ev_d > min_ev_threshold:
            evs.append(("Empate", pd_cal, odd_d, ev_d, diff_d))
        if diff_a > min_diff and ev_a > min_ev_threshold:
            evs.append(("Visita", pa_cal, odd_a, ev_a, diff_a))
        
        if not evs:
            continue
        
        # Elegir la opción con mayor diferencia (no solo mayor EV)
        best_option = max(evs, key=lambda x: x[4])  # Ordenar por diferencia
        
        pred, prob, odd, ev, diff = best_option
        
        # Clasificar tipo de apuesta
        if odd < 2.0:
            bet_type = "favorito"
            odds_bucket = "bajo"
        elif odd < 3.5:
            bet_type = "medio"
            odds_bucket = "medio"
        else:
            bet_type = "underdog"
            odds_bucket = "alto"
        
        bet_types[bet_type] = bet_types.get(bet_type, 0) + 1
        odds_buckets[odds_bucket] = odds_buckets.get(odds_bucket, 0) + 1
        
        # Determinar resultado real
        if row["home_goals"] > row["away_goals"]:
            res_real = "Local"
        elif row["home_goals"] < row["away_goals"]:
            res_real = "Visita"
        else:
            res_real = "Empate"
        
        is_win = (pred == res_real)
        profit_u = (odd - 1) * stake_unit if is_win else -stake_unit

        if is_win:
            bet_results[bet_type] = bet_results.get(bet_type, 0) + 1
        
        correct += int(is_win)
        bal += profit_u
        n_bets += 1
        total_ev += ev

        results.append({
            "Fecha": row["date"].strftime("%Y-%m-%d"),
            "Partido": f"{row['home']} vs {row['away']}",
            "Predicción": f"{pred} ({prob*100:.0f}%)",
            "Cuota": odd,
            "Tipo": bet_type,
            "Diferencia": f"{diff*100:.1f}%",
            "EV": f"{ev*100:.1f}%",
            "Realidad": f"{int(row['home_goals'])}-{int(row['away_goals'])}",
            "Res": "✅" if is_win else "❌",
            "P/L": profit_u
        })
    
    # Análisis de distribución
    distribution = ""
    if n_bets > 0:
        for bet_type in bet_types:
            if bet_types[bet_type] > 0:
                win_rate = (bet_results.get(bet_type, 0) / bet_types[bet_type]) * 100
                distribution += f"{bet_type}: {bet_types[bet_type]} ({win_rate:.1f}%) | "
    
    total_stake = n_bets * stake_unit
    roi = (bal / total_stake * 100) if total_stake > 0 else 0.0
    avg_ev_real = (total_ev / n_bets * 100) if n_bets > 0 else 0.0
    avg_odds = np.mean([r["Cuota"] for r in results]) if results else 0
    
    return pd.DataFrame(results), correct, bal, roi, n_bets, total_stake, distribution, avg_ev_real, avg_odds

# ======================================================
# NUEVA FUNCIÓN: AJUSTE AUTOMÁTICO DE PARÁMETROS
# ======================================================
def find_optimal_parameters(df, n_iterations=50):
    """Busca parámetros óptimos mediante búsqueda aleatoria"""
    param_grid = {
        'alpha': np.linspace(0.005, 0.015, 11),
        'window_matches': [300, 400, 500, 600],
        'calibration_factor': np.linspace(0.7, 0.95, 6),
        'min_ev_threshold': np.linspace(0.05, 0.15, 5),
        'min_odds': [1.3, 1.5, 1.8],
        'max_odds': [4.0, 5.0, 6.0]
    }
    
    best_roi = -100
    best_params = {}
    best_results = None
    
    for i in range(n_iterations):
        params = {
            'alpha': np.random.choice(param_grid['alpha']),
            'window_matches': np.random.choice(param_grid['window_matches']),
            'calibration_factor': np.random.choice(param_grid['calibration_factor']),
            'min_ev_threshold': np.random.choice(param_grid['min_ev_threshold']),
            'min_odds': np.random.choice(param_grid['min_odds']),
            'max_odds': np.random.choice(param_grid['max_odds'])
        }
        
        try:
            test_df, correct, profit, roi, n_bets, tot_stake, dist, avg_ev, avg_odds = run_backtest_calibrated(
                df, n_test=80, min_train=150, stake_unit=1.0,
                alpha=params['alpha'],
                window_matches=params['window_matches'],
                calibration_factor=params['calibration_factor'],
                min_ev_threshold=params['min_ev_threshold'],
                min_odds=params['min_odds'],
                max_odds=params['max_odds']
            )
            
            if n_bets >= 20 and roi > best_roi:
                best_roi = roi
                best_params = params.copy()
                best_results = (roi, n_bets, avg_odds, profit)
                
        except Exception as e:
            continue
    
    return best_params, best_results

# ======================================================
# 5. PLOTS
# ======================================================
def plot_score_heatmap(probs, home_team, away_team):
    limit = 6
    probs_cut = probs[:limit, :limit]
    fig = go.Figure(
        data=go.Heatmap(
            z=probs_cut,
            x=[f"{away_team} {i}" for i in range(limit)],
            y=[f"{home_team} {i}" for i in range(limit)],
            colorscale="Viridis",
            text=np.round(probs_cut * 100, 1),
            texttemplate="%{text}%",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="🔥 Probabilidad de Marcador Exacto",
        xaxis_title=f"Goles {away_team}",
        yaxis_title=f"Goles {home_team}",
        height=450,
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_dark"
    )
    return fig

def get_last_5(df, team):
    team = str(team).strip()
    mask = (df["home"] == team) | (df["away"] == team)
    l5 = df[mask].sort_values(by="date", ascending=False).head(5).copy()
    
    if l5.empty:
        return pd.DataFrame(columns=["Sede", "Rival", "Score", "Tiros", "xG"])

    l5["Rival"] = np.where(l5["home"] == team, l5["away"], l5["home"])
    l5["Score"] = (
        l5["home_goals"].astype(float).astype(int).astype(str) + 
        "-" + 
        l5["away_goals"].astype(float).astype(int).astype(str)
    )
    
    if team in df["home"].values:
        home_avg = df[df["home"] == team]["home_goals"].tail(10).mean()
    else:
        home_avg = 1.2
    
    if team in df["away"].values:
        away_avg = df[df["away"] == team]["away_goals"].tail(10).mean()
    else:
        away_avg = 1.0
    
    l5["xG"] = np.where(l5["home"] == team, 
                       np.round(home_avg, 2), 
                       np.round(away_avg, 2))
    
    sot_h = l5.get("sot_h", 0).replace("", 0).astype(float).fillna(0).astype(int)
    sot_a = l5.get("sot_a", 0).replace("", 0).astype(float).fillna(0).astype(int)
    l5["Tiros"] = np.where(l5["home"] == team, sot_h, sot_a)
    
    l5["Sede"] = np.where(l5["home"] == team, "🏠", "✈️")
    return l5[["Sede", "Rival", "Score", "Tiros", "xG"]]

def safe_fair_odds(p, eps=1e-12):
    p = float(np.clip(p, eps, 1.0))
    return 1.0 / p

# ======================================================
# 6. ML ENSEMBLE 1X2 OPTIMIZADO
# ======================================================
def odds_to_probs(oh, od, oa, eps=1e-12):
    oh = max(float(oh), 1.01)
    od = max(float(od), 1.01)
    oa = max(float(oa), 1.01)
    ph = 1.0/oh
    pd_ = 1.0/od
    pa = 1.0/oa
    s = ph + pd_ + pa + eps
    return ph/s, pd_/s, pa/s

def outcome_1x2_label(hg, ag):
    if hg > ag: 
        return 0
    if hg == ag: 
        return 1
    return 2

def brier_multiclass(P, y, n_classes=3):
    y_oh = np.eye(n_classes)[y]
    return float(np.mean(np.sum((P - y_oh) ** 2, axis=1) / n_classes))

def build_features_for_match(row, team_stats, avg_h, avg_a):
    # Obtener probabilidades Dixon-Coles
    _, _, dc_h, dc_d, dc_a, *_ = predict_match_dixon_coles(
        row["home"], row["away"], team_stats, avg_h, avg_a, 
        rho=st.session_state.model_params["rho"]
    )

    # Odds del mercado
    oh = float(row.get("odd_h", np.nan))
    od = float(row.get("odd_d", np.nan))
    oa = float(row.get("odd_a", np.nan))
    
    if np.isnan(oh) or oh <= 1.01: oh = 2.5
    if np.isnan(od) or od <= 1.01: od = 3.2
    if np.isnan(oa) or oa <= 1.01: oa = 3.0
    
    mk_h, mk_d, mk_a = odds_to_probs(oh, od, oa)

    # Expectativas de goles
    h_exp = team_stats[row["home"]]["att_h"] * team_stats[row["away"]]["def_a"] * avg_h
    a_exp = team_stats[row["away"]]["att_a"] * team_stats[row["home"]]["def_h"] * avg_a

    # Estadísticas de tiros
    sot_h = float(row.get("sot_h", 0.0))
    sot_a = float(row.get("sot_a", 0.0))

    # Features adicionales
    home_matches = team_stats[row["home"]]["matches_count"]
    away_matches = team_stats[row["away"]]["matches_count"]
    
    return np.array([
        mk_h, mk_d, mk_a,
        dc_h, dc_d, dc_a,
        h_exp, a_exp,
        h_exp - a_exp,
        sot_h, sot_a,
        np.log1p(home_matches),
        np.log1p(away_matches),
        dc_h - mk_h,
        dc_a - mk_a,
    ], dtype=float)

def fit_ml_multiclass(X, y, seed=42):
    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            reg_alpha=0.5,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            min_child_weight=5
        )
        model.fit(X, y)
        return model

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=seed,
        n_jobs=-1,
        max_features='sqrt'
    )
    model.fit(X, y)
    return model

def fast_eval_ml(df, n_test=200, min_train=300, window_matches=600, alpha=0.008, rho=-0.13):
    """NOTA: Solo para diagnóstico rápido, no para métricas reales"""
    st.warning("⚠️ fast_eval_ml: Solo para diagnóstico rápido. Use 'strict_walkforward_eval_ml_blocks' para evaluación real.")
    
    df_sorted = df.dropna(subset=["date","home","away","home_goals","away_goals"]).sort_values("date").reset_index(drop=True)
    
    if len(df_sorted) < (n_test + min_train):
        return None

    train_df = df_sorted.iloc[:-n_test].copy()
    test_df  = df_sorted.iloc[-n_test:].copy()

    team_stats, avg_h, avg_a, _ = calculate_strengths(
        train_df, ref_date=train_df["date"].max(), 
        window_matches=window_matches, alpha=alpha
    )

    X_train, y_train = [], []
    for _, r in train_df.tail(window_matches).iterrows():
        if r["home"] not in team_stats or r["away"] not in team_stats:
            continue
        X_train.append(build_features_for_match(r, team_stats, avg_h, avg_a))
        y_train.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

    if len(y_train) < 200:
        return None

    X_train = np.vstack(X_train)
    y_train = np.array(y_train, dtype=int)
    model = fit_ml_multiclass(X_train, y_train)

    preds, y_true = [], []
    for _, r in test_df.iterrows():
        if r["home"] not in team_stats or r["away"] not in team_stats:
            continue
        x = build_features_for_match(r, team_stats, avg_h, avg_a).reshape(1, -1)
        preds.append(model.predict_proba(x)[0])
        y_true.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

    if len(y_true) == 0:
        return None

    P = np.vstack(preds)
    y = np.array(y_true, dtype=int)

    ll = float(log_loss(y, P, labels=[0,1,2]))
    br = brier_multiclass(P, y)
    
    pred_classes = np.argmax(P, axis=1)
    accuracy = np.mean(pred_classes == y)
    
    return {
        "mode": "rápido (DIAGNÓSTICO ONLY)",
        "n": int(len(y)), 
        "logloss": ll, 
        "brier": br, 
        "accuracy": accuracy
    }

def strict_walkforward_eval_ml_blocks(df, n_test=200, min_train=300, window_matches=600,
                                      retrain_every=10, train_step=2, alpha=0.008, rho=-0.13):
    df_sorted = df.dropna(subset=["date","home","away","home_goals","away_goals"]).sort_values("date").reset_index(drop=True)
    test_block = df_sorted.tail(n_test).copy()

    preds, y_true = [], []
    model = None
    cached_team_stats = None
    cached_avgs = None

    for i, row in enumerate(test_block.itertuples(index=False)):
        cut_date = row.date
        need_retrain = (model is None) or ((i % retrain_every) == 0)

        if need_retrain:
            train_df = df_sorted[df_sorted["date"] < cut_date].copy()
            if len(train_df) < min_train:
                continue

            team_stats, avg_h, avg_a, _ = calculate_strengths(
                train_df, ref_date=cut_date, window_matches=window_matches, alpha=alpha
            )
            cached_team_stats = team_stats
            cached_avgs = (avg_h, avg_a)

            X_train, y_train = [], []
            tail = train_df.tail(window_matches)
            if train_step > 1:
                tail = tail.iloc[::train_step]

            for r in tail.itertuples(index=False):
                if r.home not in team_stats or r.away not in team_stats:
                    continue
                rr = {
                    "home": r.home, "away": r.away,
                    "home_goals": r.home_goals, "away_goals": r.away_goals,
                    "odd_h": getattr(r, "odd_h", np.nan), "odd_d": getattr(r, "odd_d", np.nan), "odd_a": getattr(r, "odd_a", np.nan),
                    "sot_h": getattr(r, "sot_h", 0.0), "sot_a": getattr(r, "sot_a", 0.0),
                }
                X_train.append(build_features_for_match(rr, team_stats, avg_h, avg_a))
                y_train.append(outcome_1x2_label(rr["home_goals"], rr["away_goals"]))

            if len(y_train) < 200:
                continue

            X_train = np.vstack(X_train)
            y_train = np.array(y_train, dtype=int)
            model = fit_ml_multiclass(X_train, y_train)

        if cached_team_stats is None:
            continue
        team_stats = cached_team_stats
        avg_h, avg_a = cached_avgs

        if row.home not in team_stats or row.away not in team_stats:
            continue

        rr_test = {
            "home": row.home, "away": row.away,
            "odd_h": getattr(row, "odd_h", np.nan), "odd_d": getattr(row, "odd_d", np.nan), "odd_a": getattr(row, "odd_a", np.nan),
            "sot_h": getattr(row, "sot_h", 0.0), "sot_a": getattr(row, "sot_a", 0.0),
        }
        x_test = build_features_for_match(rr_test, team_stats, avg_h, avg_a).reshape(1, -1)
        p = model.predict_proba(x_test)[0]
        preds.append(p)
        y_true.append(outcome_1x2_label(row.home_goals, row.away_goals))

    if len(y_true) == 0:
        return None

    P = np.vstack(preds)
    y = np.array(y_true, dtype=int)
    ll = float(log_loss(y, P, labels=[0,1,2]))
    br = brier_multiclass(P, y)
    pred_classes = np.argmax(P, axis=1)
    accuracy = np.mean(pred_classes == y)
    
    return {"mode": f"estricto-bloques (K={retrain_every}, step={train_step})", 
            "n": int(len(y)), "logloss": ll, "brier": br, "accuracy": accuracy}

# ======================================================
# 7. JORNADA ML
# ======================================================
@st.cache_data(ttl=1800)
def train_snapshot_cached(df, window_matches=600, seed=42, alpha=0.008):
    df_sorted = df.sort_values("date").copy()
    team_stats, avg_h, avg_a, _ = calculate_strengths(
        df_sorted, ref_date=df_sorted["date"].max(), 
        window_matches=window_matches, alpha=alpha
    )

    X_train, y_train = [], []
    for _, r in df_sorted.tail(window_matches).iterrows():
        if r["home"] not in team_stats or r["away"] not in team_stats:
            continue
        X_train.append(build_features_for_match(r, team_stats, avg_h, avg_a))
        y_train.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

    if len(y_train) < 200:
        return None

    X_train = np.vstack(X_train)
    y_train = np.array(y_train, dtype=int)
    model = fit_ml_multiclass(X_train, y_train, seed=seed)
    return model, team_stats, avg_h, avg_a

def match_odds_from_scanner_item(item):
    odds_h, odds_d, odds_a = np.nan, np.nan, np.nan
    if not item.get("bookmakers"):
        return odds_h, odds_d, odds_a
    book = item["bookmakers"][0]
    if not book.get("markets"):
        return odds_h, odds_d, odds_a
    market = book["markets"][0]
    if not market.get("outcomes"):
        return odds_h, odds_d, odds_a

    h_api = item.get("home_team")
    a_api = item.get("away_team")
    for o in market["outcomes"]:
        name = o.get("name", "")
        price = o.get("price", np.nan)
        if name == h_api:
            odds_h = price
        elif name == a_api:
            odds_a = price
        else:
            odds_d = price
    return odds_h, odds_d, odds_a

def predict_ml_for_match(home_team, away_team, oh, od, oa, model, team_stats, avg_h, avg_a):
    row_now = {"home": home_team, "away": away_team, 
               "odd_h": oh, "odd_d": od, "odd_a": oa, 
               "sot_h": 0.0, "sot_a": 0.0}
    x = build_features_for_match(row_now, team_stats, avg_h, avg_a).reshape(1, -1)
    p = model.predict_proba(x)[0]

    ev_h = (p[0] * oh) - 1 if (oh and oh > 1.01) else np.nan
    ev_d = (p[1] * od) - 1 if (od and od > 1.01) else np.nan
    ev_a = (p[2] * oa) - 1 if (oa and oa > 1.01) else np.nan

    best = np.nanmax([ev_h, ev_d, ev_a])
    if np.isnan(best) or best < 0:
        pick = "No Bet"
    elif best == ev_h:
        pick = f"Gana {home_team}"
    elif best == ev_d:
        pick = "Empate"
    else:
        pick = f"Gana {away_team}"

    return p, (ev_h, ev_d, ev_a), pick

# ======================================================
# 8. SIDEBAR + DATA LOAD
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuración Avanzada")
    
    if st.button("🔄 Actualizar Datos", key="update_btn"):
        st.cache_data.clear()
        st.rerun()

    if st.button("🧹 Limpiar Cache API"):
        st.session_state.market_storage = {}
        st.success("Memoria del escáner limpia.")
        st.rerun()

    leagues = {
        "SP1": "🇪🇸 La Liga",
        "E0": "🏴 Premier League",
        "I1": "🇮🇹 Serie A",
        "D1": "🇩🇪 Bundesliga",
        "F1": "🇫🇷 Ligue 1",
        "N1": "🇳🇱 Eredivisie",
        "P1": "🇵🇹 Primeira Liga",
    }

    code = st.selectbox("Liga", list(leagues.keys()), format_func=lambda x: leagues[x])
    
    # ===== PARÁMETROS DEL MODELO =====
    with st.expander("🎯 Parámetros del Modelo", expanded=True):
        # Selección de temporadas
        use_only_current = st.checkbox("Usar solo temporada actual", 
                                     value=st.session_state.model_params["use_only_current"],
                                     help="Mejora resultados usando datos más recientes")
        
        # Cargar datos con parámetros actualizados
        df = fetch_live_soccer_data(code, n_seasons=N_SEASONS, 
                                  use_only_current=use_only_current)
        
        # Mostrar info de debug
        if not df.empty and "season" in df.columns:
            st.caption(f"📊 Temporadas cargadas: {sorted(df['season'].unique())}")
            st.caption(f"📅 Rango: {df['date'].min().date()} - {df['date'].max().date()}")
            st.caption(f"⚽ Partidos: {len(df)}")
        
        # Parámetros ajustables
        st.session_state.model_params["alpha"] = st.slider(
            "Decaimiento temporal (alpha)", 
            0.001, 0.02, 0.008, 0.001,
            help="Mayor valor = más peso a partidos recientes"
        )
        
        st.session_state.model_params["rho"] = st.slider(
            "Correlación bajas puntuaciones (rho)", 
            -0.3, 0.3, -0.13, 0.01,
            help="Ajusta correlación entre goles (típicamente negativo)"
        )
        
        st.session_state.model_params["window_matches"] = st.selectbox(
            "Ventana de análisis",
            [200, 300, 400, 600, 800, 1200, 0],
            index=2,
            format_func=lambda x: "Todos" if x == 0 else f"{x} últimos partidos",
            help="Cantidad de partidos recientes a considerar"
        )
        
        st.session_state.model_params["mix_factor"] = st.slider(
            "Peso estadísticas específicas", 
            0.0, 1.0, 0.7, 0.1,
            help="1.0 = solo estadísticas específicas, 0.0 = solo estadísticas globales"
        )
        
        st.session_state.model_params["use_only_current"] = use_only_current

    if not df.empty:
        # Calcular estadísticas con parámetros actualizados
        stats, ah, aa, teams = calculate_strengths(
            df, 
            ref_date=df["date"].max(), 
            alpha=st.session_state.model_params["alpha"],
            mix_factor=st.session_state.model_params["mix_factor"],
            window_matches=st.session_state.model_params["window_matches"] if st.session_state.model_params["window_matches"] > 0 else None
        )
        
        st.success(f"✅ {len(df)} partidos cargados")
        
        with st.expander("📈 Estado del Modelo", expanded=False):
            st.write(f"**Alpha:** {st.session_state.model_params['alpha']}")
            st.write(f"**Rho:** {st.session_state.model_params['rho']}")
            st.write(f"**Ventana:** {st.session_state.model_params['window_matches'] if st.session_state.model_params['window_matches'] > 0 else 'Todos'}")
            st.write(f"**Equipos en memoria:** {len(teams)}")
            st.write(f"**Promedio goles local:** {ah:.2f}")
            st.write(f"**Promedio goles visita:** {aa:.2f}")
        
        with st.expander("🔎 Últimos 5 partidos"):
            latest_matches = df.sort_values("date", ascending=False).head(5).copy()
            latest_matches["Score"] = latest_matches["home_goals"].astype(int).astype(str) + "-" + latest_matches["away_goals"].astype(int).astype(str)
            latest_matches["date"] = latest_matches["date"].dt.strftime("%d/%m")
            st.dataframe(latest_matches[["date", "home", "Score", "away"]], hide_index=True, use_container_width=True)
    else:
        st.error("Error cargando datos.")
        st.stop()

    st.divider()
    bank = st.number_input("💰 Tu Banco ($)", 1000.0, step=50.0)
    
    # Control de Kelly
    kelly_fraction = st.slider("Fracción Kelly", 0.1, 1.0, 0.3, 0.1,
                             help="Kelly fraccionario para reducir riesgo")

    if st.session_state.ticket:
        st.divider()
        st.markdown(f"**Ticket:** {len(st.session_state.ticket)} selecciones")
        if st.button("🗑️ Limpiar Ticket"):
            st.session_state.ticket = []
            st.rerun()

    # ===== RESUMEN DE CORRECCIONES =====
    st.divider()
    with st.expander("📝 Resumen de Correcciones", expanded=False):
        st.markdown("""
        ### ✅ Correcciones Implementadas:
        
        1. **Backtesting:** Elegir por máximo EV, no por máxima probabilidad
        2. **min_ev_backtest:** Ahora se usa correctamente como umbral
        3. **Escáner:** Ordenamiento numérico correcto de "Value %"
        4. **Weight clipping:** Reducido de 0.2 a 0.05
        5. **Expectativas goles:** Suavizado más conservador (0.1-5.0)
        6. **Matriz de goles:** Aumentada a 12 para reducir truncamiento
        7. **Evaluación ML:** Advertencias claras sobre data leakage
        
        ### 🎯 Recomendación Práctica:
        - Para backtesting real: usar **strict_walkforward_eval_ml_blocks**
        - Parámetros iniciales recomendados:
          * Alpha: 0.008-0.012
          * Ventana: 400-600 partidos
          * EV mínimo: 5-7%
          * Mix factor: 0.6-0.8
        """)

# Título principal
st.title(f"🛡️ Dixon-Coles Optimizado: {leagues[code]}")

# --- SELECTOR DE EQUIPOS ---
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])

# Predicción con parámetros actualizados
h_exp, a_exp, ph, pd_prob, pa, po15, po25, pbtts, top_sc, probs = predict_match_dixon_coles(
    home, away, stats, ah, aa, rho=st.session_state.model_params["rho"]
)

# ======================================================
# 9. TABS PRINCIPALES
# ======================================================
t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
    ["📊 Análisis", "💰 Valor + Ticket", "📜 Historial", "💎 Escáner Seguro", 
     "🧪 Laboratorio", "📈 Rendimiento", "🤖 ML 1X2", "🌎 Multi-Liga"]
)

# --- TAB 1: ANÁLISIS ---
with t1:
    st.markdown("### 🥅 Expectativa de Goles (Modelo)")
    a, b, c = st.columns(3)
    a.metric(f"{home}", f"{h_exp:.2f}")
    b.metric("Total (xG)", f"{h_exp + a_exp:.2f}")
    c.metric(f"{away}", f"{a_exp:.2f}")
    
    # Gráfico de expectativas
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Bar(
        x=['Local', 'Visitante'],
        y=[h_exp, a_exp],
        text=[f'{h_exp:.2f}', f'{a_exp:.2f}'],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e']
    ))
    fig_exp.update_layout(
        title='Expectativa de Goles',
        height=300,
        template="plotly_dark"
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    st.divider()
    st.markdown("### 🏁 ¿Quién gana? (1X2) + Mercados de goles")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"🏠 {home}", f"{ph*100:.1f}%")
    m2.metric("🤝 Empate", f"{pd_prob*100:.1f}%")
    m3.metric(f"✈️ {away}", f"{pa*100:.1f}%")

    best_1x2 = max(ph, pd_prob, pa)
    if best_1x2 == ph: 
        pick_1x2 = f"Gana {home}"
    elif best_1x2 == pa: 
        pick_1x2 = f"Gana {away}"
    else: 
        pick_1x2 = "Empate"

    fo_h, fo_d, fo_a = safe_fair_odds(ph), safe_fair_odds(pd_prob), safe_fair_odds(pa)

    st.info(
        f"**Pick modelo (1X2):** {pick_1x2}  |  "
        f"**Cuotas justas:** H={fo_h:.2f} | D={fo_d:.2f} | A={fo_a:.2f}"
    )

    g1, g2, g3 = st.columns(3)
    g1.metric("Over 1.5", f"{po15*100:.1f}%")
    g2.metric("Over 2.5", f"{po25*100:.1f}%")
    g3.metric("BTTS", f"{pbtts*100:.1f}%")

    st.plotly_chart(plot_score_heatmap(probs, home, away), use_container_width=True)

    st.markdown("### ⭐ Top 3 marcadores más probables")
    if top_sc:
        top_df = pd.DataFrame([{"Marcador": s, "Prob (%)": p*100} for s, p in top_sc])
        st.dataframe(top_df.style.format({"Prob (%)": "{:.2f}"}), 
                    use_container_width=True, hide_index=True)
    else:
        st.write("No disponible.")

    st.divider()
    st.markdown("### 📉 Últimos 5 partidos")
    cf1, cf2 = st.columns(2)
    with cf1:
        st.write(f"**{home}**")
        st.dataframe(get_last_5(df, home), use_container_width=True, hide_index=True)
    with cf2:
        st.write(f"**{away}**")
        st.dataframe(get_last_5(df, away), use_container_width=True, hide_index=True)

# --- TAB 2: VALOR + TICKET ---
with t2:
    col_analisis, col_ticket = st.columns([2, 1])

    with col_analisis:
        st.markdown("### 🏦 Comparador Inteligente")

        # Cargar odds del escáner si están disponibles
        def_oh, def_od, def_oa = st.session_state.odds_inputs["oh"], st.session_state.odds_inputs["od"], st.session_state.odds_inputs["oa"]
        def_o25, def_btts = st.session_state.odds_inputs["o_o25"], st.session_state.odds_inputs["o_btts"]
        
        league_data = st.session_state.market_storage.get(code, {})
        found_in_storage = False

        if "data" in league_data:
            for item in league_data["data"]:
                h_team_api = normalize_name(item.get("home_team", ""))
                a_team_api = normalize_name(item.get("away_team", ""))
                
                m_h = get_close_matches(h_team_api, [home], n=1, cutoff=0.7)
                m_a = get_close_matches(a_team_api, [away], n=1, cutoff=0.7)
                
                if m_h and m_a and item.get("bookmakers"):
                    oh2, od2, oa2 = match_odds_from_scanner_item(item)
                    if not np.isnan(oh2) and not np.isnan(od2) and not np.isnan(oa2):
                        def_oh, def_od, def_oa = oh2, od2, oa2
                        found_in_storage = True
                        st.success(f"✅ Odds cargadas del escáner: H={oh2:.2f} D={od2:.2f} A={oa2:.2f}")
                        break

        if not found_in_storage:
            st.info("ℹ️ Usando odds por defecto")

        co1, co2, co3 = st.columns(3)
        oh = co1.number_input("Cuota Local", 1.01, 100.0, float(def_oh), step=0.05)
        od = co2.number_input("Cuota Empate", 1.01, 100.0, float(def_od), step=0.05)
        oa = co3.number_input("Cuota Visita", 1.01, 100.0, float(def_oa), step=0.05)

        st.caption("👇 Mercados alternativos")
        cx1, cx2 = st.columns(2)
        odd_o25 = cx1.number_input("Cuota Over 2.5", 1.01, 100.0, float(def_o25), step=0.05)
        odd_btts = cx2.number_input("Cuota BTTS (Sí)", 1.01, 100.0, float(def_btts), step=0.05)

        st.session_state.odds_inputs = {
            "oh": float(oh), "od": float(od), "oa": float(oa),
            "o_o25": float(odd_o25), "o_btts": float(odd_btts)
        }

        st.markdown("#### 🧠 Análisis de Valor")
        # Calcular valores esperados
        ev_h = (ph * oh) - 1
        ev_d = (pd_prob * od) - 1
        ev_a = (pa * oa) - 1
        ev_o25 = (po25 * odd_o25) - 1
        ev_btts = (pbtts * odd_btts) - 1
        
        ev_data = {
            f"Gana {home}": ev_h,
            "Empate": ev_d,
            f"Gana {away}": ev_a,
            "Over 2.5": ev_o25,
            "BTTS Sí": ev_btts
        }
        
        # Mostrar tabla de EVs
        ev_df = pd.DataFrame(list(ev_data.items()), columns=["Mercado", "Value"])
        ev_df["Value %"] = ev_df["Value"] * 100
        ev_df["Recomendación"] = ev_df["Value"].apply(
            lambda x: "✅ BUY" if x > 0.1 else "⚠️ NEUTRAL" if x > 0 else "❌ AVOID"
        )
        
        st.dataframe(ev_df.style.format({"Value": "{:.3f}", "Value %": "{:.1f}%"}), 
                    use_container_width=True, hide_index=True)

        # Kelly
        st.markdown("#### 🎯 Kelly Criterion")
        best_ev_market = max(ev_data, key=ev_data.get)
        best_ev_value = ev_data[best_ev_market]
        
        if best_ev_value > 0:
            if "Gana" in best_ev_market:
                if home in best_ev_market:
                    k_prob, k_odd = ph, oh
                else:
                    k_prob, k_odd = pa, oa
            elif "Empate" in best_ev_market:
                k_prob, k_odd = pd_prob, od
            elif "Over 2.5" in best_ev_market:
                k_prob, k_odd = po25, odd_o25
            else:  # BTTS
                k_prob, k_odd = pbtts, odd_btts
            
            k_pct = calculate_kelly(k_prob, k_odd, bank, kelly_fraction)
            k_stake = (k_pct / 100) * bank
            
            st.success(f"""
            **💎 Recomendación Kelly:** {best_ev_market}
            - **Value:** {best_ev_value*100:.1f}%
            - **Stake:** ${k_stake:.2f} ({k_pct:.2f}% del bank)
            - **Retorno esperado:** ${k_stake * best_ev_value:.2f}
            """)
        else:
            st.warning("📉 **No hay valor positivo identificado** - Kelly sugiere NO apostar")

        st.divider()
        st.markdown("### ➕ Agregar al Ticket")
        with st.form("add_to_ticket"):
            sel_pick_options = [
                f"Gana {home}", 
                "Empate", 
                f"Gana {away}",
                "Over 2.5 Goles",
                "BTTS (Ambos Anotan)"
            ]
            sel_pick = st.selectbox("Selección", sel_pick_options)
            
            # Asignar cuota y probabilidad
            if f"Gana {home}" in sel_pick: 
                sel_odd, sel_prob = oh, ph
            elif "Empate" in sel_pick: 
                sel_odd, sel_prob = od, pd_prob
            elif f"Gana {away}" in sel_pick: 
                sel_odd, sel_prob = oa, pa
            elif "Over 2.5" in sel_pick:
                sel_odd, sel_prob = odd_o25, po25
            elif "BTTS" in sel_pick:
                sel_odd, sel_prob = odd_btts, pbtts
            else:
                sel_odd, sel_prob = 1.0, 0.0
            
            sel_ev = (sel_prob * sel_odd) - 1
            
            col_stake1, col_stake2 = st.columns(2)
            with col_stake1:
                stake_type = st.radio("Tipo de stake", ["Kelly", "Fijo", "% Bank"])
            with col_stake2:
                if stake_type == "Kelly":
                    stake_amount = calculate_kelly(sel_prob, sel_odd, bank, kelly_fraction)
                    stake_amount = (stake_amount / 100) * bank
                elif stake_type == "Fijo":
                    stake_amount = st.number_input("Stake fijo ($)", 1.0, bank, 10.0, step=5.0)
                else:
                    stake_pct = st.slider("% Bank", 0.5, 10.0, 1.0, step=0.5)
                    stake_amount = (stake_pct / 100) * bank
            
            if st.form_submit_button("🎫 Añadir selección"):
                st.session_state.ticket.append({
                    "match": f"{home} vs {away}",
                    "pick": sel_pick,
                    "odd": sel_odd,
                    "prob": sel_prob,
                    "ev": sel_ev,
                    "stake": stake_amount,
                    "league": leagues[code],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.success("✅ Añadido al ticket")
                st.rerun()

    with col_ticket:
        st.markdown("### 🎫 Ticket")
        if not st.session_state.ticket:
            st.info("Ticket vacío")
        else:
            total_odd, total_prob, total_stake = 1.0, 1.0, 0.0
            total_ev = 0.0
            
            for idx, item in enumerate(st.session_state.ticket):
                st.markdown(
                    f"""
                    <div class='ticket-box'>
                        <small>{item['league']} • {item['timestamp']}</small><br>
                        <strong>{item['match']}</strong><br>
                        {item['pick']} @ {item['odd']:.2f}<br>
                        <small>Prob: {item['prob']*100:.1f}% • EV: {item['ev']*100:.1f}% • Stake: ${item['stake']:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                if st.button("❌", key=f"del_{idx}"):
                    st.session_state.ticket.pop(idx)
                    st.rerun()
                
                total_odd *= item["odd"]
                total_prob *= item["prob"]
                total_stake += item["stake"]
                total_ev += item["ev"]

            st.divider()
            st.metric("Cuota Total", f"{total_odd:.2f}")
            st.metric("Probabilidad Total", f"{total_prob*100:.1f}%")
            st.metric("Stake Total", f"${total_stake:.2f}")
            
            # Calcular retornos esperados
            win_amount = (total_stake * total_odd) - total_stake
            expected_value = total_stake * (total_ev / len(st.session_state.ticket) if st.session_state.ticket else 0)
            
            col_ret1, col_ret2 = st.columns(2)
            with col_ret1:
                st.metric("Ganancia potencial", f"${win_amount:.2f}")
            with col_ret2:
                st.metric("Value esperado", f"${expected_value:.2f}")

            if st.button("💾 Guardar Apuesta"):
                tipo_str = "Simple" if len(st.session_state.ticket) == 1 else "Parlay"
                match_str = st.session_state.ticket[0]["match"] if len(st.session_state.ticket) == 1 else f"Combinada ({len(st.session_state.ticket)})"
                pick_str = " + ".join([i["pick"] for i in st.session_state.ticket])

                manage_bets("save", {
                    "ID": pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f"),
                    "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "Liga": tipo_str,
                    "Partido": match_str,
                    "Pick": pick_str,
                    "Cuota": round(total_odd, 2),
                    "Stake": total_stake,
                    "Prob": round(total_prob, 4),
                    "EV": round(total_ev / max(1, len(st.session_state.ticket)), 4),
                    "Estado": "Pendiente",
                    "Ganancia": 0.0
                })
                st.session_state.ticket = []
                st.balloons()
                st.rerun()

# --- TAB 3: HISTORIAL ---
with t3:
    st.markdown("### 📜 Historial de Apuestas")
    db = manage_bets("load")
    
    if not db.empty:
        # Resumen general
        st.markdown("#### 📊 Resumen General")
        total_bets = len(db)
        completed_bets = db[db["Estado"].isin(["Ganada", "Perdida", "Push"])]
        pending_bets = db[db["Estado"] == "Pendiente"]
        
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        with col_sum1:
            st.metric("Total Apuestas", total_bets)
        with col_sum2:
            if len(completed_bets) > 0:
                win_rate = len(completed_bets[completed_bets["Estado"] == "Ganada"]) / len(completed_bets) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.metric("Win Rate", "0%")
        with col_sum3:
            total_profit = completed_bets["Ganancia"].sum()
            st.metric("Profit Total", f"${total_profit:.2f}")
        with col_sum4:
            if len(pending_bets) > 0:
                st.metric("Pendientes", len(pending_bets))
            else:
                st.metric("Pendientes", 0)

        # Tabla principal
        st.divider()
        st.markdown("#### 📋 Detalle de Apuestas")
        db_display = db.copy()
        db_display["Ganancia"] = db_display["Ganancia"].apply(lambda x: f"${x:.2f}")
        db_display["Cuota"] = db_display["Cuota"].apply(lambda x: f"{x:.2f}")
        
        # Color según estado
        def color_state(val):
            if val == "Ganada":
                return "background-color: #d4edda; color: #155724;"
            elif val == "Perdida":
                return "background-color: #f8d7da; color: #721c24;"
            elif val == "Pendiente":
                return "background-color: #fff3cd; color: #856404;"
            else:
                return ""
        
        st.dataframe(
            db_display.style.applymap(color_state, subset=["Estado"]),
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        st.markdown("### 🛠️ Administrar Apuestas")
        
        # Crear lista de opciones legibles
        db["Display"] = db.apply(lambda x: f"{x['ID']} | {x['Fecha']} | {x['Partido']} | {x['Pick']}", axis=1)
        opciones_apuestas = db["Display"].tolist()
        seleccion_str = st.selectbox("Selecciona la apuesta a editar:", ["-- Seleccionar --"] + opciones_apuestas)
        
        if seleccion_str != "-- Seleccionar --":
            bet_id = seleccion_str.split(" | ")[0]
            fila = db[db["ID"].astype(str) == bet_id].iloc[0]
            
            st.info(f"**Seleccionado:** {fila['Partido']} - {fila['Pick']} (Cuota: {fila['Cuota']}, Stake: ${fila['Stake']})")
            
            col_edit1, col_edit2 = st.columns(2)
            
            with col_edit1:
                nuevo_estado = st.selectbox(
                    "Actualizar Estado:", 
                    ["Pendiente", "Ganada", "Perdida", "Push"],
                    index=["Pendiente", "Ganada", "Perdida", "Push"].index(fila["Estado"]) if fila["Estado"] in ["Pendiente", "Ganada", "Perdida", "Push"] else 0
                )
                if st.button("💾 Actualizar Estado"):
                    manage_bets("update", id_bet=bet_id, status=nuevo_estado)
                    st.success(f"Apuesta {bet_id} actualizada a {nuevo_estado}.")
                    st.rerun()
            
            with col_edit2:
                st.markdown("**Zona de peligro**")
                if st.button("🗑️ Eliminar definitivamente", type="primary"):
                    manage_bets("delete", id_bet=bet_id)
                    st.warning(f"Apuesta {bet_id} eliminada.")
                    st.rerun()

    else:
        st.warning("Aún no hay historial de apuestas.")

# --- TAB 4: ESCÁNER + JORNADA ML ---
with t4:
    st.markdown("## 💎 Escáner Seguro")

    api_league_map = {
        "SP1": "soccer_spain_la_liga",
        "E0": "soccer_epl",
        "I1": "soccer_italy_serie_a",
        "D1": "soccer_germany_bundesliga",
        "F1": "soccer_france_ligue_one",
        "N1": "soccer_netherlands_eredivisie",
        "P1": "soccer_portugal_primeira_liga",
    }

    api_key_input = st.text_input("🔑 API Key:", value=st.session_state.api_key, type="password")
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    if st.session_state.api_key:
        sport_key = api_league_map.get(code)
        
        if sport_key:
            # Controles del escáner
            col_scan1, col_scan2 = st.columns(2)
            with col_scan1:
                hours_ahead = st.slider("Horas hacia adelante", 1, 168, 72, help="Partidos en las próximas X horas")
            with col_scan2:
                min_odds_quality = st.slider("Calidad mínima odds", 1.5, 10.0, 1.8, step=0.1, 
                                           help="Filtrar odds muy bajas")

            if st.button("⬇️ Descargar/Actualizar Datos API"):
                with st.spinner("Conectando a API..."):
                    resp = call_api_real(sport_key, st.session_state.api_key)
                    if resp["success"]:
                        st.session_state.market_storage[code] = {
                            "timestamp": datetime.now(), 
                            "data": resp["data"]
                        }
                        st.session_state.api_usage["used"] = resp["used"]
                        st.session_state.api_usage["remaining"] = resp["remaining"]
                        st.success(f"✅ {len(resp['data'])} partidos descargados")
                        st.info(f"📊 API Usage: {resp['used']} usadas, {resp['remaining']} restantes")
                        st.rerun()
                    else:
                        st.error(f"Error API: {resp.get('message', 'Unknown error')}")

            # Mostrar datos almacenados
            if code in st.session_state.market_storage:
                stored = st.session_state.market_storage[code]
                data_to_display = stored.get("data", [])
                st.info(f"📂 Datos en memoria ({len(data_to_display)} partidos). Actualizado: {stored['timestamp'].strftime('%H:%M:%S')}")
                
                if data_to_display:
                    # Análisis Dixon-Coles de partidos próximos
                    now_utc = pd.Timestamp.now(tz="UTC")
                    live_rows = []
                    
                    for item in data_to_display:
                        match_date = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
                        if pd.isna(match_date):
                            continue
                        
                        diff_hours = (match_date - now_utc).total_seconds() / 3600
                        if diff_hours > hours_ahead or diff_hours < -2:
                            continue

                        h_api = normalize_name(item.get("home_team", ""))
                        a_api = normalize_name(item.get("away_team", ""))
                        
                        m_h = get_close_matches(h_api, teams, n=1, cutoff=0.7)
                        m_a = get_close_matches(a_api, teams, n=1, cutoff=0.7)

                        if not m_h or not m_a:
                            continue
                        
                        h = m_h[0]
                        a = m_a[0]
                        
                        if h not in stats or a not in stats:
                            continue

                        oh2, od2, oa2 = match_odds_from_scanner_item(item)
                        if np.isnan(oh2) or np.isnan(od2) or np.isnan(oa2):
                            continue
                        
                        # Filtrar por calidad de odds
                        if min(oh2, od2, oa2) < min_odds_quality:
                            continue

                        # Predicción Dixon-Coles
                        h_exp_dc, a_exp_dc, ph2, pd2, pa2, po15_2, po25_2, pbtts_2, _, _ = predict_match_dixon_coles(
                            h, a, stats, ah, aa, rho=st.session_state.model_params["rho"]
                        )

                        # Calcular Value
                        ev_h = (ph2 * oh2) - 1
                        ev_d = (pd2 * od2) - 1
                        ev_a = (pa2 * oa2) - 1
                        
                        best_ev = max(ev_h, ev_d, ev_a)
                        pick = "No Bet"
                        ev_value = 0.0
                        
                        if best_ev > 0.05:
                            if best_ev == ev_h:
                                pick = f"Gana {h}"
                                ev_value = ev_h
                            elif best_ev == ev_d:
                                pick = "Empate"
                                ev_value = ev_d
                            else:
                                pick = f"Gana {a}"
                                ev_value = ev_a

                        live_rows.append({
                            "Hora (UTC)": match_date.strftime("%d/%m %H:%M"),
                            "Partido": f"{h} vs {a}",
                            "xG": f"{h_exp_dc:.1f}-{a_exp_dc:.1f}",
                            "Cuotas": f"{oh2:.2f}/{od2:.2f}/{oa2:.2f}",
                            "DC %": f"{ph2*100:.0f}/{pd2*100:.0f}/{pa2*100:.0f}",
                            "Value": ev_value,  # CORREGIDO: Guardar como número
                            "Pick": pick,
                            "O2.5%": f"{po25_2*100:.0f}",
                            "BTTS%": f"{pbtts_2*100:.0f}"
                        })
                    
                    if live_rows:
                        # CORRECCIÓN: Ordenar por Value numérico
                        df_live = pd.DataFrame(live_rows)
                        df_live = df_live.sort_values("Value", ascending=False)
                        
                        # Convertir Value a string para display
                        df_live["Value"] = df_live["Value"].apply(lambda x: f"{x*100:.1f}%")
                        
                        st.markdown(f"### 🎯 Oportunidades ({len(df_live)} partidos) - Ordenado por Value")
                        
                        # Color por value
                        def highlight_value(val):
                            try:
                                value_pct = float(val.strip('%'))
                                if value_pct > 10:
                                    return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                                elif value_pct > 5:
                                    return 'background-color: #fff3cd; color: #856404;'
                                else:
                                    return ''
                            except:
                                return ''
                        
                        st.dataframe(
                            df_live.style.applymap(highlight_value, subset=["Value"]),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Botón de descarga
                        csv_live = df_live.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Descargar oportunidades",
                            data=csv_live,
                            file_name=f"oportunidades_{code}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No se encontraron partidos con valor positivo en el rango de tiempo seleccionado.")
            else:
                st.warning("No hay datos descargados para esta liga. Usa el botón 'Descargar/Actualizar' primero.")

        st.divider()
        st.markdown("## 😁 Jornada ML (desde Escáner)")
        
        if code in st.session_state.market_storage and "data" in st.session_state.market_storage[code]:
            stored = st.session_state.market_storage[code]
            data_to_display = stored.get("data", [])
            
            if data_to_display:
                st.caption("Entrena un modelo ML con datos históricos y predice todos los partidos de la jornada.")
                
                col_jornada1, col_jornada2 = st.columns(2)
                with col_jornada1:
                    window_ml_j = st.slider("Ventana entrenamiento", 300, 2000, 600, step=100)
                with col_jornada2:
                    min_ev_j = st.slider("EV mínimo para mostrar", 0.0, 0.2, 0.05, step=0.01)

                if st.button("📌 Generar Pronósticos Jornada ML"):
                    with st.spinner("Entrenando modelo y analizando partidos..."):
                        snap = train_snapshot_cached(
                            df, 
                            window_matches=window_ml_j, 
                            seed=42,
                            alpha=st.session_state.model_params["alpha"]
                        )

                    if snap is None:
                        st.warning("No se pudo entrenar el modelo (datos insuficientes).")
                    else:
                        model, team_stats2, avg_h2, avg_a2 = snap
                        now_utc = pd.Timestamp.now(tz="UTC")
                        rows = []

                        for item in data_to_display:
                            match_date = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
                            if pd.isna(match_date):
                                continue
                            
                            diff_hours = (match_date - now_utc).total_seconds() / 3600
                            if diff_hours > 168 or diff_hours < -2:
                                continue

                            h_api = normalize_name(item.get("home_team", ""))
                            a_api = normalize_name(item.get("away_team", ""))

                            m_h = get_close_matches(h_api, teams, n=1, cutoff=0.7)
                            m_a = get_close_matches(a_api, teams, n=1, cutoff=0.7)
                            
                            if not m_h or not m_a:
                                continue
                            
                            h = m_h[0]
                            a = m_a[0]
                            
                            if h not in team_stats2 or a not in team_stats2:
                                continue

                            oh2, od2, oa2 = match_odds_from_scanner_item(item)
                            if np.isnan(oh2) or np.isnan(od2) or np.isnan(oa2) or oh2 <= 1.01 or od2 <= 1.01 or oa2 <= 1.01:
                                continue

                            # Predicción ML
                            p, (ev_h, ev_d, ev_a), pick = predict_ml_for_match(
                                h, a, float(oh2), float(od2), float(oa2),
                                model, team_stats2, avg_h2, avg_a2
                            )
                            
                            best_ev = np.nanmax([ev_h, ev_d, ev_a])
                            if best_ev < min_ev_j:
                                continue

                            rows.append({
                                "Hora (UTC)": match_date.strftime("%d/%m %H:%M"),
                                "Partido": f"{h} vs {a}",
                                "Cuotas": f"{oh2:.2f}|{od2:.2f}|{oa2:.2f}",
                                "ML %": f"{p[0]*100:.0f}|{p[1]*100:.0f}|{p[2]*100:.0f}",
                                "EV_H": ev_h,
                                "EV_D": ev_d,
                                "EV_A": ev_a,
                                "Mejor EV": best_ev,
                                "Pick": pick,
                                "Confianza": np.max(p) * 100
                            })

                        if not rows:
                            st.info("No se encontraron partidos con valor suficiente.")
                        else:
                            out_df = pd.DataFrame(rows).sort_values("Mejor EV", ascending=False).reset_index(drop=True)
                            st.success(f"✅ Jornada generada: {len(out_df)} partidos con valor")
                            
                            # Formatear DataFrame
                            def color_ev(val):
                                if val > 0.15:
                                    return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                                elif val > 0.05:
                                    return 'background-color: #fff3cd; color: #856404;'
                                else:
                                    return ''
                            
                            styled_df = out_df.style.format({
                                "EV_H": "{:.3f}",
                                "EV_D": "{:.3f}", 
                                "EV_A": "{:.3f}",
                                "Mejor EV": "{:.3f}",
                                "Confianza": "{:.1f}%"
                            }).applymap(color_ev, subset=["Mejor EV"])
                            
                            st.dataframe(styled_df, use_container_width=True, height=400)
                            
                            # Botón de descarga
                            st.download_button(
                                "📥 Descargar jornada (CSV)",
                                data=out_df.to_csv(index=False).encode("utf-8"),
                                file_name=f"jornada_ml_{code}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
            else:
                st.info("Primero descarga datos del escáner usando el botón superior.")
        else:
            st.warning("No hay datos de escáner disponibles para esta liga.")
    else:
        st.info("🔑 Ingresa tu API Key de The Odds API para usar el escáner.")

# --- TAB 5: LABORATORIO MEJORADO ---
with t5:
    st.markdown("## 🧪 Laboratorio - Diagnóstico y Calibración")
    
    tab_lab1, tab_lab2 = st.tabs(["🔧 Backtesting", "📊 Calibración"])
    
    with tab_lab1:
        st.markdown("### 🔧 Backtesting Avanzado")
        
        col_lab1, col_lab2 = st.columns(2)
        with col_lab1:
            n_test = st.slider("Partidos a evaluar", 20, 300, 100, step=10)
            min_train = st.slider("Mínimo entrenamiento", 50, 500, 150, step=25)
            stake_unit = st.number_input("Stake unit", 0.5, 10.0, 1.0, step=0.5)
        
        with col_lab2:
            min_ev_backtest = st.slider("EV mínimo", 0.0, 0.3, 0.08, step=0.01, 
                                       help="EV mínimo para apostar. Más alto = menos apuestas pero más calidad")
            min_prob_threshold = st.slider("Probabilidad mínima", 0.2, 0.6, 0.35, step=0.05,
                                         help="Probabilidad mínima del resultado elegido")
            use_prob_cap = st.checkbox("Limitar probabilidades máximas", value=True,
                                     help="Evita sobreconfianza limitando probabilidades")
        
        col_params1, col_params2 = st.columns(2)
        with col_params1:
            st.markdown("#### ⚙️ Parámetros Dixon-Coles (backtest)")
            alpha_bt = st.slider("Alpha (backtest)", 0.001, 0.02, st.session_state.model_params["alpha"], 0.001)
            rho_bt = st.slider("Rho (backtest)", -0.3, 0.3, st.session_state.model_params["rho"], 0.01)
        
        with col_params2:
            st.markdown("#### 📈 Filtros adicionales")
            max_odds = st.number_input("Odds máxima permitida", 1.5, 50.0, 10.0, 0.5,
                                     help="Filtrar odds muy altas (mayor riesgo)")
            min_odds = st.number_input("Odds mínima", 1.01, 3.0, 1.3, 0.05,
                                     help="Filtrar odds muy bajas (poco valor)")
        
        if st.button("▶️ Ejecutar Backtesting (Diagnóstico Completo)", type="primary"):
            with st.spinner(f"Analizando {n_test} partidos..."):
                test_df, ok, profit, roi_bt, n_bets, tot_stake, calib_info, avg_ev_real = run_backtest_no_leak(
                    df, 
                    n_test=n_test, 
                    min_train=min_train, 
                    window_matches=st.session_state.model_params["window_matches"],
                    stake_unit=stake_unit,
                    alpha=alpha_bt,
                    rho=rho_bt,
                    min_ev_threshold=min_ev_backtest,
                    min_prob_threshold=min_prob_threshold,
                    use_probability_cap=use_prob_cap
                )
            
            if test_df.empty or n_bets == 0:
                st.warning(f"No se encontraron apuestas con los criterios actuales.")
                st.info(f"**Sugerencias:** Disminuir EV mínimo o probabilidad mínima")
            else:
                st.markdown("### 📊 Resultados Backtesting")
                
                # Métricas principales
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                with col_res1:
                    st.metric("Apuestas", n_bets)
                with col_res2:
                    win_rate = (ok / n_bets) * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%", 
                             delta=f"{(win_rate - 33.3):+.1f}%" if win_rate != 33.3 else None,
                             delta_color="normal" if win_rate > 33.3 else "inverse")
                with col_res3:
                    st.metric("Profit", f"{profit:.2f} U", 
                             delta_color="normal" if profit > 0 else "inverse")
                with col_res4:
                    st.metric("ROI", f"{roi_bt:.2f}%", 
                             delta_color="normal" if roi_bt > 0 else "inverse")
                
                # Métricas secundarias
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    avg_odds = test_df["Cuota"].mean() if not test_df.empty else 0
                    st.metric("Avg Odds", f"{avg_odds:.2f}")
                with col_met2:
                    expectancy = profit / n_bets if n_bets > 0 else 0
                    st.metric("Expectancy/U", f"{expectancy:.3f}")
                with col_met3:
                    st.metric("Stake Total", f"{tot_stake:.2f} U")
                with col_met4:
                    st.metric("EV Promedio", f"{avg_ev_real:.1f}%")
                
                # Análisis de calibración
                st.divider()
                st.markdown("#### 🎯 Diagnóstico de Calibración")
                
                if calib_info:
                    if "sobreestima" in calib_info:
                        st.error(f"**{calib_info}**")
                        st.warning("""
                        **Problema:** El modelo está sobreestimando probabilidades.
                        
                        **Soluciones:**
                        1. ✅ Activar 'Limitar probabilidades máximas'
                        2. ⬆️ Aumentar 'Alpha' (da más peso a partidos recientes)
                        3. ⬇️ Disminuir 'Ventana de análisis'
                        4. ⬆️ Aumentar 'EV mínimo' para filtrar apuestas dudosas
                        """)
                    elif "subestima" in calib_info:
                        st.warning(f"**{calib_info}**")
                    else:
                        st.success(f"**{calib_info}**")
                
                # Distribución de resultados
                if not test_df.empty:
                    wins = len(test_df[test_df["Res"] == "✅"])
                    losses = len(test_df[test_df["Res"] == "❌"])
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Bar(
                        x=['Ganadas', 'Perdidas'],
                        y=[wins, losses],
                        text=[wins, losses],
                        textposition='auto',
                        marker_color=['#00ff00', '#ff4444']
                    ))
                    fig_dist.update_layout(
                        title='Distribución de Resultados',
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Equity curve
                if not test_df.empty and len(test_df) > 1:
                    test_df["Equity"] = test_df["P/L(U)"].cumsum()
                    
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(
                        x=list(range(len(test_df))),
                        y=test_df["Equity"],
                        mode='lines',
                        name='Equity',
                        line=dict(color='#00ff00', width=2)
                    ))
                    fig_equity.update_layout(
                        title='Curva de Equity',
                        xaxis_title='Número de Apuesta',
                        yaxis_title='Equity (U)',
                        height=400,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_equity, use_container_width=True)
                
                st.divider()
                st.markdown("#### 📋 Detalle de Apuestas")
                st.dataframe(test_df, use_container_width=True, height=300)
                
                # Exportar resultados
                csv = test_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Descargar resultados",
                    data=csv,
                    file_name=f"backtest_{code}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    with tab_lab2:
        st.markdown("### 📊 Análisis de Calibración")
        st.caption("Evalúa si las probabilidades predichas coinciden con las frecuencias reales")
        
        if st.button("📈 Ejecutar Análisis de Calibración"):
            with st.spinner("Evaluando calibración del modelo..."):
                calib_df = evaluate_calibration(
                    df,
                    n_test=150,
                    window_matches=st.session_state.model_params["window_matches"],
                    alpha=st.session_state.model_params["alpha"],
                    rho=st.session_state.model_params["rho"]
                )
            
            if calib_df is None or calib_df.empty:
                st.warning("No hay suficientes datos para el análisis de calibración.")
            else:
                st.dataframe(calib_df.style.format({
                    "Prob. Esperada": "{:.2f}",
                    "Tasa Acierto": "{:.2f}",
                    "Diferencia": "{:.3f}"
                }), use_container_width=True)
                
                # Gráfico de calibración
                fig_calib = go.Figure()
                fig_calib.add_trace(go.Scatter(
                    x=calib_df["Prob. Esperada"],
                    y=calib_df["Tasa Acierto"],
                    mode='markers+text',
                    text=calib_df["Muestras"].astype(str),
                    textposition="top center",
                    marker=dict(size=calib_df["Muestras"]/calib_df["Muestras"].max()*50 + 10,
                              color=calib_df["Diferencia"],
                              colorscale='RdBu',
                              showscale=True,
                              colorbar=dict(title="Sobre/Sub"))
                ))
                
                # Línea de calibración perfecta
                fig_calib.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='Calibración perfecta'
                ))
                
                fig_calib.update_layout(
                    title='Gráfico de Calibración',
                    xaxis_title='Probabilidad Predicha',
                    yaxis_title='Tasa de Acierto Real',
                    height=500,
                    template="plotly_dark"
                )
                st.plotly_chart(fig_calib, use_container_width=True)
                
                # Interpretación
                avg_error = calib_df["Diferencia"].abs().mean()
                if avg_error > 0.1:
                    st.error(f"**Problema de calibración grave:** Error promedio de {avg_error:.3f}")
                    st.markdown("""
                    **El modelo necesita recalibración:**
                    1. **Activar 'Limitar probabilidades máximas'** en backtesting
                    2. **Ajustar parámetros del modelo:**
                       - ⬆️ **Alpha**: Da más peso a partidos recientes
                       - ⬇️ **Ventana**: Usa menos partidos históricos
                       - 🔄 **Mix factor**: Ajusta balance entre estadísticas específicas y globales
                    3. **Considerar transformación logística** de las probabilidades
                    """)
                elif avg_error > 0.05:
                    st.warning(f"**Calibración aceptable:** Error promedio de {avg_error:.3f}")
                else:
                    st.success(f"**Excelente calibración:** Error promedio de {avg_error:.3f}")
# --- TAB 6: RENDIMIENTO ---
with t6:
    st.markdown("## 📈 Rendimiento - Risk Management")
    
    if os.path.exists(CSV_FILE):
        df_hist = pd.read_csv(CSV_FILE)
        df_finished = df_hist[df_hist["Estado"].isin(["Ganada", "Perdida", "Push"])].copy()
        
        if df_finished.empty:
            st.info("No hay apuestas finalizadas para analizar.")
        else:
            # Ordenar por fecha
            df_finished = df_finished.sort_values("Fecha")
            df_finished["Fecha_dt"] = pd.to_datetime(df_finished["Fecha"])
            
            # Cálculos básicos
            tot_inv = df_finished["Stake"].sum()
            tot_prof = df_finished["Ganancia"].sum()
            roi = (tot_prof / tot_inv * 100) if tot_inv > 0 else 0
            
            # Equity curve
            df_finished["Equity"] = df_finished["Ganancia"].cumsum()
            df_finished["Peak"] = df_finished["Equity"].cummax()
            df_finished["Drawdown"] = df_finished["Equity"] - df_finished["Peak"]
            max_dd = df_finished["Drawdown"].min()
            max_dd_pct = (max_dd / df_finished["Peak"].max() * 100) if df_finished["Peak"].max() > 0 else 0
            
            # Métricas
            wins = len(df_finished[df_finished["Ganancia"] > 0])
            losses = len(df_finished[df_finished["Ganancia"] < 0])
            pushes = len(df_finished[df_finished["Ganancia"] == 0])
            
            avg_win = df_finished[df_finished["Ganancia"] > 0]["Ganancia"].mean() if wins > 0 else 0
            avg_loss = df_finished[df_finished["Ganancia"] < 0]["Ganancia"].mean() if losses > 0 else 0
            win_rate = (wins / len(df_finished)) * 100 if len(df_finished) > 0 else 0
            
            # Profit Factor
            gross_profit = df_finished[df_finished["Ganancia"] > 0]["Ganancia"].sum()
            gross_loss = abs(df_finished[df_finished["Ganancia"] < 0]["Ganancia"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Sharpe-like ratio (simplificado)
            returns_std = df_finished["Ganancia"].std()
            sharpe_ratio = (df_finished["Ganancia"].mean() / returns_std) if returns_std > 0 else 0
            
            st.markdown("### 📊 Métricas de Desempeño")
            
            # Primera fila de métricas
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            with col_met1:
                st.metric("Beneficio Neto", f"${tot_prof:,.2f}", f"{roi:.1f}% ROI")
            with col_met2:
                st.metric("Win Rate", f"{win_rate:.1f}%", f"{wins}-{losses}-{pushes}")
            with col_met3:
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            with col_met4:
                st.metric("Max Drawdown", f"${max_dd:,.2f}", f"{max_dd_pct:.1f}%")
            
            # Segunda fila
            col_met5, col_met6, col_met7, col_met8 = st.columns(4)
            with col_met5:
                st.metric("Apuestas", len(df_finished))
            with col_met6:
                st.metric("Avg Win", f"${avg_win:.2f}")
            with col_met7:
                st.metric("Avg Loss", f"${avg_loss:.2f}")
            with col_met8:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            st.divider()
            
            # Gráficos
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Equity curve
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=df_finished["Fecha_dt"],
                    y=df_finished["Equity"],
                    mode='lines',
                    name='Equity',
                    line=dict(color='#00ff00', width=2)
                ))
                fig_eq.add_trace(go.Scatter(
                    x=df_finished["Fecha_dt"],
                    y=df_finished["Peak"],
                    mode='lines',
                    name='Peak',
                    line=dict(color='#ff9900', width=1, dash='dash')
                ))
                fig_eq.update_layout(
                    title='Curva de Equity',
                    xaxis_title='Fecha',
                    yaxis_title='Equity ($)',
                    height=400,
                    template="plotly_dark",
                    showlegend=True
                )
                st.plotly_chart(fig_eq, use_container_width=True)
            
            with col_chart2:
                # Drawdown
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Bar(
                    x=df_finished["Fecha_dt"],
                    y=df_finished["Drawdown"],
                    name='Drawdown',
                    marker_color='#ff4444'
                ))
                fig_dd.update_layout(
                    title='Drawdown',
                    xaxis_title='Fecha',
                    yaxis_title='Drawdown ($)',
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig_dd, use_container_width=True)
            
            st.divider()
            st.markdown("#### 📋 Historial Detallado")
            st.dataframe(df_finished, use_container_width=True)
    else:
        st.info("Aún no hay historial de apuestas guardado.")

# --- TAB 7: ML EVALUATION ---
with t7:
    st.markdown("## 🤖 ML 1X2: Ensemble Avanzado")
    
    if HAS_XGB:
        st.success("✅ XGBoost disponible - Usando modelo optimizado")
    else:
        st.warning("⚠️ XGBoost no instalado - Usando RandomForest (instala xgboost para mejor rendimiento)")
    
    st.markdown("### ⚠️ Importante sobre evaluación ML")
    st.warning("""
    **fast_eval_ml()** tiene potential data leakage (usa team_stats calculados una sola vez).  
    **Solo úsalo para diagnóstico rápido.**  
    
    Para métricas reales, usa **strict_walkforward_eval_ml_blocks()** que reentrena en cada bloque.
    """)
    
    st.markdown("### 📊 Evaluación del Modelo")
    
    col_eval1, col_eval2, col_eval3 = st.columns(3)
    with col_eval1:
        n_test_ml = st.slider("Partidos test", 50, 500, 200, step=25)
    with col_eval2:
        min_train_ml = st.slider("Mínimo train", 200, 1000, 300, step=50)
    with col_eval3:
        window_ml = st.slider("Ventana train", 300, 1500, 600, step=100)
    
    eval_mode = st.radio(
        "Modo de evaluación",
        ["⚡ Rápido (1 train + test final)", "🧪 Estricto (walk-forward por bloques)"],
        horizontal=True,
        index=0
    )
    
    if eval_mode == "🧪 Estricto (walk-forward por bloques)":
        col_strict1, col_strict2 = st.columns(2)
        with col_strict1:
            retrain_every = st.slider("Re-entrenar cada K partidos", 1, 50, 10, step=1)
        with col_strict2:
            train_step = st.slider("Submuestreo train", 1, 5, 2, step=1)
    
    if st.button("▶️ Evaluar Modelo ML"):
        # Mostrar ambas evaluaciones pero con advertencias claras
        col_eval_fast, col_eval_strict = st.columns(2)
        
        with col_eval_fast:
            st.markdown("#### ⚡ Rápido (Diagnóstico)")
            with st.spinner("Evaluación rápida..."):
                out_fast = fast_eval_ml(
                    df, 
                    n_test=n_test_ml, 
                    min_train=min_train_ml, 
                    window_matches=window_ml,
                    alpha=st.session_state.model_params["alpha"],
                    rho=st.session_state.model_params["rho"]
                )
            
            if out_fast:
                st.metric("LogLoss (fast)", f"{out_fast['logloss']:.4f}")
                st.metric("Accuracy (fast)", f"{out_fast.get('accuracy', 0)*100:.1f}%")
                st.caption("⚠️ Puede estar optimista por data leakage")
        
        with col_eval_strict:
            st.markdown("#### 🧪 Estricto (Métrica real)")
            with st.spinner("Evaluación walkforward..."):
                if eval_mode == "⚡ Rápido (1 train + test final)":
                    # Usar parámetros por defecto para evaluación estricta
                    out_strict = strict_walkforward_eval_ml_blocks(
                        df, 
                        n_test=n_test_ml, 
                        min_train=min_train_ml, 
                        window_matches=window_ml,
                        retrain_every=10,
                        train_step=2,
                        alpha=st.session_state.model_params["alpha"],
                        rho=st.session_state.model_params["rho"]
                    )
                else:
                    out_strict = strict_walkforward_eval_ml_blocks(
                        df, 
                        n_test=n_test_ml, 
                        min_train=min_train_ml, 
                        window_matches=window_ml,
                        retrain_every=retrain_every,
                        train_step=train_step,
                        alpha=st.session_state.model_params["alpha"],
                        rho=st.session_state.model_params["rho"]
                    )
            
            if out_strict:
                st.metric("LogLoss (strict)", f"{out_strict['logloss']:.4f}")
                st.metric("Accuracy (strict)", f"{out_strict.get('accuracy', 0)*100:.1f}%")
                st.caption("✅ Métrica real sin data leakage")
    
    st.divider()
    st.markdown("### 🎯 Predicción ML para Partido Actual")
    
    # Usar odds actuales
    oh = st.session_state.odds_inputs["oh"]
    od = st.session_state.odds_inputs["od"]
    oa = st.session_state.odds_inputs["oa"]
    
    st.info(f"Usando odds actuales: H={oh:.2f} D={od:.2f} A={oa:.2f}")
    
    if st.button("🧠 Predecir con ML Ensemble"):
        with st.spinner("Entrenando snapshot y prediciendo..."):
            snap = train_snapshot_cached(
                df, 
                window_matches=window_ml, 
                seed=42,
                alpha=st.session_state.model_params["alpha"]
            )
        
        if snap is None:
            st.warning("No se pudo entrenar el modelo.")
        else:
            model, team_stats2, avg_h2, avg_a2 = snap
            
            if home not in team_stats2 or away not in team_stats2:
                st.warning("Equipos no encontrados en el histórico.")
            else:
                # Construir features
                row_now = {
                    "home": home, "away": away, 
                    "odd_h": oh, "odd_d": od, "odd_a": oa, 
                    "sot_h": 0.0, "sot_a": 0.0
                }
                x_now = build_features_for_match(row_now, team_stats2, avg_h2, avg_a2).reshape(1, -1)
                
                # Predecir
                p = model.predict_proba(x_now)[0]
                
                # Calcular EVs
                ev_h = (p[0] * oh) - 1
                ev_d = (p[1] * od) - 1
                ev_a = (p[2] * oa) - 1
                
                st.markdown("#### 📊 Probabilidades por Modelo")
                
                # Comparativa entre modelos
                comp_data = {
                    "Modelo": ["Mercado (sin margen)", "Dixon-Coles", "ML Ensemble"],
                    f"Gana {home}": [odds_to_probs(oh, od, oa)[0], ph, p[0]],
                    "Empate": [odds_to_probs(oh, od, oa)[1], pd_prob, p[1]],
                    f"Gana {away}": [odds_to_probs(oh, od, oa)[2], pa, p[2]]
                }
                
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(
                    comp_df.style.format({
                        f"Gana {home}": "{:.3f}",
                        "Empate": "{:.3f}",
                        f"Gana {away}": "{:.3f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Recomendación
                best_model_ev = max(ev_h, ev_d, ev_a)
                if best_model_ev > 0:
                    if best_model_ev == ev_h:
                        rec = f"✅ Gana {home} (EV: {ev_h*100:.1f}%)"
                    elif best_model_ev == ev_d:
                        rec = f"✅ Empate (EV: {ev_d*100:.1f}%)"
                    else:
                        rec = f"✅ Gana {away} (EV: {ev_a*100:.1f}%)"
                    st.success(f"**Recomendación ML:** {rec}")
                else:
                    st.warning("**Recomendación ML:** No Bet (sin valor positivo)")
                
                # Feature Importance si está disponible
                if hasattr(model, "feature_importances_"):
                    st.markdown("#### 🔍 Importancia de Variables")
                    
                    feature_names = [
                        "Mkt Home", "Mkt Draw", "Mkt Away",
                        "DC Home", "DC Draw", "DC Away",
                        "xG Home", "xG Away", 
                        "Diff xG",
                        "SOT Home", "SOT Away",
                        "log(Matches Home)", "log(Matches Away)",
                        "DC-Mkt Home", "DC-Mkt Away"
                    ]
                    
                    if len(model.feature_importances_) == len(feature_names):
                        imp_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": model.feature_importances_
                        }).sort_values("Importance", ascending=True)
                        
                        fig_imp = go.Figure(go.Bar(
                            x=imp_df["Importance"],
                            y=imp_df["Feature"],
                            orientation='h',
                            marker_color='#76b900'
                        ))
                        fig_imp.update_layout(
                            title='Feature Importance',
                            height=500,
                            margin=dict(l=0, r=0, t=30, b=0),
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

# --- TAB 8: MULTI-LIGA ---
with t8:
    st.markdown("## 🌎 Super Escáner: Todas las Ligas")
    st.caption("Analiza simultáneamente las 7 ligas principales usando datos del escáner.")
    
    if not st.session_state.api_key:
        st.warning("🔑 Necesitas una API Key para usar el escáner multi-liga.")
    else:
        # Configuración multi-liga
        col_multi1, col_multi2 = st.columns(2)
        with col_multi1:
            win_multi = st.slider("Ventana entrenamiento", 300, 2000, 600, step=100)
        with col_multi2:
            min_ev_multi = st.slider("EV mínimo", 0.0, 0.2, 0.07, step=0.01)
        
        # Selección de ligas
        st.markdown("### ⚽ Ligas a Analizar")
        selected_leagues = []
        cols_leagues = st.columns(4)
        
        league_names = list(leagues.values())
        league_codes = list(leagues.keys())
        
        for idx, (code_l, name_l) in enumerate(zip(league_codes, league_names)):
            col_idx = idx % 4
            with cols_leagues[col_idx]:
                if st.checkbox(name_l, value=True, key=f"chk_{code_l}"):
                    selected_leagues.append((code_l, name_l))
        
        if st.button("🚀 Ejecutar Análisis Masivo", type="primary"):
            if not selected_leagues:
                st.warning("Selecciona al menos una liga.")
            else:
                master_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_leagues = len(selected_leagues)
                
                for idx, (l_code, l_name) in enumerate(selected_leagues):
                    progress = int(((idx + 1) / total_leagues) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Analizando {l_name}...")
                    
                    # Verificar si hay datos de esta liga
                    if l_code not in st.session_state.market_storage:
                        continue
                    
                    stored = st.session_state.market_storage[l_code]
                    data_api = stored.get("data", [])
                    if not data_api:
                        continue
                    
                    # Cargar datos históricos
                    df_loop = fetch_live_soccer_data(
                        l_code, 
                        n_seasons=N_SEASONS,
                        use_only_current=st.session_state.model_params["use_only_current"]
                    )
                    
                    if df_loop.empty or len(df_loop) < 200:
                        continue
                    
                    # Entrenar modelo para esta liga
                    snap_loop = train_snapshot_cached(
                        df_loop, 
                        window_matches=win_multi, 
                        seed=42,
                        alpha=st.session_state.model_params["alpha"]
                    )
                    
                    if snap_loop is None:
                        continue
                    
                    model_loop, stats_loop, avgh_loop, avga_loop = snap_loop
                    
                    # Analizar partidos
                    now_utc = pd.Timestamp.now(tz="UTC")
                    league_rows = []
                    
                    for item in data_api:
                        match_date = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
                        if pd.isna(match_date):
                            continue
                        
                        diff_hours = (match_date - now_utc).total_seconds() / 3600
                        if diff_hours > 168 or diff_hours < -2:
                            continue

                        h_api = normalize_name(item.get("home_team", ""))
                        a_api = normalize_name(item.get("away_team", ""))
                        
                        teams_loop = sorted(list(set(df_loop["home"].unique()) | set(df_loop["away"].unique())))
                        
                        m_h = get_close_matches(h_api, teams_loop, n=1, cutoff=0.7)
                        m_a = get_close_matches(a_api, teams_loop, n=1, cutoff=0.7)
                        
                        if not m_h or not m_a:
                            continue
                        
                        h_team, a_team = m_h[0], m_a[0]
                        
                        if h_team not in stats_loop or a_team not in stats_loop:
                            continue

                        oh2, od2, oa2 = match_odds_from_scanner_item(item)
                        if np.isnan(oh2) or np.isnan(od2) or np.isnan(oa2) or oh2 <= 1.01:
                            continue

                        # Predicción ML
                        p, (ev_h, ev_d, ev_a), pick = predict_ml_for_match(
                            h_team, a_team, float(oh2), float(od2), float(oa2),
                            model_loop, stats_loop, avgh_loop, avga_loop
                        )
                        
                        best_ev = np.nanmax([ev_h, ev_d, ev_a])
                        if best_ev < min_ev_multi:
                            continue

                        league_rows.append({
                            "Liga": l_name,
                            "Fecha": match_date.strftime("%d/%m %H:%M"),
                            "Partido": f"{h_team} vs {a_team}",
                            "Cuotas": f"{oh2:.2f}|{od2:.2f}|{oa2:.2f}",
                            "ML Prob": f"{p[0]*100:.0f}|{p[1]*100:.0f}|{p[2]*100:.0f}",
                            "EV": best_ev,
                            "Pick": pick,
                            "Confianza": f"{np.max(p)*100:.0f}%"
                        })
                    
                    if league_rows:
                        df_res_league = pd.DataFrame(league_rows).sort_values("EV", ascending=False)
                        master_results.extend(league_rows)
                        
                        with st.expander(f"⚽ {l_name} ({len(league_rows)} picks)", expanded=True):
                            st.dataframe(
                                df_res_league.style.format({"EV": "{:.3f}"}), 
                                use_container_width=True, 
                                hide_index=True
                            )
                
                progress_bar.progress(100)
                status_text.text("✅ Análisis completo")
                
                if master_results:
                    st.divider()
                    st.markdown(f"### 📊 Resumen General: {len(master_results)} picks encontrados")
                    
                    # Resumen por liga
                    summary_df = pd.DataFrame(master_results)
                    if not summary_df.empty:
                        liga_counts = summary_df["Liga"].value_counts()
                        st.bar_chart(liga_counts)
                    
                    # Tabla consolidada
                    st.markdown("### 📋 Todos los Picks")
                    df_master = pd.DataFrame(master_results).sort_values("EV", ascending=False)
                    
                    def color_ev_multi(val):
                        try:
                            if val > 0.15:
                                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                            elif val > 0.07:
                                return 'background-color: #fff3cd; color: #856404;'
                            else:
                                return ''
                        except:
                            return ''
                    
                    st.dataframe(
                        df_master.style.applymap(color_ev_multi, subset=["EV"]),
                        use_container_width=True,
                        height=500
                    )
                    
                    # Botón de descarga
                    st.download_button(
                        "📥 Descargar CSV Completo",
                        data=df_master.to_csv(index=False).encode("utf-8"),
                        file_name=f"super_jornada_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No se encontraron picks con valor suficiente en las ligas seleccionadas.")

# ======================================================
# 10. FOOTER
# ======================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em;">
    <p>🛡️ Dixon-Coles Pro v6.1 • Corregido y optimizado</p>
    <p>✅ Backtesting corregido: Elegir por máximo EV (no por probabilidad)</p>
    <p>⚠️ Disclaimer: Las apuestas deportivas conllevan riesgo. Este es un tool de análisis, no garantía de ganancias.</p>
</div>
""", unsafe_allow_html=True)



