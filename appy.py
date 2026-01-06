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
st.set_page_config(page_title="Dixon-Coles Pro v6.2 (Final)", layout="wide", page_icon="🛡️")
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
# --- NUEVO: ESTADO PARA RESULTADOS BACKTEST ---
if "backtest_results" not in st.session_state: st.session_state.backtest_results = None

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
    df = df.copy().dropna(subset=["date", "home", "away", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    
    if window_matches and window_matches > 0 and len(df) > window_matches:
        df = df.tail(window_matches).reset_index(drop=True)
    
    if df.empty:
        return {}, 0.0, 0.0, []
    
    last_date = pd.to_datetime(ref_date) if ref_date is not None else df["date"].max()
    df["days_ago"] = (last_date - df["date"]).dt.days.clip(lower=0)
    df["weight"] = np.exp(-alpha * df["days_ago"])
    
    # Weight clipping
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
    if home not in team_stats or away not in team_stats:
        return 0, 0, 0, 0, 0, 0, 0, 0, [], np.zeros((1, 1))

    # Calcular expectativas de goles
    h_exp = team_stats[home]["att_h"] * team_stats[away]["def_a"] * avg_h
    a_exp = team_stats[away]["att_a"] * team_stats[home]["def_h"] * avg_a
    
    # Suavizado conservador
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
    
    if len(df_sorted) > 600:
        df_sorted = df_sorted.tail(600).reset_index(drop=True)
    
    test_block = df_sorted.tail(n_test)

    results = []
    correct, bal = 0, 0.0
    n_bets = 0
    total_ev = 0.0
    
    bet_types = {"favorito": 0, "underdog": 0, "empate": 0}
    bet_results = {"favorito": 0, "underdog": 0, "empate": 0}

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

        h_exp, a_exp, ph, pd_prob, pa, *_ = predict_match_dixon_coles(
            row["home"], row["away"], team_stats, avg_h, avg_a, rho=rho
        )

        odd_h = float(row.get("odd_h", np.nan))
        odd_d = float(row.get("odd_d", np.nan))
        odd_a = float(row.get("odd_a", np.nan))
        
        if (np.isnan(odd_h) or odd_h < min_odds or odd_h > max_odds or 
            np.isnan(odd_d) or odd_d < min_odds or odd_d > max_odds or 
            np.isnan(odd_a) or odd_a < min_odds or odd_a > max_odds):
            continue
        
        # Calibración diferencial
        odds_list = [odd_h, odd_d, odd_a]
        probs_raw = [ph, pd_prob, pa]
        
        probs_calibrated = []
        for prob, odd in zip(probs_raw, odds_list):
            if odd > 2.5: 
                calibrated = prob * calibration_factor * 0.9
            elif odd > 1.8:
                calibrated = prob * calibration_factor
            else:
                calibrated = prob * (calibration_factor * 1.1)
            probs_calibrated.append(max(0.05, min(calibrated, 0.85)))
        
        total = sum(probs_calibrated)
        ph_cal, pd_cal, pa_cal = [p/total for p in probs_calibrated]
        
        ev_h = (ph_cal * odd_h) - 1
        ev_d = (pd_cal * odd_d) - 1
        ev_a = (pa_cal * odd_a) - 1
        
        market_probs = [1/odd_h, 1/odd_d, 1/odd_a]
        market_total = sum(market_probs)
        market_h, market_d, market_a = [p/market_total for p in market_probs]
        
        diff_h = ph_cal - market_h
        diff_d = pd_cal - market_d
        diff_a = pa_cal - market_a
        
        min_diff = 0.03
        
        evs = []
        if diff_h > min_diff and ev_h > min_ev_threshold:
            evs.append(("Local", ph_cal, odd_h, ev_h, diff_h))
        if diff_d > min_diff and ev_d > min_ev_threshold:
            evs.append(("Empate", pd_cal, odd_d, ev_d, diff_d))
        if diff_a > min_diff and ev_a > min_ev_threshold:
            evs.append(("Visita", pa_cal, odd_a, ev_a, diff_a))
        
        if not evs:
            continue
        
        best_option = max(evs, key=lambda x: x[4])
        
        pred, prob, odd, ev, diff = best_option
        
        if odd < 2.0:
            bet_type = "favorito"
        elif odd < 3.5:
            bet_type = "medio"
        else:
            bet_type = "underdog"
        
        bet_types[bet_type] = bet_types.get(bet_type, 0) + 1
        
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

        # --- MODIFICADO: GUARDAR VALORES NUMÉRICOS PARA DIAGNÓSTICO ---
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
            "P/L": profit_u,
            # Campos para diagnóstico
            "Prob_Modelo": prob,
            "Resultado_Num": 1 if is_win else 0,
            "Odds_Num": odd
        })
    
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

def find_optimal_parameters(df, n_iterations=50):
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
    _, _, dc_h, dc_d, dc_a, *_ = predict_match_dixon_coles(
        row["home"], row["away"], team_stats, avg_h, avg_a, 
        rho=st.session_state.model_params["rho"]
    )

    oh = float(row.get("odd_h", np.nan))
    od = float(row.get("odd_d", np.nan))
    oa = float(row.get("odd_a", np.nan))
    
    if np.isnan(oh) or oh <= 1.01: oh = 2.5
    if np.isnan(od) or od <= 1.01: od = 3.2
    if np.isnan(oa) or oa <= 1.01: oa = 3.0
    
    mk_h, mk_d, mk_a = odds_to_probs(oh, od, oa)

    h_exp = team_stats[row["home"]]["att_h"] * team_stats[row["away"]]["def_a"] * avg_h
    a_exp = team_stats[row["away"]]["att_a"] * team_stats[row["home"]]["def_h"] * avg_a

    sot_h = float(row.get("sot_h", 0.0))
    sot_a = float(row.get("sot_a", 0.0))

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
    
    with st.expander("🎯 Parámetros del Modelo", expanded=True):
        use_only_current = st.checkbox("Usar solo temporada actual", 
                                     value=st.session_state.model_params["use_only_current"],
                                     help="Mejora resultados usando datos más recientes")
        
        df = fetch_live_soccer_data(code, n_seasons=N_SEASONS, 
                                  use_only_current=use_only_current)
        
        if not df.empty and "season" in df.columns:
            st.caption(f"📊 Temporadas: {sorted(df['season'].unique())}")
            st.caption(f"⚽ Partidos: {len(df)}")
        
        st.session_state.model_params["alpha"] = st.slider(
            "Decaimiento (alpha)", 
            0.001, 0.02, 0.008, 0.001
        )
        
        st.session_state.model_params["rho"] = st.slider(
            "Correlación (rho)", 
            -0.3, 0.3, -0.13, 0.01
        )
        
        st.session_state.model_params["window_matches"] = st.selectbox(
            "Ventana análisis",
            [200, 300, 400, 600, 800, 1200, 0],
            index=2,
            format_func=lambda x: "Todos" if x == 0 else f"{x} partidos"
        )
        
        st.session_state.model_params["mix_factor"] = st.slider(
            "Peso stats específicas", 
            0.0, 1.0, 0.7, 0.1
        )
        
        st.session_state.model_params["use_only_current"] = use_only_current

    if not df.empty:
        stats, ah, aa, teams = calculate_strengths(
            df, 
            ref_date=df["date"].max(), 
            alpha=st.session_state.model_params["alpha"],
            mix_factor=st.session_state.model_params["mix_factor"],
            window_matches=st.session_state.model_params["window_matches"] if st.session_state.model_params["window_matches"] > 0 else None
        )
        
        st.success(f"✅ {len(df)} partidos cargados")
        
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
    
    kelly_fraction = st.slider("Fracción Kelly", 0.1, 1.0, 0.3, 0.1,
                             help="Kelly fraccionario para reducir riesgo")

    if st.session_state.ticket:
        st.divider()
        st.markdown(f"**Ticket:** {len(st.session_state.ticket)} selecciones")
        if st.button("🗑️ Limpiar Ticket"):
            st.session_state.ticket = []
            st.rerun()

st.title(f"🛡️ Dixon-Coles Optimizado: {leagues[code]}")

# --- SELECTOR DE EQUIPOS ---
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])

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
    
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Bar(
        x=['Local', 'Visitante'],
        y=[h_exp, a_exp],
        text=[f'{h_exp:.2f}', f'{a_exp:.2f}'],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e']
    ))
    fig_exp.update_layout(title='Expectativa de Goles', height=300, template="plotly_dark")
    st.plotly_chart(fig_exp, use_container_width=True)

    st.divider()
    st.markdown("### 🏁 ¿Quién gana? (1X2) + Mercados")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"🏠 {home}", f"{ph*100:.1f}%")
    m2.metric("🤝 Empate", f"{pd_prob*100:.1f}%")
    m3.metric(f"✈️ {away}", f"{pa*100:.1f}%")

    st.plotly_chart(plot_score_heatmap(probs, home, away), use_container_width=True)

    st.markdown("### ⭐ Top 3 marcadores más probables")
    if top_sc:
        top_df = pd.DataFrame([{"Marcador": s, "Prob (%)": p*100} for s, p in top_sc])
        st.dataframe(top_df.style.format({"Prob (%)": "{:.2f}"}), 
                    use_container_width=True, hide_index=True)

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

        # Calcular valor
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
        
        ev_df = pd.DataFrame(list(ev_data.items()), columns=["Mercado", "Value"])
        ev_df["Value %"] = ev_df["Value"] * 100
        ev_df["Recomendación"] = ev_df["Value"].apply(
            lambda x: "✅ BUY" if x > 0.1 else "⚠️ NEUTRAL" if x > 0 else "❌ AVOID"
        )
        
        st.dataframe(ev_df.style.format({"Value": "{:.3f}", "Value %": "{:.1f}%"}), 
                    use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### ➕ Agregar al Ticket")
        with st.form("add_to_ticket"):
            sel_pick_options = [
                f"Gana {home}", "Empate", f"Gana {away}",
                "Over 2.5 Goles", "BTTS (Ambos Anotan)"
            ]
            sel_pick = st.selectbox("Selección", sel_pick_options)
            
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
            st.metric("Stake Total", f"${total_stake:.2f}")
            
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
        total_bets = len(db)
        completed_bets = db[db["Estado"].isin(["Ganada", "Perdida", "Push"])]
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
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

        st.divider()
        st.markdown("#### 📋 Detalle")
        
        def color_state(val):
            if val == "Ganada": return "background-color: #d4edda; color: #155724;"
            elif val == "Perdida": return "background-color: #f8d7da; color: #721c24;"
            elif val == "Pendiente": return "background-color: #fff3cd; color: #856404;"
            else: return ""
        
        st.dataframe(db.style.applymap(color_state, subset=["Estado"]), use_container_width=True, hide_index=True)
        
        st.divider()
        st.markdown("### 🛠️ Editar Estado")
        
        db["Display"] = db.apply(lambda x: f"{x['ID']} | {x['Fecha']} | {x['Partido']} | {x['Pick']}", axis=1)
        seleccion_str = st.selectbox("Selecciona apuesta:", ["-- Seleccionar --"] + db["Display"].tolist())
        
        if seleccion_str != "-- Seleccionar --":
            bet_id = seleccion_str.split(" | ")[0]
            fila = db[db["ID"].astype(str) == bet_id].iloc[0]
            
            col_edit1, col_edit2 = st.columns(2)
            with col_edit1:
                nuevo_estado = st.selectbox("Actualizar Estado:", ["Pendiente", "Ganada", "Perdida", "Push"])
                if st.button("💾 Actualizar"):
                    manage_bets("update", id_bet=bet_id, status=nuevo_estado)
                    st.rerun()
            with col_edit2:
                if st.button("🗑️ Eliminar"):
                    manage_bets("delete", id_bet=bet_id)
                    st.rerun()

    else:
        st.warning("Aún no hay historial.")

# --- TAB 4: ESCÁNER ---
with t4:
    st.markdown("## 💎 Escáner Seguro")

    api_league_map = {
        "SP1": "soccer_spain_la_liga", "E0": "soccer_epl", "I1": "soccer_italy_serie_a",
        "D1": "soccer_germany_bundesliga", "F1": "soccer_france_ligue_one",
        "N1": "soccer_netherlands_eredivisie", "P1": "soccer_portugal_primeira_liga",
    }

    api_key_input = st.text_input("🔑 API Key:", value=st.session_state.api_key, type="password")
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    if st.session_state.api_key:
        sport_key = api_league_map.get(code)
        
        if sport_key:
            col_scan1, col_scan2 = st.columns(2)
            with col_scan1:
                hours_ahead = st.slider("Horas hacia adelante", 1, 168, 72)
            with col_scan2:
                min_odds_quality = st.slider("Calidad mínima odds", 1.5, 10.0, 1.8)

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
                        st.rerun()
                    else:
                        st.error(f"Error API: {resp.get('message', 'Unknown error')}")

            if code in st.session_state.market_storage:
                stored = st.session_state.market_storage[code]
                data_to_display = stored.get("data", [])
                st.info(f"📂 Datos en memoria ({len(data_to_display)} partidos). Actualizado: {stored['timestamp'].strftime('%H:%M:%S')}")
                
                if data_to_display:
                    now_utc = pd.Timestamp.now(tz="UTC")
                    live_rows = []
                    
                    for item in data_to_display:
                        match_date = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
                        if pd.isna(match_date): continue
                        
                        diff_hours = (match_date - now_utc).total_seconds() / 3600
                        if diff_hours > hours_ahead or diff_hours < -2: continue

                        h_api = normalize_name(item.get("home_team", ""))
                        a_api = normalize_name(item.get("away_team", ""))
                        
                        m_h = get_close_matches(h_api, teams, n=1, cutoff=0.7)
                        m_a = get_close_matches(a_api, teams, n=1, cutoff=0.7)

                        if not m_h or not m_a: continue
                        h, a = m_h[0], m_a[0]
                        
                        if h not in stats or a not in stats: continue

                        oh2, od2, oa2 = match_odds_from_scanner_item(item)
                        if np.isnan(oh2) or min(oh2, od2, oa2) < min_odds_quality: continue

                        h_exp_dc, a_exp_dc, ph2, pd2, pa2, po15_2, po25_2, pbtts_2, _, _ = predict_match_dixon_coles(
                            h, a, stats, ah, aa, rho=st.session_state.model_params["rho"]
                        )

                        ev_h = (ph2 * oh2) - 1
                        ev_d = (pd2 * od2) - 1
                        ev_a = (pa2 * oa2) - 1
                        
                        best_ev = max(ev_h, ev_d, ev_a)
                        pick = "No Bet"
                        ev_value = 0.0
                        
                        if best_ev > 0.05:
                            if best_ev == ev_h: pick, ev_value = f"Gana {h}", ev_h
                            elif best_ev == ev_d: pick, ev_value = "Empate", ev_d
                            else: pick, ev_value = f"Gana {a}", ev_a

                        live_rows.append({
                            "Hora (UTC)": match_date.strftime("%d/%m %H:%M"),
                            "Partido": f"{h} vs {a}",
                            "xG": f"{h_exp_dc:.1f}-{a_exp_dc:.1f}",
                            "Cuotas": f"{oh2:.2f}/{od2:.2f}/{oa2:.2f}",
                            "Value": ev_value,
                            "Pick": pick,
                            "O2.5%": f"{po25_2*100:.0f}"
                        })
                    
                    if live_rows:
                        df_live = pd.DataFrame(live_rows).sort_values("Value", ascending=False)
                        df_live["Value"] = df_live["Value"].apply(lambda x: f"{x*100:.1f}%")
                        
                        def highlight_value(val):
                            try:
                                if float(val.strip('%')) > 10: return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                                elif float(val.strip('%')) > 5: return 'background-color: #fff3cd; color: #856404;'
                                else: return ''
                            except: return ''
                        
                        st.dataframe(df_live.style.applymap(highlight_value, subset=["Value"]), use_container_width=True, hide_index=True)
                    else:
                        st.info("No se encontraron partidos con valor positivo.")
            else:
                st.warning("No hay datos descargados.")
    else:
        st.info("🔑 Ingresa tu API Key para usar el escáner.")

# --- TAB 5: LABORATORIO CON OPTIMIZACIÓN (CORREGIDO) ---
with t5:
    st.markdown("## 🧪 Laboratorio - Optimización Avanzada")
    
    tab_lab1, tab_lab2, tab_lab3 = st.tabs(["🔧 Backtesting", "⚙️ Optimización", "📊 Diagnóstico Real"])
    
    with tab_lab1:
        st.markdown("### 🔧 Backtesting Inteligente")
        
        col1, col2 = st.columns(2)
        with col1:
            n_test = st.slider("Partidos test", 50, 600, 200, 10)
            min_train = st.slider("Mínimo train", 100, 300, 150, 25)
            stake_unit = st.number_input("Stake unit", 0.5, 5.0, 1.0, 0.5)
        
        with col2:
            min_ev = st.slider("EV mínimo", 0.01, 0.20, 0.08, 0.01)
            calibration_factor = st.slider("Factor calibración", 0.60, 1.00, 0.85, 0.02,
                                         help="<1.0 reduce sobreestimación. Si tu ROI es negativo, BAJA esto.")
            min_diff = st.slider("Diferencia mínima vs mercado", 0.01, 0.10, 0.02, 0.01)
        
        col3, col4 = st.columns(2)
        with col3:
            min_odds = st.number_input("Odds mínima", 1.1, 2.5, 1.5, 0.1)
            max_odds = st.number_input("Odds máxima", 3.0, 10.0, 4.0, 0.5)
        
        with col4:
            st.markdown("#### ⚙️ Parámetros Modelo")
            alpha_bt = st.slider("Alpha", 0.005, 0.015, 0.010, 0.001)
            window_bt = st.selectbox("Ventana", [300, 400, 500, 600], index=1)
        
        if st.button("▶️ Ejecutar Backtesting Calibrado", type="primary"):
            with st.spinner("Ejecutando backtesting con calibración..."):
                test_df, correct, profit, roi, n_bets, tot_stake, distribution, avg_ev, avg_odds = run_backtest_calibrated(
                    df, 
                    n_test=n_test,
                    min_train=min_train,
                    window_matches=window_bt,
                    stake_unit=stake_unit,
                    alpha=alpha_bt,
                    min_ev_threshold=min_ev,
                    calibration_factor=calibration_factor,
                    min_odds=min_odds,
                    max_odds=max_odds
                )
                
                # GUARDAR EN SESSION STATE PARA USAR EN OTRAS TABS
                st.session_state.backtest_results = {
                    "df": test_df,
                    "metrics": (n_bets, correct, profit, roi, avg_odds, avg_ev, tot_stake, distribution)
                }
            
        # MOSTRAR RESULTADOS SI EXISTEN EN MEMORIA
        if st.session_state.backtest_results is not None:
            test_df = st.session_state.backtest_results["df"]
            n_bets, correct, profit, roi, avg_odds, avg_ev, tot_stake, distribution = st.session_state.backtest_results["metrics"]

            if test_df.empty or n_bets == 0:
                st.warning("No se encontraron apuestas con los criterios actuales.")
            else:
                # Métricas principales
                st.markdown("### 📊 Resultados Backtesting")
                
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                with col_res1:
                    st.metric("Apuestas", n_bets)
                with col_res2:
                    win_rate = (correct / n_bets) * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with col_res3:
                    st.metric("ROI", f"{roi:.2f}%", 
                             delta_color="normal" if roi > 0 else "inverse")
                with col_res4:
                    st.metric("Profit", f"{profit:.2f} U")
                
                # Métricas secundarias
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric("Avg Odds", f"{avg_odds:.2f}")
                with col_met2:
                    expectancy = profit / n_bets if n_bets > 0 else 0
                    st.metric("Expectancy", f"{expectancy:.3f}")
                with col_met3:
                    st.metric("EV Promedio", f"{avg_ev:.1f}%")
                with col_met4:
                    st.metric("Stake Total", f"{tot_stake:.2f} U")
                
                st.markdown(f"**Distribución:** {distribution}")
                
                # Gráficos de Equity
                test_df["Equity"] = test_df["P/L"].cumsum()
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=test_df.index, y=test_df["Equity"],
                    mode='lines', name='Equity',
                    line=dict(color='#00ff00' if profit > 0 else '#ff0000', width=2)
                ))
                fig_equity.update_layout(title='Curva de Equity', height=300, template="plotly_dark")
                st.plotly_chart(fig_equity, use_container_width=True)
                
                st.dataframe(test_df[["Fecha", "Partido", "Predicción", "Cuota", "EV", "Res", "P/L"]], use_container_width=True, height=300)

    with tab_lab2:
        st.markdown("### ⚙️ Optimización Automática")
        if st.button("🔍 Buscar Parámetros (Aleatorio)"):
            with st.spinner("Optimizando..."):
                best_params, best_results = find_optimal_parameters(df, n_iterations=20)
            
            if best_params:
                roi_opt, n_bets_opt, avg_odds_opt, profit_opt = best_results
                st.success(f"✅ Mejor ROI encontrado: {roi_opt:.2f}%")
                st.json(best_params)
            else:
                st.warning("No se encontró configuración rentable.")

    with tab_lab3:
        st.markdown("### 📊 Diagnóstico Real del Modelo")
        
        if st.session_state.backtest_results is None or st.session_state.backtest_results["df"].empty:
            st.info("⚠️ Ejecuta el Backtesting en la primera pestaña para ver el diagnóstico.")
        else:
            df_diag = st.session_state.backtest_results["df"].copy()
            
            # Verificar que tengamos las columnas necesarias
            if "Prob_Modelo" not in df_diag.columns:
                st.error("Faltan datos numéricos. Asegúrate de haber actualizado la función 'run_backtest_calibrated' como se indicó.")
            else:
                # 1. Crear rangos de cuotas (Bins)
                bins = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 10.0]
                labels = ['1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-4.0', '4.0+']
                df_diag['Odds_Range'] = pd.cut(df_diag['Odds_Num'], bins=bins, labels=labels)
                
                # 2. Calcular Win Rate Real vs Esperado
                calibration = df_diag.groupby('Odds_Range', observed=True).agg({
                    'Resultado_Num': ['count', 'mean'],  # mean = win rate real
                    'Prob_Modelo': 'mean'               # mean = win rate esperado
                }).reset_index()
                
                # Aplanar columnas
                calibration.columns = ['Rango', 'Apuestas', 'WinRate_Real', 'WinRate_Predicho']
                calibration['Diferencia'] = calibration['WinRate_Predicho'] - calibration['WinRate_Real']
                
                # 3. Visualización
                c_d1, c_d2 = st.columns([2, 1])
                
                with c_d1:
                    fig_calib = go.Figure()
                    # Barra Realidad
                    fig_calib.add_trace(go.Bar(
                        x=calibration['Rango'], y=calibration['WinRate_Real'],
                        name='Realidad (Win %)', marker_color='#00CC96'
                    ))
                    # Línea Modelo
                    fig_calib.add_trace(go.Scatter(
                        x=calibration['Rango'], y=calibration['WinRate_Predicho'],
                        name='Modelo (Prob %)', line=dict(color='#EF553B', width=3, dash='dot')
                    ))
                    fig_calib.update_layout(title='Calibración: Realidad vs Expectativa', barmode='group', template="plotly_dark")
                    st.plotly_chart(fig_calib, use_container_width=True)
                
                with c_d2:
                    st.markdown("#### 📉 Tabla de Error")
                    st.dataframe(calibration.style.format({
                        "WinRate_Real": "{:.1%}",
                        "WinRate_Predicho": "{:.1%}",
                        "Diferencia": "{:+.1%}"
                    }), use_container_width=True)
                    
                    # Diagnóstico automático
                    total_diff = calibration['Diferencia'].mean()
                    if total_diff > 0.05:
                        st.error(f"🚨 **SOBREESTIMACIÓN SEVERA ({total_diff:+.1%})**")
                        st.markdown(f"El modelo es demasiado optimista. **Baja el 'Factor Calibración' a {calibration_factor - total_diff:.2f}**")
                    elif total_diff > 0:
                        st.warning(f"⚠️ Sobreestimación leve ({total_diff:+.1%})")
                    else:
                        st.success("✅ El modelo está bien calibrado o es conservador.")

# --- TAB 6: RENDIMIENTO ---
with t6:
    st.markdown("## 📈 Rendimiento - Risk Management")
    if os.path.exists(CSV_FILE):
        df_hist = pd.read_csv(CSV_FILE)
        df_finished = df_hist[df_hist["Estado"].isin(["Ganada", "Perdida", "Push"])].copy()
        
        if not df_finished.empty:
            df_finished = df_finished.sort_values("Fecha")
            df_finished["Fecha_dt"] = pd.to_datetime(df_finished["Fecha"])
            
            tot_inv = df_finished["Stake"].sum()
            tot_prof = df_finished["Ganancia"].sum()
            roi = (tot_prof / tot_inv * 100) if tot_inv > 0 else 0
            
            df_finished["Equity"] = df_finished["Ganancia"].cumsum()
            df_finished["Drawdown"] = df_finished["Equity"] - df_finished["Equity"].cummax()
            
            st.metric("Beneficio Neto", f"${tot_prof:,.2f}", f"{roi:.1f}% ROI")
            
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=df_finished["Fecha_dt"], y=df_finished["Equity"], mode='lines', name='Equity', line=dict(color='#00ff00')))
            fig_eq.update_layout(title='Curva de Equity', height=400, template="plotly_dark")
            st.plotly_chart(fig_eq, use_container_width=True)
            
            st.dataframe(df_finished, use_container_width=True)
        else:
            st.info("No hay apuestas finalizadas.")
    else:
        st.info("Aún no hay historial.")

# --- TAB 7: ML EVALUATION ---
with t7:
    st.markdown("## 🤖 ML 1X2: Ensemble Avanzado")
    if not HAS_XGB: st.warning("⚠️ XGBoost no instalado. Usando RandomForest.")
    
    col_eval1, col_eval2, col_eval3 = st.columns(3)
    with col_eval1: n_test_ml = st.slider("Partidos test", 50, 500, 200)
    with col_eval2: min_train_ml = st.slider("Mínimo train", 200, 1000, 300)
    with col_eval3: window_ml = st.slider("Ventana train", 300, 1500, 600)
    
    eval_mode = st.radio("Modo", ["⚡ Rápido", "🧪 Estricto"])
    
    if st.button("▶️ Evaluar"):
        with st.spinner("Evaluando..."):
            if eval_mode == "⚡ Rápido":
                out = fast_eval_ml(df, n_test_ml, min_train_ml, window_ml, st.session_state.model_params["alpha"], st.session_state.model_params["rho"])
            else:
                out = strict_walkforward_eval_ml_blocks(df, n_test_ml, min_train_ml, window_ml, alpha=st.session_state.model_params["alpha"], rho=st.session_state.model_params["rho"])
            
            if out:
                c1, c2, c3 = st.columns(3)
                c1.metric("LogLoss", f"{out['logloss']:.4f}")
                c2.metric("Accuracy", f"{out['accuracy']*100:.1f}%")
                c3.metric("N", out['n'])
    
    st.divider()
    if st.button("🧠 Predecir Partido Actual con ML"):
        snap = train_snapshot_cached(df, window_ml, alpha=st.session_state.model_params["alpha"])
        if snap:
            model, ts, ah2, aa2 = snap
            oh, od, oa = st.session_state.odds_inputs["oh"], st.session_state.odds_inputs["od"], st.session_state.odds_inputs["oa"]
            p, _, pick = predict_ml_for_match(home, away, oh, od, oa, model, ts, ah2, aa2)
            
            st.success(f"Predicción ML: {pick}")
            st.write(f"Probabilidades: Local {p[0]:.2f} | Empate {p[1]:.2f} | Visita {p[2]:.2f}")

# --- TAB 8: MULTI-LIGA ---
with t8:
    st.markdown("## 🌎 Super Escáner: Todas las Ligas")
    if not st.session_state.api_key:
        st.warning("Requieres API Key.")
    else:
        win_multi = st.slider("Ventana entrenamiento", 300, 2000, 600)
        min_ev_multi = st.slider("EV mínimo", 0.0, 0.2, 0.07)
        
        selected_leagues = []
        cols_leagues = st.columns(4)
        for idx, (code_l, name_l) in enumerate(leagues.items()):
            with cols_leagues[idx % 4]:
                if st.checkbox(name_l, value=True, key=f"chk_{code_l}"):
                    selected_leagues.append((code_l, name_l))
        
        if st.button("🚀 Ejecutar Análisis Masivo"):
            master_results = []
            progress_bar = st.progress(0)
            
            for idx, (l_code, l_name) in enumerate(selected_leagues):
                progress_bar.progress(int(((idx + 1) / len(selected_leagues)) * 100))
                
                if l_code not in st.session_state.market_storage: continue
                
                df_loop = fetch_live_soccer_data(l_code, N_SEASONS, use_only_current)
                snap_loop = train_snapshot_cached(df_loop, win_multi, alpha=st.session_state.model_params["alpha"])
                if not snap_loop: continue
                
                model_loop, stats_loop, avgh_loop, avga_loop = snap_loop
                data_api = st.session_state.market_storage[l_code].get("data", [])
                
                now_utc = pd.Timestamp.now(tz="UTC")
                teams_loop = sorted(list(set(df_loop["home"].unique()) | set(df_loop["away"].unique())))

                for item in data_api:
                    match_date = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
                    if pd.isna(match_date) or (match_date - now_utc).total_seconds()/3600 > 168: continue
                    
                    h_api, a_api = normalize_name(item.get("home_team", "")), normalize_name(item.get("away_team", ""))
                    m_h = get_close_matches(h_api, teams_loop, n=1, cutoff=0.7)
                    m_a = get_close_matches(a_api, teams_loop, n=1, cutoff=0.7)
                    
                    if not m_h or not m_a: continue
                    h_team, a_team = m_h[0], m_a[0]
                    if h_team not in stats_loop or a_team not in stats_loop: continue
                    
                    oh2, od2, oa2 = match_odds_from_scanner_item(item)
                    if np.isnan(oh2) or oh2 <= 1.01: continue
                    
                    p, (ev_h, ev_d, ev_a), pick = predict_ml_for_match(h_team, a_team, oh2, od2, oa2, model_loop, stats_loop, avgh_loop, avga_loop)
                    best_ev = np.nanmax([ev_h, ev_d, ev_a])
                    
                    if best_ev > min_ev_multi:
                        master_results.append({
                            "Liga": l_name, "Partido": f"{h_team} vs {a_team}",
                            "Cuotas": f"{oh2}|{od2}|{oa2}", "EV": best_ev, "Pick": pick
                        })
            
            if master_results:
                df_master = pd.DataFrame(master_results).sort_values("EV", ascending=False)
                st.dataframe(df_master.style.format({"EV": "{:.3f}"}), use_container_width=True)
            else:
                st.info("No se encontraron picks.")

st.divider()
st.markdown("<div style='text-align: center; color: #888;'>🛡️ Dixon-Coles Pro v6.2 • Final</div>", unsafe_allow_html=True)
