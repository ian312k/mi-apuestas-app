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
st.set_page_config(page_title="Dixon-Coles Pro v5.4 (Monitor Liga)", layout="wide", page_icon="🛡️")
CSV_FILE = "mis_apuestas_pro.csv"
N_SEASONS = 3

# --- TRADUCTOR DE EQUIPOS (API -> CSV HISTÓRICO) ---
# Esto corrige el problema de "Man City" vs "Manchester City" con el filtro estricto
TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Leeds United": "Leeds",
    "Sheffield United": "Sheffield United", # A veces la API manda 'Sheffield Utd'
    "Brighton & Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle"
}

def normalize_name(name):
    # Si el nombre exacto está en el mapa, lo traduce. Si no, lo deja igual.
    return TEAM_MAP.get(name, name)

# --- SESSION STATE ---
if "ticket" not in st.session_state: st.session_state.ticket = []
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "api_usage" not in st.session_state: st.session_state.api_usage = {"used": 0, "remaining": 500}
if "market_storage" not in st.session_state: st.session_state.market_storage = {}
if "odds_inputs" not in st.session_state:
    st.session_state.odds_inputs = {"oh": 2.0, "od": 3.2, "oa": 3.5}

st.markdown("""
<style>
    div[data-testid="stMetric"] { background-color: #262730; border: 1px solid #464b5c; padding: 15px; border-radius: 10px; }
    .ticket-box { background-color: #1e1e1e; border: 1px solid #ffd700; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. DATA + API (CORREGIDO Y LIMPIO)
# ======================================================
@st.cache_data(ttl=3600)
def fetch_live_soccer_data(league_code="SP1", n_seasons=3):
    def season_code(start_year: int) -> str:
        yy = start_year % 100
        yy2 = (start_year + 1) % 100
        return f"{yy:02d}{yy2:02d}"

    today = datetime.now()
    # Ajuste temporada
    current_start_year = today.year if today.month >= 7 else (today.year - 1)
    seasons = [season_code(current_start_year - i) for i in range(n_seasons)]

    frames = []
    for s in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{league_code}.csv"
        try:
            # Encoding latin1 para caracteres especiales
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

            # --- LIMPIEZA DE NOMBRES (TRIM) ---
            tmp["home"] = tmp["home"].astype(str).str.strip()
            tmp["away"] = tmp["away"].astype(str).str.strip()
            # ----------------------------------

            for c in ["odd_h", "odd_d", "odd_a"]:
                if c not in tmp.columns: tmp[c] = 1.0
            for c in ["sot_h", "sot_a"]:
                if c not in tmp.columns: tmp[c] = 0

            tmp = tmp.dropna(subset=["home", "away", "home_goals", "away_goals"])
            
            # Formato fecha robusto
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
# 3. DIXON-COLES
# ======================================================
def calculate_strengths(df, ref_date=None, alpha=0.004, mix_factor=0.7, window_matches=None):
    df = df.copy().dropna(subset=["date", "home", "away", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    if window_matches is not None and len(df) > window_matches:
        df = df.tail(window_matches).reset_index(drop=True)

    last_date = pd.to_datetime(ref_date) if ref_date is not None else df["date"].max()
    df["days_ago"] = (last_date - df["date"]).dt.days.clip(lower=0)
    df["weight"] = np.exp(-alpha * df["days_ago"])

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
            att_global = (np.average(team_matches["goals_scored"], weights=team_matches["weight"]) / avg_global) if avg_global > 0 else 1.0
            def_global = (np.average(team_matches["goals_conceded"], weights=team_matches["weight"]) / avg_global) if avg_global > 0 else 1.0
        else:
            att_global, def_global = 1.0, 1.0

        h_m = df[df["home"] == team]
        if not h_m.empty:
            att_h_pure = (np.average(h_m["home_goals"], weights=h_m["weight"]) / avg_home) if avg_home > 0 else 1.0
            def_h_pure = (np.average(h_m["away_goals"], weights=h_m["weight"]) / avg_away) if avg_away > 0 else 1.0
            sot_h_avg = np.average(h_m["sot_h"], weights=h_m["weight"]) if "sot_h" in h_m.columns else 0.0
        else:
            att_h_pure, def_h_pure, sot_h_avg = 1.0, 1.0, 0.0

        a_m = df[df["away"] == team]
        if not a_m.empty:
            att_a_pure = (np.average(a_m["away_goals"], weights=a_m["weight"]) / avg_away) if avg_away > 0 else 1.0
            def_a_pure = (np.average(a_m["home_goals"], weights=a_m["weight"]) / avg_home) if avg_home > 0 else 1.0
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
        }

    return team_stats, avg_home, avg_away, all_teams

def predict_match_dixon_coles(home, away, team_stats, avg_h, avg_a, rho=-0.13, max_goals=10):
    if home not in team_stats or away not in team_stats:
        return 0,0,0,0,0,0,0,0,[],np.zeros((1,1))

    h_exp = team_stats[home]["att_h"] * team_stats[away]["def_a"] * avg_h
    a_exp = team_stats[away]["att_a"] * team_stats[home]["def_h"] * avg_a

    probs = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            p_base = poisson.pmf(x, h_exp) * poisson.pmf(y, a_exp)
            correction = 1.0
            if x == 0 and y == 0: correction = 1.0 - (h_exp * a_exp * rho)
            elif x == 0 and y == 1: correction = 1.0 + (h_exp * rho)
            elif x == 1 and y == 0: correction = 1.0 + (a_exp * rho)
            elif x == 1 and y == 1: correction = 1.0 - (rho)
            probs[x][y] = p_base * correction

    probs = np.maximum(0, probs)
    probs = probs / probs.sum()

    p_home = np.tril(probs, -1).sum()
    p_draw = np.diag(probs).sum()
    p_away = np.triu(probs, 1).sum()

    p_o15 = probs[(np.add.outer(np.arange(max_goals), np.arange(max_goals)) > 1.5)].sum()
    p_o25 = probs[(np.add.outer(np.arange(max_goals), np.arange(max_goals)) > 2.5)].sum()
    p_btts = probs[(np.arange(max_goals)[:, None] > 0) & (np.arange(max_goals)[None, :] > 0)].sum()

    flat_indices = np.argsort(probs.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, probs.shape)
        top_scores.append((f"{i}-{j}", probs[i][j]))

    return h_exp, a_exp, p_home, p_draw, p_away, p_o15, p_o25, p_btts, top_scores, probs

# ======================================================
# 4. APUESTAS / HISTORIAL
# ======================================================
def calculate_kelly(prob, odd):
    if prob <= 0 or odd <= 1: return 0.0
    b = odd - 1
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f * 0.5) * 100

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
            profit = (df.at[i, "Stake"] * df.at[i, "Cuota"]) - df.at[i, "Stake"] if status == "Ganada" else (-df.at[i, "Stake"] if status == "Perdida" else 0)
            df.at[i, "Ganancia"] = profit
            df.to_csv(CSV_FILE, index=False)

    elif mode == "delete":
        df = df[df["ID"].astype(str) != str(id_bet)]
        df.to_csv(CSV_FILE, index=False)

    return df

def run_backtest_no_leak(df, n_test=50, min_train=200, window_matches=800, stake_unit=1.0):
    df_sorted = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    test_block = df_sorted.tail(n_test)

    results = []
    correct, bal = 0, 0.0
    n_bets = 0

    for _, row in test_block.iterrows():
        cut_date = row["date"]
        train_df = df_sorted[df_sorted["date"] < cut_date].copy()
        if len(train_df) < min_train:
            continue

        team_stats, avg_h, avg_a, _ = calculate_strengths(train_df, ref_date=cut_date, window_matches=window_matches)
        if row["home"] not in team_stats or row["away"] not in team_stats:
            continue

        _, _, ph, pd_prob, pa, *_ = predict_match_dixon_coles(row["home"], row["away"], team_stats, avg_h, avg_a)

        odd_h = float(row.get("odd_h", np.nan))
        odd_d = float(row.get("odd_d", np.nan))
        odd_a = float(row.get("odd_a", np.nan))
        if (np.isnan(odd_h) or odd_h <= 1.01) or (np.isnan(odd_d) or odd_d <= 1.01) or (np.isnan(odd_a) or odd_a <= 1.01):
            continue

        if ph > pd_prob and ph > pa:
            pred, prob, odd = "Local", ph, odd_h
            res_real = "Local" if row["home_goals"] > row["away_goals"] else "Fallo"
        elif pa > ph and pa > pd_prob:
            pred, prob, odd = "Visita", pa, odd_a
            res_real = "Visita" if row["away_goals"] > row["home_goals"] else "Fallo"
        else:
            pred, prob, odd = "Empate", pd_prob, odd_d
            res_real = "Empate" if row["home_goals"] == row["away_goals"] else "Fallo"

        is_win = (pred == res_real)
        profit_u = (odd - 1) * stake_unit if is_win else -stake_unit

        correct += int(is_win)
        bal += profit_u
        n_bets += 1

        results.append({
            "Fecha": row["date"].strftime("%Y-%m-%d"),
            "Temporada": row.get("season", ""),
            "Partido": f"{row['home']} vs {row['away']}",
            "Predicción": f"{pred} ({prob*100:.0f}%)",
            "Realidad": f"{int(row['home_goals'])}-{int(row['away_goals'])}",
            "Cuota": odd,
            "Res": "✅" if is_win else "❌",
            "Stake(U)": stake_unit,
            "P/L(U)": profit_u
        })

    total_stake = n_bets * stake_unit
    roi = (bal / total_stake * 100) if total_stake > 0 else 0.0
    return pd.DataFrame(results), correct, bal, roi, n_bets, total_stake

# ======================================================
# 5. PLOTS (Y FUNCION GET_LAST_5 CORREGIDA)
# ======================================================
def plot_gauge(val, title, color):
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val * 100,
            title={"text": title},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}, "bgcolor": "white"},
        )
    ).update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))

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
    )
    return fig

def get_last_5(df, team):
    # CORRECCIÓN: Limpiar espacios y string
    team = str(team).strip()
    
    # Filtrar
    mask = (df["home"] == team) | (df["away"] == team)
    l5 = df[mask].sort_values(by="date", ascending=False).head(5).copy()
    
    if l5.empty:
        return pd.DataFrame(columns=["Sede", "Rival", "Score", "Tiros"])

    l5["Rival"] = np.where(l5["home"] == team, l5["away"], l5["home"])
    
    # Formateo Score seguro
    l5["Score"] = (
        l5["home_goals"].astype(float).astype(int).astype(str) + 
        "-" + 
        l5["away_goals"].astype(float).astype(int).astype(str)
    )
    
    # Manejo seguro de tiros
    sot_h = l5.get("sot_h", 0).replace("", 0).astype(float).fillna(0).astype(int)
    sot_a = l5.get("sot_a", 0).replace("", 0).astype(float).fillna(0).astype(int)
    l5["Tiros"] = np.where(l5["home"] == team, sot_h, sot_a)
    
    l5["Sede"] = np.where(l5["home"] == team, "🏠", "✈️")
    return l5[["Sede", "Rival", "Score", "Tiros"]]

def safe_fair_odds(p, eps=1e-12):
    p = float(np.clip(p, eps, 1.0))
    return 1.0 / p

# ======================================================
# 6. ML ENSEMBLE 1X2 (Odds + DC + XGB/RF)
# ======================================================
def odds_to_probs(oh, od, oa, eps=1e-12):
    oh = max(float(oh), 1.01); od = max(float(od), 1.01); oa = max(float(oa), 1.01)
    ph = 1.0/oh; pd_ = 1.0/od; pa = 1.0/oa
    s = ph + pd_ + pa + eps
    return ph/s, pd_/s, pa/s

def outcome_1x2_label(hg, ag):
    if hg > ag: return 0
    if hg == ag: return 1
    return 2

def brier_multiclass(P, y, n_classes=3):
    y_oh = np.eye(n_classes)[y]
    return float(np.mean(np.sum((P - y_oh) ** 2, axis=1) / n_classes))

def build_features_for_match(row, team_stats, avg_h, avg_a):
    _, _, dc_h, dc_d, dc_a, *_ = predict_match_dixon_coles(row["home"], row["away"], team_stats, avg_h, avg_a)

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

    return np.array([
        mk_h, mk_d, mk_a,
        dc_h, dc_d, dc_a,
        h_exp, a_exp,
        h_exp - a_exp,
        sot_h, sot_a
    ], dtype=float)

def fit_ml_multiclass(X, y, seed=42):
    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X, y)
        return model

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def fast_eval_ml(df, n_test=200, min_train=500, window_matches=1200):
    df_sorted = df.dropna(subset=["date","home","away","home_goals","away_goals"]).sort_values("date").reset_index(drop=True)
    if len(df_sorted) < (n_test + min_train):
        return None

    train_df = df_sorted.iloc[:-n_test].copy()
    test_df  = df_sorted.iloc[-n_test:].copy()

    team_stats, avg_h, avg_a, _ = calculate_strengths(train_df, ref_date=train_df["date"].max(), window_matches=window_matches)

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
    return {"mode": "rápido", "n": int(len(y)), "logloss": ll, "brier": br}

def strict_walkforward_eval_ml_blocks(df, n_test=200, min_train=500, window_matches=1200,
                                      retrain_every=10, train_step=2):
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

            team_stats, avg_h, avg_a, _ = calculate_strengths(train_df, ref_date=cut_date, window_matches=window_matches)
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
    return {"mode": f"estricto-bloques (K={retrain_every}, step={train_step})", "n": int(len(y)), "logloss": ll, "brier": br}

# ======================================================
# 7. JORNADA ML (desde escáner)
# ======================================================
@st.cache_data(ttl=1800)
def train_snapshot_cached(df, window_matches=1200, seed=42):
    df_sorted = df.sort_values("date").copy()
    team_stats, avg_h, avg_a, _ = calculate_strengths(df_sorted, ref_date=df_sorted["date"].max(), window_matches=window_matches)

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
    row_now = {"home": home_team, "away": away_team, "odd_h": oh, "odd_d": od, "odd_a": oa, "sot_h": 0.0, "sot_a": 0.0}
    x = build_features_for_match(row_now, team_stats, avg_h, avg_a).reshape(1, -1)
    p = model.predict_proba(x)[0]  # [H,D,A]

    ev_h = (p[0] * oh) - 1 if (oh and oh > 1.01) else np.nan
    ev_d = (p[1] * od) - 1 if (od and od > 1.01) else np.nan
    ev_a = (p[2] * oa) - 1 if (oa and oa > 1.01) else np.nan

    best = np.nanmax([ev_h, ev_d, ev_a])
    if np.isnan(best):
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
    st.header("⚙️ Configuración")
    if st.button("🔄 Actualizar Datos"):
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
    df = fetch_live_soccer_data(code, n_seasons=N_SEASONS)

    if not df.empty:
        stats, ah, aa, teams = calculate_strengths(df, ref_date=df["date"].max(), window_matches=1200)
        st.success(f"✅ {len(df)} partidos cargados")
        
        # --- SECCIÓN NUEVA: MONITOR DE LA LIGA ---
        st.divider()
        st.markdown("### 🗓️ Estado de la Liga")
        last_date = df["date"].max()
        st.write(f"**Última actualización:** {last_date.strftime('%d/%m/%Y')}")

        with st.expander("🔎 Ver últimos 5 partidos globales"):
            # Obtenemos los últimos 5 partidos globales de la base de datos
            latest_matches = df.sort_values("date", ascending=False).head(5).copy()
            latest_matches["Score"] = latest_matches["home_goals"].astype(int).astype(str) + "-" + latest_matches["away_goals"].astype(int).astype(str)
            latest_matches["date"] = latest_matches["date"].dt.strftime("%d/%m")
            st.dataframe(latest_matches[["date", "home", "Score", "away"]], hide_index=True, use_container_width=True)
        # -----------------------------------------
    else:
        st.error("Error cargando datos.")
        st.stop()

    st.divider()
    bank = st.number_input("💰 Tu Banco ($)", 1000.0, step=50.0)

    if st.session_state.ticket:
        st.divider()
        st.markdown(f"**Ticket:** {len(st.session_state.ticket)} selecciones")
        if st.button("🗑️ Limpiar"):
            st.session_state.ticket = []
            st.rerun()

st.title(f"🛡️ Dixon-Coles: {leagues[code]}")

# --- SELECTOR ---
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])

h_exp, a_exp, ph, pd_prob, pa, po15, po25, pbtts, top_sc, probs = predict_match_dixon_coles(home, away, stats, ah, aa)

# ======================================================
# 9. TABS
# ======================================================
t1, t2, t3, t4, t5, t6, t7 = st.tabs(
    ["📊 Análisis", "💰 Valor", "📜 Historial", "💎 Escáner Seguro", "🧪 Laboratorio", "📈 Rendimiento (Risk)", "🤖 ML 1X2 + Jornada"]
)

# --- TAB 1: ANÁLISIS ---
with t1:
    st.markdown("### 🥅 Expectativa de Goles (Modelo)")
    a, b, c = st.columns(3)
    a.metric(home, f"{h_exp:.2f}")
    b.metric("Total (xG)", f"{h_exp + a_exp:.2f}")
    c.metric(away, f"{a_exp:.2f}")

    st.divider()
    st.markdown("### 🏁 ¿Quién gana? (1X2) + Mercados de goles")

    # 1X2
    m1, m2, m3 = st.columns(3)
    m1.metric(f"🏠 {home}", f"{ph*100:.1f}%")
    m2.metric("🤝 Empate", f"{pd_prob*100:.1f}%")
    m3.metric(f"✈️ {away}", f"{pa*100:.1f}%")

    # Pick "quién gana"
    best_1x2 = max(ph, pd_prob, pa)
    if best_1x2 == ph:
        pick_1x2 = f"Gana {home}"
    elif best_1x2 == pa:
        pick_1x2 = f"Gana {away}"
    else:
        pick_1x2 = "Empate"

    # Fair odds (sin margen) de DC
    fo_h, fo_d, fo_a = safe_fair_odds(ph), safe_fair_odds(pd_prob), safe_fair_odds(pa)

    st.info(
        f"**Pick modelo (1X2):** {pick_1x2}  |  "
        f"**Cuotas justas (DC, sin margen):** H={fo_h:.2f}  D={fo_d:.2f}  A={fo_a:.2f}"
    )

    # Overs / BTTS
    g1, g2, g3 = st.columns(3)
    g1.metric("Over 1.5", f"{po15*100:.1f}%")
    g2.metric("Over 2.5", f"{po25*100:.1f}%")
    g3.metric("BTTS (Ambos anotan)", f"{pbtts*100:.1f}%")

    st.plotly_chart(plot_score_heatmap(probs, home, away), use_container_width=True)

    st.markdown("### ⭐ Top 3 marcadores más probables")
    if top_sc:
        top_df = pd.DataFrame([{"Marcador": s, "Prob (%)": p*100} for s, p in top_sc])
        st.dataframe(top_df.style.format({"Prob (%)":"{:.2f}"}), use_container_width=True, hide_index=True)
    else:
        st.write("No disponible.")

    st.divider()
    st.markdown("### 📉 Últimos 5")
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
        league_data = st.session_state.market_storage.get(code, {})
        found_in_storage = False

        if "data" in league_data:
            for item in league_data["data"]:
                # --- APLICAR MAPEO AQUÍ ---
                h_team_api = normalize_name(item.get("home_team", ""))
                a_team_api = normalize_name(item.get("away_team", ""))
                
                # FIX: CUTOFF MAS ESTRICTO (0.8) PERO YA CON NOMBRES NORMALIZADOS
                m_h = get_close_matches(h_team_api, [home], n=1, cutoff=0.8)
                m_a = get_close_matches(a_team_api, [away], n=1, cutoff=0.8)
                
                if m_h and m_a and item.get("bookmakers"):
                    oh2, od2, oa2 = match_odds_from_scanner_item(item)
                    if not np.isnan(oh2) and not np.isnan(od2) and not np.isnan(oa2):
                        def_oh, def_od, def_oa = oh2, od2, oa2
                        found_in_storage = True
                        break

        if found_in_storage: st.success("✅ Momios cargados automáticamente (Escáner).")
        else: st.info("ℹ️ Momios por defecto (No encontrados en escáner).")

        co1, co2, co3 = st.columns(3)
        oh = co1.number_input("Cuota Local", 1.01, 100.0, float(def_oh))
        od = co2.number_input("Cuota Empate", 1.01, 100.0, float(def_od))
        oa = co3.number_input("Cuota Visita", 1.01, 100.0, float(def_oa))

        st.session_state.odds_inputs = {"oh": float(oh), "od": float(od), "oa": float(oa)}

        st.markdown("#### 🧠 Kelly")
        k_ev_h = (ph * oh) - 1
        k_ev_d = (pd_prob * od) - 1
        k_ev_a = (pa * oa) - 1
        k_max_ev = max(k_ev_h, k_ev_d, k_ev_a)

        if k_max_ev > 0:
            if k_max_ev == k_ev_h: k_sel, k_p, k_o = f"Gana {home}", ph, oh
            elif k_max_ev == k_ev_d: k_sel, k_p, k_o = "Empate", pd_prob, od
            else: k_sel, k_p, k_o = f"Gana {away}", pa, oa

            k_pct = calculate_kelly(k_p, k_o)
            k_stake = (k_pct / 100) * bank
            st.success(f"💎 **Recomendación Kelly:** {k_sel} | Stake: ${k_stake:.2f} ({k_pct:.2f}%)")
        else:
            st.warning("📉 Kelly sugiere: **No apostar** (Sin valor esperado positivo)")

        st.divider()
        st.markdown("### ➕ Agregar al Ticket")
        with st.form("add_to_ticket"):
            sel_pick = st.selectbox("Selección", [f"Gana {home}", "Empate", f"Gana {away}"])
            if f"Gana {home}" in sel_pick: sel_odd, sel_prob = oh, ph
            elif "Empate" in sel_pick: sel_odd, sel_prob = od, pd_prob
            else: sel_odd, sel_prob = oa, pa

            if st.form_submit_button("Añadir selección"):
                st.session_state.ticket.append({
                    "match": f"{home} vs {away}",
                    "pick": sel_pick,
                    "odd": sel_odd,
                    "prob": sel_prob,
                    "league": leagues[code],
                })
                st.success("Añadido")
                st.rerun()

    with col_ticket:
        st.markdown("### 🎫 Ticket")
        if not st.session_state.ticket:
            st.info("Vacío")
        else:
            total_odd, total_prob = 1.0, 1.0
            for idx, item in enumerate(st.session_state.ticket):
                st.markdown(
                    f"<div class='ticket-box'><small>{item['league']}</small><br><strong>{item['match']}</strong><br>{item['pick']} @ {item['odd']}</div>",
                    unsafe_allow_html=True
                )
                if st.button("❌", key=f"del_{idx}"):
                    st.session_state.ticket.pop(idx)
                    st.rerun()
                total_odd *= item["odd"]
                total_prob *= item["prob"]

            st.divider()
            st.metric("Cuota Total", f"{total_odd:.2f}")
            stake_parlay = st.number_input("Stake ($)", 1.0, 5000.0, 50.0)
            st.success(f"Ganancia: ${(stake_parlay * total_odd) - stake_parlay:.2f}")

            if st.button("💾 Guardar"):
                tipo_str = "Simple" if len(st.session_state.ticket) == 1 else "Parlay"
                match_str = st.session_state.ticket[0]["match"] if len(st.session_state.ticket) == 1 else f"Combinada ({len(st.session_state.ticket)})"
                pick_str = " + ".join([i["pick"] for i in st.session_state.ticket])

                manage_bets("save", {
                    "ID": pd.Timestamp.now().strftime("%Y%m%d%H%M%S"),
                    "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "Liga": tipo_str,
                    "Partido": match_str,
                    "Pick": pick_str,
                    "Cuota": round(total_odd, 2),
                    "Stake": stake_parlay,
                    "Prob": round(total_prob, 4),
                    "Estado": "Pendiente",
                    "Ganancia": 0.0
                })
                st.session_state.ticket = []
                st.balloons()
                st.rerun()

# --- TAB 3: HISTORIAL ---
with t3:
    st.markdown("### 📜 Historial")
    db = manage_bets("load")
    if not db.empty:
        st.dataframe(db.sort_values(by="Fecha", ascending=False), use_container_width=True)
    else:
        st.warning("Aún no hay historial.")

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

        if code in st.session_state.market_storage:
            stored = st.session_state.market_storage[code]
            data_to_display = stored.get("data", [])
            st.info(f"📂 Datos en memoria. Actualizado: {stored['timestamp'].strftime('%H:%M:%S')}")
        else:
            data_to_display = []
            st.warning("⚠️ Sin datos descargados.")

        if st.button("⬇️ Descargar/Actualizar Datos (1 llamada)"):
            with st.spinner("Conectando..."):
                resp = call_api_real(sport_key, st.session_state.api_key)
                if resp["success"]:
                    st.session_state.market_storage[code] = {"timestamp": datetime.now(), "data": resp["data"]}
                    st.session_state.api_usage["used"] = resp["used"]
                    st.session_state.api_usage["remaining"] = resp["remaining"]
                    st.success("✅ Descargado.")
                    st.rerun()
                else:
                    st.error(f"Error API: {resp['message']}")

        # Mostrar oportunidades (DC EV)
        if data_to_display:
            now_utc = pd.Timestamp.now(tz="UTC")
            live_rows = []
            for item in data_to_display:
                match_date = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
                if pd.isna(match_date):
                    continue
                diff_hours = (match_date - now_utc).total_seconds()/3600
                if diff_hours > 168 or diff_hours < -5:
                    continue

                # --- APLICAR MAPEO AQUÍ ---
                h_api = normalize_name(item.get("home_team",""))
                a_api = normalize_name(item.get("away_team",""))
                
                oh2, od2, oa2 = match_odds_from_scanner_item(item)
                if np.isnan(oh2) or np.isnan(od2) or np.isnan(oa2):
                    continue

                # FIX: CUTOFF MAS ESTRICTO (0.8) PERO YA CON NOMBRES NORMALIZADOS
                m_h = get_close_matches(h_api, teams, n=1, cutoff=0.8)
                m_a = get_close_matches(a_api, teams, n=1, cutoff=0.8)

                if not m_h or not m_a:
                    continue
                h = m_h[0]; a = m_a[0]
                if h not in stats or a not in stats:
                    continue

                _, _, ph2, pd2, pa2, *_ = predict_match_dixon_coles(h, a, stats, ah, aa)
                ev_h = (ph2*oh2)-1
                ev_d = (pd2*od2)-1
                ev_a = (pa2*oa2)-1
                best_ev = max(ev_h, ev_d, ev_a)
                pick = "No Bet"
                if best_ev > 0:
                    pick = f"Gana {h}" if best_ev==ev_h else ("Empate" if best_ev==ev_d else f"Gana {a}")

                live_rows.append({
                    "Hora (UTC)": match_date.strftime("%d/%m %H:%M"),
                    "Partido": f"{h} vs {a}",
                    "Cuotas": f"H:{oh2:.2f} D:{od2:.2f} A:{oa2:.2f}",
                    "DC Prob": f"H:{ph2:.3f} D:{pd2:.3f} A:{pa2:.3f}",
                    "Mejor EV": best_ev,
                    "Pick": pick
                })
            if live_rows:
                st.markdown("### 🎯 Oportunidades (Dixon-Coles vs Cuotas)")
                df_live = pd.DataFrame(live_rows).sort_values("Mejor EV", ascending=False)
                st.dataframe(df_live, use_container_width=True)

        # ===== Jornada ML (desde escáner) =====
        st.divider()
        st.markdown("## 😁 Jornada ML (desde Escáner)")
        st.caption("Entrena 1 snapshot (cacheado) y predice TODOS los partidos de la semana (próx 7 días).")

        window_ml_j = st.slider("Ventana train snapshot (matches)", 300, 3000, 1200, step=100, key="window_ml_j")
        only_positive_ev = st.checkbox("Solo EV > 0", value=False)
        min_ev = st.slider("EV mínimo", 0.0, 0.20, 0.00, step=0.01)

        if st.button("📌 Generar pronósticos jornada (ML + Escáner)"):
            if code not in st.session_state.market_storage:
                st.warning("Primero descarga/actualiza datos en el escáner.")
            else:
                stored = st.session_state.market_storage[code]
                data_to_display = stored.get("data", [])
                if not data_to_display:
                    st.warning("No hay partidos en el escáner.")
                else:
                    with st.spinner("Entrenando snapshot (cache) y prediciendo..."):
                        snap = train_snapshot_cached(df, window_matches=window_ml_j, seed=42)

                    if snap is None:
                        st.warning("No se pudo entrenar snapshot (historial insuficiente).")
                    else:
                        model, team_stats2, avg_h2, avg_a2 = snap
                        now_utc = pd.Timestamp.now(tz="UTC")
                        rows = []

                        for item in data_to_display:
                            match_date = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
                            if pd.isna(match_date):
                                continue
                            diff_hours = (match_date - now_utc).total_seconds()/3600
                            if diff_hours > 168 or diff_hours < -5:
                                continue

                            # --- APLICAR MAPEO AQUÍ ---
                            h_api = normalize_name(item.get("home_team",""))
                            a_api = normalize_name(item.get("away_team",""))

                            # FIX: CUTOFF MAS ESTRICTO (0.8) PERO YA CON NOMBRES NORMALIZADOS
                            m_h = get_close_matches(h_api, teams, n=1, cutoff=0.8)
                            m_a = get_close_matches(a_api, teams, n=1, cutoff=0.8)
                            
                            if not m_h or not m_a:
                                continue
                            h = m_h[0]; a = m_a[0]
                            if h not in team_stats2 or a not in team_stats2:
                                continue

                            oh2, od2, oa2 = match_odds_from_scanner_item(item)
                            if np.isnan(oh2) or np.isnan(od2) or np.isnan(oa2) or oh2<=1.01 or od2<=1.01 or oa2<=1.01:
                                continue

                            p, (ev_h, ev_d, ev_a), pick = predict_ml_for_match(h, a, float(oh2), float(od2), float(oa2),
                                                                                model, team_stats2, avg_h2, avg_a2)
                            best_ev = np.nanmax([ev_h, ev_d, ev_a])
                            if only_positive_ev and (np.isnan(best_ev) or best_ev <= 0):
                                continue
                            if best_ev < min_ev:
                                continue

                            rows.append({
                                "Hora (UTC)": match_date.strftime("%d/%m %H:%M"),
                                "Partido": f"{h} vs {a}",
                                "Cuotas": f"H:{oh2:.2f} D:{od2:.2f} A:{oa2:.2f}",
                                "ML Prob": f"H:{p[0]:.3f} D:{p[1]:.3f} A:{p[2]:.3f}",
                                "EV_H": ev_h,
                                "EV_D": ev_d,
                                "EV_A": ev_a,
                                "Mejor EV": best_ev,
                                "Pick": pick,
                            })

                        if not rows:
                            st.info("No se encontraron partidos válidos (nombres/odds/ventana).")
                        else:
                            out_df = pd.DataFrame(rows).sort_values("Mejor EV", ascending=False).reset_index(drop=True)
                            st.success(f"✅ Jornada generada: {len(out_df)} partidos")
                            st.dataframe(out_df.style.format({"EV_H":"{:.3f}","EV_D":"{:.3f}","EV_A":"{:.3f}","Mejor EV":"{:.3f}"}),
                                         use_container_width=True)
                            st.download_button("📥 Descargar jornada (CSV)",
                                                data=out_df.to_csv(index=False).encode("utf-8"),
                                                file_name=f"jornada_ml_{code}.csv",
                                                mime="text/csv")
    else:
        st.info("Pon tu API key para usar el escáner y jornada.")

# --- TAB 5: LABORATORIO (BACKTEST DC) ---
with t5:
    st.markdown("## 🧪 Laboratorio")
    n_test = st.slider("Partidos a evaluar", 20, 250, 100, step=10)
    min_train = st.slider("Mínimo de partidos para entrenar", 50, 900, 250, step=25)
    if st.button("▶️ Validar DC (walk-forward)"):
        with st.spinner("Backtesteando sin fugas..."):
            test_df, ok, profit, roi_bt, n_bets, tot_stake = run_backtest_no_leak(
                df, n_test=n_test, min_train=min_train, window_matches=900, stake_unit=1.0
            )
        if test_df.empty:
            st.warning("No se pudo backtestear.")
        else:
            a,b,c,d = st.columns(4)
            a.metric("Apuestas", f"{n_bets}")
            b.metric("Aciertos", f"{ok}/{n_bets} ({(ok/max(1,n_bets))*100:.0f}%)")
            c.metric("Profit", f"{profit:.2f} U")
            d.metric("ROI", f"{roi_bt:.2f}%")
            st.dataframe(test_df, use_container_width=True)

# --- TAB 6: RISK ---
with t6:
    st.markdown("## 📈 Rendimiento (Risk)")
    if os.path.exists(CSV_FILE):
        df_hist = pd.read_csv(CSV_FILE)
        df_finished = df_hist[df_hist["Estado"].isin(["Ganada", "Perdida", "Push"])].copy()
        if df_finished.empty:
            st.info("No hay apuestas finalizadas.")
        else:
            df_finished = df_finished.sort_values("ID")
            tot_inv = df_finished["Stake"].sum()
            tot_prof = df_finished["Ganancia"].sum()
            roi = (tot_prof / tot_inv * 100) if tot_inv > 0 else 0
            df_finished["Equity"] = df_finished["Ganancia"].cumsum()
            df_finished["Peak"] = df_finished["Equity"].cummax()
            df_finished["Drawdown"] = df_finished["Equity"] - df_finished["Peak"]
            max_dd = df_finished["Drawdown"].min()

            a,b,c,d = st.columns(4)
            a.metric("Beneficio Neto", f"${tot_prof:,.2f}")
            b.metric("ROI", f"{roi:.2f}%")
            c.metric("Max Drawdown", f"{max_dd:.2f} U")
            d.metric("Apuestas", len(df_finished))
            st.dataframe(df_finished, use_container_width=True)
    else:
        st.info("Aún no hay historial.")

# --- TAB 7: ML EVAL + PRED MATCH ---
with t7:
    st.markdown("## 🤖 ML 1X2: Odds pre-match + Dixon-Coles + XGB/RF")
    if HAS_XGB:
        st.success("✅ XGBoost detectado.")
    else:
        st.warning("⚠️ XGBoost NO instalado: usando RandomForest (fallback).")

    eval_mode = st.radio(
        "Modo de evaluación",
        ["⚡ Rápido (1 train + test final)", "🧪 Estricto (walk-forward sin fuga, por bloques)"],
        horizontal=True,
        index=0
    )

    cml1, cml2, cml3 = st.columns(3)
    n_test_ml = cml1.slider("Partidos test", 50, 400, 200, step=25)
    min_train_ml = cml2.slider("Mínimo train", 200, 2000, 500, step=50)
    window_ml = cml3.slider("Ventana train (matches)", 300, 3000, 1200, step=100)

    if "Estricto" in eval_mode:
        cs1, cs2 = st.columns(2)
        retrain_every = cs1.slider("Re-entrenar cada K (más K=más rápido)", 1, 25, 10, step=1)
        train_step = cs2.slider("Submuestreo train (1=todo, 2=mitad)", 1, 5, 2, step=1)
    else:
        retrain_every, train_step = 10, 2

    if st.button("▶️ Evaluar ML (LogLoss/Brier)"):
        with st.spinner("Evaluando..."):
            if "Rápido" in eval_mode:
                out = fast_eval_ml(df, n_test=n_test_ml, min_train=min_train_ml, window_matches=window_ml)
            else:
                out = strict_walkforward_eval_ml_blocks(
                    df, n_test=n_test_ml, min_train=min_train_ml, window_matches=window_ml,
                    retrain_every=retrain_every, train_step=train_step
                )

        if out is None:
            st.warning("No se pudo evaluar (historial insuficiente o datos faltantes).")
        else:
            a,b,c,d = st.columns(4)
            a.metric("Modo", out["mode"])
            b.metric("Partidos evaluados", out["n"])
            c.metric("LogLoss", f"{out['logloss']:.4f}")
            d.metric("Brier", f"{out['brier']:.4f}")

    st.divider()
    st.markdown("### 🎯 Predicción ML para el partido seleccionado")
    st.info("Usa las cuotas capturadas en '💰 Valor' (se guardan automáticamente).")

    oh = st.session_state.odds_inputs["oh"]
    od = st.session_state.odds_inputs["od"]
    oa = st.session_state.odds_inputs["oa"]
    mk_h, mk_d, mk_a = odds_to_probs(oh, od, oa)

    st.markdown("**Inputs (pre-match):**")
    st.write(f"• Mercado (sin margen): H={mk_h:.3f} D={mk_d:.3f} A={mk_a:.3f}")
    st.write(f"• Dixon-Coles: H={ph:.3f} D={pd_prob:.3f} A={pa:.3f}")

    if st.button("🧠 Predecir con ML (snapshot hasta hoy)"):
        with st.spinner("Entrenando snapshot y prediciendo..."):
            snap = train_snapshot_cached(df, window_matches=window_ml, seed=42)

        if snap is None:
            st.warning("No se pudo entrenar snapshot.")
        else:
            model, team_stats2, avg_h2, avg_a2 = snap
            if home not in team_stats2 or away not in team_stats2:
                st.warning("Faltan equipos en histórico.")
            else:
                row_now = {"home": home, "away": away, "odd_h": oh, "odd_d": od, "odd_a": oa, "sot_h": 0.0, "sot_a": 0.0}
                x_now = build_features_for_match(row_now, team_stats2, avg_h2, avg_a2).reshape(1, -1)
                p = model.predict_proba(x_now)[0]

                a,b,c = st.columns(3)
                a.metric(f"Gana {home}", f"{p[0]*100:.1f}%")
                b.metric("Empate", f"{p[1]*100:.1f}%")
                c.metric(f"Gana {away}", f"{p[2]*100:.1f}%")

                comp = pd.DataFrame({
                    "Modelo": ["Mercado (sin margen)", "Dixon-Coles", "ML Ensemble"],
                    "H": [mk_h, ph, p[0]],
                    "D": [mk_d, pd_prob, p[1]],
                    "A": [mk_a, pa, p[2]],
                })
                st.dataframe(comp.style.format({"H":"{:.3f}","D":"{:.3f}","A":"{:.3f}"}), use_container_width=True)
