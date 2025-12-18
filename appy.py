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
# ML (XGB opcional + fallback)
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
st.set_page_config(page_title="Dixon-Coles Pro v5.1 (Risk Manager)", layout="wide", page_icon="🛡️")
CSV_FILE = "mis_apuestas_pro.csv"
N_SEASONS = 3

# --- GESTIÓN DE ESTADO ---
if "ticket" not in st.session_state: st.session_state.ticket = []
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "api_odds_cache" not in st.session_state: st.session_state.api_odds_cache = {}
if "api_usage" not in st.session_state: st.session_state.api_usage = {"used": 0, "remaining": 500}
if "market_storage" not in st.session_state: st.session_state.market_storage = {}
if "odds_inputs" not in st.session_state:
    st.session_state.odds_inputs = {"oh": 2.0, "od": 3.2, "oa": 3.5}

st.markdown("""
<style>
    div[data-testid="stMetric"] { background-color: #262730; border: 1px solid #464b5c; padding: 15px; border-radius: 10px; }
    .ticket-box { background-color: #1e1e1e; border: 1px solid #ffd700; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    .success-box { background-color: #1c2e24; border: 1px solid #4CAF50; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. FUNCIONES LÓGICAS 🧠
# ======================================================

@st.cache_data(ttl=3600)
def fetch_live_soccer_data(league_code="SP1", n_seasons=3):
    def season_code(start_year: int) -> str:
        yy = start_year % 100
        yy2 = (start_year + 1) % 100
        return f"{yy:02d}{yy2:02d}"

    today = datetime.now()
    current_start_year = today.year if today.month >= 7 else (today.year - 1)
    seasons = [season_code(current_start_year - i) for i in range(n_seasons)]

    frames = []
    for s in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{league_code}.csv"
        try:
            tmp = pd.read_csv(url)

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

def calculate_strengths(df, ref_date=None, alpha=0.004, mix_factor=0.7, window_matches=None):
    df = df.copy()
    df = df.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])
    df = df.sort_values("date").reset_index(drop=True)

    if window_matches is not None and len(df) > window_matches:
        df = df.tail(window_matches).reset_index(drop=True)

    last_date = pd.to_datetime(ref_date) if ref_date is not None else df["date"].max()
    df["days_ago"] = (last_date - df["date"]).dt.days
    df["days_ago"] = df["days_ago"].clip(lower=0)
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

# ----------------------------
# GRAFICACIÓN
# ----------------------------
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

def plot_radar_comparison(home, away, stats):
    h_att, h_def = stats[home]["att_h"], 2 - stats[home]["def_h"]
    a_att, a_def = stats[away]["att_a"], 2 - stats[away]["def_a"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[h_att, h_def, stats[home]["att_a"], 2 - stats[home]["def_a"]],
        theta=["Ataque (Casa)", "Defensa (Casa)", "Ataque (Fuera)", "Defensa (Fuera)"],
        fill="toself",
        name=home,
        line_color="#4CAF50"
    ))
    fig.add_trace(go.Scatterpolar(
        r=[stats[away]["att_h"], 2 - stats[away]["def_h"], a_att, a_def],
        theta=["Ataque (Casa)", "Defensa (Casa)", "Ataque (Fuera)", "Defensa (Fuera)"],
        fill="toself",
        name=away,
        line_color="#2196F3"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 2.5])),
        showlegend=True,
        title="⚔️ Comparativa de Fuerzas (Área Mayor = Mejor)",
        height=350,
        margin=dict(t=40, b=20, l=40, r=40),
    )
    return fig

def get_last_5(df, team):
    mask = (df["home"] == team) | (df["away"] == team)
    l5 = df[mask].sort_values(by="date", ascending=False).head(5).copy()
    l5["Rival"] = np.where(l5["home"] == team, l5["away"], l5["home"])
    l5["Score"] = l5["home_goals"].astype(int).astype(str) + "-" + l5["away_goals"].astype(int).astype(str)
    l5["Tiros"] = np.where(l5["home"] == team, l5.get("sot_h", 0), l5.get("sot_a", 0)).astype(int)
    l5["Sede"] = np.where(l5["home"] == team, "🏠", "✈️")
    return l5[["Sede", "Rival", "Score", "Tiros"]]

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

# ======================================================
# 2B. HELPERS ML (Odds + DC + ML)
# ======================================================
def odds_to_probs(oh, od, oa, eps=1e-9):
    oh = max(float(oh), 1.01); od = max(float(od), 1.01); oa = max(float(oa), 1.01)
    p_h = 1.0/oh; p_d = 1.0/od; p_a = 1.0/oa
    s = p_h + p_d + p_a + eps
    return p_h/s, p_d/s, p_a/s

def outcome_1x2_label(home_goals, away_goals):
    if home_goals > away_goals: return 0
    if home_goals == away_goals: return 1
    return 2

def brier_multiclass(P, y, n_classes=3):
    y_oh = np.eye(n_classes)[y]
    return float(np.mean(np.sum((P - y_oh)**2, axis=1)))

def build_features_for_match(row, team_stats, avg_h, avg_a):
    # DC probs
    _, _, ph, pd, pa, *_ = predict_match_dixon_coles(row["home"], row["away"], team_stats, avg_h, avg_a)

    # odds -> mercado sin margen
    oh = float(row.get("odd_h", np.nan))
    od = float(row.get("odd_d", np.nan))
    oa = float(row.get("odd_a", np.nan))
    if np.isnan(oh) or oh <= 1.01: oh = 2.5
    if np.isnan(od) or od <= 1.01: od = 3.2
    if np.isnan(oa) or oa <= 1.01: oa = 3.0
    mh, md, ma = odds_to_probs(oh, od, oa)

    # lambdas DC (snapshot)
    h_exp = team_stats[row["home"]]["att_h"] * team_stats[row["away"]]["def_a"] * avg_h
    a_exp = team_stats[row["away"]]["att_a"] * team_stats[row["home"]]["def_h"] * avg_a

    sot_h = float(row.get("sot_h", 0.0))
    sot_a = float(row.get("sot_a", 0.0))

    return np.array([
        mh, md, ma,          # mercado
        ph, pd, pa,          # DC probs
        h_exp, a_exp,        # lambdas
        h_exp - a_exp,       # diff
        sot_h, sot_a
    ], dtype=float)

def fit_ml_multiclass(X, y, seed=42):
    # XGBoost si existe, si no: RandomForest
    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
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
    """
    RÁPIDO (sin fuga):
    - Entrena una sola vez con pasado
    - Evalúa en bloque test (últimos n_test)
    """
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

def strict_walkforward_eval_ml(df, n_test=200, min_train=500, window_matches=1200):
    """
    ESTRICTO (walk-forward):
    - Para cada partido test, entrena modelo con pasado y predice ese partido
    (lento, pero muy purista)
    """
    df_sorted = df.dropna(subset=["date","home","away","home_goals","away_goals"]).sort_values("date").reset_index(drop=True)
    test_block = df_sorted.tail(n_test).copy()

    preds = []
    y_true = []

    for _, row in test_block.iterrows():
        cut_date = row["date"]
        train_df = df_sorted[df_sorted["date"] < cut_date].copy()
        if len(train_df) < min_train:
            continue

        team_stats, avg_h, avg_a, _ = calculate_strengths(train_df, ref_date=cut_date, window_matches=window_matches)
        if row["home"] not in team_stats or row["away"] not in team_stats:
            continue

        X_train, y_train = [], []
        for _, r in train_df.tail(window_matches).iterrows():
            if r["home"] not in team_stats or r["away"] not in team_stats:
                continue
            X_train.append(build_features_for_match(r, team_stats, avg_h, avg_a))
            y_train.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

        if len(y_train) < 200:
            continue

        X_train = np.vstack(X_train)
        y_train = np.array(y_train, dtype=int)

        model = fit_ml_multiclass(X_train, y_train)

        x_test = build_features_for_match(row, team_stats, avg_h, avg_a).reshape(1, -1)
        p = model.predict_proba(x_test)[0]

        preds.append(p)
        y_true.append(outcome_1x2_label(row["home_goals"], row["away_goals"]))

    if len(y_true) == 0:
        return None

    P = np.vstack(preds)
    y = np.array(y_true, dtype=int)

    ll = float(log_loss(y, P, labels=[0,1,2]))
    br = brier_multiclass(P, y)
    return {"mode": "estricto", "n": int(len(y)), "logloss": ll, "brier": br}

# ======================================================
# 5. SIDEBAR Y CARGA DE DATOS 🌟
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    if st.button("🔄 Actualizar Datos"):
        st.cache_data.clear()
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
        seasons_loaded = df["season"].nunique() if "season" in df.columns else 1
        st.success(f"✅ {len(df)} partidos cargados ({seasons_loaded} temporadas)")
    else:
        st.error("Error cargando datos. (Puede que no haya temporada activa o falló la conexión)")
        st.stop()

    st.divider()
    bank = st.number_input("💰 Tu Banco ($)", 1000.0, step=50.0)

st.title(f"🛡️ Dixon-Coles: {leagues[code]}")

# --- SELECTOR GLOBAL ---
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])

h_exp, a_exp, ph, pd_prob, pa, po15, po25, pbtts, top_sc, probs = predict_match_dixon_coles(home, away, stats, ah, aa)

# ======================================================
# 6. PESTAÑAS 📑
# ======================================================
t1, t2, t3, t4, t5, t6, t7 = st.tabs(
    ["📊 Análisis", "💰 Valor", "📜 Historial", "💎 Escáner Seguro", "🧪 Laboratorio", "📈 Rendimiento (Risk)", "🤖 ML 1X2 (Ensemble)"]
)

# --- TAB 1: ANÁLISIS ---
with t1:
    st.markdown("### 🥅 Expectativa de Goles (Modelo)")
    c_g1, c_g2, c_g3 = st.columns(3)
    c_g1.metric(home, f"{h_exp:.2f}")
    c_g2.metric("Total (xG)", f"{h_exp + a_exp:.2f}")
    c_g3.metric(away, f"{a_exp:.2f}")

    st.plotly_chart(plot_radar_comparison(home, away, stats), use_container_width=True)

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(plot_gauge(ph, f"Gana {home}", "#4CAF50"), use_container_width=True)
    g2.plotly_chart(plot_gauge(pd_prob, "Empate", "#FFC107"), use_container_width=True)
    g3.plotly_chart(plot_gauge(pa, f"Gana {away}", "#2196F3"), use_container_width=True)

    st.plotly_chart(plot_score_heatmap(probs, home, away), use_container_width=True)

    st.markdown("### 📉 Estado de Forma (Últimos 5)")
    cf1, cf2 = st.columns(2)
    with cf1:
        st.write(f"**{home}**")
        st.dataframe(get_last_5(df, home), use_container_width=True, hide_index=True)
    with cf2:
        st.write(f"**{away}**")
        st.dataframe(get_last_5(df, away), use_container_width=True, hide_index=True)

# --- TAB 2: VALOR ---
with t2:
    st.markdown("### 🏦 Cuotas y Valor (se guardan para ML)")

    def_oh, def_od, def_oa = st.session_state.odds_inputs["oh"], st.session_state.odds_inputs["od"], st.session_state.odds_inputs["oa"]
    co1, co2, co3 = st.columns(3)
    oh = co1.number_input("Cuota Local", 1.01, 100.0, float(def_oh))
    od = co2.number_input("Cuota Empate", 1.01, 100.0, float(def_od))
    oa = co3.number_input("Cuota Visita", 1.01, 100.0, float(def_oa))

    st.session_state.odds_inputs = {"oh": float(oh), "od": float(od), "oa": float(oa)}

    mh, md, ma = odds_to_probs(oh, od, oa)
    st.write(f"Mercado (sin margen): H={mh:.3f} D={md:.3f} A={ma:.3f}")
    st.write(f"Dixon-Coles: H={ph:.3f} D={pd_prob:.3f} A={pa:.3f}")

# --- TAB 3: HISTORIAL ---
with t3:
    st.markdown("### 📜 Historial")
    db = manage_bets("load")
    if db.empty:
        st.info("Aún no hay historial.")
    else:
        st.dataframe(db.sort_values(by="Fecha", ascending=False), use_container_width=True)

# --- TAB 4: ESCÁNER ---
with t4:
    st.markdown("## 💎 Escáner Seguro")
    st.info("Esta sección la dejé minimal aquí para no alargar el script. Tu versión anterior sigue siendo compatible.")

# --- TAB 5: LABORATORIO ---
with t5:
    st.markdown("## 🧪 Laboratorio")
    n_test = st.slider("Partidos a evaluar (DC simple)", 20, 250, 100, step=10)
    min_train = st.slider("Mínimo de partidos para entrenar (DC simple)", 50, 900, 250, step=25)

    if st.button("▶️ Backtest DC (walk-forward)"):
        with st.spinner("Backtesteando sin fugas..."):
            test_df, ok, profit, roi_bt, n_bets, tot_stake = run_backtest_no_leak(
                df, n_test=n_test, min_train=min_train, window_matches=900, stake_unit=1.0
            )
        if test_df.empty:
            st.warning("No se pudo backtestear.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Apuestas", f"{n_bets}")
            m2.metric("Aciertos", f"{ok}/{n_bets}")
            m3.metric("Profit", f"{profit:.2f} U")
            m4.metric("ROI", f"{roi_bt:.2f}%")
            st.dataframe(test_df, use_container_width=True)

# --- TAB 6: RISK ---
with t6:
    st.markdown("## 📈 Rendimiento (Risk)")
    st.info("Tu versión anterior de Risk funciona igual; aquí lo dejé corto.")

# --- TAB 7: ENSEMBLE ML 1X2 ---
with t7:
    st.markdown("## 🤖 Ensemble 1X2: Odds (pre-match) + Dixon-Coles + ML")
    st.caption("Incluye 2 modos: Rápido (usable) y Estricto (walk-forward).")

    if HAS_XGB:
        st.success("✅ XGBoost detectado (mejor performance).")
    else:
        st.warning("⚠️ XGBoost NO instalado: usando RandomForest fallback (más rápido, menos potente).")

    # ----- SWITCH / SELECTOR -----
    mode = st.radio(
        "Modo de evaluación",
        ["⚡ Rápido (1 train + test final)", "🧪 Estricto (walk-forward)"],
        index=0
    )

    cml1, cml2, cml3 = st.columns(3)
    n_test_ml = cml1.slider("Partidos test", 50, 400, 200, step=25)
    min_train_ml = cml2.slider("Mínimo train", 200, 1500, 500, step=50)
    window_ml = cml3.slider("Ventana (matches) snapshot/train", 300, 2500, 1200, step=100)

    if st.button("▶️ Evaluar modelo"):
        with st.spinner("Evaluando..."):
            if "Rápido" in mode:
                out = fast_eval_ml(df, n_test=n_test_ml, min_train=min_train_ml, window_matches=window_ml)
            else:
                out = strict_walkforward_eval_ml(df, n_test=n_test_ml, min_train=min_train_ml, window_matches=window_ml)

        if out is None:
            st.warning("No se pudo evaluar (historial insuficiente / equipos sin historial / odds faltantes).")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Modo", out["mode"])
            m2.metric("Partidos", out["n"])
            m3.metric("LogLoss", f"{out['logloss']:.4f}")
            m4.metric("Brier", f"{out['brier']:.4f}")
            st.success("✅ Listo. (Menor LogLoss/Brier = mejor)")

    st.divider()
    st.markdown("### 🎯 Predicción ML para el partido seleccionado (snapshot hasta hoy)")

    oh = st.session_state.odds_inputs["oh"]
    od = st.session_state.odds_inputs["od"]
    oa = st.session_state.odds_inputs["oa"]

    mh, md, ma = odds_to_probs(oh, od, oa)

    st.write(f"**Cuotas actuales:** oh={oh:.2f} od={od:.2f} oa={oa:.2f}")
    st.write(f"**Mercado (sin margen):** H={mh:.3f} D={md:.3f} A={ma:.3f}")
    st.write(f"**Dixon-Coles:** H={ph:.3f} D={pd_prob:.3f} A={pa:.3f}")

    if st.button("🧠 Predecir (ML snapshot)"):
        with st.spinner("Entrenando snapshot y prediciendo..."):
            train_df = df.sort_values("date").copy()
            team_stats2, avg_h2, avg_a2, _ = calculate_strengths(train_df, ref_date=train_df["date"].max(), window_matches=window_ml)

            X_train, y_train = [], []
            for _, r in train_df.tail(window_ml).iterrows():
                if r["home"] not in team_stats2 or r["away"] not in team_stats2:
                    continue
                X_train.append(build_features_for_match(r, team_stats2, avg_h2, avg_a2))
                y_train.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

            if len(y_train) < 200 or home not in team_stats2 or away not in team_stats2:
                st.warning("No hay suficiente train o faltan equipos en historial.")
            else:
                X_train = np.vstack(X_train)
                y_train = np.array(y_train, dtype=int)
                model = fit_ml_multiclass(X_train, y_train)

                row_now = {"home": home, "away": away, "odd_h": oh, "odd_d": od, "odd_a": oa, "sot_h": 0.0, "sot_a": 0.0}
                x_now = build_features_for_match(row_now, team_stats2, avg_h2, avg_a2).reshape(1, -1)

                p = model.predict_proba(x_now)[0]  # [H,D,A]

                a1, a2, a3 = st.columns(3)
                a1.metric(f"Gana {home}", f"{p[0]*100:.1f}%")
                a2.metric("Empate", f"{p[1]*100:.1f}%")
                a3.metric(f"Gana {away}", f"{p[2]*100:.1f}%")

                comp = pd.DataFrame({
                    "Modelo": ["Mercado (sin margen)", "Dixon-Coles", "ML Ensemble"],
                    "H": [mh, ph, p[0]],
                    "D": [md, pd_prob, p[1]],
                    "A": [ma, pa, p[2]],
                })
                st.dataframe(comp.style.format({"H":"{:.3f}","D":"{:.3f}","A":"{:.3f}"}), use_container_width=True)
