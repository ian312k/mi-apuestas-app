import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os
import requests
from difflib import get_close_matches
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import log_loss

# ======================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS (DARK MODE) 🎨
# ======================================================
st.set_page_config(page_title="Dixon-Coles Pro v6.0 (Hybrid AI)", layout="wide", page_icon="🧠")
CSV_FILE = "mis_apuestas_pro.csv"
N_SEASONS = 3  # ✅ Toma las últimas 3 temporadas

# --- GESTIÓN DE ESTADO (SESSION STATE) ---
if "ticket" not in st.session_state: st.session_state.ticket = []
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "api_odds_cache" not in st.session_state: st.session_state.api_odds_cache = {}
if "api_usage" not in st.session_state: st.session_state.api_usage = {"used": 0, "remaining": 500}
if "market_storage" not in st.session_state: st.session_state.market_storage = {}
if "ml_model" not in st.session_state: st.session_state.ml_model = None
if "ml_metrics" not in st.session_state: st.session_state.ml_metrics = None

# Estilos CSS
st.markdown("""
<style>
    div[data-testid="stMetric"] { background-color: #262730; border: 1px solid #464b5c; padding: 15px; border-radius: 10px; }
    .ticket-box { background-color: #1e1e1e; border: 1px solid #ffd700; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    .success-box { background-color: #1c2e24; border: 1px solid #4CAF50; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. FUNCIONES LÓGICAS (DATA & CORE) 🧠
# ======================================================

@st.cache_data(ttl=3600)
def fetch_live_soccer_data(league_code="SP1", n_seasons=3):
    """ Descarga y concatena N temporadas desde football-data.co.uk """
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

    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def call_api_real(sport_key, api_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h&oddsFormat=decimal&apiKey={api_key}"
    try:
        res = requests.get(url, timeout=20)
        if res.status_code == 200:
            return {"success": True, "data": res.json(), "used": int(res.headers.get("x-requests-used", 0)), "remaining": int(res.headers.get("x-requests-remaining", 500))}
        return {"success": False, "error": f"Error {res.status_code}", "message": res.text}
    except Exception as e:
        return {"success": False, "error": "Excepción", "message": str(e)}

# ----------------------------
# STRENGTHS (Calculadora de Fuerza)
# ----------------------------
def calculate_strengths(df, ref_date=None, alpha=0.004, mix_factor=0.7, window_matches=None):
    df = df.copy().dropna(subset=["date", "home", "away", "home_goals", "away_goals"])
    df = df.sort_values("date").reset_index(drop=True)

    if window_matches is not None and len(df) > window_matches:
        df = df.tail(window_matches).reset_index(drop=True)

    last_date = pd.to_datetime(ref_date) if ref_date is not None else df["date"].max()
    df["days_ago"] = (last_date - df["date"]).dt.days
    df["days_ago"] = df["days_ago"].clip(lower=0)
    df["weight"] = np.exp(-alpha * df["days_ago"])

    if df.empty or df["weight"].sum() == 0: return {}, 0.0, 0.0, []

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
        else: att_global, def_global = 1.0, 1.0

        h_m = df[df["home"] == team]
        if not h_m.empty:
            att_h_pure = (np.average(h_m["home_goals"], weights=h_m["weight"]) / avg_home) if avg_home > 0 else 1.0
            def_h_pure = (np.average(h_m["away_goals"], weights=h_m["weight"]) / avg_away) if avg_away > 0 else 1.0
            sot_h_avg = np.average(h_m["sot_h"], weights=h_m["weight"]) if "sot_h" in h_m.columns else 0.0
        else: att_h_pure, def_h_pure, sot_h_avg = 1.0, 1.0, 0.0

        a_m = df[df["away"] == team]
        if not a_m.empty:
            att_a_pure = (np.average(a_m["away_goals"], weights=a_m["weight"]) / avg_away) if avg_away > 0 else 1.0
            def_a_pure = (np.average(a_m["home_goals"], weights=a_m["weight"]) / avg_home) if avg_home > 0 else 1.0
            sot_a_avg = np.average(a_m["sot_a"], weights=a_m["weight"]) if "sot_a" in a_m.columns else 0.0
        else: att_a_pure, def_a_pure, sot_a_avg = 1.0, 1.0, 0.0

        team_stats[team] = {
            "att_h": (att_h_pure * mix_factor) + (att_global * (1 - mix_factor)),
            "def_h": (def_h_pure * mix_factor) + (def_global * (1 - mix_factor)),
            "att_a": (att_a_pure * mix_factor) + (att_global * (1 - mix_factor)),
            "def_a": (def_a_pure * mix_factor) + (def_global * (1 - mix_factor)),
            "sot_h_avg": sot_h_avg, "sot_a_avg": sot_a_avg,
        }
    return team_stats, avg_home, avg_away, all_teams

# ----------------------------
# MODELO DIXON-COLES
# ----------------------------
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
            if x==0 and y==0: correction = 1.0 - (h_exp*a_exp*rho)
            elif x==0 and y==1: correction = 1.0 + (h_exp*rho)
            elif x==1 and y==0: correction = 1.0 + (a_exp*rho)
            elif x==1 and y==1: correction = 1.0 - (rho)
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
    top_scores = [(f"{np.unravel_index(i, probs.shape)[0]}-{np.unravel_index(i, probs.shape)[1]}", probs[np.unravel_index(i, probs.shape)]) for i in flat_indices]
    
    return h_exp, a_exp, p_home, p_draw, p_away, p_o15, p_o25, p_btts, top_scores, probs

# ----------------------------
# BACKTEST SIN FUGAS (Walk-Forward)
# ----------------------------
def run_backtest_no_leak(df, n_test=50, min_train=200, window_matches=800, stake_unit=1.0):
    df_sorted = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    test_block = df_sorted.tail(n_test)
    results = []
    correct, bal, n_bets = 0, 0.0, 0

    for _, row in test_block.iterrows():
        cut_date = row["date"]
        train_df = df_sorted[df_sorted["date"] < cut_date].copy()
        if len(train_df) < min_train: continue

        team_stats, avg_h, avg_a, _ = calculate_strengths(train_df, ref_date=cut_date, window_matches=window_matches)
        if row["home"] not in team_stats or row["away"] not in team_stats: continue

        _, _, ph, pd_prob, pa, *_ = predict_match_dixon_coles(row["home"], row["away"], team_stats, avg_h, avg_a)
        
        odd_h, odd_d, odd_a = float(row.get("odd_h", 1.0)), float(row.get("odd_d", 1.0)), float(row.get("odd_a", 1.0))
        if odd_h <= 1.01 or odd_d <= 1.01 or odd_a <= 1.01: continue

        if ph > pd_prob and ph > pa: pred, prob, odd = "Local", ph, odd_h
        elif pa > ph and pa > pd_prob: pred, prob, odd = "Visita", pa, odd_a
        else: pred, prob, odd = "Empate", pd_prob, odd_d
        
        if row["home_goals"] > row["away_goals"]: res_real = "Local"
        elif row["away_goals"] > row["home_goals"]: res_real = "Visita"
        else: res_real = "Empate"

        is_win = (pred == res_real)
        profit_u = (odd - 1) * stake_unit if is_win else -stake_unit
        correct += int(is_win); bal += profit_u; n_bets += 1

        results.append({
            "Fecha": row["date"].strftime("%Y-%m-%d"), "Partido": f"{row['home']} vs {row['away']}",
            "Predicción": f"{pred} ({prob*100:.0f}%)", "Realidad": f"{int(row['home_goals'])}-{int(row['away_goals'])}",
            "Cuota": odd, "Res": "✅" if is_win else "❌", "P/L(U)": profit_u
        })

    total_stake = n_bets * stake_unit
    roi = (bal / total_stake * 100) if total_stake > 0 else 0.0
    return pd.DataFrame(results), correct, bal, roi, n_bets, total_stake

# ======================================================
# 🧠 FUNCIÓN PARA ENTRENAR XGBOOST HÍBRIDO (NUEVO)
# ======================================================
def train_hybrid_model(df, window_matches=800, train_size=500):
    """
    Genera dataset para ML: [Probs Dixon-Coles, Cuotas Mercado] -> [Resultado]
    Evalúa con LogLoss.
    """
    df_sorted = df.dropna(subset=["date", "odd_h", "odd_d", "odd_a"]).sort_values("date").reset_index(drop=True)
    if len(df_sorted) > train_size + 100:
        dataset = df_sorted.tail(train_size + 100).copy()
    else:
        dataset = df_sorted.copy()
        
    X, y = [], []
    progress_text = "Generando features para IA..."
    my_bar = st.progress(0, text=progress_text)
    total_rows = len(dataset)
    start_idx = 100 
    
    for i in range(start_idx, total_rows):
        if i % 10 == 0: my_bar.progress(i / total_rows, text=f"{progress_text} ({i}/{total_rows})")
        
        row = dataset.iloc[i]
        cut_date = row['date']
        
        # 1. Dixon-Coles al momento (Sin Fugas)
        history = dataset[dataset['date'] < cut_date]
        if len(history) < 50: continue
        
        stats_mom, ah, aa, _ = calculate_strengths(history, ref_date=cut_date, window_matches=window_matches)
        if row['home'] not in stats_mom or row['away'] not in stats_mom: continue
            
        _, _, ph, pd_prob, pa, _, _, _, _, _ = predict_match_dixon_coles(row['home'], row['away'], stats_mom, ah, aa)
        
        # 2. Features de Mercado
        if row['odd_h'] <= 1.01: continue 
        margen = (1/row['odd_h'] + 1/row['odd_d'] + 1/row['odd_a'])
        mkt_h = (1/row['odd_h']) / margen
        mkt_d = (1/row['odd_d']) / margen
        mkt_a = (1/row['odd_a']) / margen
        
        # 3. Vector de Features
        features = [ph, pd_prob, pa, mkt_h, mkt_d, mkt_a, ph - mkt_h, pa - mkt_a]
        
        # 4. Target (0:Local, 1:Empate, 2:Visita)
        if row['home_goals'] > row['away_goals']: target = 0
        elif row['home_goals'] == row['away_goals']: target = 1
        else: target = 2
        
        X.append(features)
        y.append(target)

    my_bar.empty()
    if len(X) < 50: return None, None, {}, "Insuficientes datos para ML"

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Entrenar XGBoost
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, objective='multi:softprob', num_class=3, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    
    # Métricas
    preds_test = model.predict_proba(X_test)
    ll_xgb = log_loss(y_test, preds_test)
    
    mkt_probs_test = X_test[:, 3:6]
    ll_mkt = log_loss(y_test, mkt_probs_test)
    
    dc_probs_test = X_test[:, 0:3]
    ll_dc = log_loss(y_test, dc_probs_test)
    
    metrics = {"LogLoss XGB": ll_xgb, "LogLoss Mkt": ll_mkt, "LogLoss DC": ll_dc, "Mejora vs Mkt": (ll_mkt - ll_xgb) / ll_mkt * 100}
    return model, (X_test, y_test), metrics, "OK"

# ----------------------------
# GRAFICACIÓN & UTILS
# ----------------------------
def plot_gauge(val, title, color):
    return go.Figure(go.Indicator(mode="gauge+number", value=val * 100, title={"text": title}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}, "bgcolor": "white"})).update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))

def plot_score_heatmap(probs, home_team, away_team):
    limit = 6
    probs_cut = probs[:limit, :limit]
    fig = go.Figure(data=go.Heatmap(z=probs_cut, x=[f"{away_team} {i}" for i in range(limit)], y=[f"{home_team} {i}" for i in range(limit)], colorscale="Viridis", text=np.round(probs_cut * 100, 1), texttemplate="%{text}%", hoverongaps=False))
    fig.update_layout(title="🔥 Probabilidad de Marcador Exacto", height=450, margin=dict(l=40, r=40, t=40, b=40))
    return fig

def plot_radar_comparison(home, away, stats):
    h_att, h_def = stats[home]["att_h"], 2 - stats[home]["def_h"]
    a_att, a_def = stats[away]["att_a"], 2 - stats[away]["def_a"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[h_att, h_def, stats[home]["att_a"], 2 - stats[home]["def_a"]], theta=["Ataque (Casa)", "Defensa (Casa)", "Ataque (Fuera)", "Defensa (Fuera)"], fill="toself", name=home, line_color="#4CAF50"))
    fig.add_trace(go.Scatterpolar(r=[stats[away]["att_h"], 2 - stats[away]["def_h"], a_att, a_def], theta=["Ataque (Casa)", "Defensa (Casa)", "Ataque (Fuera)", "Defensa (Fuera)"], fill="toself", name=away, line_color="#2196F3"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 2.5])), showlegend=True, title="⚔️ Comparativa de Fuerzas", height=350, margin=dict(t=40, b=20, l=40, r=40))
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
    return max(0.0, ((odd - 1) * prob - (1 - prob)) / (odd - 1) * 0.5) * 100

def manage_bets(mode, data=None, id_bet=None, status=None):
    if os.path.exists(CSV_FILE): df = pd.read_csv(CSV_FILE)
    else: df = pd.DataFrame(columns=["ID", "Fecha", "Liga", "Partido", "Pick", "Cuota", "Stake", "Prob", "Estado", "Ganancia"])
    
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
# 5. SIDEBAR Y CARGA DE DATOS 🌟
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    if st.button("🔄 Actualizar Datos"):
        st.cache_data.clear()
        st.rerun()

    leagues = {"SP1": "🇪🇸 La Liga", "E0": "🏴 Premier League", "I1": "🇮🇹 Serie A", "D1": "🇩🇪 Bundesliga", "F1": "🇫🇷 Ligue 1", "N1": "🇳🇱 Eredivisie", "P1": "🇵🇹 Primeira Liga"}
    code = st.selectbox("Liga", list(leagues.keys()), format_func=lambda x: leagues[x])
    df = fetch_live_soccer_data(code, n_seasons=N_SEASONS)

    if not df.empty:
        stats, ah, aa, teams = calculate_strengths(df, ref_date=df["date"].max(), window_matches=1200)
        st.success(f"✅ {len(df)} partidos cargados ({df['season'].nunique()} temporadas)")
        st.markdown("---")
        last_5 = df.sort_values("date").tail(5).copy().iloc[::-1]
        last_5["Fecha"] = last_5["date"].dt.strftime("%d/%m")
        last_5["Partido"] = last_5["home"] + " vs " + last_5["away"]
        last_5["Score"] = last_5["home_goals"].astype(int).astype(str) + "-" + last_5["away_goals"].astype(int).astype(str)
        st.dataframe(last_5[["Fecha", "Partido", "Score", "season"]], hide_index=True, use_container_width=True)
    else:
        st.error("Error cargando datos."); st.stop()

    st.divider()
    bank = st.number_input("💰 Tu Banco ($)", 1000.0, step=50.0)
    if st.session_state.ticket:
        st.divider(); st.markdown(f"**Ticket:** {len(st.session_state.ticket)} selecciones")
        if st.button("🗑️ Limpiar"): st.session_state.ticket = []; st.rerun()

st.title(f"🛡️ Dixon-Coles Pro: {leagues[code]}")

# --- SELECTOR GLOBAL ---
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])
h_exp, a_exp, ph, pd_prob, pa, po15, po25, pbtts, top_sc, probs = predict_match_dixon_coles(home, away, stats, ah, aa)

# ======================================================
# 6. PESTAÑAS (TABS) 📑
# ======================================================
t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📊 Análisis", "💰 Valor", "📜 Historial", "💎 Escáner Seguro", "🧪 Laboratorio", "📈 Rendimiento (Risk)", "🧠 ML Híbrido"])

# --- TAB 1: ANÁLISIS ---
with t1:
    st.markdown("### 🥅 Expectativa de Goles (Modelo)")
    c_g1, c_g2, c_g3 = st.columns(3)
    c_g1.metric(home, f"{h_exp:.2f}")
    c_g2.metric("Total (xG)", f"{h_exp + a_exp:.2f}")
    c_g3.metric(away, f"{a_exp:.2f}")

    st.markdown("### 🎯 Realidad Ofensiva (Goles vs Tiros al Arco)")
    sot_h_val = stats[home].get("sot_h_avg", 0)
    sot_a_val = stats[away].get("sot_a_avg", 0)
    fig_shot = go.Figure(data=[
        go.Bar(name="Goles Esperados (Modelo)", x=[home, away], y=[h_exp, a_exp], marker_color="#FFA726"),
        go.Bar(name="Prom. Tiros a Puerta (Real)", x=[home, away], y=[sot_h_val, sot_a_val], marker_color="#29B6F6"),
    ])
    fig_shot.update_layout(barmode="group", title="¿Suerte o Talento?", height=300, margin=dict(t=30, b=20))
    st.plotly_chart(fig_shot, use_container_width=True)

    st.plotly_chart(plot_radar_comparison(home, away, stats), use_container_width=True)

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(plot_gauge(ph, f"Gana {home}", "#4CAF50"), use_container_width=True)
    g2.plotly_chart(plot_gauge(pd_prob, "Empate", "#FFC107"), use_container_width=True)
    g3.plotly_chart(plot_gauge(pa, f"Gana {away}", "#2196F3"), use_container_width=True)

    st.plotly_chart(plot_score_heatmap(probs, home, away), use_container_width=True)
    
    st.markdown("### 📉 Estado de Forma (Últimos 5)")
    cf1, cf2 = st.columns(2)
    with cf1: st.write(f"**{home}**"); st.dataframe(get_last_5(df, home), use_container_width=True, hide_index=True)
    with cf2: st.write(f"**{away}**"); st.dataframe(get_last_5(df, away), use_container_width=True, hide_index=True)

# --- TAB 2: VALOR ---
with t2:
    col_analisis, col_ticket = st.columns([2, 1])
    with col_analisis:
        st.markdown("### 🏦 Comparador Inteligente")
        def_oh, def_od, def_oa = 2.0, 3.2, 3.5
        league_data = st.session_state.market_storage.get(code, {})
        found_in_storage = False

        if "data" in league_data:
            for item in league_data["data"]:
                if get_close_matches(item["home_team"], [home], n=1, cutoff=0.5) and get_close_matches(item["away_team"], [away], n=1, cutoff=0.5):
                    if item.get("bookmakers"):
                        book = item["bookmakers"][0]
                        for m in book["markets"][0]["outcomes"]:
                            if m["name"] == item["home_team"]: def_oh = m["price"]
                            elif m["name"] == item["away_team"]: def_oa = m["price"]
                            else: def_od = m["price"]
                        found_in_storage = True
                        break

        if found_in_storage: st.success("✅ Momios cargados (Escáner).")
        else: st.info("ℹ️ Momios por defecto.")

        co1, co2, co3 = st.columns(3)
        oh = co1.number_input("Cuota Local", 1.01, 100.0, float(def_oh))
        od = co2.number_input("Cuota Empate", 1.01, 100.0, float(def_od))
        oa = co3.number_input("Cuota Visita", 1.01, 100.0, float(def_oa))

        st.markdown("#### 🧠 Estrategia Kelly")
        k_ev_h, k_ev_d, k_ev_a = (ph * oh) - 1, (pd_prob * od) - 1, (pa * oa) - 1
        k_max_ev = max(k_ev_h, k_ev_d, k_ev_a)
        
        if k_max_ev > 0:
            if k_max_ev == k_ev_h: k_sel, k_p, k_o = f"Gana {home}", ph, oh
            elif k_max_ev == k_ev_d: k_sel, k_p, k_o = "Empate", pd_prob, od
            else: k_sel, k_p, k_o = f"Gana {away}", pa, oa
            k_pct = calculate_kelly(k_p, k_o)
            st.success(f"💎 **Kelly:** {k_sel} | Stake: ${(k_pct/100)*bank:.2f} ({k_pct:.2f}%)")
        else: st.warning("📉 Kelly sugiere: **No apostar**")

        fig_val = go.Figure(data=[
            go.Bar(name="Tu Modelo", x=[home, "Empate", away], y=[ph, pd_prob, pa], marker_color="#00CC96"),
            go.Bar(name="Casa (Sin Margen)", x=[home, "Empate", away], y=[(1/oh)/(1/oh+1/od+1/oa), (1/od)/(1/oh+1/od+1/oa), (1/oa)/(1/oh+1/od+1/oa)], marker_color="#EF553B"),
        ])
        fig_val.update_layout(barmode="group", height=250, margin=dict(t=20, b=20, l=20, r=20), title="⚖️ Detector de Valor")
        st.plotly_chart(fig_val, use_container_width=True)

        st.divider()
        st.markdown("### ➕ Agregar al Ticket")
        with st.form("add_to_ticket"):
            sel_pick = st.selectbox("Selección", [f"Gana {home}", "Empate", f"Gana {away}"])
            if f"Gana {home}" in sel_pick: sel_odd, sel_prob = oh, ph
            elif "Empate" in sel_pick: sel_odd, sel_prob = od, pd_prob
            else: sel_odd, sel_prob = oa, pa
            if st.form_submit_button("Añadir selección"):
                st.session_state.ticket.append({"match": f"{home} vs {away}", "pick": sel_pick, "odd": sel_odd, "prob": sel_prob, "league": leagues[code]})
                st.rerun()

    with col_ticket:
        st.markdown("### 🎫 Ticket")
        if not st.session_state.ticket: st.info("Vacío")
        else:
            total_odd = 1.0
            for idx, item in enumerate(st.session_state.ticket):
                st.markdown(f"<div class='ticket-box'><small>{item['league']}</small><br><strong>{item['match']}</strong><br>{item['pick']} @ {item['odd']}</div>", unsafe_allow_html=True)
                if st.button("❌", key=f"del_{idx}"): st.session_state.ticket.pop(idx); st.rerun()
                total_odd *= item["odd"]
            st.metric("Cuota Total", f"{total_odd:.2f}")
            stake_parlay = st.number_input("Stake ($)", 1.0, 5000.0, 50.0)
            if st.button("💾 Guardar"):
                tipo_str = "Simple" if len(st.session_state.ticket) == 1 else "Parlay"
                match_str = st.session_state.ticket[0]["match"] if len(st.session_state.ticket) == 1 else f"Combinada ({len(st.session_state.ticket)})"
                pick_str = " + ".join([i["pick"] for i in st.session_state.ticket])
                manage_bets("save", {"ID": pd.Timestamp.now().strftime("%Y%m%d%H%M%S"), "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d"), "Liga": tipo_str, "Partido": match_str, "Pick": pick_str, "Cuota": round(total_odd, 2), "Stake": stake_parlay, "Prob": 0, "Estado": "Pendiente", "Ganancia": 0.0})
                st.session_state.ticket = []; st.balloons(); st.rerun()

# --- TAB 3: HISTORIAL ---
with t3:
    st.markdown("### 📜 Historial")
    db = manage_bets("load")
    if not db.empty:
        df_plot = db.copy().sort_values(by="ID")
        df_plot["Balance Acumulado"] = df_plot["Ganancia"].cumsum()
        fig_bal = go.Figure()
        fig_bal.add_trace(go.Scatter(x=pd.to_datetime(df_plot["Fecha"]), y=df_plot["Balance Acumulado"], mode="lines+markers", name="Balance", line=dict(color="#00ff00" if df_plot["Balance Acumulado"].iloc[-1] >= 0 else "#ff0000", width=3)))
        st.plotly_chart(fig_bal, use_container_width=True)
        st.dataframe(db.sort_values(by="Fecha", ascending=False), use_container_width=True)
        
        c_upd, c_del = st.columns(2)
        with c_upd:
            with st.expander("📝 Actualizar Resultado"):
                pen = db[db["Estado"] == "Pendiente"]
                if not pen.empty:
                    bid = st.selectbox("ID", pen["ID"].unique())
                    res = st.selectbox("Resultado", ["Ganada", "Perdida", "Push"])
                    if st.button("Actualizar"): manage_bets("update", id_bet=bid, status=res); st.rerun()
        with c_del:
            if st.button("🗑️ Borrar Historial"):
                if os.path.exists(CSV_FILE): os.remove(CSV_FILE); st.rerun()

# --- TAB 4: ESCÁNER ---
with t4:
    st.markdown("## 💎 Escáner Seguro")
    api_key_input = st.text_input("🔑 API Key:", value=st.session_state.api_key, type="password")
    if api_key_input != st.session_state.api_key: st.session_state.api_key = api_key_input; st.rerun()
    
    if st.button("⬇️ Descargar Datos"):
        api_league_map = {"SP1": "soccer_spain_la_liga", "E0": "soccer_epl", "I1": "soccer_italy_serie_a", "D1": "soccer_germany_bundesliga"}
        res = call_api_real(api_league_map.get(code, "soccer_spain_la_liga"), st.session_state.api_key)
        if res["success"]:
             st.session_state.market_storage[code] = {"timestamp": datetime.now(), "data": res["data"]}
             st.success("Actualizado"); st.rerun()
        else: st.error(res["message"])
    
    if code in st.session_state.market_storage:
        data = st.session_state.market_storage[code]["data"]
        # Mostrar lógica simplificada
        st.info(f"Datos en memoria: {len(data)} eventos.")

# --- TAB 5: LABORATORIO ---
with t5:
    st.markdown("## 🧪 Backtest Blindado"); n_test = st.slider("Partidos Test", 20, 250, 100)
    if st.button("▶️ Validar (Walk-Forward)"):
        with st.spinner("Procesando..."):
            test_df, ok, profit, roi_bt, n_bets, _ = run_backtest_no_leak(df, n_test=n_test)
        if not test_df.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Bets", n_bets); m2.metric("WinRate", f"{(ok/n_bets)*100:.1f}%"); m3.metric("Profit", f"{profit:.2f} U"); m4.metric("ROI", f"{roi_bt:.2f}%")
            st.dataframe(test_df, use_container_width=True)

# --- TAB 6: RENDIMIENTO (RISK) ---
with t6:
    st.markdown("## 📈 Estadísticas de Rendimiento")
    if os.path.exists(CSV_FILE):
        df_hist = pd.read_csv(CSV_FILE)
        df_finished = df_hist[df_hist["Estado"].isin(["Ganada", "Perdida", "Push"])].copy().sort_values("ID")
        if not df_finished.empty:
            tot_prof = df_finished["Ganancia"].sum()
            roi = (tot_prof / df_finished["Stake"].sum() * 100)
            
            df_finished["Equity"] = df_finished["Ganancia"].cumsum()
            df_finished["Peak"] = df_finished["Equity"].cummax()
            df_finished["Drawdown"] = df_finished["Equity"] - df_finished["Peak"]
            max_dd = df_finished["Drawdown"].min()
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Profit", f"${tot_prof:.2f}"); k2.metric("ROI", f"{roi:.2f}%"); k3.metric("Max Drawdown", f"{max_dd:.2f} U", delta="Riesgo", delta_color="off")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🌊 Drawdown")
                fig_dd = go.Figure(go.Scatter(x=pd.to_datetime(df_finished["Fecha"]), y=df_finished["Drawdown"], fill='tozeroy', line=dict(color='#FF5252')))
                st.plotly_chart(fig_dd, use_container_width=True)
            with c2:
                prof_l = df_finished.groupby("Liga")["Ganancia"].sum().sort_values()
                st.plotly_chart(go.Figure(go.Bar(x=prof_l.values, y=prof_l.index, orientation='h', marker_color=['#FF5252' if x<0 else '#4CAF50' for x in prof_l.values])), use_container_width=True)

# --- TAB 7: ML HIBRIDO (XGBOOST) ---
with t7:
    st.markdown("## 🧠 Dixon-Coles + XGBoost")
    st.info("Inteligencia Artificial que aprende de los errores de Dixon-Coles y del Mercado.")
    
    c_ml1, c_ml2 = st.columns([1, 2])
    with c_ml1:
        st.markdown("### ⚙️ Entrenamiento")
        train_size = st.slider("Partidos para entrenar", 200, 1000, 400)
        
        if st.button("🚀 Entrenar Modelo Híbrido"):
            with st.spinner("Generando dataset y entrenando IA..."):
                model_xgb, test_data, metrics, msg = train_hybrid_model(df, train_size=train_size)
                if msg == "OK":
                    st.session_state['ml_model'] = model_xgb; st.session_state['ml_metrics'] = metrics
                    st.success("Modelo Entrenado")
                else: st.error(msg)
    
    with c_ml2:
        if 'ml_metrics' in st.session_state:
            met = st.session_state['ml_metrics']
            st.markdown("### 📉 Evaluación (LogLoss - Menor es mejor)")
            k1, k2, k3 = st.columns(3)
            k1.metric("LogLoss Mercado", f"{met['LogLoss Mkt']:.4f}")
            k2.metric("LogLoss DC", f"{met['LogLoss DC']:.4f}")
            k3.metric("LogLoss XGB", f"{met['LogLoss XGB']:.4f}", delta=f"{met['Mejora vs Mkt']:.2f}% vs Mkt")
            
            fig_metrics = go.Figure(go.Bar(x=['Dixon-Coles', 'Mercado', 'XGBoost'], y=[met['LogLoss DC'], met['LogLoss Mkt'], met['LogLoss XGB']], marker_color=['#FFC107', '#FF5252', '#4CAF50'], textposition='auto'))
            st.plotly_chart(fig_metrics, use_container_width=True)
            
    st.divider()
    
    if 'ml_model' in st.session_state and st.session_state['ml_model']:
        st.markdown(f"### 🤖 Predicción Híbrida: {home} vs {away}")
        c_p1, c_p2, c_p3 = st.columns(3)
        cur_oh = c_p1.number_input("Cuota Local (Actual)", 1.01, 100.0, 2.0)
        cur_od = c_p2.number_input("Cuota Empate (Actual)", 1.01, 100.0, 3.2)
        cur_oa = c_p3.number_input("Cuota Visita (Actual)", 1.01, 100.0, 3.5)
        
        if st.button("🔮 Predecir con IA"):
            margen_now = (1/cur_oh + 1/cur_od + 1/cur_oa)
            input_vector = np.array([[ph, pd_prob, pa, (1/cur_oh)/margen_now, (1/cur_od)/margen_now, (1/cur_oa)/margen_now, ph - (1/cur_oh)/margen_now, pa - (1/cur_oa)/margen_now]])
            xgb_probs = st.session_state['ml_model'].predict_proba(input_vector)[0]
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write("**Probabilidades Dixon-Coles**")
                st.progress(ph, text=f"Local: {ph*100:.1f}%"); st.progress(pd_prob, text=f"Empate: {pd_prob*100:.1f}%"); st.progress(pa, text=f"Visita: {pa*100:.1f}%")
            with col_res2:
                st.write("**Probabilidades Híbridas (IA)**")
                st.progress(float(xgb_probs[0]), text=f"Local: {xgb_probs[0]*100:.1f}%"); st.progress(float(xgb_probs[1]), text=f"Empate: {xgb_probs[1]*100:.1f}%"); st.progress(float(xgb_probs[2]), text=f"Visita: {xgb_probs[2]*100:.1f}%")
                
            if (xgb_probs[0] * cur_oh) - 1 > 0.05: st.success(f"💎 Valor IA: LOCAL (EV: {((xgb_probs[0]*cur_oh)-1)*100:.1f}%)")
            if (xgb_probs[2] * cur_oa) - 1 > 0.05: st.success(f"💎 Valor IA: VISITA (EV: {((xgb_probs[2]*cur_oa)-1)*100:.1f}%)")
    else: st.warning("⚠️ Entrena el modelo arriba primero.")
