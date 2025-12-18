import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os
import requests
from difflib import get_close_matches
from datetime import datetime

# ======================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS (DARK MODE) 🎨
# ======================================================
st.set_page_config(page_title="Dixon-Coles Pro v5.2 (Validator)", layout="wide", page_icon="🛡️")
CSV_FILE = "mis_apuestas_pro.csv"
N_SEASONS = 3  # ✅ últimas 3 temporadas

# --- GESTIÓN DE ESTADO (SESSION STATE) ---
if "ticket" not in st.session_state: st.session_state.ticket = []
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "api_odds_cache" not in st.session_state: st.session_state.api_odds_cache = {}
if "api_usage" not in st.session_state: st.session_state.api_usage = {"used": 0, "remaining": 500}
if "market_storage" not in st.session_state: st.session_state.market_storage = {}

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
# 2. FUNCIONES LÓGICAS 🧠
# ======================================================

@st.cache_data(ttl=3600)
def fetch_live_soccer_data(league_code="SP1", n_seasons=3):
    """
    Descarga y concatena N temporadas desde football-data.co.uk
    Ej: n_seasons=3 -> 2526 + 2425 + 2324 (según fecha actual)
    """
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

            # defaults
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

# ----------------------------
# STRENGTHS (ref_date evita fuga por ponderación)
# ----------------------------
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

# ----------------------------
# MODELO DIXON-COLES (Poisson + Corrección)
# ----------------------------
def predict_match_dixon_coles(home, away, team_stats, avg_h, avg_a, rho=-0.13, max_goals=10):
    if home not in team_stats or away not in team_stats:
        return 0, 0, 0, 0, 0, 0, 0, 0, [], np.zeros((1, 1))

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
    s = probs.sum()
    if s <= 0:
        probs = np.zeros_like(probs)
        probs[0, 0] = 1.0
    else:
        probs = probs / s

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

# ----------------------------
# MÉTRICAS (probabilísticas)
# ----------------------------
def _clip_prob(p, eps=1e-12):
    return float(np.clip(p, eps, 1 - eps))

def multiclass_log_loss(p_vec, y_idx):
    p = _clip_prob(p_vec[y_idx])
    return -np.log(p)

def multiclass_brier(p_vec, y_idx):
    y = np.zeros(3)
    y[y_idx] = 1.0
    p = np.array(p_vec, dtype=float)
    return float(np.mean((p - y) ** 2))

def implied_probs_no_margin(odd_h, odd_d, odd_a):
    ih, id_, ia = 1.0/odd_h, 1.0/odd_d, 1.0/odd_a
    s = ih + id_ + ia
    return (ih/s, id_/s, ia/s)

# ----------------------------
# BACKTEST SIN FUGAS + ROI + LOGLOSS/BRIER + CALIBRACIÓN + SALTADOS
# ----------------------------
def run_backtest_no_leak(df, n_test=50, min_train=200, window_matches=800, stake_unit=1.0, min_ev=0.0):
    """
    Walk-forward sin fugas.
    Evalúa:
      - ROI, Profit, Accuracy
      - LogLoss / Brier del modelo (1X2)
      - LogLoss / Brier baseline (prob implícita del book sin margen)
      - Calibración (binning por prob del pick)
      - Contadores de saltados
    Estrategia:
      - pick = argmax(model_probs)
      - bet sólo si EV >= min_ev
    """
    df_sorted = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    test_block = df_sorted.tail(n_test)

    results = []
    correct, bal = 0, 0.0
    n_bets = 0

    # métricas prob
    ll_model_list, ll_book_list = [], []
    br_model_list, br_book_list = [], []

    # calibración (por prob del pick)
    calib_rows = []  # (p_pick, is_win)

    skipped = {"min_train": 0, "missing_team": 0, "bad_odds": 0, "no_value": 0}

    for _, row in test_block.iterrows():
        cut_date = row["date"]
        train_df = df_sorted[df_sorted["date"] < cut_date].copy()
        if len(train_df) < min_train:
            skipped["min_train"] += 1
            continue

        team_stats, avg_h, avg_a, _ = calculate_strengths(train_df, ref_date=cut_date, window_matches=window_matches)
        if row["home"] not in team_stats or row["away"] not in team_stats:
            skipped["missing_team"] += 1
            continue

        _, _, ph, pd_prob, pa, *_ = predict_match_dixon_coles(row["home"], row["away"], team_stats, avg_h, avg_a)

        odd_h = float(row.get("odd_h", np.nan))
        odd_d = float(row.get("odd_d", np.nan))
        odd_a = float(row.get("odd_a", np.nan))

        if (np.isnan(odd_h) or odd_h <= 1.01) or (np.isnan(odd_d) or odd_d <= 1.01) or (np.isnan(odd_a) or odd_a <= 1.01):
            skipped["bad_odds"] += 1
            continue

        # outcome real -> índice (0=H,1=D,2=A)
        if row["home_goals"] > row["away_goals"]:
            y_idx = 0
            res_real = "Local"
        elif row["home_goals"] == row["away_goals"]:
            y_idx = 1
            res_real = "Empate"
        else:
            y_idx = 2
            res_real = "Visita"

        p_model = (float(ph), float(pd_prob), float(pa))
        p_book = implied_probs_no_margin(odd_h, odd_d, odd_a)

        ll_model_list.append(multiclass_log_loss(p_model, y_idx))
        ll_book_list.append(multiclass_log_loss(p_book, y_idx))
        br_model_list.append(multiclass_brier(p_model, y_idx))
        br_book_list.append(multiclass_brier(p_book, y_idx))

        # pick del modelo
        pick_idx = int(np.argmax(p_model))
        pick_name = ["Local", "Empate", "Visita"][pick_idx]
        pick_prob = p_model[pick_idx]
        pick_odd = [odd_h, odd_d, odd_a][pick_idx]
        ev = (pick_prob * pick_odd) - 1.0

        # filtro de valor
        if ev < min_ev:
            skipped["no_value"] += 1
            calib_rows.append((pick_prob, 0.0))  # lo contamos para calibración si quieres, pero sin apuesta real
            continue

        is_win = (pick_name == res_real)
        profit_u = (pick_odd - 1) * stake_unit if is_win else -stake_unit

        correct += int(is_win)
        bal += profit_u
        n_bets += 1
        calib_rows.append((pick_prob, float(is_win)))

        results.append({
            "Fecha": row["date"].strftime("%Y-%m-%d"),
            "Temporada": row.get("season", ""),
            "Partido": f"{row['home']} vs {row['away']}",
            "Pick": pick_name,
            "Prob(Pick)": round(pick_prob, 4),
            "EV": round(ev, 4),
            "Cuota": pick_odd,
            "Realidad": f"{int(row['home_goals'])}-{int(row['away_goals'])}",
            "Resultado": res_real,
            "Res": "✅" if is_win else "❌",
            "Stake(U)": stake_unit,
            "P/L(U)": round(profit_u, 3),
        })

    total_stake = n_bets * stake_unit
    roi = (bal / total_stake * 100) if total_stake > 0 else 0.0

    metrics = {
        "logloss_model": float(np.mean(ll_model_list)) if ll_model_list else np.nan,
        "logloss_book": float(np.mean(ll_book_list)) if ll_book_list else np.nan,
        "brier_model": float(np.mean(br_model_list)) if br_model_list else np.nan,
        "brier_book": float(np.mean(br_book_list)) if br_book_list else np.nan,
    }

    df_res = pd.DataFrame(results)

    # Equity + drawdown (del backtest)
    if not df_res.empty:
        df_res["Equity"] = df_res["P/L(U)"].cumsum()
        df_res["Peak"] = df_res["Equity"].cummax()
        df_res["Drawdown"] = df_res["Equity"] - df_res["Peak"]
        max_dd = float(df_res["Drawdown"].min())
    else:
        max_dd = 0.0

    # Calibración por bins (prob del pick)
    calib_df = pd.DataFrame(calib_rows, columns=["p_pick", "is_win"])
    calib_plot = pd.DataFrame()
    if not calib_df.empty:
        bins = np.linspace(0, 1, 6)  # 5 bins
        calib_df["bin"] = pd.cut(calib_df["p_pick"], bins=bins, include_lowest=True)
        calib_plot = calib_df.groupby("bin", dropna=False).agg(
            n=("p_pick", "count"),
            p_avg=("p_pick", "mean"),
            acc=("is_win", "mean")
        ).reset_index()

    # ROI por temporada (solo bets hechos)
    season_summary = pd.DataFrame()
    if not df_res.empty and "Temporada" in df_res.columns:
        tmp = df_res.copy()
        tmp["StakeTot"] = tmp["Stake(U)"]
        season_summary = tmp.groupby("Temporada").agg(
            apuestas=("Stake(U)", "count"),
            profit=("P/L(U)", "sum"),
            stake=("StakeTot", "sum")
        ).reset_index()
        season_summary["ROI%"] = np.where(season_summary["stake"] > 0, season_summary["profit"] / season_summary["stake"] * 100, 0.0)

    return df_res, correct, float(bal), float(roi), int(n_bets), float(total_stake), metrics, skipped, float(max_dd), calib_plot, season_summary

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
    return max(0.0, f * 0.5) * 100  # Kelly fraccional (0.5)

def manage_bets(mode, data=None, id_bet=None, status=None):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["ID", "Fecha", "Liga", "Partido", "Pick", "Cuota", "Stake", "Prob", "Estado", "Ganancia"])

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

    # mode == "load" (o cualquier otro) regresa df
    return df

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

        st.markdown("---")
        st.markdown("###### 🕒 Últimos 5 Registrados:")
        last_5 = df.sort_values("date").tail(5).copy().iloc[::-1]
        last_5["Fecha"] = last_5["date"].dt.strftime("%d/%m")
        last_5["Partido"] = last_5["home"] + " vs " + last_5["away"]
        last_5["Score"] = last_5["home_goals"].astype(int).astype(str) + "-" + last_5["away_goals"].astype(int).astype(str)
        st.dataframe(last_5[["Fecha", "Partido", "Score", "season"]], hide_index=True, use_container_width=True)
    else:
        st.error("Error cargando datos. (Puede que no haya temporada activa o falló la conexión)")
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

# --- SELECTOR GLOBAL ---
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])

h_exp, a_exp, ph, pd_prob, pa, po15, po25, pbtts, top_sc, probs = predict_match_dixon_coles(home, away, stats, ah, aa)

# ======================================================
# 6. PESTAÑAS 📑
# ======================================================
t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Análisis", "💰 Valor", "📜 Historial", "💎 Escáner Seguro", "🧪 Laboratorio (Validador)", "📈 Rendimiento (Risk)"])

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
    fig_shot.update_layout(barmode="group", title="¿Suerte o Talento? (Barra Azul debe ser alta)", height=300, margin=dict(t=30, b=20))
    st.plotly_chart(fig_shot, use_container_width=True)

    st.plotly_chart(plot_radar_comparison(home, away, stats), use_container_width=True)

    mg1, mg2, mg3 = st.columns(3)
    mg1.metric("Over 1.5", f"{po15*100:.1f}%")
    mg2.metric("Over 2.5", f"{po25*100:.1f}%")
    mg3.metric("BTTS", f"{pbtts*100:.1f}%")

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
    col_analisis, col_ticket = st.columns([2, 1])

    with col_analisis:
        st.markdown("### 🏦 Comparador Inteligente")

        def_oh, def_od, def_oa = 2.0, 3.2, 3.5
        league_data = st.session_state.market_storage.get(code, {})
        found_in_storage = False

        if "data" in league_data:
            for item in league_data["data"]:
                h_team_api = item["home_team"]
                a_team_api = item["away_team"]

                m_h = get_close_matches(h_team_api, [home], n=1, cutoff=0.5)
                m_a = get_close_matches(a_team_api, [away], n=1, cutoff=0.5)
                if m_h and m_a:
                    if item.get("bookmakers"):
                        book = item["bookmakers"][0]
                        for m in book["markets"][0]["outcomes"]:
                            if m["name"] == h_team_api: def_oh = m["price"]
                            elif m["name"] == a_team_api: def_oa = m["price"]
                            else: def_od = m["price"]
                        found_in_storage = True
                        break

        if found_in_storage: st.success("✅ Momios cargados automáticamente (Escáner).")
        else: st.info("ℹ️ Momios por defecto (No encontrados en escáner).")

        co1, co2, co3 = st.columns(3)
        oh = co1.number_input("Cuota Local", 1.01, 100.0, float(def_oh))
        od = co2.number_input("Cuota Empate", 1.01, 100.0, float(def_od))
        oa = co3.number_input("Cuota Visita", 1.01, 100.0, float(def_oa))

        st.markdown("#### 🧠 Estrategia Kelly")
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

        # barras modelo vs book sin margen
        imp_h, imp_d, imp_a = implied_probs_no_margin(oh, od, oa)
        fig_val = go.Figure(data=[
            go.Bar(name="Tu Modelo", x=[home, "Empate", away], y=[ph, pd_prob, pa], marker_color="#00CC96"),
            go.Bar(name="Book (Sin Margen)", x=[home, "Empate", away], y=[imp_h, imp_d, imp_a], marker_color="#EF553B"),
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
        df_plot = db.copy().sort_values(by="ID")
        df_plot["Balance Acumulado"] = df_plot["Ganancia"].cumsum()

        fig_bal = go.Figure()
        last_bal = df_plot["Balance Acumulado"].iloc[-1] if not df_plot.empty else 0
        fig_bal.add_trace(go.Scatter(
            x=pd.to_datetime(df_plot["Fecha"], errors="coerce"),
            y=df_plot["Balance Acumulado"],
            mode="lines+markers",
            name="Balance",
            line=dict(color="#00ff00" if last_bal >= 0 else "#ff0000", width=3),
        ))
        st.plotly_chart(fig_bal, use_container_width=True)
        st.dataframe(db.sort_values(by="Fecha", ascending=False), use_container_width=True)

        csv = db.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar Historial (CSV)", data=csv, file_name="mis_apuestas_backup.csv", mime="text/csv")

        c_upd, c_del = st.columns(2)
        with c_upd:
            with st.expander("📝 Actualizar Resultado"):
                pen = db[db["Estado"] == "Pendiente"]
                if not pen.empty:
                    bid = st.selectbox("ID", pen["ID"].unique())
                    res = st.selectbox("Resultado", ["Ganada", "Perdida", "Push"])
                    if st.button("Actualizar"):
                        manage_bets("update", id_bet=bid, status=res)
                        st.rerun()
                else:
                    st.info("No hay pendientes")

        with c_del:
            with st.expander("🗑️ Eliminar Apuesta"):
                ids_all = db["ID"].unique()
                id_del = st.selectbox("Seleccionar ID para borrar", ids_all)
                if st.button("Borrar definitivamente"):
                    manage_bets("delete", id_bet=id_del)
                    st.warning("Apuesta eliminada")
                    st.rerun()

# --- TAB 4: ESCÁNER BLINDADO ---
with t4:
    st.markdown("## 💎 Escáner Seguro")

    if st.session_state.api_usage["used"] > 0:
        pct_used = st.session_state.api_usage["used"] / 500
        st.progress(pct_used, text=f"Llamadas API: {st.session_state.api_usage['used']} / 500 usadas")

    api_league_map = {
        "SP1": "soccer_spain_la_liga",
        "E0": "soccer_epl",
        "I1": "soccer_italy_serie_a",
        "D1": "soccer_germany_bundesliga",
        "F1": "soccer_france_ligue_one",
        "N1": "soccer_netherlands_eredivisie",
        "P1": "soccer_portugal_primeira_liga",
    }

    api_key_input = st.text_input("🔑 API Key (Pega y presiona Enter):", value=st.session_state.api_key, type="password", key="api_key_input")

    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.rerun()

    if st.button("💾 Guardar Key Manualmente"):
        st.session_state.api_key = api_key_input
        st.success("Guardado.")
        st.rerun()

    st.divider()

    if st.session_state.api_key:
        sport_key = api_league_map.get(code)
        has_data = False
        data_to_display = []

        if code in st.session_state.market_storage:
            stored = st.session_state.market_storage[code]
            data_to_display = stored["data"]
            has_data = True
            st.info(f"📂 Datos en memoria. Actualizado: {stored['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.warning("⚠️ Sin datos descargados.")

        if st.button(f"{'🔄 Actualizar' if has_data else '⬇️ Descargar'} Datos (Gasta 1 llamada)"):
            with st.spinner("Conectando..."):
                response = call_api_real(sport_key, st.session_state.api_key)
                if response["success"]:
                    st.session_state.market_storage[code] = {"timestamp": datetime.now(), "data": response["data"]}
                    st.session_state.api_usage["used"] = response["used"]
                    st.session_state.api_usage["remaining"] = response["remaining"]
                    st.success("✅ Descargado.")
                    st.rerun()
                else:
                    st.error(f"Error API: {response['message']}")

        if has_data and data_to_display:
            live_results = []
            for item in data_to_display:
                match_date = pd.to_datetime(item["commence_time"], utc=True, errors="coerce")
                if pd.isna(match_date): continue

                now_utc = pd.Timestamp.now(tz="UTC")
                diff_hours = (match_date - now_utc).total_seconds() / 3600
                if diff_hours > 168 or diff_hours < -5: continue

                h_team_api = item["home_team"]
                a_team_api = item["away_team"]

                odds_h, odds_d, odds_a = 0, 0, 0
                if item.get("bookmakers"):
                    book = item["bookmakers"][0]
                    for m in book["markets"][0]["outcomes"]:
                        if m["name"] == h_team_api: odds_h = m["price"]
                        elif m["name"] == a_team_api: odds_a = m["price"]
                        else: odds_d = m["price"]

                m_h = get_close_matches(h_team_api, teams, n=1, cutoff=0.5)
                m_a = get_close_matches(a_team_api, teams, n=1, cutoff=0.5)

                if m_h and m_a:
                    real_home, real_away = m_h[0], m_a[0]
                    if real_home in stats and real_away in stats:
                        _, _, ph2, pd2, pa2, *_ = predict_match_dixon_coles(real_home, real_away, stats, ah, aa)
                        ev_h = (ph2 * odds_h) - 1
                        ev_a = (pa2 * odds_a) - 1
                        ev_d = (pd2 * odds_d) - 1

                        best_pick, best_ev = "No Bet", -10.0
                        if ev_h > 0: best_pick, best_ev = f"Gana {real_home}", ev_h
                        if ev_a > best_ev and ev_a > 0: best_pick, best_ev = f"Gana {real_away}", ev_a
                        if ev_d > best_ev and ev_d > 0: best_pick, best_ev = "Empate", ev_d

                        live_results.append({
                            "Hora": match_date.strftime("%d/%m %H:%M UTC"),
                            "Partido": f"{real_home} vs {real_away}",
                            "Prob": f"L:{ph2:.2f} E:{pd2:.2f} V:{pa2:.2f}",
                            "Cuotas": f"L:{odds_h} E:{odds_d} V:{odds_a}",
                            "Pick Valor": best_pick,
                            "EV": best_ev,
                        })

            if live_results:
                df_live = pd.DataFrame(live_results).sort_values(by="EV", ascending=False)
                st.markdown(f"### 🎯 Oportunidades (Memoria) - {len(df_live)} Partidos")

                for _, row in df_live.iterrows():
                    color = "#4CAF50" if row["EV"] > 0 else "#FF5252"
                    val_txt = f"+{row['EV']*100:.1f}%" if row["EV"] > 0 else f"{row['EV']*100:.1f}%"
                    st.markdown(f"""
                    <div style="background-color: #262730; border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <div style="display:flex; justify-content:space-between;">
                            <strong>⏰ {row['Hora']} | {row['Partido']}</strong>
                            <span style="color:{color}; font-weight:bold; font-size:1.2em">EV: {val_txt}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.9em; margin-top:5px; color:#ccc;">
                            <span>🧠 {row['Prob']}</span>
                            <span>🏦 {row['Cuotas']}</span>
                        </div>
                        <div style="margin-top:5px; font-size:1.1em;">
                            👉 Recomendación: <strong>{row['Pick Valor']}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Datos descargados, pero no se encontraron partidos compatibles para esta semana.")

# --- TAB 5: LABORATORIO (VALIDACIÓN REAL) ---
with t5:
    st.markdown("## 🧪 Validador: ¿Predice bien de verdad?")
    st.markdown("Aquí validamos **probabilidades**, no solo ‘aciertos’. Si tu modelo es bueno, debería tener **LogLoss/Brier** mejores que el book (sin margen) y buena calibración.")

    n_test = st.slider("Partidos a evaluar", 20, 400, 150, step=10)
    min_train = st.slider("Mínimo de partidos para entrenar", 50, 1200, 300, step=25)
    min_ev = st.slider("Filtro de Valor (EV mínimo)", -0.10, 0.20, 0.00, step=0.01, help="EV = p*odd - 1. Si subes esto, apuestas menos pero más ‘selectivo’.")

    if st.button("▶️ Validar (walk-forward sin fugas)"):
        with st.spinner("Backtesteando sin fugas + métricas..."):
            test_df, ok, profit, roi_bt, n_bets, tot_stake, metrics, skipped, max_dd, calib_plot, season_summary = run_backtest_no_leak(
                df, n_test=n_test, min_train=min_train, window_matches=900, stake_unit=1.0, min_ev=min_ev
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Apuestas (bet)", f"{n_bets}")
        c2.metric("Aciertos", f"{ok}/{max(1,n_bets)} ({(ok/max(1,n_bets))*100:.0f}%)")
        c3.metric("Profit", f"{profit:.2f} U")
        c4.metric("ROI", f"{roi_bt:.2f}%")
        c5.metric("Max Drawdown", f"{max_dd:.2f} U")

        st.markdown("### 🧾 ¿Cuántos partidos se saltaron y por qué?")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Saltados: min_train", skipped["min_train"])
        s2.metric("Saltados: equipos sin historial", skipped["missing_team"])
        s3.metric("Saltados: cuotas malas", skipped["bad_odds"])
        s4.metric("Saltados: sin valor (EV)", skipped["no_value"])

        st.markdown("### 📉 Métricas probabilísticas (lo que mide si ‘predice bien’)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LogLoss (Modelo)", f"{metrics['logloss_model']:.4f}" if np.isfinite(metrics["logloss_model"]) else "N/A")
        m2.metric("LogLoss (Book)", f"{metrics['logloss_book']:.4f}" if np.isfinite(metrics["logloss_book"]) else "N/A")
        m3.metric("Brier (Modelo)", f"{metrics['brier_model']:.4f}" if np.isfinite(metrics["brier_model"]) else "N/A")
        m4.metric("Brier (Book)", f"{metrics['brier_book']:.4f}" if np.isfinite(metrics["brier_book"]) else "N/A")

        st.caption("Interpretación rápida: **más bajo = mejor** (en LogLoss y Brier). Si tu modelo no mejora al book, ‘acertar’ puede ser engañoso.")

        if not calib_plot.empty:
            st.markdown("### 🎯 Calibración (¿cuando dices 60% realmente ganas ~60%?)")
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Bar(x=calib_plot["bin"].astype(str), y=calib_plot["n"], name="N (muestras)"))
            fig_cal.add_trace(go.Scatter(x=calib_plot["bin"].astype(str), y=calib_plot["p_avg"], mode="lines+markers", name="Prob media"))
            fig_cal.add_trace(go.Scatter(x=calib_plot["bin"].astype(str), y=calib_plot["acc"], mode="lines+markers", name="Acierto real"))
            fig_cal.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), title="Calibración por bins (Prob vs Acierto)")
            st.plotly_chart(fig_cal, use_container_width=True)

        if not season_summary.empty:
            st.markdown("### 🗓️ ROI por temporada (solo bets hechos)")
            st.dataframe(season_summary.sort_values("Temporada"), use_container_width=True, hide_index=True)

        if not test_df.empty:
            st.markdown("### 📈 Curva de equity (backtest)")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=test_df["Fecha"], y=test_df["Equity"], mode="lines+markers", name="Equity"))
            fig_eq.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Fecha", yaxis_title="Unidades")
            st.plotly_chart(fig_eq, use_container_width=True)

            st.markdown("### 📜 Detalle de apuestas (backtest)")
            st.dataframe(test_df, use_container_width=True)

        else:
            st.warning("No se generaron apuestas (con esos filtros). Baja min_train o min_ev, o sube n_test.")

# --- TAB 6: RENDIMIENTO (BI + RIESGO) ---
with t6:
    st.markdown("## 📈 Estadísticas de Rendimiento (tu historial real)")
    if os.path.exists(CSV_FILE):
        df_hist = pd.read_csv(CSV_FILE)
        df_finished = df_hist[df_hist["Estado"].isin(["Ganada", "Perdida", "Push"])].copy()

        if not df_finished.empty:
            df_finished = df_finished.sort_values("ID")

            tot_inv = df_finished["Stake"].sum()
            tot_prof = df_finished["Ganancia"].sum()
            roi = (tot_prof / tot_inv * 100) if tot_inv > 0 else 0.0

            df_finished["Equity"] = df_finished["Ganancia"].cumsum()
            df_finished["Peak"] = df_finished["Equity"].cummax()
            df_finished["Drawdown"] = df_finished["Equity"] - df_finished["Peak"]
            max_dd = df_finished["Drawdown"].min()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Beneficio Neto", f"${tot_prof:,.2f}")
            k2.metric("ROI", f"{roi:.2f}%")
            k3.metric("Max Drawdown", f"{max_dd:.2f} U")
            k4.metric("Apuestas", len(df_finished))

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🌊 Drawdown (Riesgo)")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=pd.to_datetime(df_finished["Fecha"], errors="coerce"),
                    y=df_finished["Drawdown"],
                    fill="tozeroy",
                    mode="lines",
                    name="Drawdown"
                ))
                fig_dd.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Unidades bajo el pico")
                st.plotly_chart(fig_dd, use_container_width=True)

            with c2:
                st.markdown("##### 📊 Ganancia por Liga")
                prof_league = df_finished.groupby("Liga")["Ganancia"].sum().sort_values()
                fig_l = go.Figure(go.Bar(x=prof_league.values, y=prof_league.index, orientation="h"))
                fig_l.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_l, use_container_width=True)

        else:
            st.info("No hay apuestas finalizadas para analizar.")
    else:
        st.warning("Aún no hay historial.")
