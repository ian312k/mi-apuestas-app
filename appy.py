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
st.set_page_config(page_title="Dixon-Coles Pro v3.4 (xG Lite)", layout="wide", page_icon="⚽")
CSV_FILE = "mis_apuestas_pro.csv"

# --- GESTIÓN DE ESTADO (SESSION STATE) ---
if "ticket" not in st.session_state: st.session_state.ticket = []
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "api_odds_cache" not in st.session_state: st.session_state.api_odds_cache = {}
if "api_usage" not in st.session_state: st.session_state.api_usage = {"used": 0, "remaining": 500}
if "market_storage" not in st.session_state: st.session_state.market_storage = {}  # almacén blindado

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
def fetch_live_soccer_data(league_code="SP1"):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    try:
        df = pd.read_csv(url)

        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A', 'HST', 'AST']
        actual_cols = [c for c in cols if c in df.columns]
        df = df[actual_cols]

        rename_map = {
            'Date': 'date', 'HomeTeam': 'home', 'AwayTeam': 'away',
            'FTHG': 'home_goals', 'FTAG': 'away_goals',
            'B365H': 'odd_h', 'B365D': 'odd_d', 'B365A': 'odd_a',
            'HST': 'sot_h', 'AST': 'sot_a'
        }
        df = df.rename(columns=rename_map)

        # defaults
        if 'odd_h' not in df.columns: df['odd_h'] = 1.0
        if 'odd_d' not in df.columns: df['odd_d'] = 1.0
        if 'odd_a' not in df.columns: df['odd_a'] = 1.0
        if 'sot_h' not in df.columns: df['sot_h'] = 0
        if 'sot_a' not in df.columns: df['sot_a'] = 0

        df = df.dropna(subset=['home', 'away', 'home_goals', 'away_goals'])
        df = df.fillna(0)

        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
        return df
    except Exception:
        return pd.DataFrame()

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
# STRENGTHS (SIN FUGAS)
# ref_date = fecha de corte (para que los pesos no usen futuro)
# ----------------------------
def calculate_strengths(df, ref_date=None, alpha=0.004, mix_factor=0.7, window_matches=None):
    df = df.copy()
    df = df.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])

    df = df.sort_values("date").reset_index(drop=True)
    if window_matches is not None and len(df) > window_matches:
        df = df.tail(window_matches).reset_index(drop=True)

    last_date = pd.to_datetime(ref_date) if ref_date is not None else df["date"].max()
    df["days_ago"] = (last_date - df["date"]).dt.days

    # Por seguridad: si algo raro deja days_ago negativo, lo recortamos
    df["days_ago"] = df["days_ago"].clip(lower=0)

    df["weight"] = np.exp(-alpha * df["days_ago"])

    # Evitar promedio ponderado sobre vacío
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

# ----------------------------
# BACKTEST SIN FUGAS (walk-forward)
# Recalcula strengths SOLO con partidos previos al partido evaluado
# ----------------------------
def run_backtest_no_leak(df, n_test=50, min_train=200, window_matches=600):
    df_sorted = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    test_block = df_sorted.tail(n_test)

    results = []
    correct, bal = 0, 0

    for _, row in test_block.iterrows():
        cut_date = row["date"]
        train_df = df_sorted[df_sorted["date"] < cut_date].copy()

        if len(train_df) < min_train:
            continue

        team_stats, avg_h, avg_a, _ = calculate_strengths(train_df, ref_date=cut_date, window_matches=window_matches)

        if row["home"] not in team_stats or row["away"] not in team_stats:
            continue

        _, _, ph, pd_prob, pa, _, _, _, _, _ = predict_match_dixon_coles(
            row["home"], row["away"], team_stats, avg_h, avg_a
        )

        # cuotas: si vienen “rellenas” o faltan, no sirven para P/L realista
        odd_h = float(row.get("odd_h", np.nan))
        odd_d = float(row.get("odd_d", np.nan))
        odd_a = float(row.get("odd_a", np.nan))
        if (np.isnan(odd_h) or odd_h <= 1.01) or (np.isnan(odd_d) or odd_d <= 1.01) or (np.isnan(odd_a) or odd_a <= 1.01):
            # igual guardamos el resultado sin P/L si quieres, pero aquí lo saltamos
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
        profit = (odd - 1) if is_win else -1

        correct += int(is_win)
        bal += profit

        results.append({
            "Partido": f"{row['home']} vs {row['away']}",
            "Predicción": f"{pred} ({prob*100:.0f}%)",
            "Realidad": f"{int(row['home_goals'])}-{int(row['away_goals'])}",
            "Cuota": odd,
            "Res": "✅" if is_win else "❌",
            "P/L": profit
        })

    return pd.DataFrame(results), correct, bal

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

    # "load" o cualquier otro modo regresa df
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
    df = fetch_live_soccer_data(code)

    if not df.empty:
        # stats para "partido actual": usa todo el historial (esto NO es fuga para predicción en vivo)
        stats, ah, aa, teams = calculate_strengths(df, ref_date=df["date"].max(), window_matches=900)
        st.success(f"✅ {len(df)} partidos cargados")
        st.markdown("---")
        st.markdown("###### 🕒 Últimos 5 Registrados (Liga):")
        last_5 = df.sort_values("date").tail(5).copy().iloc[::-1]
        last_5["Fecha"] = last_5["date"].dt.strftime("%d/%m")
        last_5["Partido"] = last_5["home"] + " vs " + last_5["away"]
        last_5["Score"] = last_5["home_goals"].astype(int).astype(str) + "-" + last_5["away_goals"].astype(int).astype(str)
        st.dataframe(last_5[["Fecha", "Partido", "Score"]], hide_index=True, use_container_width=True)
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

st.title(f"🤖 Dixon-Coles: {leagues[code]}")

# --- SELECTOR GLOBAL ---
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])

h_exp, a_exp, ph, pd_prob, pa, po15, po25, pbtts, top_sc, probs = predict_match_dixon_coles(home, away, stats, ah, aa)

# ======================================================
# 6. PESTAÑAS 📑
# ======================================================
t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Análisis", "💰 Valor", "📜 Historial", "💎 Escáner Seguro", "🧪 Laboratorio", "📈 Rendimiento"])

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

        imp_h = (1 / oh); imp_d = (1 / od); imp_a = (1 / oa)
        total_imp = imp_h + imp_d + imp_a
        imp_h /= total_imp; imp_d /= total_imp; imp_a /= total_imp

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

        fig_val = go.Figure(data=[
            go.Bar(name="Tu Modelo", x=[home, "Empate", away], y=[ph, pd_prob, pa], marker_color="#00CC96"),
            go.Bar(name="Casa (Sin Margen)", x=[home, "Empate", away], y=[imp_h, imp_d, imp_a], marker_color="#EF553B"),
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
                if pd.isna(match_date):
                    continue

                now_utc = pd.Timestamp.now(tz="UTC")
                diff_hours = (match_date - now_utc).total_seconds() / 3600
                if diff_hours > 168 or diff_hours < -5:
                    continue

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
                        _, _, ph2, pd2, pa2, _, _, _, _, _ = predict_match_dixon_coles(real_home, real_away, stats, ah, aa)
                        ev_h = (ph2 * odds_h) - 1
                        ev_a = (pa2 * odds_a) - 1
                        ev_d = (pd2 * odds_d) - 1

                        best_pick, best_ev = "No Bet", -10.0
                        if ev_h > 0: best_pick, best_ev = f"Gana {real_home}", ev_h
                        if ev_a > best_ev and ev_a > 0: best_pick, best_ev = f"Gana {real_away}", ev_a
                        if ev_d > best_ev and ev_d > 0: best_pick, best_ev = "Empate", ev_d

                        live_results.append({
                            "Hora": match_date.tz_convert("UTC").strftime("%d/%m %H:%M"),
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

# --- TAB 5: LABORATORIO ---
with t5:
    st.markdown("## 🧪 Laboratorio de Simulación")

    st.markdown("### 🎲 Simulador Monte Carlo (Partido Actual)")
    st.info(f"Simulando: **{home} vs {away}**")
    if st.button("▶️ Ejecutar Monte Carlo (1,000 Partidos)"):
        sim_h = np.random.poisson(h_exp, 1000)
        sim_a = np.random.poisson(a_exp, 1000)
        sim_diff = sim_h - sim_a

        wins_h = np.sum(sim_diff > 0)
        draws = np.sum(sim_diff == 0)
        wins_a = np.sum(sim_diff < 0)

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Local Gana", f"{wins_h/10:.1f}%")
        sc2.metric("Empate", f"{draws/10:.1f}%")
        sc3.metric("Visita Gana", f"{wins_a/10:.1f}%")

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Histogram(x=sim_h, name=home, marker_color="#4CAF50", opacity=0.75))
        fig_sim.add_trace(go.Histogram(x=sim_a, name=away, marker_color="#2196F3", opacity=0.75))
        fig_sim.update_layout(barmode="overlay", title="Distribución de Goles Simulados", xaxis_title="Goles")
        st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()
    st.markdown("### 📜 Backtest Histórico (SIN FUGAS)")
    n_test = st.slider("Partidos a evaluar", 20, 120, 50, step=10)
    min_train = st.slider("Mínimo de partidos para entrenar", 50, 400, 200, step=25)

    if st.button("▶️ Validar (walk-forward sin fuga)"):
        with st.spinner("Backtesteando sin fugas..."):
            test_df, ok, profit = run_backtest_no_leak(df, n_test=n_test, min_train=min_train, window_matches=600)

        if test_df.empty:
            st.warning("No se pudo backtestear (faltan cuotas reales o historial insuficiente).")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Aciertos", f"{ok}/{len(test_df)} ({ok/max(1,len(test_df))*100:.0f}%)")
            m2.metric("Profit", f"{profit:.2f} U")
            m3.metric("Estado", "🔥 Rentable" if profit > 0 else "❄️ Pérdidas")
            st.dataframe(test_df, use_container_width=True)

# --- TAB 6: RENDIMIENTO (BI) ---
with t6:
    st.markdown("## 📈 Estadísticas de Rendimiento")
    if os.path.exists(CSV_FILE):
        df_hist = pd.read_csv(CSV_FILE)
        df_finished = df_hist[df_hist["Estado"].isin(["Ganada", "Perdida", "Push"])].copy()

        if not df_finished.empty:
            tot_inv = df_finished["Stake"].sum()
            tot_prof = df_finished["Ganancia"].sum()
            roi = (tot_prof / tot_inv * 100) if tot_inv > 0 else 0

            k1, k2, k3 = st.columns(3)
            k1.metric("Beneficio Neto", f"${tot_prof:,.2f}")
            k2.metric("ROI", f"{roi:.2f}%")
            k3.metric("Apuestas", len(df_finished))

            c1, c2 = st.columns(2)
            with c1:
                prof_league = df_finished.groupby("Liga")["Ganancia"].sum().sort_values()
                fig_l = go.Figure(go.Bar(x=prof_league.values, y=prof_league.index, orientation="h"))
                st.plotly_chart(fig_l, use_container_width=True)

            with c2:
                st.write("**Distribución**")
                st.plotly_chart(go.Figure(go.Pie(labels=df_finished["Estado"].unique(), values=df_finished["Estado"].value_counts())), use_container_width=True)
        else:
            st.info("No hay apuestas finalizadas para analizar.")
    else:
        st.warning("Aún no hay historial.")
