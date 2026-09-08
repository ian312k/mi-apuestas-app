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
# 1. CONFIGURACIÓN Y ESTILOS CSS
# ======================================================
st.set_page_config(page_title="Analisis predictivo de futbol (football-data.org)", layout="wide", page_icon="🛡️")
CSV_FILE = "mis_apuestas_pro.csv"
DEFAULT_API_KEY = "67fbbbfe88854afcba6116b35df04daa"

# Códigos oficiales de competiciones en football-data.org
COMPETITION_MAP = {
    "PD": "🇪🇸 La Liga",
    "PL": "🏴 Premier League",
    "SA": "🇮🇹 Serie A",
    "BL1": "🇩🇪 Bundesliga",
    "FL1": "🇫🇷 Ligue 1",
    "DED": "🇳🇱 Eredivisie",
    "PPL": "🇵🇹 Primeira Liga",
}

# Normalizador de nombres oficiales a formato legible
TEAM_MAP = {
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Nottingham Forest FC": "Nott'm Forest",
    "Wolverhampton Wanderers FC": "Wolves",
    "Brighton & Hove Albion FC": "Brighton",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Newcastle United FC": "Newcastle",
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Aston Villa FC": "Aston Villa",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Brentford FC": "Brentford",
    "Crystal Palace FC": "Crystal Palace",
    "AFC Bournemouth": "Bournemouth",
    "Leicester City FC": "Leicester",
    "Ipswich Town FC": "Ipswich",
    "Southampton FC": "Southampton",
    
    "Athletic Club": "Ath Bilbao",
    "Club Atlético de Madrid": "Ath Madrid",
    "Real Betis Balompié": "Betis",
    "RC Celta de Vigo": "Celta",
    "RCD Espanyol de Barcelona": "Espanol",
    "Real Sociedad de Fútbol": "Sociedad",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Deportivo Alavés": "Alaves",
    "FC Barcelona": "Barcelona",
    "Real Madrid CF": "Real Madrid",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "Girona FC": "Girona",
    "Getafe CF": "Getafe",
    "CA Osasuna": "Osasuna",
    "RCD Mallorca": "Mallorca",
    "UD Las Palmas": "Las Palmas",
    "CD Leganés": "Leganes",
    "Real Valladolid CF": "Valladolid",
    
    "FC Internazionale Milano": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Juventus FC": "Juventus",
    "SSC Napoli": "Napoli",
    "SS Lazio": "Lazio",
    "ACF Fiorentina": "Fiorentina",
    "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna",
    "Torino FC": "Torino",
    "Hellas Verona FC": "Verona",
    "Parma Calcio 1913": "Parma",
}

def normalize_name(name):
    clean = str(name).strip()
    return TEAM_MAP.get(clean, clean)

# --- SESSION STATE ---
if "ticket" not in st.session_state: st.session_state.ticket = []
if "api_key" not in st.session_state: st.session_state.api_key = DEFAULT_API_KEY
if "odds_inputs" not in st.session_state:
    st.session_state.odds_inputs = {"oh": 2.0, "od": 3.2, "oa": 3.5, "o_o25": 1.90, "o_btts": 1.90}

st.markdown("""
<style>
    div[data-testid="stMetric"] { background-color: #262730; border: 1px solid #464b5c; padding: 15px; border-radius: 10px; }
    .ticket-box { background-color: #1e1e1e; border: 1px solid #ffd700; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. DATA VIA FOOTBALL-DATA.ORG (MULTITEMPORADA)
# ======================================================
@st.cache_data(ttl=3600, show_spinner="Descargando histórico de varias temporadas desde football-data.org...")
def fetch_competition_data(comp_code, api_key, n_seasons=3):
    today = datetime.now()
    current_season_year = today.year if today.month >= 7 else (today.year - 1)
    target_years = [current_season_year - i for i in range(n_seasons)]

    headers = {"X-Auth-Token": api_key.strip()}
    finished_rows = []
    upcoming_matches = []

    for year in target_years:
        url = f"https://api.football-data.org/v4/competitions/{comp_code}/matches?season={year}"
        try:
            res = requests.get(url, headers=headers, timeout=12)
            if res.status_code == 429:
                return pd.DataFrame(), [], "Error 429: Límite de peticiones alcanzado (10 req/min). Espera un minuto."
            if res.status_code != 200:
                continue

            payload = res.json()
            for m in payload.get("matches", []):
                status = m.get("status")
                h_name = normalize_name(m.get("homeTeam", {}).get("name", ""))
                a_name = normalize_name(m.get("awayTeam", {}).get("name", ""))
                utc_date = m.get("utcDate", "")

                if status == "FINISHED":
                    score = m.get("score", {}).get("fullTime", {})
                    hg = score.get("home")
                    ag = score.get("away")
                    if hg is not None and ag is not None:
                        finished_rows.append({
                            "date": pd.to_datetime(utc_date),
                            "home": h_name,
                            "away": a_name,
                            "home_goals": float(hg),
                            "away_goals": float(ag),
                            "odd_h": 2.5,
                            "odd_d": 3.2,
                            "odd_a": 3.0,
                            "sot_h": 0.0,
                            "sot_a": 0.0,
                            "season": str(year)
                        })
                elif status in ["SCHEDULED", "TIMED"] and year == current_season_year:
                    upcoming_matches.append({
                        "home_team": h_name,
                        "away_team": a_name,
                        "commence_time": utc_date,
                        "matchday": m.get("matchday", 0)
                    })
        except Exception:
            continue

    df_hist = pd.DataFrame(finished_rows)
    if not df_hist.empty:
        df_hist = df_hist.drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)

    return df_hist, upcoming_matches, None

# ======================================================
# 3. DIXON-COLES
# ======================================================
@st.cache_data(ttl=3600)
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
        att_h_pure = (np.average(h_m["home_goals"], weights=h_m["weight"]) / avg_home) if (not h_m.empty and avg_home > 0) else 1.0
        def_h_pure = (np.average(h_m["away_goals"], weights=h_m["weight"]) / avg_away) if (not h_m.empty and avg_away > 0) else 1.0

        a_m = df[df["away"] == team]
        att_a_pure = (np.average(a_m["away_goals"], weights=a_m["weight"]) / avg_away) if (not a_m.empty and avg_away > 0) else 1.0
        def_a_pure = (np.average(a_m["home_goals"], weights=a_m["weight"]) / avg_home) if (not a_m.empty and avg_home > 0) else 1.0

        team_stats[team] = {
            "att_h": (att_h_pure * mix_factor) + (att_global * (1 - mix_factor)),
            "def_h": (def_h_pure * mix_factor) + (def_global * (1 - mix_factor)),
            "att_a": (att_a_pure * mix_factor) + (att_global * (1 - mix_factor)),
            "def_a": (def_a_pure * mix_factor) + (def_global * (1 - mix_factor)),
            "sot_h_avg": 0.0,
            "sot_a_avg": 0.0,
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
# 5. VISUALIZACIÓN
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
    )
    return fig

def safe_fair_odds(p, eps=1e-12):
    p = float(np.clip(p, eps, 1.0))
    return 1.0 / p

# ======================================================
# 6. ML PREDICTION ENGINE
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

def build_features_for_match(row, team_stats, avg_h, avg_a):
    _, _, dc_h, dc_d, dc_a, *_ = predict_match_dixon_coles(row["home"], row["away"], team_stats, avg_h, avg_a)
    oh = float(row.get("odd_h", 2.5))
    od = float(row.get("odd_d", 3.2))
    oa = float(row.get("odd_a", 3.0))
    mk_h, mk_d, mk_a = odds_to_probs(oh, od, oa)

    h_exp = team_stats[row["home"]]["att_h"] * team_stats[row["away"]]["def_a"] * avg_h
    a_exp = team_stats[row["away"]]["att_a"] * team_stats[row["home"]]["def_h"] * avg_a

    return np.array([
        mk_h, mk_d, mk_a,
        dc_h, dc_d, dc_a,
        h_exp, a_exp,
        h_exp - a_exp,
        0.0, 0.0
    ], dtype=float)

@st.cache_data(ttl=1800)
def train_snapshot_cached(df, window_matches=600, seed=42):
    df_sorted = df.sort_values("date").copy()
    team_stats, avg_h, avg_a, _ = calculate_strengths(df_sorted, ref_date=df_sorted["date"].max(), window_matches=window_matches)

    X_train, y_train = [], []
    for _, r in df_sorted.tail(window_matches).iterrows():
        if r["home"] not in team_stats or r["away"] not in team_stats:
            continue
        X_train.append(build_features_for_match(r, team_stats, avg_h, avg_a))
        y_train.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

    if len(y_train) < 30:
        return None

    X_train = np.vstack(X_train)
    y_train = np.array(y_train, dtype=int)

    if HAS_XGB:
        model = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, objective="multi:softprob", num_class=3, random_state=seed, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)

    model.fit(X_train, y_train)
    return model, team_stats, avg_h, avg_a

# ======================================================
# 7. SIDEBAR & DATA LOAD
# ======================================================
with st.sidebar:
    st.header("⚙️ football-data.org API")
    api_key_input = st.text_input("API Token:", value=st.session_state.api_key, type="password")
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.cache_data.clear()
        st.rerun()

    code = st.selectbox("Liga", list(COMPETITION_MAP.keys()), format_func=lambda x: COMPETITION_MAP[x])
    n_seasons_load = st.slider("Temporadas históricas a cargar", min_value=1, max_value=4, value=3, step=1)

    if st.button("🔄 Recargar Datos"):
        st.cache_data.clear()
        st.rerun()

    df, upcoming_matches, error_msg = fetch_competition_data(code, st.session_state.api_key, n_seasons=n_seasons_load)

    if error_msg:
        st.error(error_msg)
        st.stop()

    if not df.empty:
        stats, ah, aa, teams = calculate_strengths(df, ref_date=df["date"].max(), window_matches=800)
        st.success(f"✅ {len(df)} partidos terminados cargados")
        st.info(f"📅 {len(upcoming_matches)} próximos partidos")
    else:
        st.warning("No hay partidos terminados para esta liga en las temporadas consultadas.")
        st.stop()

    st.divider()
    bank = st.number_input("💰 Tu Banco ($)", 1000.0, step=50.0)

    if st.session_state.ticket:
        st.divider()
        st.markdown(f"**Ticket:** {len(st.session_state.ticket)} selecciones")
        if st.button("🗑️ Limpiar"):
            st.session_state.ticket = []
            st.rerun()

st.title(f"Pronósticos de Fútbol: {COMPETITION_MAP[code]}")

# --- SELECTOR DE EQUIPOS ---
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])

h_exp, a_exp, ph, pd_prob, pa, po15, po25, pbtts, top_sc, probs = predict_match_dixon_coles(home, away, stats, ah, aa)

# ======================================================
# 8. TABS PRINCIPALES
# ======================================================
t1, t2, t3, t4, t5 = st.tabs(["📊 Análisis Partido", "💰 Valor & Ticket", "📅 Próximos Partidos", "🤖 Machine Learning", "📜 Historial"])

# --- TAB 1: ANÁLISIS ---
with t1:
    st.markdown("### 🥅 Expectativa de Goles (xG Modelo)")
    a, b, c = st.columns(3)
    a.metric(home, f"{h_exp:.2f}")
    b.metric("Total xG", f"{h_exp + a_exp:.2f}")
    c.metric(away, f"{a_exp:.2f}")

    st.divider()
    st.markdown("### 🏁 Probabilidades 1X2")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"🏠 {home}", f"{ph*100:.1f}%")
    m2.metric("🤝 Empate", f"{pd_prob*100:.1f}%")
    m3.metric(f"✈️ {away}", f"{pa*100:.1f}%")

    fo_h, fo_d, fo_a = safe_fair_odds(ph), safe_fair_odds(pd_prob), safe_fair_odds(pa)
    st.info(f"💡 **Cuotas Justas del Modelo (Sin margen de la casa):** Local = {fo_h:.2f} | Empate = {fo_d:.2f} | Visitante = {fo_a:.2f}")

    g1, g2, g3 = st.columns(3)
    g1.metric("Over 1.5", f"{po15*100:.1f}%")
    g2.metric("Over 2.5", f"{po25*100:.1f}%")
    g3.metric("BTTS (Ambos anotan)", f"{pbtts*100:.1f}%")

    st.plotly_chart(plot_score_heatmap(probs, home, away), use_container_width=True)

# --- TAB 2: VALOR & TICKET ---
with t2:
    col_analisis, col_ticket = st.columns([2, 1])

    with col_analisis:
        st.markdown("### 🏦 Comparador de Cuotas Reales vs Modelo")
        st.caption("Ingresa los momios de tu casa de apuestas para verificar si existe Valor Esperado (+EV):")

        co1, co2, co3 = st.columns(3)
        oh = co1.number_input("Cuota Local", 1.01, 100.0, float(st.session_state.odds_inputs["oh"]))
        od = co2.number_input("Cuota Empate", 1.01, 100.0, float(st.session_state.odds_inputs["od"]))
        oa = co3.number_input("Cuota Visitante", 1.01, 100.0, float(st.session_state.odds_inputs["oa"]))

        cx1, cx2 = st.columns(2)
        odd_o25 = cx1.number_input("Cuota Over 2.5", 1.01, 100.0, float(st.session_state.odds_inputs["o_o25"]))
        odd_btts = cx2.number_input("Cuota BTTS (Sí)", 1.01, 100.0, float(st.session_state.odds_inputs["o_btts"]))

        st.session_state.odds_inputs = {"oh": float(oh), "od": float(od), "oa": float(oa), "o_o25": float(odd_o25), "o_btts": float(odd_btts)}

        k_ev_h = (ph * oh) - 1
        k_ev_d = (pd_prob * od) - 1
        k_ev_a = (pa * oa) - 1
        k_max_ev = max(k_ev_h, k_ev_d, k_ev_a)

        if k_max_ev > 0:
            if k_max_ev == k_ev_h: sel, p_sel, o_sel = f"Gana {home}", ph, oh
            elif k_max_ev == k_ev_d: sel, p_sel, o_sel = "Empate", pd_prob, od
            else: sel, p_sel, o_sel = f"Gana {away}", pa, oa

            pct_k = calculate_kelly(p_sel, o_sel)
            st.success(f"💎 **Apuesta de Valor encontrada:** {sel} (+{(k_max_ev*100):.1f}% EV) | Kelly Stake: ${(pct_k/100)*bank:.2f}")
        else:
            st.warning("📉 Sin valor esperado positivo en el 1X2.")

        st.divider()
        with st.form("add_ticket_form"):
            pick_sel = st.selectbox("Añadir al Ticket", [f"Gana {home}", "Empate", f"Gana {away}", "Over 2.5", "BTTS"])
            if st.form_submit_button("➕ Agregar Selección"):
                odd_val = oh if "Gana " + home in pick_sel else (od if pick_sel == "Empate" else oa)
                pr_val = ph if "Gana " + home in pick_sel else (pd_prob if pick_sel == "Empate" else pa)
                st.session_state.ticket.append({"match": f"{home} vs {away}", "pick": pick_sel, "odd": odd_val, "prob": pr_val, "league": COMPETITION_MAP[code]})
                st.rerun()

    with col_ticket:
        st.markdown("### 🎫 Ticket Activo")
        if not st.session_state.ticket:
            st.info("Sin selecciones en el ticket.")
        else:
            tot_odd, tot_prob = 1.0, 1.0
            for idx, item in enumerate(st.session_state.ticket):
                st.markdown(f"<div class='ticket-box'><strong>{item['match']}</strong><br>{item['pick']} @ {item['odd']:.2f}</div>", unsafe_allow_html=True)
                if st.button("❌", key=f"btn_del_{idx}"):
                    st.session_state.ticket.pop(idx)
                    st.rerun()
                tot_odd *= item["odd"]
                tot_prob *= item["prob"]

            st.metric("Cuota Combinada", f"{tot_odd:.2f}")
            stake_val = st.number_input("Monto a apostar ($)", 1.0, 5000.0, 50.0)

            if st.button("💾 Guardar en Historial"):
                manage_bets("save", {
                    "ID": pd.Timestamp.now().strftime("%Y%m%d%H%M%S"),
                    "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "Liga": COMPETITION_MAP[code],
                    "Partido": st.session_state.ticket[0]["match"] if len(st.session_state.ticket) == 1 else "Combinada",
                    "Pick": " + ".join([x["pick"] for x in st.session_state.ticket]),
                    "Cuota": round(tot_odd, 2),
                    "Stake": stake_val,
                    "Prob": round(tot_prob, 4),
                    "Estado": "Pendiente",
                    "Ganancia": 0.0
                })
                st.session_state.ticket = []
                st.balloons()
                st.rerun()

# --- TAB 3: PRÓXIMOS PARTIDOS ---
with t3:
    st.markdown("### 📅 Fixture Oficial de Próximos Partidos (football-data.org)")
    if not upcoming_matches:
        st.info("No hay partidos programados próximamente para esta liga.")
    else:
        fix_rows = []
        for match in upcoming_matches:
            h_name = match["home_team"]
            a_name = match["away_team"]

            m_h = get_close_matches(h_name, teams, n=1, cutoff=0.6)
            m_a = get_close_matches(a_name, teams, n=1, cutoff=0.6)

            if m_h and m_a and m_h[0] in stats and m_a[0] in stats:
                _, _, p_h, p_d, p_a, *_ = predict_match_dixon_coles(m_h[0], m_a[0], stats, ah, aa)
                fix_rows.append({
                    "Fecha (UTC)": pd.to_datetime(match["commence_time"]).strftime("%d/%m %H:%M"),
                    "Partido": f"{m_h[0]} vs {m_a[0]}",
                    "Prob. Local": f"{p_h*100:.1f}%",
                    "Prob. Empate": f"{p_d*100:.1f}%",
                    "Prob. Visita": f"{p_a*100:.1f}%",
                    "Cuotas Justas": f"{safe_fair_odds(p_h):.2f} | {safe_fair_odds(p_d):.2f} | {safe_fair_odds(p_a):.2f}"
                })

        if fix_rows:
            st.dataframe(pd.DataFrame(fix_rows), use_container_width=True)
        else:
            st.write("Equipos de la próxima jornada no coinciden aún con los nombres históricos cargados.")

# --- TAB 4: ML ENGINE ---
with t4:
    st.markdown("### 🤖 Predicción con Ensamble Machine Learning")
    if st.button("🧠 Entrenar y Predecir Partido Actual"):
        with st.spinner("Entrenando modelo sobre histórico..."):
            snap = train_snapshot_cached(df, window_matches=800)
        if snap is None:
            st.warning("Datos históricos insuficientes para entrenar el modelo ML.")
        else:
            model, t_stats, a_h, a_a = snap
            row_now = {"home": home, "away": away, "odd_h": st.session_state.odds_inputs["oh"], "odd_d": st.session_state.odds_inputs["od"], "odd_a": st.session_state.odds_inputs["oa"]}
            x_vec = build_features_for_match(row_now, t_stats, a_h, a_a).reshape(1, -1)
            probs_ml = model.predict_proba(x_vec)[0]

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Gana {home}", f"{probs_ml[0]*100:.1f}%")
            col2.metric("Empate", f"{probs_ml[1]*100:.1f}%")
            col3.metric(f"Gana {away}", f"{probs_ml[2]*100:.1f}%")

# --- TAB 5: HISTORIAL ---
with t5:
    st.markdown("### 📜 Historial de Apuestas Registradas")
    db = manage_bets("load")
    if not db.empty:
        st.dataframe(db.sort_values(by="Fecha", ascending=False), use_container_width=True)
    else:
        st.info("Sin registros guardados todavía.")
