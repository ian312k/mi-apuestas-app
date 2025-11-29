import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os

# ======================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS 🎨
# ======================================================
st.set_page_config(page_title="Poisson Pro", layout="wide", page_icon="⚽")
CSV_FILE = 'mis_apuestas_poisson.csv'

# Inyectamos CSS para que se vea bonito (Tarjetas y métricas)
st.markdown("""
<style>
    /* Estilo para las métricas (cajitas de números) */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    /* Centrar títulos */
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. LÓGICA MATEMÁTICA 🧠 (Igual que antes)
# ======================================================
@st.cache_data
def fetch_live_soccer_data(league_code="SP1"):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
        df.columns = ['date', 'home', 'away', 'home_goals', 'away_goals']
        df = df.dropna()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        return df
    except: return pd.DataFrame()

def calculate_strengths(df):
    last_date = df['date'].max()
    df['days_ago'] = (last_date - df['date']).dt.days
    alpha = 0.005 
    df['weight'] = np.exp(-alpha * df['days_ago'])
    avg_home = np.average(df['home_goals'], weights=df['weight'])
    avg_away = np.average(df['away_goals'], weights=df['weight'])
    
    team_stats = {}
    all_teams = sorted(list(set(df['home'].unique()) | set(df['away'].unique())))
    
    for team in all_teams:
        home_matches = df[df['home'] == team]
        if not home_matches.empty:
            att_h = np.average(home_matches['home_goals'], weights=home_matches['weight']) / avg_home
            def_h = np.average(home_matches['away_goals'], weights=home_matches['weight']) / avg_away
        else: att_h, def_h = 1.0, 1.0

        away_matches = df[df['away'] == team]
        if not away_matches.empty:
            att_a = np.average(away_matches['away_goals'], weights=away_matches['weight']) / avg_away
            def_a = np.average(away_matches['home_goals'], weights=away_matches['weight']) / avg_home
        else: att_a, def_a = 1.0, 1.0
        team_stats[team] = {'att_h': att_h, 'def_h': def_h, 'att_a': att_a, 'def_a': def_a}
    return team_stats, avg_home, avg_away, all_teams

def get_last_5_matches(df, team_name):
    mask = (df['home'] == team_name) | (df['away'] == team_name)
    last_5 = df[mask].sort_values(by='date', ascending=False).head(5).copy()
    last_5['Fecha'] = last_5['date'].dt.strftime('%d/%m')
    last_5['Rival'] = np.where(last_5['home'] == team_name, last_5['away'], last_5['home'])
    last_5['Res'] = last_5['home_goals'].astype(int).astype(str) + "-" + last_5['away_goals'].astype(int).astype(str)
    last_5['L/V'] = np.where(last_5['home'] == team_name, '🏠', '✈️')
    return last_5[['Fecha', 'L/V', 'Rival', 'Res']]

def predict_match_poisson(home, away, team_stats, avg_h, avg_a):
    h_exp = team_stats[home]['att_h'] * team_stats[away]['def_a'] * avg_h
    a_exp = team_stats[away]['att_a'] * team_stats[home]['def_h'] * avg_a
    max_goals = 10
    probs = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            probs[i][j] = poisson.pmf(i, h_exp) * poisson.pmf(j, a_exp)
    p_home = np.tril(probs, -1).sum()
    p_draw = np.diag(probs).sum()
    p_away = np.triu(probs, 1).sum()
    p_o25 = 0
    for i in range(max_goals):
        for j in range(max_goals):
            if (i + j) > 2.5: p_o25 += probs[i][j]
    
    flat_indices = np.argsort(probs.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, probs.shape)
        top_scores.append((f"{i}-{j}", probs[i][j]))
    return h_exp, a_exp, p_home, p_draw, p_away, p_o25, top_scores

def calculate_kelly_criterion(prob_real, odd_bookie):
    if prob_real <= 0 or odd_bookie <= 1: return 0.0
    b = odd_bookie - 1
    f = (b * prob_real - (1 - prob_real)) / b
    return max(0.0, f * 0.5) * 100

# ======================================================
# 3. FUNCIONES VISUALES NUEVAS (VELOCÍMETROS) 🏎️
# ======================================================
def plot_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        title = {'text': title, 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [{'range': [0, 100], 'color': '#f0f2f6'}]
        }
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# ======================================================
# 4. GESTIÓN DE APUESTAS 💰
# ======================================================
def load_bets():
    if os.path.exists(CSV_FILE): return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["ID", "Fecha", "Liga", "Partido", "Pick", "Cuota", "Stake", "Prob_Modelo", "Estado", "Ganancia"])

def save_bet(bet_data):
    df = load_bets()
    df = pd.concat([df, pd.DataFrame([bet_data])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def update_bet_status(bet_id, new_status):
    df = load_bets()
    idx = df[df['ID'].astype(str) == str(bet_id)].index
    if not idx.empty:
        i = idx[0]
        df.at[i, 'Estado'] = new_status
        stake, odd = float(df.at[i, 'Stake']), float(df.at[i, 'Cuota'])
        df.at[i, 'Ganancia'] = (stake * odd) - stake if new_status == 'Ganada' else (-stake if new_status == 'Perdida' else 0.0)
        df.to_csv(CSV_FILE, index=False)
        return True
    return False

# ======================================================
# 5. INTERFAZ GRÁFICA (UI) MEJORADA 🌟
# ======================================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    league_map = {"SP1": "🇪🇸 La Liga", "E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "I1": "🇮🇹 Serie A", "D1": "🇩🇪 Bundesliga", "F1": "🇫🇷 Ligue 1"}
    league_code = st.selectbox("Liga", list(league_map.keys()), format_func=lambda x: league_map[x])
    df_data = fetch_live_soccer_data(league_code)
    
    if not df_data.empty:
        stats, avg_h, avg_a, teams_list = calculate_strengths(df_data)
        st.success(f"✅ {len(df_data)} partidos cargados")
    else:
        st.error("Error de conexión"); st.stop()
    
    st.divider()
    bankroll = st.number_input("💰 Tu Banco Actual ($)", value=1000.0, step=50.0)

# --- TÍTULO Y SELECCIÓN ---
st.title(f"⚽ {league_map[league_code]} Dashboard")

c1, c2 = st.columns(2)
home_team = c1.selectbox("Local", teams_list, index=0)
away_team = c2.selectbox("Visitante", [t for t in teams_list if t != home_team], index=0)

h_exp, a_exp, p_home, p_draw, p_away, p_o25, top_scores = predict_match_poisson(home_team, away_team, stats, avg_h, avg_a)

# --- VISUALIZACIÓN PRINCIPAL (PESTAÑAS) ---
tab1, tab2, tab3 = st.tabs(["📊 Análisis Visual", "💰 Valor & Apuesta", "📜 Historial"])

with tab1:
    # 1. TARJETAS DE GOLES
    st.markdown("### 🥅 Expectativa de Goles")
    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.metric(f"{home_team}", f"{h_exp:.2f}", delta="Goles Esperados")
    col_g2.metric("Total Partido", f"{h_exp + a_exp:.2f}", delta="Overs")
    col_g3.metric(f"{away_team}", f"{a_exp:.2f}", delta="Goles Esperados")
    
    st.divider()
    
    # 2. VELOCÍMETROS (GAUGES)
    st.markdown("### 🏆 Probabilidades de Victoria")
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(plot_gauge(p_home, f"Gana {home_team}", "#4CAF50"), use_container_width=True)
    g2.plotly_chart(plot_gauge(p_draw, "Empate", "#FFC107"), use_container_width=True)
    g3.plotly_chart(plot_gauge(p_away, f"Gana {away_team}", "#2196F3"), use_container_width=True)

    # 3. MARCADORES EXACTOS
    st.info(f"🎯 **Marcador más probable:** {top_scores[0][0]} ({top_scores[0][1]*100:.1f}%) | **Segunda opción:** {top_scores[1][0]} ({top_scores[1][1]*100:.1f}%)")
    
    # 4. ESTADO DE FORMA
    st.markdown("### 📉 Estado de Forma (Últimos 5)")
    cf1, cf2 = st.columns(2)
    with cf1: st.dataframe(get_last_5_matches(df_data, home_team), hide_index=True, use_container_width=True)
    with cf2: st.dataframe(get_last_5_matches(df_data, away_team), hide_index=True, use_container_width=True)

with tab2:
    st.markdown("### 🏦 Cazador de Valor")
    col_odd1, col_odd2, col_odd3 = st.columns(3)
    odd_h = col_odd1.number_input("Cuota Local", 1.01, 20.0, 2.00)
    odd_d = col_odd2.number_input("Cuota Empate", 1.01, 20.0, 3.20)
    odd_a = col_odd3.number_input("Cuota Visita", 1.01, 20.0, 3.50)
    
    # Kelly & Valor
    ev_h, kelly_h = (p_home * odd_h) - 1, calculate_kelly_criterion(p_home, odd_h)
    ev_d, kelly_d = (p_draw * odd_d) - 1, calculate_kelly_criterion(p_draw, odd_d)
    ev_a, kelly_a = (p_away * odd_a) - 1, calculate_kelly_criterion(p_away, odd_a)
    
    st.divider()
    
    # Mostrar alertas de Valor
    def show_card(label, ev, kelly, odd):
        if ev > 0:
            st.success(f"✅ **{label}** (Cuota {odd})")
            st.markdown(f"**Rentabilidad:** +{ev*100:.1f}% | **Apostar:** ${bankroll * (kelly/100):.2f}")
        else:
            st.error(f"❌ **{label}**: No apostar (EV {ev*100:.1f}%)")

    c_val1, c_val2, c_val3 = st.columns(3)
    with c_val1: show_card(f"Gana {home_team}", ev_h, kelly_h, odd_h)
    with c_val2: show_card("Empate", ev_d, kelly_d, odd_d)
    with c_val3: show_card(f"Gana {away_team}", ev_a, kelly_a, odd_a)

    st.divider()
    st.subheader("📝 Registrar Apuesta")
    with st.form("bet_form"):
        pick_sel = st.selectbox("Tu Elección", [f"Gana {home_team}", "Empate", f"Gana {away_team}"])
        stake_in = st.number_input("Dinero Apostado ($)", 1.0, 5000.0, 50.0)
        
        if "Gana " + home_team in pick_sel: f_odd, f_prob = odd_h, p_home
        elif "Empate" in pick_sel: f_odd, f_prob = odd_d, p_draw
        else: f_odd, f_prob = odd_a, p_away
        
        if st.form_submit_button("💾 Guardar Apuesta"):
            save_bet({"ID": pd.Timestamp.now().strftime('%Y%m%d%H%M%S'), "Fecha": pd.Timestamp.now().strftime('%Y-%m-%d'), 
                      "Liga": league_code, "Partido": f"{home_team} vs {away_team}", "Pick": pick_sel, 
                      "Cuota": f_odd, "Stake": stake_in, "Prob_Modelo": round(f_prob, 4), "Estado": "Pendiente", "Ganancia": 0.0})
            st.success("Guardada!"); st.rerun()

with tab3:
    st.markdown("### 📜 Historial")
    df_bets = load_bets()
    if not df_bets.empty:
        profit = df_bets['Ganancia'].sum()
        st.metric("Ganancia Neta", f"${profit:.2f}", delta_color="normal")
        st.dataframe(df_bets.sort_values(by="Fecha", ascending=False), use_container_width=True)
        
        with st.expander("🔄 Actualizar Resultados"):
            pends = df_bets[df_bets['Estado'] == 'Pendiente']
            if not pends.empty:
                b_id = st.selectbox("ID Apuesta", pends['ID'].values)
                res = st.selectbox("Resultado", ["Ganada", "Perdida", "Push"])
                if st.button("Actualizar"): update_bet_status(b_id, res); st.rerun()
            else: st.info("No hay pendientes")
    else: st.info("Historial vacío")