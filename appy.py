import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os

# ======================================================
# 1. CONFIGURACIÓN DE PÁGINA Y CONSTANTES
# ======================================================
st.set_page_config(page_title="Poisson Betting Pro", layout="wide", page_icon="⚽")
CSV_FILE = 'mis_apuestas_poisson.csv'

# ======================================================
# 2. LÓGICA MATEMÁTICA Y DE DATOS 🧠
# ======================================================
@st.cache_data
def fetch_live_soccer_data(league_code="SP1"):
    """Descarga datos, limpia y cachea."""
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
        df.columns = ['date', 'home', 'away', 'home_goals', 'away_goals']
        df = df.dropna()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_strengths(df):
    """
    Calcula fuerza de ataque y defensa con Time Decay.
    """
    last_date = df['date'].max()
    df['days_ago'] = (last_date - df['date']).dt.days
    
    # Decaimiento temporal (Weighted Average)
    alpha = 0.005 
    df['weight'] = np.exp(-alpha * df['days_ago'])

    avg_home = np.average(df['home_goals'], weights=df['weight'])
    avg_away = np.average(df['away_goals'], weights=df['weight'])
    
    team_stats = {}
    all_teams = sorted(list(set(df['home'].unique()) | set(df['away'].unique())))
    
    for team in all_teams:
        # Local
        home_matches = df[df['home'] == team]
        if not home_matches.empty:
            att_h = np.average(home_matches['home_goals'], weights=home_matches['weight']) / avg_home
            def_h = np.average(home_matches['away_goals'], weights=home_matches['weight']) / avg_away
        else:
            att_h, def_h = 1.0, 1.0

        # Visitante
        away_matches = df[df['away'] == team]
        if not away_matches.empty:
            att_a = np.average(away_matches['away_goals'], weights=away_matches['weight']) / avg_away
            def_a = np.average(away_matches['home_goals'], weights=away_matches['weight']) / avg_home
        else:
            att_a, def_a = 1.0, 1.0
            
        team_stats[team] = {'att_h': att_h, 'def_h': def_h, 'att_a': att_a, 'def_a': def_a}
        
    return team_stats, avg_home, avg_away, all_teams

def get_last_5_matches(df, team_name):
    mask = (df['home'] == team_name) | (df['away'] == team_name)
    last_5 = df[mask].sort_values(by='date', ascending=False).head(5).copy()
    last_5['Fecha'] = last_5['date'].dt.strftime('%d/%m/%Y')
    last_5['Rival'] = np.where(last_5['home'] == team_name, last_5['away'], last_5['home'])
    last_5['Marcador'] = last_5['home_goals'].astype(int).astype(str) + " - " + last_5['away_goals'].astype(int).astype(str)
    last_5['Condición'] = np.where(last_5['home'] == team_name, 'Casa', 'Fuera')
    return last_5[['Fecha', 'Condición', 'Rival', 'Marcador']]

def predict_match_poisson(home, away, team_stats, avg_h, avg_a):
    h_exp = team_stats[home]['att_h'] * team_stats[away]['def_a'] * avg_h
    a_exp = team_stats[away]['att_a'] * team_stats[home]['def_h'] * avg_a

    max_goals = 10
    probs = np.zeros((max_goals, max_goals))

    # Matriz de probabilidades
    for i in range(max_goals):
        for j in range(max_goals):
            probs[i][j] = poisson.pmf(i, h_exp) * poisson.pmf(j, a_exp)

    # 1X2
    p_home = np.tril(probs, -1).sum()
    p_draw = np.diag(probs).sum()
    p_away = np.triu(probs, 1).sum()

    # Overs
    p_o25 = 0
    p_btts = 0 
    for i in range(max_goals):
        for j in range(max_goals):
            if (i + j) > 2.5: p_o25 += probs[i][j]
            if i > 0 and j > 0: p_btts += probs[i][j]
            
    # --- NUEVO: TOP 3 MARCADORES EXACTOS ---
    # Aplanamos la matriz y buscamos los índices de los valores más altos
    flat_indices = np.argsort(probs.ravel())[::-1][:3] # Top 3
    top_scores = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, probs.shape)
        score_prob = probs[i][j]
        top_scores.append((f"{i}-{j}", score_prob))

    return h_exp, a_exp, p_home, p_draw, p_away, p_o25, p_btts, top_scores

def calculate_kelly_criterion(prob_real, odd_bookie):
    """
    Calcula el % del bankroll a apostar usando Kelly Fraccional (más seguro).
    f = (bp - q) / b
    donde: b = cuota - 1, p = probabilidad real, q = 1 - p
    """
    if prob_real <= 0 or odd_bookie <= 1:
        return 0.0
    
    b = odd_bookie - 1
    p = prob_real
    q = 1 - p
    
    f = (b * p - q) / b
    
    # Usamos "Half Kelly" (Kelly / 2) para ser conservadores y reducir volatilidad
    f_safe = f * 0.5
    
    return max(0.0, f_safe) * 100 # Retornamos porcentaje

# ======================================================
# 3. GESTIÓN DE APUESTAS (BANKROLL) 💰
# ======================================================
def load_bets():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
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
        stake = float(df.at[i, 'Stake'])
        odd = float(df.at[i, 'Cuota'])
        
        if new_status == 'Ganada':
            df.at[i, 'Ganancia'] = (stake * odd) - stake
        elif new_status == 'Perdida':
            df.at[i, 'Ganancia'] = -stake
        else: # Push
            df.at[i, 'Ganancia'] = 0.0
            
        df.to_csv(CSV_FILE, index=False)
        return True
    return False

# ======================================================
# 4. INTERFAZ DE USUARIO (FRONTEND) 🖥️
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    league_map = {"SP1": "🇪🇸 La Liga", "E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "I1": "🇮🇹 Serie A", "D1": "🇩🇪 Bundesliga", "F1": "🇫🇷 Ligue 1"}
    league_code = st.selectbox("Seleccionar Competición", list(league_map.keys()), format_func=lambda x: league_map[x])
    
    with st.spinner("Analizando datos..."):
        df_data = fetch_live_soccer_data(league_code)
    
    if not df_data.empty:
        stats, avg_h, avg_a, teams_list = calculate_strengths(df_data)
        st.success(f"Datos cargados: {len(df_data)} partidos")
        with st.expander("👀 Ver últimos resultados globales"):
            last_matches = df_data.sort_values(by='date', ascending=False).head(5)
            last_matches['date'] = last_matches['date'].dt.strftime('%d-%m-%Y')
            st.dataframe(last_matches[['date', 'home', 'away', 'home_goals', 'away_goals']], hide_index=True)
    else:
        st.error("Error cargando datos.")
        st.stop()

    st.divider()
    # Input de Bankroll para Kelly
    st.header("💰 Gestión de Banca")
    bankroll = st.number_input("Tu Bankroll Actual ($)", value=1000.0, step=50.0)
    st.caption("Usado para calcular el stake sugerido (Criterio de Kelly).")

# --- ÁREA PRINCIPAL ---
st.title(f"🤖 Poisson Betting: {league_map[league_code]}")

col_team1, col_team2 = st.columns(2)
with col_team1:
    home_team = st.selectbox("🏠 Equipo Local", teams_list, index=0)
with col_team2:
    away_list = [t for t in teams_list if t != home_team]
    away_team = st.selectbox("✈️ Equipo Visitante", away_list, index=0)

# CÁLCULOS
h_exp, a_exp, p_home, p_draw, p_away, p_o25, p_btts, top_scores = predict_match_poisson(home_team, away_team, stats, avg_h, avg_a)

st.divider()

col_analysis, col_market = st.columns([1.2, 1])

with col_analysis:
    st.subheader("📊 Predicción del Modelo")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Goles {home_team}", f"{h_exp:.2f}")
    m2.metric("Total Goles", f"{h_exp + a_exp:.2f}")
    m3.metric(f"Goles {away_team}", f"{a_exp:.2f}")
    
    st.write("#### Probabilidades 1X2 (Weighted)")
    st.progress(int(p_home*100))
    st.caption(f"🏠 {home_team}: **{p_home*100:.1f}%**")
    st.progress(int(p_draw*100))
    st.caption(f"🤝 Empate: **{p_draw*100:.1f}%**")
    st.progress(int(p_away*100))
    st.caption(f"✈️ {away_team}: **{p_away*100:.1f}%**")

    # --- NUEVO: TOP 3 MARCADORES ---
    st.write("#### 🎯 Marcadores Exactos Más Probables")
    cols_score = st.columns(3)
    for i, (score, prob) in enumerate(top_scores):
        with cols_score[i]:
            st.metric(label=f"Opción #{i+1}", value=score, delta=f"{prob*100:.1f}%")

with col_market:
    st.subheader("🏦 Comparar con Casa de Apuestas")
    st.write("Introduce las cuotas de tu web:")
    c_odd1, c_odd2, c_odd3 = st.columns(3)
    odd_h = c_odd1.number_input("Cuota Local", 1.01, 20.0, 2.00, step=0.05)
    odd_d = c_odd2.number_input("Cuota Empate", 1.01, 20.0, 3.20, step=0.05)
    odd_a = c_odd3.number_input("Cuota Visita", 1.01, 20.0, 3.50, step=0.05)
    
    # Cálculo de Valor y Kelly
    ev_h = (p_home * odd_h) - 1
    ev_d = (p_draw * odd_d) - 1
    ev_a = (p_away * odd_a) - 1
    
    kelly_h = calculate_kelly_criterion(p_home, odd_h)
    kelly_d = calculate_kelly_criterion(p_draw, odd_d)
    kelly_a = calculate_kelly_criterion(p_away, odd_a)

    st.write("#### 💎 Valor & Stake (Kelly)")
    
    def show_value_kelly(label, ev, kelly_pct, bankroll):
        if ev > 0:
            st.success(f"✅ **{label}**: +{ev*100:.1f}% EV")
            money_bet = (kelly_pct/100) * bankroll
            st.markdown(f"👉 **Apostar:** {kelly_pct:.1f}% del banco (**${money_bet:.2f}**)")
        else:
            st.error(f"❌ **{label}**: Sin valor")

    show_value_kelly(f"Gana {home_team}", ev_h, kelly_h, bankroll)
    show_value_kelly("Empate", ev_d, kelly_d, bankroll)
    show_value_kelly(f"Gana {away_team}", ev_a, kelly_a, bankroll)

st.divider()

# ESTADO DE FORMA
st.subheader(f"📉 Estado de Forma (Últimos 5 Partidos)")
col_hist_h, col_hist_a = st.columns(2)
with col_hist_h:
    st.markdown(f"**{home_team}**")
    st.dataframe(get_last_5_matches(df_data, home_team), hide_index=True, use_container_width=True)
with col_hist_a:
    st.markdown(f"**{away_team}**")
    st.dataframe(get_last_5_matches(df_data, away_team), hide_index=True, use_container_width=True)

st.divider()

# Gráfico y Registro
col_graph, col_track = st.columns(2)
with col_graph:
    st.subheader("📈 Visualización")
    imp_h, imp_d, imp_a = 1/odd_h, 1/odd_d, 1/odd_a
    fig = go.Figure(data=[
        go.Bar(name='Modelo Poisson', x=[home_team, 'Empate', away_team], y=[p_home*100, p_draw*100, p_away*100], marker_color='#00CC96'),
        go.Bar(name='Implícita Casa', x=[home_team, 'Empate', away_team], y=[imp_h*100, imp_d*100, imp_a*100], marker_color='#EF553B')
    ])
    fig.update_layout(barmode='group', height=300, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

with col_track:
    st.subheader("📝 Registrar Apuesta")
    with st.form("betting_form"):
        options = [f"Gana {home_team}", "Empate", f"Gana {away_team}", "Over 2.5 Goles"]
        pick_sel = st.selectbox("Selecciona tu Pick", options)
        stake_in = st.number_input("Stake (Dinero $)", 1.0, 5000.0, 50.0) # Modificado para usar Dinero Real
        
        if "Gana " + home_team in pick_sel: final_odd = odd_h; prob_ref = p_home
        elif "Empate" in pick_sel: final_odd = odd_d; prob_ref = p_draw
        elif "Gana " + away_team in pick_sel: final_odd = odd_a; prob_ref = p_away
        else: final_odd = 1.90; prob_ref = p_o25 # Default
        
        submitted = st.form_submit_button("💾 Guardar en Historial")
        if submitted:
            new_bet = {
                "ID": pd.Timestamp.now().strftime('%Y%m%d%H%M%S'),
                "Fecha": pd.Timestamp.now().strftime('%Y-%m-%d'),
                "Liga": league_code,
                "Partido": f"{home_team} vs {away_team}",
                "Pick": pick_sel,
                "Cuota": final_odd,
                "Stake": stake_in,
                "Prob_Modelo": round(prob_ref, 4),
                "Estado": "Pendiente",
                "Ganancia": 0.0
            }
            save_bet(new_bet)
            st.success("Apuesta guardada.")
            st.rerun()

st.divider()
st.subheader("📜 Tu Historial y Rendimiento")
df_bets = load_bets()
if not df_bets.empty:
    total_stake = df_bets['Stake'].sum()
    total_profit = df_bets['Ganancia'].sum()
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
    k1, k2, k3 = st.columns(3)
    k1.metric("Beneficio Neto ($)", f"${total_profit:.2f}", delta_color="normal")
    k2.metric("Volumen Jugado ($)", f"${total_stake:.2f}")
    k3.metric("ROI (Retorno)", f"{roi:.2f}%", delta=f"{roi:.2f}%")
    st.dataframe(df_bets.sort_values(by="Fecha", ascending=False).style.format({"Prob_Modelo": "{:.1%}", "Ganancia": "${:.2f}", "Stake": "${:.2f}"}), use_container_width=True)
    with st.expander("🔄 Actualizar Resultado de Apuesta"):
        pends = df_bets[df_bets['Estado'] == 'Pendiente']
        if not pends.empty:
            c_up1, c_up2, c_up3 = st.columns([3, 2, 1])
            id_to_up = c_up1.selectbox("Apuesta Pendiente", pends['ID'].values)
            res_to_up = c_up2.selectbox("Resultado", ["Ganada", "Perdida", "Push"])
            if c_up3.button("Actualizar"):
                update_bet_status(id_to_up, res_to_up)
                st.rerun()