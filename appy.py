import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os
import requests 
from difflib import get_close_matches
from datetime import datetime, timezone

# ======================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS (DARK MODE) 🎨
# ======================================================
st.set_page_config(page_title="Dixon-Coles Pro + API Semanal", layout="wide", page_icon="⚽")
CSV_FILE = 'mis_apuestas_pro.csv'

# Inicializar Session States (CON TU CLAVE YA CARGADA)
if 'ticket' not in st.session_state: st.session_state.ticket = []
if 'daily_matches' not in st.session_state: st.session_state.daily_matches = []
if 'api_key' not in st.session_state: st.session_state.api_key = "f8b57bf9dc94df0f21b95752a4897c98"

# Estilos CSS
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .ticket-box {
        background-color: #1e1e1e;
        border: 1px solid #ffd700;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .best-bet-card {
        background-color: #1c2e24;
        border-left: 5px solid #00ff00;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. LÓGICA DE DATOS Y MODELO MATEMÁTICO 🧠
# ======================================================

@st.cache_data(ttl=3600)
def fetch_live_soccer_data(league_code="SP1"):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']
        actual_cols = [c for c in cols if c in df.columns]
        df = df[actual_cols]
        
        new_names = ['date', 'home', 'away', 'home_goals', 'away_goals', 'odd_h', 'odd_d', 'odd_a']
        if len(actual_cols) == 8:
            df.columns = new_names
        else:
            df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            df.columns = ['date', 'home', 'away', 'home_goals', 'away_goals']
            df['odd_h'] = 1.0; df['odd_d'] = 1.0; df['odd_a'] = 1.0 

        df = df.dropna()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        return df
    except: return pd.DataFrame()

def calculate_strengths(df):
    last_date = df['date'].max()
    df['days_ago'] = (last_date - df['date']).dt.days
    alpha = 0.004
    df['weight'] = np.exp(-alpha * df['days_ago'])
    
    avg_home = np.average(df['home_goals'], weights=df['weight'])
    avg_away = np.average(df['away_goals'], weights=df['weight'])
    avg_global = (avg_home + avg_away) / 2
    
    team_stats = {}
    all_teams = sorted(list(set(df['home'].unique()) | set(df['away'].unique())))
    MIX_FACTOR = 0.7 
    
    for team in all_teams:
        team_matches = df[(df['home'] == team) | (df['away'] == team)].copy()
        if not team_matches.empty:
            team_matches['goals_scored'] = np.where(team_matches['home'] == team, team_matches['home_goals'], team_matches['away_goals'])
            team_matches['goals_conceded'] = np.where(team_matches['home'] == team, team_matches['away_goals'], team_matches['home_goals'])
            att_global = np.average(team_matches['goals_scored'], weights=team_matches['weight']) / avg_global
            def_global = np.average(team_matches['goals_conceded'], weights=team_matches['weight']) / avg_global
        else: att_global, def_global = 1.0, 1.0

        h_m = df[df['home'] == team]
        if not h_m.empty:
            att_h_pure = np.average(h_m['home_goals'], weights=h_m['weight']) / avg_home
            def_h_pure = np.average(h_m['away_goals'], weights=h_m['weight']) / avg_away
        else: att_h_pure, def_h_pure = 1.0, 1.0

        a_m = df[df['away'] == team]
        if not a_m.empty:
            att_a_pure = np.average(a_m['away_goals'], weights=a_m['weight']) / avg_away
            def_a_pure = np.average(a_m['home_goals'], weights=a_m['weight']) / avg_home
        else: att_a_pure, def_a_pure = 1.0, 1.0
            
        team_stats[team] = {
            'att_h': (att_h_pure * MIX_FACTOR) + (att_global * (1 - MIX_FACTOR)),
            'def_h': (def_h_pure * MIX_FACTOR) + (def_global * (1 - MIX_FACTOR)),
            'att_a': (att_a_pure * MIX_FACTOR) + (att_global * (1 - MIX_FACTOR)),
            'def_a': (def_a_pure * MIX_FACTOR) + (def_global * (1 - MIX_FACTOR))
        }
    return team_stats, avg_home, avg_away, all_teams

def predict_match_dixon_coles(home, away, team_stats, avg_h, avg_a):
    h_exp = team_stats[home]['att_h'] * team_stats[away]['def_a'] * avg_h
    a_exp = team_stats[away]['att_a'] * team_stats[home]['def_h'] * avg_a
    
    max_goals = 10
    probs = np.zeros((max_goals, max_goals))
    rho = -0.13 

    for x in range(max_goals):
        for y in range(max_goals):
            p_base = poisson.pmf(x, h_exp) * poisson.pmf(y, a_exp)
            correction = 1.0
            if x==0 and y==0: correction = 1.0 - (h_exp * a_exp * rho)
            elif x==0 and y==1: correction = 1.0 + (h_exp * rho)
            elif x==1 and y==0: correction = 1.0 + (a_exp * rho)
            elif x==1 and y==1: correction = 1.0 - (rho)
            probs[x][y] = p_base * correction
            
    probs = np.maximum(0, probs)
    probs = probs / probs.sum()

    p_home = np.tril(probs, -1).sum()
    p_draw = np.diag(probs).sum()
    p_away = np.triu(probs, 1).sum()
    
    p_o15, p_o25, p_btts = 0, 0, 0
    for i in range(max_goals):
        for j in range(max_goals):
            if (i+j) > 1.5: p_o15 += probs[i][j]
            if (i+j) > 2.5: p_o25 += probs[i][j]
            if i > 0 and j > 0: p_btts += probs[i][j]

    flat_indices = np.argsort(probs.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, probs.shape)
        top_scores.append((f"{i}-{j}", probs[i][j]))

    return h_exp, a_exp, p_home, p_draw, p_away, p_o15, p_o25, p_btts, top_scores, probs

def run_backtest(df, team_stats, avg_h, avg_a):
    recent = df.tail(50).copy() 
    results = []
    correct, bal = 0, 0
    for _, row in recent.iterrows():
        _, _, ph, pd_prob, pa, _, _, _, _, _ = predict_match_dixon_coles(row['home'], row['away'], team_stats, avg_h, avg_a)
        if ph > pd_prob and ph > pa: pred, prob, odd, res_real = "Local", ph, row['odd_h'], ("Local" if row['home_goals'] > row['away_goals'] else "Fallo")
        elif pa > ph and pa > pd_prob: pred, prob, odd, res_real = "Visita", pa, row['odd_a'], ("Visita" if row['away_goals'] > row['home_goals'] else "Fallo")
        else: pred, prob, odd, res_real = "Empate", pd_prob, row['odd_d'], ("Empate" if row['home_goals'] == row['away_goals'] else "Fallo")
        is_win = (pred == res_real)
        profit = (odd - 1) if is_win else -1
        if is_win: correct += 1
        bal += profit
        results.append({"Partido": f"{row['home']} vs {row['away']}", "Predicción": f"{pred} ({prob*100:.0f}%)", "Realidad": f"{int(row['home_goals'])}-{int(row['away_goals'])}", "Cuota": odd, "Res": "✅" if is_win else "❌", "P/L": profit})
    return pd.DataFrame(results), correct, bal

# ======================================================
# 3. UTILIDADES VISUALES Y GESTIÓN 🛠️
# ======================================================
def plot_gauge(val, title, color):
    return go.Figure(go.Indicator(mode="gauge+number", value=val*100, title={'text': title}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "white"})).update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))

def plot_score_heatmap(probs, home_team, away_team):
    limit = 6
    probs_cut = probs[:limit, :limit]
    fig = go.Figure(data=go.Heatmap(z=probs_cut, x=[f"{away_team} {i}" for i in range(limit)], y=[f"{home_team} {i}" for i in range(limit)], colorscale='Viridis', text=np.round(probs_cut * 100, 1), texttemplate="%{text}%", hoverongaps=False))
    fig.update_layout(title="🔥 Probabilidad de Marcador Exacto", xaxis_title=f"Goles {away_team}", yaxis_title=f"Goles {home_team}", height=450, margin=dict(l=40, r=40, t=40, b=40))
    return fig

def plot_radar_comparison(home, away, stats):
    h_att, h_def = stats[home]['att_h'], 2 - stats[home]['def_h'] 
    a_att, a_def = stats[away]['att_a'], 2 - stats[away]['def_a']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[h_att, h_def, stats[home]['att_a'], 2-stats[home]['def_a']], theta=['Ataque (Casa)', 'Defensa (Casa)', 'Ataque (Fuera)', 'Defensa (Fuera)'], fill='toself', name=home, line_color='#4CAF50'))
    fig.add_trace(go.Scatterpolar(r=[stats[away]['att_h'], 2-stats[away]['def_h'], a_att, a_def], theta=['Ataque (Casa)', 'Defensa (Casa)', 'Ataque (Fuera)', 'Defensa (Fuera)'], fill='toself', name=away, line_color='#2196F3'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 2.5])), showlegend=True, title="⚔️ Comparativa de Fuerzas (Área Mayor = Mejor)", height=350, margin=dict(t=40, b=20, l=40, r=40))
    return fig

def get_last_5(df, team):
    mask = (df['home'] == team) | (df['away'] == team)
    l5 = df[mask].sort_values(by='date', ascending=False).head(5).copy()
    l5['Rival'] = np.where(l5['home'] == team, l5['away'], l5['home'])
    l5['Score'] = l5['home_goals'].astype(int).astype(str) + "-" + l5['away_goals'].astype(int).astype(str)
    l5['Sede'] = np.where(l5['home'] == team, '🏠', '✈️')
    return l5[['Sede', 'Rival', 'Score']]

def calculate_kelly(prob, odd):
    if prob <= 0 or odd <= 1: return 0.0
    b = odd - 1
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f * 0.5) * 100

def manage_bets(mode, data=None, id_bet=None, status=None):
    if os.path.exists(CSV_FILE): df = pd.read_csv(CSV_FILE)
    else: df = pd.DataFrame(columns=["ID", "Fecha", "Liga", "Partido", "Pick", "Cuota", "Stake", "Prob", "Estado", "Ganancia"])
    if mode == "save":
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
    elif mode == "update":
        idx = df[df['ID'].astype(str) == str(id_bet)].index
        if not idx.empty:
            i = idx[0]
            df.at[i, 'Estado'] = status
            profit = (df.at[i, 'Stake'] * df.at[i, 'Cuota']) - df.at[i, 'Stake'] if status == 'Ganada' else (-df.at[i, 'Stake'] if status == 'Perdida' else 0)
            df.at[i, 'Ganancia'] = profit
            df.to_csv(CSV_FILE, index=False)
    return df

# ======================================================
# 4. INTERFAZ GRÁFICA (UI) 🌟
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    if st.button("🔄 Actualizar Datos"): st.cache_data.clear(); st.rerun()
    leagues = {"SP1": "🇪🇸 La Liga", "E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "I1": "🇮🇹 Serie A", "D1": "🇩🇪 Bundesliga", "F1": "🇫🇷 Ligue 1", "N1": "🇳🇱 Eredivisie", "P1": "🇵🇹 Primeira Liga"}
    code = st.selectbox("Liga", list(leagues.keys()), format_func=lambda x: leagues[x])
    df = fetch_live_soccer_data(code)
    
    if not df.empty:
        stats, ah, aa, teams = calculate_strengths(df)
        st.success(f"✅ {len(df)} partidos cargados")
        
        st.markdown("---")
        st.markdown("###### 🕒 Últimos 5 Registrados (Liga):")
        last_5 = df.tail(5).copy().iloc[::-1]
        last_5['Fecha'] = last_5['date'].dt.strftime('%d/%m')
        last_5['Partido'] = last_5['home'] + " vs " + last_5['away']
        last_5['Score'] = last_5['home_goals'].astype(int).astype(str) + "-" + last_5['away_goals'].astype(int).astype(str)
        st.dataframe(last_5[['Fecha', 'Partido', 'Score']], hide_index=True, use_container_width=True)
    else: st.error("Error cargando datos"); st.stop()

    st.divider()
    bank = st.number_input("💰 Tu Banco ($)", 1000.0, step=50.0)
    if st.session_state.ticket:
        st.divider()
        st.markdown(f"**Ticket:** {len(st.session_state.ticket)} selecciones")
        if st.button("🗑️ Limpiar"): st.session_state.ticket = []; st.rerun()

st.title(f"🤖 Dixon-Coles: {leagues[code]}")

# --- SELECTOR DE PARTIDO PRINCIPAL ---
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])
# -------------------------------------

h_exp, a_exp, ph, pd_prob, pa, po15, po25, pbtts, top_sc, probs = predict_match_dixon_coles(home, away, stats, ah, aa)

# PESTAÑAS
t1, t2, t3, t4, t5 = st.tabs(["📊 Análisis", "💰 Valor y Parlay", "📜 Historial", "💎 Escáner En Vivo (API)", "🧪 Backtest"])

with t1:
    st.markdown("### 🥅 Expectativa de Goles")
    c_g1, c_g2, c_g3 = st.columns(3)
    c_g1.metric(home, f"{h_exp:.2f}")
    c_g2.metric("Total (xG)", f"{h_exp+a_exp:.2f}") 
    c_g3.metric(away, f"{a_exp:.2f}")

    st.plotly_chart(plot_radar_comparison(home, away, stats), use_container_width=True)
    
    st.markdown("### 📊 Probabilidades")
    mg1, mg2, mg3 = st.columns(3) 
    mg1.metric("Over 1.5", f"{po15*100:.1f}%"); mg2.metric("Over 2.5", f"{po25*100:.1f}%"); mg3.metric("BTTS", f"{pbtts*100:.1f}%")
    
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

with t2:
    col_analisis, col_ticket = st.columns([2, 1])
    with col_analisis:
        st.markdown("### 🏦 Comparador")
        co1, co2, co3 = st.columns(3)
        oh = co1.number_input("Cuota 1", 1.01, 20.0, 2.0)
        od = co2.number_input("Cuota X", 1.01, 20.0, 3.2)
        oa = co3.number_input("Cuota 2", 1.01, 20.0, 3.5)

        imp_h = (1/oh); imp_d = (1/od); imp_a = (1/oa) 
        total_imp = imp_h + imp_d + imp_a
        imp_h /= total_imp; imp_d /= total_imp; imp_a /= total_imp

        fig_val = go.Figure(data=[
            go.Bar(name='Tu Modelo', x=[home, 'Empate', away], y=[ph, pd_prob, pa], marker_color='#00CC96'),
            go.Bar(name='Casa (Sin Margen)', x=[home, 'Empate', away], y=[imp_h, imp_d, imp_a], marker_color='#EF553B')
        ])
        fig_val.update_layout(barmode='group', height=250, margin=dict(t=20, b=20, l=20, r=20), title="⚖️ Detector de Valor")
        st.plotly_chart(fig_val, use_container_width=True)
        
        st.divider()
        st.markdown("### ➕ Agregar al Ticket")
        with st.form("add_to_ticket"):
            sel_pick = st.selectbox("Selección", [f"Gana {home}", "Empate", f"Gana {away}"])
            if "Gana "+home in sel_pick: sel_odd, sel_prob = oh, ph
            elif "Empate" in sel_pick: sel_odd, sel_prob = od, pd_prob
            else: sel_odd, sel_prob = oa, pa
            if st.form_submit_button("Añadir selección"):
                st.session_state.ticket.append({"match": f"{home} vs {away}", "pick": sel_pick, "odd": sel_odd, "prob": sel_prob, "league": leagues[code]})
                st.success("Añadido"); st.rerun()

    with col_ticket:
        st.markdown("### 🎫 Ticket")
        if not st.session_state.ticket: st.info("Vacío")
        else:
            total_odd, total_prob = 1.0, 1.0
            for idx, item in enumerate(st.session_state.ticket):
                st.markdown(f"<div class='ticket-box'><small>{item['league']}</small><br><strong>{item['match']}</strong><br>{item['pick']} @ {item['odd']}</div>", unsafe_allow_html=True)
                if st.button("❌", key=f"del_{idx}"): st.session_state.ticket.pop(idx); st.rerun()
                total_odd *= item['odd']; total_prob *= item['prob']
            st.divider()
            st.metric("Cuota Total", f"{total_odd:.2f}")
            stake_parlay = st.number_input("Stake ($)", 1.0, 5000.0, 50.0)
            st.success(f"Ganancia: ${(stake_parlay * total_odd) - stake_parlay:.2f}")
            if st.button("💾 Guardar"):
                tipo_str = "Simple" if len(st.session_state.ticket) == 1 else "Parlay"
                match_str = st.session_state.ticket[0]['match'] if len(st.session_state.ticket) == 1 else f"Combinada ({len(st.session_state.ticket)})"
                pick_str = " + ".join([i['pick'] for i in st.session_state.ticket])
                manage_bets("save", {"ID": pd.Timestamp.now().strftime('%Y%m%d%H%M%S'), "Fecha": pd.Timestamp.now().strftime('%Y-%m-%d'), "Liga": tipo_str, "Partido": match_str, "Pick": pick_str, "Cuota": round(total_odd, 2), "Stake": stake_parlay, "Prob": round(total_prob, 4), "Estado": "Pendiente", "Ganancia": 0.0})
                st.session_state.ticket = []; st.balloons(); st.rerun()

with t3:
    st.markdown("### 📜 Historial")
    db = manage_bets("load")
    if not db.empty:
        df_plot = db.copy().sort_values(by='ID')
        df_plot['Balance Acumulado'] = df_plot['Ganancia'].cumsum()
        fig_bal = go.Figure()
        fig_bal.add_trace(go.Scatter(x=pd.to_datetime(df_plot['Fecha']), y=df_plot['Balance Acumulado'], mode='lines+markers', name='Balance', line=dict(color='#00ff00' if df_plot['Balance Acumulado'].iloc[-1] >= 0 else '#ff0000', width=3)))
        st.plotly_chart(fig_bal, use_container_width=True)
        st.dataframe(db.sort_values(by="Fecha", ascending=False), use_container_width=True)
        with st.expander("Actualizar Estado"):
            pen = db[db['Estado']=='Pendiente']
            if not pen.empty:
                bid = st.selectbox("ID", pen['ID'].unique())
                res = st.selectbox("Resultado", ["Ganada", "Perdida", "Push"])
                if st.button("Actualizar"): manage_bets("update", id_bet=bid, status=res); st.rerun()

# --- ESCÁNER CON API ---
with t4:
    st.markdown("## 💎 Escáner En Vivo (The Odds API)")
    
    # Mapeo de códigos de tu App a los códigos de la API
    api_league_map = {
        "SP1": "soccer_spain_la_liga",
        "E0": "soccer_epl",
        "I1": "soccer_italy_serie_a",
        "D1": "soccer_germany_bundesliga",
        "F1": "soccer_france_ligue_one",
        "N1": "soccer_netherlands_eredivisie",
        "P1": "soccer_portugal_primeira_liga"
    }

    # Input de Key (Pre-cargada con tu clave)
    api_key_input = st.text_input("🔑 API Key:", value=st.session_state.api_key, type="password")
    if st.button("💾 Guardar Key / Actualizar"):
        st.session_state.api_key = api_key_input
        st.success("Clave guardada.")
        st.rerun()

    st.divider()

    if st.session_state.api_key:
        st.write(f"📡 Buscando partidos **(Próximos 7 días)** para: **{leagues[code]}**...")
        
        if st.button("🚀 Escanear Mercado Real"):
            sport_key = api_league_map.get(code)
            
            try:
                # 1. Llamada a la API
                url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h&oddsFormat=decimal&apiKey={st.session_state.api_key}"
                res = requests.get(url)
                data = res.json()
                
                if 'message' in data:
                    st.error(f"Error de API: {data['message']}")
                elif not data:
                    st.warning("✅ Conexión exitosa, pero NO hay partidos programados en el corto plazo para esta liga.")
                else:
                    live_results = []
                    
                    # 2. Procesar partidos
                    for item in data:
                        # --- FILTRO DE FECHA (1 SEMANA = 168 HORAS) ---
                        match_date = pd.to_datetime(item['commence_time'])
                        now = pd.Timestamp.now(tz='UTC')
                        diff_hours = (match_date - now).total_seconds() / 3600
                        
                        if diff_hours > 168 or diff_hours < -5: # Filtro extendido a 7 días
                            continue
                        # ------------------------------------------------

                        h_team_api = item['home_team']
                        a_team_api = item['away_team']
                        
                        odds_h, odds_d, odds_a = 0, 0, 0
                        
                        if item['bookmakers']:
                            book = item['bookmakers'][0] # Usamos el primero disponible
                            markets = book['markets'][0]['outcomes']
                            for m in markets:
                                if m['name'] == h_team_api: odds_h = m['price']
                                elif m['name'] == a_team_api: odds_a = m['price']
                                else: odds_d = m['price']
                        
                        # 3. Match de Nombres (Fuzzy Logic)
                        match_home = get_close_matches(h_team_api, teams, n=1, cutoff=0.5)
                        match_away = get_close_matches(a_team_api, teams, n=1, cutoff=0.5)
                        
                        if match_home and match_away:
                            real_home, real_away = match_home[0], match_away[0]
                            
                            # Validar que existan en datos históricos
                            if real_home in stats and real_away in stats:
                                # Ejecutar Modelo
                                _, _, ph, pd_prob, pa, o25, _, btts, _, _ = predict_match_dixon_coles(real_home, real_away, stats, ah, aa)
                                
                                # Calcular Valor (EV)
                                ev_h = (ph * odds_h) - 1
                                ev_a = (pa * odds_a) - 1
                                ev_d = (pd_prob * odds_d) - 1
                                
                                best_pick, best_ev = "No Bet", -10.0
                                if ev_h > 0: best_pick, best_ev = f"Gana {real_home}", ev_h
                                if ev_a > ev_h and ev_a > 0: best_pick, best_ev = f"Gana {real_away}", ev_a
                                if ev_d > ev_h and ev_d > ev_a and ev_d > 0: best_pick, best_ev = "Empate", ev_d
                                
                                live_results.append({
                                    "Hora": pd.to_datetime(item['commence_time']).strftime('%d/%m %H:%M'),
                                    "Partido": f"{real_home} vs {real_away}",
                                    "Prob Modelo": f"L:{ph*100:.0f}% E:{pd_prob*100:.0f}% V:{pa*100:.0f}%",
                                    "Cuotas Reales": f"L:{odds_h} E:{odds_d} V:{odds_a}",
                                    "Pick Valor": best_pick,
                                    "EV": best_ev
                                })
                    
                    if live_results:
                        df_live = pd.DataFrame(live_results).sort_values(by="EV", ascending=False)
                        
                        st.markdown(f"### 🎯 Oportunidades (Próximos 7 días) - {len(df_live)} Partidos")
                        for i, row in df_live.iterrows():
                            color = "#4CAF50" if row['EV'] > 0 else "#FF5252" # Verde si hay valor, Rojo si no
                            val_txt = f"+{row['EV']*100:.1f}%" if row['EV'] > 0 else f"{row['EV']*100:.1f}%"
                            
                            st.markdown(f"""
                            <div style="background-color: #262730; border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                                <div style="display:flex; justify-content:space-between;">
                                    <strong>⏰ {row['Hora']} | {row['Partido']}</strong>
                                    <span style="color:{color}; font-weight:bold; font-size:1.2em">EV: {val_txt}</span>
                                </div>
                                <div style="display:flex; justify-content:space-between; font-size:0.9em; margin-top:5px; color:#ccc;">
                                    <span>🧠 {row['Prob Modelo']}</span>
                                    <span>🏦 {row['Cuotas Reales']}</span>
                                </div>
                                <div style="margin-top:5px; font-size:1.1em;">
                                    👉 Recomendación: <strong>{row['Pick Valor']}</strong>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("La API no encontró partidos para los próximos 7 días en esta liga, o no pude coincidir los nombres de los equipos.")

            except Exception as e:
                st.error(f"Error de conexión: {e}")
    else:
        st.warning("Por favor ingresa una API Key válida.")

with t5:
    st.markdown("### 🧪 Backtest")
    if st.button("▶️ Simular 50 Partidos"):
        test_df, ok, profit = run_backtest(df, stats, ah, aa) 
        m1, m2, m3 = st.columns(3)
        m1.metric("Aciertos", f"{ok}/50 ({ok/50*100:.0f}%)") 
        m2.metric("Profit", f"{profit:.2f} U"); m3.metric("Estado", "🔥 Rentable" if profit > 0 else "❄️ Pérdidas")
        st.dataframe(test_df, use_container_width=True)
