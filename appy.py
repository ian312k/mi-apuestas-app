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
st.set_page_config(page_title="Dixon-Coles Pro v3.3 Blindada", layout="wide", page_icon="⚽")
CSV_FILE = 'mis_apuestas_pro.csv'

# --- GESTIÓN DE ESTADO (SESSION STATE) ---
if 'ticket' not in st.session_state: st.session_state.ticket = []
if 'api_key' not in st.session_state: st.session_state.api_key = "f8b57bf9dc94df0f21b95752a4897c98"
if 'api_odds_cache' not in st.session_state: st.session_state.api_odds_cache = {} 
if 'api_usage' not in st.session_state: st.session_state.api_usage = {"used": 0, "remaining": 500}

# ESTE ES EL ALMACÉN BLINDADO
if 'market_storage' not in st.session_state: st.session_state.market_storage = {}

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
def fetch_live_soccer_data(league_code="SP1"):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']
        actual_cols = [c for c in cols if c in df.columns]
        df = df[actual_cols]
        new_names = ['date', 'home', 'away', 'home_goals', 'away_goals', 'odd_h', 'odd_d', 'odd_a']
        if len(actual_cols) == 8: df.columns = new_names
        else:
            df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            df.columns = ['date', 'home', 'away', 'home_goals', 'away_goals']
            df['odd_h'] = 1.0; df['odd_d'] = 1.0; df['odd_a'] = 1.0 
        df = df.dropna()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        return df
    except: return pd.DataFrame()

# API MANUAL (CONTROL TOTAL)
def call_api_real(sport_key, api_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h&oddsFormat=decimal&apiKey={api_key}"
    try:
        res = requests.get(url)
        used = res.headers.get('x-requests-used', 0)
        remaining = res.headers.get('x-requests-remaining', 500)
        
        if res.status_code == 200:
            return {"success": True, "data": res.json(), "used": int(used), "remaining": int(remaining)}
        else:
            return {"success": False, "error": f"Error {res.status_code}", "message": res.text}
    except Exception as e:
        return {"success": False, "error": "Excepción", "message": str(e)}

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
            
    elif mode == "delete":
        df = df[df['ID'].astype(str) != str(id_bet)]
        df.to_csv(CSV_FILE, index=False)
        
    return df

# ======================================================
# 5. SIDEBAR Y CARGA DE DATOS 🌟
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
    st.markdown("### 🥅 Expectativa de Goles")
    c_g1, c_g2, c_g3 = st.columns(3)
    c_g1.metric(home, f"{h_exp:.2f}")
    c_g2.metric("Total (xG)", f"{h_exp+a_exp:.2f}") 
    c_g3.metric(away, f"{a_exp:.2f}")

    st.plotly_chart(plot_radar_comparison(home, away, stats), use_container_width=True)
    
    mg1, mg2, mg3 = st.columns(3) 
    mg1.metric("Over 1.5", f"{po15*100:.1f}%"); mg2.metric("Over 2.5", f"{po25*100:.1f}%"); mg3.metric("BTTS", f"{pbtts*100:.1f}%")
    
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
        match_key = f"{home} vs {away}"
        league_data = st.session_state.market_storage.get(code, {})
        found_in_storage = False
        
        if 'data' in league_data:
            for item in league_data['data']:
                h_team_api = item['home_team']; a_team_api = item['away_team']
                m_h = get_close_matches(h_team_api, [home], n=1, cutoff=0.5)
                m_a = get_close_matches(a_team_api, [away], n=1, cutoff=0.5)
                if m_h and m_a:
                    if item['bookmakers']:
                        book = item['bookmakers'][0]
                        for m in book['markets'][0]['outcomes']:
                            if m['name'] == h_team_api: def_oh = m['price']
                            elif m['name'] == a_team_api: def_oa = m['price']
                            else: def_od = m['price']
                        found_in_storage = True; break
        
        if found_in_storage: st.success("✅ Momios cargados automáticamente (Escáner).")
        else: st.info("ℹ️ Momios por defecto (No encontrados en escáner).")

        co1, co2, co3 = st.columns(3)
        oh = co1.number_input("Cuota Local", 1.01, 100.0, float(def_oh))
        od = co2.number_input("Cuota Empate", 1.01, 100.0, float(def_od))
        oa = co3.number_input("Cuota Visita", 1.01, 100.0, float(def_oa))

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

# --- TAB 3: HISTORIAL ---
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
        csv = db.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Historial (CSV)", data=csv, file_name="mis_apuestas_backup.csv", mime="text/csv")
        
        c_upd, c_del = st.columns(2)
        with c_upd:
            with st.expander("📝 Actualizar Resultado"):
                pen = db[db['Estado']=='Pendiente']
                if not pen.empty:
                    bid = st.selectbox("ID", pen['ID'].unique())
                    res = st.selectbox("Resultado", ["Ganada", "Perdida", "Push"])
                    if st.button("Actualizar"): manage_bets("update", id_bet=bid, status=res); st.rerun()
                else: st.info("No hay pendientes")
        
        with c_del:
            with st.expander("🗑️ Eliminar Apuesta"):
                ids_all = db['ID'].unique()
                id_del = st.selectbox("Seleccionar ID para borrar", ids_all)
                if st.button("Borrar definitivamente"):
                    manage_bets("delete", id_bet=id_del)
                    st.warning("Apuesta eliminada")
                    st.rerun()

# --- TAB 4: ESCÁNER BLINDADO ---
with t4:
    st.markdown("## 💎 Escáner Seguro")
    if st.session_state.api_usage['used'] > 0:
        pct_used = st.session_state.api_usage['used'] / 500
        st.progress(pct_used, text=f"Llamadas API: {st.session_state.api_usage['used']} / 500 usadas")
    
    api_league_map = { "SP1": "soccer_spain_la_liga", "E0": "soccer_epl", "I1": "soccer_italy_serie_a", "D1": "soccer_germany_bundesliga", "F1": "soccer_france_ligue_one", "N1": "soccer_netherlands_eredivisie", "P1": "soccer_portugal_primeira_liga" }
    api_key_input = st.text_input("🔑 API Key:", value=st.session_state.api_key, type="password")
    if st.button("💾 Guardar Key"): st.session_state.api_key = api_key_input; st.success("Guardado."); st.rerun()
    st.divider()

    if st.session_state.api_key:
        sport_key = api_league_map.get(code)
        has_data = False; data_to_display = []
        if code in st.session_state.market_storage:
            stored = st.session_state.market_storage[code]
            data_to_display = stored['data']; has_data = True
            st.info(f"📂 Datos en memoria. Actualizado: {stored['timestamp'].strftime('%H:%M:%S')}")
        else: st.warning("⚠️ Sin datos descargados.")

        if st.button(f"{'🔄 Actualizar' if has_data else '⬇️ Descargar'} Datos (Gasta 1 llamada)"):
            with st.spinner("Conectando..."):
                response = call_api_real(sport_key, st.session_state.api_key)
                if response['success']:
                    st.session_state.market_storage[code] = {'timestamp': datetime.now(), 'data': response['data']}
                    st.session_state.api_usage['used'] = response['used']; st.session_state.api_usage['remaining'] = response['remaining']
                    st.success("✅ Descargado."); st.rerun()
                else: st.error(f"Error API: {response['message']}")

        if has_data and data_to_display:
            live_results = []
            for item in data_to_display:
                match_date = pd.to_datetime(item['commence_time'])
                now = pd.Timestamp.now(tz='UTC')
                diff_hours = (match_date - now).total_seconds() / 3600
                if diff_hours > 168 or diff_hours < -5: continue

                h_team_api = item['home_team']; a_team_api = item['away_team']
                odds_h, odds_d, odds_a = 0, 0, 0
                if item['bookmakers']:
                    book = item['bookmakers'][0] 
                    for m in book['markets'][0]['outcomes']:
                        if m['name'] == h_team_api: odds_h = m['price']
                        elif m['name'] == a_team_api: odds_a = m['price']
                        else: odds_d = m['price']
                
                m_h = get_close_matches(h_team_api, teams, n=1, cutoff=0.5)
                m_a = get_close_matches(a_team_api, teams, n=1, cutoff=0.5)
                
                if m_h and m_a:
                    real_home, real_away = m_h[0], m_a[0]
                    if real_home in stats and real_away in stats:
                        _, _, ph, pd_prob, pa, _, _, _, _, _ = predict_match_dixon_coles(real_home, real_away, stats, ah, aa)
                        ev_h = (ph * odds_h) - 1; ev_a = (pa * odds_a) - 1; ev_d = (pd_prob * odds_d) - 1
                        best_pick, best_ev = "No Bet", -10.0
                        if ev_h > 0: best_pick, best_ev = f"Gana {real_home}", ev_h
                        if ev_a > ev_h and ev_a > 0: best_pick, best_ev = f"Gana {real_away}", ev_a
                        if ev_d > ev_h and ev_d > ev_a and ev_d > 0: best_pick, best_ev = "Empate", ev_d
                        
                        live_results.append({
                            "Hora": pd.to_datetime(item['commence_time']).strftime('%d/%m %H:%M'),
                            "Partido": f"{real_home} vs {real_away}",
                            "Prob": f"L:{ph:.2f} E:{pd_prob:.2f} V:{pa:.2f}",
                            "Cuotas": f"L:{odds_h} E:{odds_d} V:{odds_a}",
                            "Pick Valor": best_pick, "EV": best_ev
                        })
            
            if live_results:
                df_live = pd.DataFrame(live_results).sort_values(by="EV", ascending=False)
                st.markdown(f"### 🎯 Oportunidades (Memoria) - {len(df_live)} Partidos")
                for i, row in df_live.iterrows():
                    color = "#4CAF50" if row['EV'] > 0 else "#FF5252"
                    val_txt = f"+{row['EV']*100:.1f}%" if row['EV'] > 0 else f"{row['EV']*100:.1f}%"
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
            else: st.info("Datos descargados, pero no se encontraron partidos compatibles para esta semana.")

# --- TAB 5: LABORATORIO ---
with t5:
    st.markdown("## 🧪 Laboratorio de Simulación")
    
    st.markdown("### 🎲 Simulador Monte Carlo (Partido Actual)")
    st.info(f"Simulando: **{home} vs {away}**")
    if st.button("▶️ Ejecutar Monte Carlo (1,000 Partidos)"):
        sim_h = np.random.poisson(h_exp, 1000)
        sim_a = np.random.poisson(a_exp, 1000)
        sim_diff = sim_h - sim_a
        wins_h = np.sum(sim_diff > 0); draws = np.sum(sim_diff == 0); wins_a = np.sum(sim_diff < 0)
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Local Gana", f"{wins_h/10:.1f}%")
        sc2.metric("Empate", f"{draws/10:.1f}%")
        sc3.metric("Visita Gana", f"{wins_a/10:.1f}%")
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Histogram(x=sim_h, name=home, marker_color='#4CAF50', opacity=0.75))
        fig_sim.add_trace(go.Histogram(x=sim_a, name=away, marker_color='#2196F3', opacity=0.75))
        fig_sim.update_layout(barmode='overlay', title="Distribución de Goles Simulados", xaxis_title="Goles")
        st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()
    st.markdown("### 📜 Backtest Histórico")
    if st.button("▶️ Validar con últimos 50 partidos"):
        test_df, ok, profit = run_backtest(df, stats, ah, aa) 
        m1, m2, m3 = st.columns(3)
        m1.metric("Aciertos", f"{ok}/50 ({ok/50*100:.0f}%)") 
        m2.metric("Profit", f"{profit:.2f} U"); m3.metric("Estado", "🔥 Rentable" if profit > 0 else "❄️ Pérdidas")
        st.dataframe(test_df, use_container_width=True)

# --- TAB 6: RENDIMIENTO (BI) ---
with t6:
    st.markdown("## 📈 Estadísticas de Rendimiento")
    if os.path.exists(CSV_FILE):
        df_hist = pd.read_csv(CSV_FILE)
        df_finished = df_hist[df_hist['Estado'].isin(['Ganada', 'Perdida', 'Push'])].copy()
        
        if not df_finished.empty:
            tot_inv = df_finished['Stake'].sum(); tot_prof = df_finished['Ganancia'].sum()
            roi = (tot_prof / tot_inv * 100) if tot_inv > 0 else 0
            k1, k2, k3 = st.columns(3)
            k1.metric("Beneficio Neto", f"${tot_prof:,.2f}"); k2.metric("ROI", f"{roi:.2f}%"); k3.metric("Apuestas", len(df_finished))
            
            c1, c2 = st.columns(2)
            with c1:
                prof_league = df_finished.groupby('Liga')['Ganancia'].sum().sort_values()
                fig_l = go.Figure(go.Bar(x=prof_league.values, y=prof_league.index, orientation='h'))
                st.plotly_chart(fig_l, use_container_width=True)
            with c2:
                st.write("**Distribución**")
                st.plotly_chart(go.Figure(go.Pie(labels=df_finished['Estado'].unique(), values=df_finished['Estado'].value_counts())), use_container_width=True)
        else: st.info("No hay apuestas finalizadas para analizar.")
    else: st.warning("Aún no hay historial.")
