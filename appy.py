import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os

# ======================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS (DARK MODE) 🎨
# ======================================================
st.set_page_config(page_title="Dixon-Coles Pro", layout="wide", page_icon="⚽")
CSV_FILE = 'mis_apuestas_pro.csv'

# Estilos CSS para modo oscuro (Tarjetas grises elegantes)
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. LÓGICA DE DATOS Y MODELO MATEMÁTICO 🧠
# ======================================================
@st.cache_data
def fetch_live_soccer_data(league_code="SP1"):
    """Descarga datos incluyendo cuotas históricas (B365)"""
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        # Seleccionamos columnas clave + Cuotas de Bet365 para el Backtest
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']
        # Filtramos solo las que existan (por seguridad)
        actual_cols = [c for c in cols if c in df.columns]
        df = df[actual_cols]
        
        # Renombramos para facilitar uso
        new_names = ['date', 'home', 'away', 'home_goals', 'away_goals', 'odd_h', 'odd_d', 'odd_a']
        # Ajustamos si faltan columnas de cuotas
        if len(actual_cols) == 8:
            df.columns = new_names
        else:
            # Si no hay cuotas, usamos nombres básicos
            df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            df.columns = ['date', 'home', 'away', 'home_goals', 'away_goals']
            df['odd_h'] = 1.0; df['odd_d'] = 1.0; df['odd_a'] = 1.0 # Placeholder

        df = df.dropna()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        return df
    except: return pd.DataFrame()

def calculate_strengths(df):
    """Calcula fuerza de ataque/defensa con Time Decay"""
    last_date = df['date'].max()
    df['days_ago'] = (last_date - df['date']).dt.days
    alpha = 0.005 # Decaimiento temporal
    df['weight'] = np.exp(-alpha * df['days_ago'])
    
    avg_home = np.average(df['home_goals'], weights=df['weight'])
    avg_away = np.average(df['away_goals'], weights=df['weight'])
    
    team_stats = {}
    all_teams = sorted(list(set(df['home'].unique()) | set(df['away'].unique())))
    
    for team in all_teams:
        h_m = df[df['home'] == team]
        if not h_m.empty:
            att_h = np.average(h_m['home_goals'], weights=h_m['weight']) / avg_home
            def_h = np.average(h_m['away_goals'], weights=h_m['weight']) / avg_away
        else: att_h, def_h = 1.0, 1.0

        a_m = df[df['away'] == team]
        if not a_m.empty:
            att_a = np.average(a_m['away_goals'], weights=a_m['weight']) / avg_away
            def_a = np.average(a_m['home_goals'], weights=a_m['weight']) / avg_home
        else: att_a, def_a = 1.0, 1.0
            
        team_stats[team] = {'att_h': att_h, 'def_h': def_h, 'att_a': att_a, 'def_a': def_a}
        
    return team_stats, avg_home, avg_away, all_teams

def predict_match_dixon_coles(home, away, team_stats, avg_h, avg_a):
    """Modelo Matemático Avanzado con corrección de empates"""
    h_exp = team_stats[home]['att_h'] * team_stats[away]['def_a'] * avg_h
    a_exp = team_stats[away]['att_a'] * team_stats[home]['def_h'] * avg_a
    
    max_goals = 10
    probs = np.zeros((max_goals, max_goals))
    rho = -0.13 # Factor de corrección Dixon-Coles

    for x in range(max_goals):
        for y in range(max_goals):
            p_base = poisson.pmf(x, h_exp) * poisson.pmf(y, a_exp)
            correction = 1.0
            if x==0 and y==0: correction = 1.0 - (h_exp * a_exp * rho)
            elif x==0 and y==1: correction = 1.0 + (h_exp * rho)
            elif x==1 and y==0: correction = 1.0 + (a_exp * rho)
            elif x==1 and y==1: correction = 1.0 - (rho)
            probs[x][y] = p_base * correction
            
    probs = np.maximum(0, probs) # Corrección de negativos
    probs = probs / probs.sum() # Normalización

    p_home = np.tril(probs, -1).sum()
    p_draw = np.diag(probs).sum()
    p_away = np.triu(probs, 1).sum()
    
    p_o25 = 0
    for i in range(max_goals):
        for j in range(max_goals):
            if (i+j) > 2.5: p_o25 += probs[i][j]

    # Top 3 Scores
    flat_indices = np.argsort(probs.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, probs.shape)
        top_scores.append((f"{i}-{j}", probs[i][j]))

    return h_exp, a_exp, p_home, p_draw, p_away, p_o25, top_scores

def run_backtest(df, team_stats, avg_h, avg_a):
    """Prueba el modelo con los últimos 20 partidos reales"""
    recent = df.tail(20).copy()
    results = []
    correct, bal = 0, 0
    
    for _, row in recent.iterrows():
        _, _, ph, pd_prob, pa, _, _ = predict_match_dixon_coles(row['home'], row['away'], team_stats, avg_h, avg_a)
        
        # Determinar predicción del modelo
        if ph > pd_prob and ph > pa: pred, prob, odd, res_real = "Local", ph, row['odd_h'], ("Local" if row['home_goals'] > row['away_goals'] else "Fallo")
        elif pa > ph and pa > pd_prob: pred, prob, odd, res_real = "Visita", pa, row['odd_a'], ("Visita" if row['away_goals'] > row['home_goals'] else "Fallo")
        else: pred, prob, odd, res_real = "Empate", pd_prob, row['odd_d'], ("Empate" if row['home_goals'] == row['away_goals'] else "Fallo")
        
        is_win = (pred == res_real)
        profit = (odd - 1) if is_win else -1
        if is_win: correct += 1
        bal += profit
        
        results.append({
            "Partido": f"{row['home']} vs {row['away']}",
            "Predicción": f"{pred} ({prob*100:.0f}%)",
            "Realidad": f"{int(row['home_goals'])}-{int(row['away_goals'])}",
            "Cuota": odd,
            "Res": "✅" if is_win else "❌",
            "P/L": round(profit, 2)
        })
    return pd.DataFrame(results), correct, bal

# ======================================================
# 3. UTILIDADES VISUALES Y GESTIÓN 🛠️
# ======================================================
def plot_gauge(val, title, color):
    return go.Figure(go.Indicator(
        mode="gauge+number", value=val*100, title={'text': title},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "white"}
    )).update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))

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
    leagues = {"SP1": "🇪🇸 La Liga", "E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "I1": "🇮🇹 Serie A", "D1": "🇩🇪 Bundesliga", "F1": "🇫🇷 Ligue 1"}
    code = st.selectbox("Liga", list(leagues.keys()), format_func=lambda x: leagues[x])
    
    df = fetch_live_soccer_data(code)
    if not df.empty:
        stats, ah, aa, teams = calculate_strengths(df)
        st.success(f"✅ {len(df)} partidos cargados")
    else: st.error("Error cargando datos"); st.stop()
    
    st.divider()
    bank = st.number_input("💰 Tu Banco ($)", 1000.0, step=50.0)

st.title(f"🤖 Dixon-Coles: {leagues[code]}")

c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])

# EJECUCIÓN DEL MODELO
h_exp, a_exp, ph, pd_prob, pa, po25, top_sc = predict_match_dixon_coles(home, away, stats, ah, aa)

# PESTAÑAS
t1, t2, t3, t4 = st.tabs(["📊 Análisis", "💰 Valor", "📜 Historial", "🧪 Laboratorio"])

with t1:
    st.markdown("### 🥅 Expectativa de Goles")
    c_g1, c_g2, c_g3 = st.columns(3)
    c_g1.metric(home, f"{h_exp:.2f}")
    c_g2.metric("Total", f"{h_exp+a_exp:.2f}", delta="Over 2.5: "+f"{po25*100:.0f}%")
    c_g3.metric(away, f"{a_exp:.2f}")
    
    st.markdown("### 🏆 Probabilidades")
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(plot_gauge(ph, f"Gana {home}", "#4CAF50"), use_container_width=True)
    g2.plotly_chart(plot_gauge(pd_prob, "Empate", "#FFC107"), use_container_width=True)
    g3.plotly_chart(plot_gauge(pa, f"Gana {away}", "#2196F3"), use_container_width=True)
    
    st.info(f"🎯 **Marcador Exacto:** {top_sc[0][0]} ({top_sc[0][1]*100:.1f}%) | **Opción 2:** {top_sc[1][0]}")
    
    st.markdown("### 📉 Estado de Forma")
    cf1, cf2 = st.columns(2)
    with cf1: st.dataframe(get_last_5(df, home), use_container_width=True, hide_index=True)
    with cf2: st.dataframe(get_last_5(df, away), use_container_width=True, hide_index=True)

with t2:
    st.markdown("### 🏦 Buscador de Valor")
    co1, co2, co3 = st.columns(3)
    oh = co1.number_input("Cuota 1", 1.01, 20.0, 2.0)
    od = co2.number_input("Cuota X", 1.01, 20.0, 3.2)
    oa = co3.number_input("Cuota 2", 1.01, 20.0, 3.5)
    
    ev_h, kh = (ph*oh)-1, calculate_kelly(ph, oh)
    ev_d, kd = (pd_prob*od)-1, calculate_kelly(pd_prob, od)
    ev_a, ka = (pa*oa)-1, calculate_kelly(pa, oa)
    
    def card(lab, ev, k, odd):
        if ev > 0:
            st.success(f"✅ **{lab}** (+{ev*100:.1f}%)")
            st.markdown(f"**Apostar:** ${bank*(k/100):.2f} ({k:.1f}%)")
        else: st.error(f"❌ **{lab}** (EV {ev*100:.1f}%)")
            
    cv1, cv2, cv3 = st.columns(3)
    with cv1: card(home, ev_h, kh, oh)
    with cv2: card("Empate", ev_d, kd, od)
    with cv3: card(away, ev_a, ka, oa)
    
    st.divider()
    with st.form("bet"):
        pk = st.selectbox("Pick", [f"Gana {home}", "Empate", f"Gana {away}"])
        stk = st.number_input("Stake $", 1.0, 5000.0, 50.0)
        if "Gana "+home in pk: fo, fp = oh, ph
        elif "Empate" in pk: fo, fp = od, pd_prob
        else: fo, fp = oa, pa
        
        if st.form_submit_button("💾 Guardar"):
            manage_bets("save", {"ID": pd.Timestamp.now().strftime('%Y%m%d%H%M%S'), "Fecha": pd.Timestamp.now().strftime('%Y-%m-%d'), 
                                 "Liga": code, "Partido": f"{home}-{away}", "Pick": pk, "Cuota": fo, "Stake": stk, "Prob": round(fp, 4), "Estado": "Pendiente", "Ganancia": 0.0})
            st.success("Guardado!"); st.rerun()

with t3:
    st.markdown("### 📜 Historial")
    db = manage_bets("load")
    if not db.empty:
        st.metric("Balance Total", f"${db['Ganancia'].sum():.2f}")
        st.dataframe(db.sort_values(by="Fecha", ascending=False), use_container_width=True)
        with st.expander("Actualizar"):
            pen = db[db['Estado']=='Pendiente']
            if not pen.empty:
                bid = st.selectbox("ID", pen['ID'].unique())
                res = st.selectbox("Resultado", ["Ganada", "Perdida", "Push"])
                if st.button("Actualizar "): manage_bets("update", id_bet=bid, status=res); st.rerun()
            else: st.info("No hay pendientes")

with t4:
    st.markdown("### 🧪 Laboratorio de Backtesting")
    st.info("Pon a prueba el modelo con los últimos 20 partidos REALES de esta liga.")
    if st.button("▶️ Ejecutar Simulación"):
        test_df, ok, profit = run_backtest(df, stats, ah, aa)
        m1, m2, m3 = st.columns(3)
        m1.metric("Aciertos", f"{ok}/20 ({ok/20*100:.0f}%)")
        m2.metric("Profit (Stake 1U)", f"{profit:.2f} U", delta_color="normal")
        m3.metric("Estado", "🔥 Rentable" if profit > 0 else "❄️ Pérdidas")
        st.dataframe(test_df, use_container_width=True)
