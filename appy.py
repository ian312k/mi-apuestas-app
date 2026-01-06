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
# ML imports
# =========================
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss

# ======================================================
# 1. CONFIGURACIÓN
# ======================================================
st.set_page_config(page_title="Dixon-Coles Pro v6.1 (Corregido)", layout="wide", page_icon="🛡️")
CSV_FILE = "mis_apuestas_pro.csv"

# --- TRADUCTOR DE EQUIPOS ---
TEAM_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest", "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton", "Leeds United": "Leeds",
    "West Ham United": "West Ham", "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham", "Leicester City": "Leicester",
    "Sheffield United": "Sheffield United", "Athletic Club": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid", "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis", "Celta Vigo": "Celta", "RCD Espanyol": "Espanol",
    "Espanyol": "Espanol", "Real Sociedad": "Sociedad", "Rayo Vallecano": "Vallecano",
    "Deportivo Alavés": "Alaves", "Alavés": "Alaves", "Internazionale": "Inter",
    "Inter Milan": "Inter", "AC Milan": "Milan", "AS Roma": "Roma",
    "Hellas Verona": "Verona", "Parma Calcio 1913": "Parma",
}

def normalize_name(name):
    return TEAM_MAP.get(name, name)

# --- SESSION STATE ---
if "market_storage" not in st.session_state: st.session_state.market_storage = {}
if "model_params" not in st.session_state:
    st.session_state.model_params = {
        "alpha": 0.008, "rho": -0.13,
        "window_matches": 400, "mix_factor": 0.7,
        "use_only_current": True
    }

# ======================================================
# 2. FUNCIONES AUXILIARES (Faltantes en tu snippet)
# ======================================================
@st.cache_data
def load_data():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=["date", "home", "away", "home_goals", "away_goals", "season", "odd_h", "odd_d", "odd_a"])
    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df

def match_odds_from_scanner_item(item):
    # Mock para evitar errores si no tienes la función definida
    try:
        # Ajusta esto a la estructura real de tu JSON del scanner
        markets = item.get("bookmakers", [])[0].get("markets", [])
        odds = [m for m in markets if m["key"] == "h2h"][0]["outcomes"]
        oh = next((x["price"] for x in odds if x["name"] == item["home_team"]), np.nan)
        oa = next((x["price"] for x in odds if x["name"] == item["away_team"]), np.nan)
        od = next((x["price"] for x in odds if x["name"] == "Draw"), np.nan)
        return oh, od, oa
    except:
        return np.nan, np.nan, np.nan

def build_features_for_match(row, team_stats, avg_h, avg_a):
    # Mock simple para que funcione fast_eval_ml
    return np.array([1.0, 1.0]) 

def outcome_1x2_label(hg, ag):
    if hg > ag: return 0
    if hg < ag: return 2
    return 1

def fit_ml_multiclass(X, y):
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, y)
    return clf

def brier_multiclass(probs, y_true):
    # Implementación simple del Brier score multiclass
    y_ohe = np.zeros_like(probs)
    y_ohe[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((probs - y_ohe)**2, axis=1))

def strict_walkforward_eval_ml_blocks(df, n_test, min_train, window_matches, retrain_every, train_step, alpha, rho):
    # Placeholder para la función estricta
    return {"logloss": 0.0, "accuracy": 0.0, "note": "Función placeholder"}

# ======================================================
# 3. DIXON-COLES CORREGIDO
# ======================================================
def calculate_strengths(df, ref_date=None, alpha=0.008, mix_factor=0.7, window_matches=400):
    df = df.copy().dropna(subset=["date", "home", "away", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    
    if window_matches and window_matches > 0 and len(df) > window_matches:
        df = df.tail(window_matches).reset_index(drop=True)
    
    if df.empty:
        return {}, 0.0, 0.0, []
    
    last_date = pd.to_datetime(ref_date) if ref_date is not None else df["date"].max()
    df["days_ago"] = (last_date - df["date"]).dt.days.clip(lower=0)
    df["weight"] = np.exp(-alpha * df["days_ago"])
    
    df["weight"] = df["weight"].clip(lower=0.05, upper=1.0)

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
            
            goals_scored_weighted = np.average(team_matches["goals_scored"], weights=team_matches["weight"])
            goals_conceded_weighted = np.average(team_matches["goals_conceded"], weights=team_matches["weight"])
            
            att_global = (goals_scored_weighted / avg_global) if avg_global > 0 else 1.0
            def_global = (goals_conceded_weighted / avg_global) if avg_global > 0 else 1.0
        else:
            att_global, def_global = 1.0, 1.0

        h_m = df[df["home"] == team]
        if not h_m.empty:
            home_goals_weighted = np.average(h_m["home_goals"], weights=h_m["weight"])
            away_goals_weighted = np.average(h_m["away_goals"], weights=h_m["weight"])
            att_h_pure = (home_goals_weighted / avg_home) if avg_home > 0 else 1.0
            def_h_pure = (away_goals_weighted / avg_away) if avg_away > 0 else 1.0
            sot_h_avg = np.average(h_m["sot_h"], weights=h_m["weight"]) if "sot_h" in h_m.columns else 0.0
        else:
            att_h_pure, def_h_pure, sot_h_avg = 1.0, 1.0, 0.0

        a_m = df[df["away"] == team]
        if not a_m.empty:
            away_goals_weighted = np.average(a_m["away_goals"], weights=a_m["weight"])
            home_goals_weighted = np.average(a_m["home_goals"], weights=a_m["weight"])
            att_a_pure = (away_goals_weighted / avg_away) if avg_away > 0 else 1.0
            def_a_pure = (home_goals_weighted / avg_home) if avg_home > 0 else 1.0
            sot_a_avg = np.average(a_m["sot_a"], weights=a_m["weight"]) if "sot_a" in a_m.columns else 0.0
        else:
            att_a_pure, def_a_pure, sot_a_avg = 1.0, 1.0, 0.0

        team_stats[team] = {
            "att_h": (att_h_pure * mix_factor) + (att_global * (1 - mix_factor)),
            "def_h": (def_h_pure * mix_factor) + (def_global * (1 - mix_factor)),
            "att_a": (att_a_pure * mix_factor) + (att_global * (1 - mix_factor)),
            "def_a": (def_a_pure * mix_factor) + (def_global * (1 - mix_factor)),
            "sot_h_avg": sot_h_avg, "sot_a_avg": sot_a_avg,
            "matches_count": len(team_matches)
        }

    return team_stats, avg_home, avg_away, all_teams

def predict_match_dixon_coles(home, away, team_stats, avg_h, avg_a, rho=-0.13, max_goals=12):
    if home not in team_stats or away not in team_stats:
        return 0, 0, 0, 0, 0, 0, 0, 0, [], np.zeros((1, 1))

    h_exp = team_stats[home]["att_h"] * team_stats[away]["def_a"] * avg_h
    a_exp = team_stats[away]["att_a"] * team_stats[home]["def_h"] * avg_a
    
    h_exp = np.log1p(np.exp(h_exp)) - 0.5 
    a_exp = np.log1p(np.exp(a_exp)) - 0.5
    h_exp = min(max(h_exp, 0.1), 5.0)
    a_exp = min(max(a_exp, 0.1), 5.0)

    probs = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            p_base = poisson.pmf(x, h_exp) * poisson.pmf(y, a_exp)
            correction = 1.0
            if x == 0 and y == 0: correction = 1.0 - (h_exp * a_exp * rho)
            elif x == 0 and y == 1: correction = 1.0 + (h_exp * rho)
            elif x == 1 and y == 0: correction = 1.0 + (a_exp * rho)
            elif x == 1 and y == 1: correction = 1.0 - rho
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
# 4. BACKTEST CORREGIDO
# ======================================================
def run_backtest_no_leak(df, n_test=50, min_train=100, window_matches=400, stake_unit=1.0, alpha=0.008, rho=-0.13, min_ev_threshold=0.05):
    df_sorted = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df_sorted) > 800: df_sorted = df_sorted.tail(800).reset_index(drop=True)
    test_block = df_sorted.tail(n_test)

    results = []
    correct, bal = 0, 0.0
    n_bets = 0

    for _, row in test_block.iterrows():
        cut_date = row["date"]
        train_df = df_sorted[df_sorted["date"] < cut_date].copy()
        if len(train_df) < min_train: continue

        team_stats, avg_h, avg_a, _ = calculate_strengths(train_df, ref_date=cut_date, window_matches=window_matches, alpha=alpha)
        if row["home"] not in team_stats or row["away"] not in team_stats: continue

        _, _, ph, pd_prob, pa, *_ = predict_match_dixon_coles(row["home"], row["away"], team_stats, avg_h, avg_a, rho=rho)

        odd_h = float(row.get("odd_h", np.nan))
        odd_d = float(row.get("odd_d", np.nan))
        odd_a = float(row.get("odd_a", np.nan))
        
        if (np.isnan(odd_h) or odd_h <= 1.01) or (np.isnan(odd_d) or odd_d <= 1.01) or (np.isnan(odd_a) or odd_a <= 1.01): continue

        ev_h, ev_d, ev_a = (ph * odd_h) - 1, (pd_prob * odd_d) - 1, (pa * odd_a) - 1
        evs = {"Local": ev_h, "Empate": ev_d, "Visita": ev_a}
        probs = {"Local": ph, "Empate": pd_prob, "Visita": pa}
        odds = {"Local": odd_h, "Empate": odd_d, "Visita": odd_a}
        
        best_option = max(evs.items(), key=lambda x: x[1])
        if best_option[1] < min_ev_threshold: continue
            
        pred = best_option[0]
        prob, odd, ev = probs[pred], odds[pred], best_option[1]
        
        if row["home_goals"] > row["away_goals"]: res_real = "Local"
        elif row["home_goals"] < row["away_goals"]: res_real = "Visita"
        else: res_real = "Empate"
        
        is_win = (pred == res_real)
        profit_u = (odd - 1) * stake_unit if is_win else -stake_unit
        correct += int(is_win)
        bal += profit_u
        n_bets += 1

        results.append({
            "Fecha": row["date"].strftime("%Y-%m-%d"),
            "Partido": f"{row['home']} vs {row['away']}",
            "Predicción": f"{pred} ({prob*100:.0f}%)",
            "Realidad": f"{int(row['home_goals'])}-{int(row['away_goals'])}",
            "Cuota": odd, "EV": f"{ev*100:.1f}%",
            "Res": "✅" if is_win else "❌",
            "Stake(U)": stake_unit, "P/L(U)": profit_u
        })

    total_stake = n_bets * stake_unit
    roi = (bal / total_stake * 100) if total_stake > 0 else 0.0
    return pd.DataFrame(results), correct, bal, roi, n_bets, total_stake

# ======================================================
# 5. ESCÁNER CORREGIDO
# ======================================================
def run_scanner_analysis(code, teams, stats, ah, aa):
    if code not in st.session_state.market_storage: return []
    
    stored = st.session_state.market_storage[code]
    data_to_display = stored.get("data", [])
    if not data_to_display: return []
    
    now_utc = pd.Timestamp.now(tz="UTC")
    live_rows = []
    
    for item in data_to_display:
        match_date = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(match_date): continue
        
        diff_hours = (match_date - now_utc).total_seconds() / 3600
        if diff_hours > 168 or diff_hours < -2: continue

        h_api = normalize_name(item.get("home_team", ""))
        a_api = normalize_name(item.get("away_team", ""))
        
        m_h = get_close_matches(h_api, teams, n=1, cutoff=0.7)
        m_a = get_close_matches(a_api, teams, n=1, cutoff=0.7)

        if not m_h or not m_a: continue
        h, a = m_h[0], m_a[0]
        if h not in stats or a not in stats: continue

        oh2, od2, oa2 = match_odds_from_scanner_item(item)
        if np.isnan(oh2) or np.isnan(od2) or np.isnan(oa2): continue
        
        h_exp_dc, a_exp_dc, ph2, pd2, pa2, po15_2, po25_2, pbtts_2, _, _ = predict_match_dixon_coles(
            h, a, stats, ah, aa, rho=st.session_state.model_params["rho"]
        )

        ev_h, ev_d, ev_a = (ph2 * oh2) - 1, (pd2 * od2) - 1, (pa2 * oa2) - 1
        best_ev = max(ev_h, ev_d, ev_a)
        
        pick = "No Bet"
        ev_value = 0.0
        
        if best_ev > 0.05:
            if best_ev == ev_h: pick, ev_value = f"Gana {h}", ev_h
            elif best_ev == ev_d: pick, ev_value = "Empate", ev_d
            else: pick, ev_value = f"Gana {a}", ev_a

        live_rows.append({
            "Hora (UTC)": match_date.strftime("%d/%m %H:%M"),
            "Partido": f"{h} vs {a}",
            "xG": f"{h_exp_dc:.1f}-{a_exp_dc:.1f}",
            "Cuotas": f"{oh2:.2f}/{od2:.2f}/{oa2:.2f}",
            "DC %": f"{ph2*100:.0f}/{pd2*100:.0f}/{pa2*100:.0f}",
            "Value": ev_value,
            "Pick": pick,
            "O2.5%": f"{po25_2*100:.0f}",
            "BTTS%": f"{pbtts_2*100:.0f}"
        })
    
    if live_rows:
        df_live = pd.DataFrame(live_rows)
        df_live["Value_num"] = df_live["Value"]
        df_live = df_live.sort_values("Value_num", ascending=False)
        df_live = df_live.drop("Value_num", axis=1)
        df_live["Value"] = df_live["Value"].apply(lambda x: f"{x*100:.1f}%")
        return df_live.to_dict('records')
    
    return []

# ======================================================
# 6. EVALUACIÓN ML
# ======================================================
def fast_eval_ml(df, n_test=200, min_train=300, window_matches=600, alpha=0.008, rho=-0.13):
    st.warning("⚠️ fast_eval_ml: Solo para diagnóstico rápido.")
    df_sorted = df.dropna(subset=["date","home","away","home_goals","away_goals"]).sort_values("date").reset_index(drop=True)
    if len(df_sorted) < (n_test + min_train): return None

    train_df = df_sorted.iloc[:-n_test].copy()
    test_df  = df_sorted.iloc[-n_test:].copy()

    team_stats, avg_h, avg_a, _ = calculate_strengths(train_df, ref_date=train_df["date"].max(), window_matches=window_matches, alpha=alpha)

    X_train, y_train = [], []
    for _, r in train_df.tail(window_matches).iterrows():
        if r["home"] not in team_stats or r["away"] not in team_stats: continue
        X_train.append(build_features_for_match(r, team_stats, avg_h, avg_a))
        y_train.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

    if len(y_train) < 200: return None

    X_train = np.vstack(X_train)
    y_train = np.array(y_train, dtype=int)
    model = fit_ml_multiclass(X_train, y_train)

    preds, y_true = [], []
    for _, r in test_df.iterrows():
        if r["home"] not in team_stats or r["away"] not in team_stats: continue
        x = build_features_for_match(r, team_stats, avg_h, avg_a).reshape(1, -1)
        preds.append(model.predict_proba(x)[0])
        y_true.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

    if len(y_true) == 0: return None

    P = np.vstack(preds)
    y = np.array(y_true, dtype=int)
    ll = float(log_loss(y, P, labels=[0,1,2]))
    br = brier_multiclass(P, y)
    accuracy = np.mean(np.argmax(P, axis=1) == y)
    
    return {"mode": "rápido", "n": int(len(y)), "logloss": ll, "brier": br, "accuracy": accuracy}

# ======================================================
# 7. MAIN UI STRUCTURE
# ======================================================
def main():
    st.sidebar.title("⚙️ Panel de Control")
    
    # Cargar datos
    df = load_data()
    
    if df.empty:
        st.error(f"No se encontró el archivo {CSV_FILE} o está vacío.")
        st.stop()

    # Calcular estadísticas base globales (necesario para el scanner)
    stats, ah, aa, teams_list = calculate_strengths(
        df, 
        window_matches=st.session_state.model_params["window_matches"],
        alpha=st.session_state.model_params["alpha"]
    )

    # ----------------------------------------------------
    # AQUÍ ES DONDE OCURRE LA CORRECCIÓN PRINCIPAL
    # Definimos las pestañas ANTES de usarlas
    # ----------------------------------------------------
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🏠 Inicio", 
        "📊 Datos", 
        "🧮 Calculadora", 
        "💎 Escáner", 
        "🧪 Laboratorio", 
        "📈 Rendimiento", 
        "🤖 ML Evaluación"
    ])

    # --- PESTAÑAS PLACEHOLDER (t1-t3) ---
    with t1: st.title("Bienvenido al Dashboard Pro v6.1")
    with t2: st.dataframe(df.head())
    with t3: st.write("Calculadora Dixon-Coles aquí.")
    with t6: st.write("Análisis de rendimiento aquí.")

    # ==========================
    # TAB 4: ESCÁNER CORREGIDO
    # ==========================
    with t4:
        st.markdown("## 💎 Escáner Seguro (CORREGIDO)")
        col_scan1, col_scan2 = st.columns([1,3])
        with col_scan1:
            code = st.selectbox("Seleccionar Liga", ["PL", "LL", "SA", "BL", "L1"], key="scan_code")
            if st.button("Actualizar Scanner"):
                # Aquí iría tu lógica de requests a la API
                st.info("Simulando actualización... (conecta tu API aquí)")

        if code in st.session_state.market_storage:
            stored = st.session_state.market_storage[code]
            data_to_display = stored.get("data", [])
            st.info(f"📂 Datos en memoria ({len(data_to_display)} partidos). Actualizado: {stored['timestamp'].strftime('%H:%M:%S')}")
            
            if data_to_display:
                live_rows_data = run_scanner_analysis(code, teams_list, stats, ah, aa)
                if live_rows_data:
                    df_live = pd.DataFrame(live_rows_data)
                    # El ordenamiento ya se hace dentro de run_scanner_analysis
                    
                    st.markdown(f"### 🎯 Oportunidades ({len(df_live)} partidos) - Ordenado por Value")
                    
                    def highlight_value(val):
                        try:
                            val_str = str(val).strip('%')
                            value_pct = float(val_str)
                            if value_pct > 10: return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                            elif value_pct > 5: return 'background-color: #fff3cd; color: #856404;'
                            else: return ''
                        except: return ''
                    
                    st.dataframe(df_live.style.applymap(highlight_value, subset=["Value"]), use_container_width=True, hide_index=True)
                    
                    csv_live = df_live.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar", data=csv_live, file_name=f"oportunidades_{code}.csv", mime="text/csv")
                else:
                    st.info("No se encontraron partidos con valor positivo.")
        else:
            st.write("No hay datos almacenados para esta liga. Por favor actualiza.")

    # ==========================
    # TAB 5: LABORATORIO (TU CÓDIGO)
    # ==========================
    with t5:
        st.markdown("## 🧪 Laboratorio - Backtesting Avanzado")
        
        col_lab1, col_lab2 = st.columns(2)
        with col_lab1:
            n_test = st.slider("Partidos a evaluar", 20, 300, 100, step=10)
            min_train = st.slider("Mínimo entrenamiento", 50, 500, 150, step=25)
        with col_lab2:
            stake_unit = st.number_input("Stake unit", 0.5, 10.0, 1.0, step=0.5)
            min_ev_backtest = st.slider("EV mínimo", 0.0, 0.2, 0.05, step=0.01)
        
        if st.button("▶️ Ejecutar Backtesting Dixon-Coles (CORREGIDO)"):
            with st.spinner(f"Backtesteando {n_test} partidos (eligiendo por EV)..."):
                test_df, ok, profit, roi_bt, n_bets, tot_stake = run_backtest_no_leak(
                    df, 
                    n_test=n_test, 
                    min_train=min_train, 
                    window_matches=st.session_state.model_params["window_matches"],
                    stake_unit=stake_unit,
                    alpha=st.session_state.model_params["alpha"],
                    rho=st.session_state.model_params["rho"],
                    min_ev_threshold=min_ev_backtest
                )
            
            if test_df.empty or n_bets == 0:
                st.warning(f"No se encontraron apuestas con EV > {min_ev_backtest*100:.1f}%")
            else:
                st.markdown("### 📊 Resultados Backtesting (CORREGIDO: Elegir por EV)")
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                with col_res1: st.metric("Apuestas", n_bets)
                with col_res2: st.metric("Win Rate", f"{(ok / n_bets) * 100:.1f}%")
                with col_res3: st.metric("Profit", f"{profit:.2f} U")
                with col_res4: st.metric("ROI", f"{roi_bt:.2f}%")
                
                avg_ev = test_df["EV"].str.rstrip('%').astype(float).mean()
                col_met1, col_met2 = st.columns(2)
                with col_met1: st.metric("Avg EV", f"{avg_ev:.1f}%")
                with col_met2: st.metric("Stake Total", f"{tot_stake:.2f} U")
                
                st.info(f"**Estrategia:** Apostar cuando EV > {min_ev_backtest*100:.1f}%, eligiendo la opción con máximo EV")
                st.divider()
                st.markdown("#### 📋 Detalle de Apuestas")
                st.dataframe(test_df, use_container_width=True, height=300)

    # ==========================
    # TAB 7: EVALUACIÓN ML (TU CÓDIGO)
    # ==========================
    with t7:
        st.markdown("## 🤖 ML 1X2: Ensemble Avanzado")
        st.warning("**fast_eval_ml()** tiene potential data leakage. Use 'strict' para métricas reales.")
        
        col_ml_in1, col_ml_in2 = st.columns(2)
        with col_ml_in1:
            n_test_ml = st.slider("Test Size (ML)", 50, 500, 200)
            window_ml = st.slider("Window (ML)", 200, 1000, 600)
        with col_ml_in2:
            min_train_ml = st.number_input("Min Train", 100, 1000, 300)
            retrain_every = st.number_input("Retrain Every", 1, 100, 10)
            train_step = 1

        if st.button("▶️ Evaluar Modelo ML"):
            col_eval_fast, col_eval_strict = st.columns(2)
            
            with col_eval_fast:
                st.markdown("#### ⚡ Rápido (Diagnóstico)")
                with st.spinner("Evaluación rápida..."):
                    out_fast = fast_eval_ml(
                        df, 
                        n_test=n_test_ml, 
                        min_train=min_train_ml, 
                        window_matches=window_ml,
                        alpha=st.session_state.model_params["alpha"],
                        rho=st.session_state.model_params["rho"]
                    )
                if out_fast:
                    st.metric("LogLoss", f"{out_fast['logloss']:.4f}")
                    st.metric("Accuracy", f"{out_fast.get('accuracy', 0)*100:.1f}%")
            
            with col_eval_strict:
                st.markdown("#### 🧪 Estricto (Placeholder)")
                with st.spinner("Evaluación walkforward..."):
                    out_strict = strict_walkforward_eval_ml_blocks(
                        df, n_test_ml, min_train_ml, window_ml, retrain_every, train_step,
                        st.session_state.model_params["alpha"], st.session_state.model_params["rho"]
                    )
                if out_strict:
                    st.info("Función estricta no implementada en este snippet.")

if __name__ == "__main__":
    main()
