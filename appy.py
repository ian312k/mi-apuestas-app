# app.py - VERSION CORREGIDA
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

from sklearn.metrics import log_loss

# ======================================================
# 1. CONFIGURACIÓN
# ======================================================
st.set_page_config(page_title="Dixon-Coles Pro v6.1 (Corregido)", layout="wide", page_icon="🛡️")
CSV_FILE = "mis_apuestas_pro.csv"
N_SEASONS = 3

# --- TRADUCTOR DE EQUIPOS ---
TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Leicester City": "Leicester",
    "Sheffield United": "Sheffield United",
    "Athletic Club": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "RCD Espanyol": "Espanol",
    "Espanyol": "Espanol",
    "Real Sociedad": "Sociedad",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alavés": "Alaves",
    "Alavés": "Alaves",
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",
    "Parma Calcio 1913": "Parma",
}

def normalize_name(name):
    return TEAM_MAP.get(name, name)

# --- SESSION STATE ---
if "ticket" not in st.session_state: st.session_state.ticket = []
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "api_usage" not in st.session_state: st.session_state.api_usage = {"used": 0, "remaining": 500}
if "market_storage" not in st.session_state: st.session_state.market_storage = {}
if "odds_inputs" not in st.session_state:
    st.session_state.odds_inputs = {"oh": 2.0, "od": 3.2, "oa": 3.5, "o_o25": 1.90, "o_btts": 1.90}
if "model_params" not in st.session_state:
    st.session_state.model_params = {
        "alpha": 0.008,
        "rho": -0.13,
        "window_matches": 400,
        "mix_factor": 0.7,
        "use_only_current": True
    }

# ======================================================
# 2. DIXON-COLES CORREGIDO
# ======================================================
def calculate_strengths(df, ref_date=None, alpha=0.008, mix_factor=0.7, window_matches=400):
    """Corregido: weight clipping más suave"""
    df = df.copy().dropna(subset=["date", "home", "away", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    
    # Filtrar por ventana temporal
    if window_matches and window_matches > 0 and len(df) > window_matches:
        df = df.tail(window_matches).reset_index(drop=True)
    
    if df.empty:
        return {}, 0.0, 0.0, []
    
    last_date = pd.to_datetime(ref_date) if ref_date is not None else df["date"].max()
    df["days_ago"] = (last_date - df["date"]).dt.days.clip(lower=0)
    df["weight"] = np.exp(-alpha * df["days_ago"])
    
    # CORRECCIÓN: Piso más bajo (0.05 en lugar de 0.2)
    # También permitir que partidos muy viejos pesen casi cero
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

        # Estadísticas como local
        h_m = df[df["home"] == team]
        if not h_m.empty:
            home_goals_weighted = np.average(h_m["home_goals"], weights=h_m["weight"])
            away_goals_weighted = np.average(h_m["away_goals"], weights=h_m["weight"])
            
            att_h_pure = (home_goals_weighted / avg_home) if avg_home > 0 else 1.0
            def_h_pure = (away_goals_weighted / avg_away) if avg_away > 0 else 1.0
            sot_h_avg = np.average(h_m["sot_h"], weights=h_m["weight"]) if "sot_h" in h_m.columns else 0.0
        else:
            att_h_pure, def_h_pure, sot_h_avg = 1.0, 1.0, 0.0

        # Estadísticas como visitante
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
            "sot_h_avg": sot_h_avg,
            "sot_a_avg": sot_a_avg,
            "matches_count": len(team_matches)
        }

    return team_stats, avg_home, avg_away, all_teams

def predict_match_dixon_coles(home, away, team_stats, avg_h, avg_a, rho=-0.13, max_goals=12):  # CORREGIDO: max_goals=12
    if home not in team_stats or away not in team_stats:
        return 0, 0, 0, 0, 0, 0, 0, 0, [], np.zeros((1, 1))

    # Calcular expectativas de goles
    h_exp = team_stats[home]["att_h"] * team_stats[away]["def_a"] * avg_h
    a_exp = team_stats[away]["att_a"] * team_stats[home]["def_h"] * avg_a
    
    # CORRECCIÓN: Suavizado más conservador
    # En lugar de clip duro, usar transformación logarítmica suave
    h_exp = np.log1p(np.exp(h_exp)) - 0.5  # Softplus centrado
    a_exp = np.log1p(np.exp(a_exp)) - 0.5
    # Límites más amplios
    h_exp = min(max(h_exp, 0.1), 5.0)
    a_exp = min(max(a_exp, 0.1), 5.0)

    probs = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            p_base = poisson.pmf(x, h_exp) * poisson.pmf(y, a_exp)
            correction = 1.0
            if x == 0 and y == 0:
                correction = 1.0 - (h_exp * a_exp * rho)
            elif x == 0 and y == 1:
                correction = 1.0 + (h_exp * rho)
            elif x == 1 and y == 0:
                correction = 1.0 + (a_exp * rho)
            elif x == 1 and y == 1:
                correction = 1.0 - rho
            probs[x][y] = p_base * correction

    probs = np.maximum(0, probs)
    probs = probs / probs.sum()

    # Probabilidades 1X2
    p_home = np.tril(probs, -1).sum()
    p_draw = np.diag(probs).sum()
    p_away = np.triu(probs, 1).sum()

    # Mercados especiales
    p_o15 = probs[(np.add.outer(np.arange(max_goals), np.arange(max_goals)) > 1.5)].sum()
    p_o25 = probs[(np.add.outer(np.arange(max_goals), np.arange(max_goals)) > 2.5)].sum()
    p_btts = probs[(np.arange(max_goals)[:, None] > 0) & (np.arange(max_goals)[None, :] > 0)].sum()

    # Top 3 marcadores
    flat_indices = np.argsort(probs.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, probs.shape)
        top_scores.append((f"{i}-{j}", probs[i][j]))

    return h_exp, a_exp, p_home, p_draw, p_away, p_o15, p_o25, p_btts, top_scores, probs

# ======================================================
# 3. BACKTEST CORREGIDO (Elegir por EV, no por probabilidad)
# ======================================================
def run_backtest_no_leak(df, n_test=50, min_train=100, window_matches=400, stake_unit=1.0, alpha=0.008, rho=-0.13, min_ev_threshold=0.05):
    """CORREGIDO: Elegir por máximo EV en lugar de máxima probabilidad"""
    df_sorted = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    
    if len(df_sorted) > 800:
        df_sorted = df_sorted.tail(800).reset_index(drop=True)
    
    test_block = df_sorted.tail(n_test)

    results = []
    correct, bal = 0, 0.0
    n_bets = 0

    for _, row in test_block.iterrows():
        cut_date = row["date"]
        train_df = df_sorted[df_sorted["date"] < cut_date].copy()
        if len(train_df) < min_train:
            continue

        team_stats, avg_h, avg_a, _ = calculate_strengths(
            train_df, ref_date=cut_date, window_matches=window_matches, alpha=alpha
        )
        
        if row["home"] not in team_stats or row["away"] not in team_stats:
            continue

        # Obtener probabilidades
        _, _, ph, pd_prob, pa, *_ = predict_match_dixon_coles(
            row["home"], row["away"], team_stats, avg_h, avg_a, rho=rho
        )

        # Obtener odds
        odd_h = float(row.get("odd_h", np.nan))
        odd_d = float(row.get("odd_d", np.nan))
        odd_a = float(row.get("odd_a", np.nan))
        
        if (np.isnan(odd_h) or odd_h <= 1.01) or (np.isnan(odd_d) or odd_d <= 1.01) or (np.isnan(odd_a) or odd_a <= 1.01):
            continue

        # CORRECCIÓN: Calcular EVs y elegir el máximo
        ev_h = (ph * odd_h) - 1
        ev_d = (pd_prob * odd_d) - 1
        ev_a = (pa * odd_a) - 1
        
        evs = {"Local": ev_h, "Empate": ev_d, "Visita": ev_a}
        probs = {"Local": ph, "Empate": pd_prob, "Visita": pa}
        odds = {"Local": odd_h, "Empate": odd_d, "Visita": odd_a}
        
        # Elegir la opción con máximo EV
        best_option = max(evs.items(), key=lambda x: x[1])
        
        # Solo apostar si el EV supera el umbral
        if best_option[1] < min_ev_threshold:
            continue
            
        pred = best_option[0]
        prob = probs[pred]
        odd = odds[pred]
        ev = best_option[1]
        
        # Determinar resultado real
        if row["home_goals"] > row["away_goals"]:
            res_real = "Local"
        elif row["home_goals"] < row["away_goals"]:
            res_real = "Visita"
        else:
            res_real = "Empate"
        
        is_win = (pred == res_real)
        profit_u = (odd - 1) * stake_unit if is_win else -stake_unit

        correct += int(is_win)
        bal += profit_u
        n_bets += 1

        results.append({
            "Fecha": row["date"].strftime("%Y-%m-%d"),
            "Temporada": row.get("season", ""),
            "Partido": f"{row['home']} vs {row['away']}",
            "Predicción": f"{pred} ({prob*100:.0f}%)",
            "Realidad": f"{int(row['home_goals'])}-{int(row['away_goals'])}",
            "Cuota": odd,
            "EV": f"{ev*100:.1f}%",
            "Res": "✅" if is_win else "❌",
            "Stake(U)": stake_unit,
            "P/L(U)": profit_u
        })

    total_stake = n_bets * stake_unit
    roi = (bal / total_stake * 100) if total_stake > 0 else 0.0
    return pd.DataFrame(results), correct, bal, roi, n_bets, total_stake

# ======================================================
# 4. ESCÁNER CORREGIDO (Ordenamiento numérico)
# ======================================================
def run_scanner_analysis():
    """Función auxiliar para el escáner con ordenamiento correcto"""
    # Esta función se llamará desde la pestaña del escáner
    
    if code not in st.session_state.market_storage:
        return []
    
    stored = st.session_state.market_storage[code]
    data_to_display = stored.get("data", [])
    
    if not data_to_display:
        return []
    
    now_utc = pd.Timestamp.now(tz="UTC")
    live_rows = []
    
    for item in data_to_display:
        match_date = pd.to_datetime(item.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(match_date):
            continue
        
        diff_hours = (match_date - now_utc).total_seconds() / 3600
        if diff_hours > 168 or diff_hours < -2:
            continue

        h_api = normalize_name(item.get("home_team", ""))
        a_api = normalize_name(item.get("away_team", ""))
        
        m_h = get_close_matches(h_api, teams, n=1, cutoff=0.7)
        m_a = get_close_matches(a_api, teams, n=1, cutoff=0.7)

        if not m_h or not m_a:
            continue
        
        h = m_h[0]
        a = m_a[0]
        
        if h not in stats or a not in stats:
            continue

        oh2, od2, oa2 = match_odds_from_scanner_item(item)
        if np.isnan(oh2) or np.isnan(od2) or np.isnan(oa2):
            continue
        
        # Predicción Dixon-Coles
        h_exp_dc, a_exp_dc, ph2, pd2, pa2, po15_2, po25_2, pbtts_2, _, _ = predict_match_dixon_coles(
            h, a, stats, ah, aa, rho=st.session_state.model_params["rho"]
        )

        # Calcular Value
        ev_h = (ph2 * oh2) - 1
        ev_d = (pd2 * od2) - 1
        ev_a = (pa2 * oa2) - 1
        
        best_ev = max(ev_h, ev_d, ev_a)
        pick = "No Bet"
        ev_value = 0.0
        
        if best_ev > 0.05:
            if best_ev == ev_h:
                pick = f"Gana {h}"
                ev_value = ev_h
            elif best_ev == ev_d:
                pick = "Empate"
                ev_value = ev_d
            else:
                pick = f"Gana {a}"
                ev_value = ev_a

        live_rows.append({
            "Hora (UTC)": match_date.strftime("%d/%m %H:%M"),
            "Partido": f"{h} vs {a}",
            "xG": f"{h_exp_dc:.1f}-{a_exp_dc:.1f}",
            "Cuotas": f"{oh2:.2f}/{od2:.2f}/{oa2:.2f}",
            "DC %": f"{ph2*100:.0f}/{pd2*100:.0f}/{pa2*100:.0f}",
            "Value": ev_value,  # CORREGIDO: Guardar como número, no string
            "Pick": pick,
            "O2.5%": f"{po25_2*100:.0f}",
            "BTTS%": f"{pbtts_2*100:.0f}"
        })
    
    # CORRECCIÓN: Ordenar por Value numérico
    if live_rows:
        df_live = pd.DataFrame(live_rows)
        # Convertir Value a numérico para ordenar
        df_live["Value_num"] = df_live["Value"]
        df_live = df_live.sort_values("Value_num", ascending=False)
        df_live = df_live.drop("Value_num", axis=1)
        
        # Convertir Value a string para display
        df_live["Value"] = df_live["Value"].apply(lambda x: f"{x*100:.1f}%")
        
        return df_live.to_dict('records')
    
    return []

# ======================================================
# 5. EVALUACIÓN ML CORREGIDA
# ======================================================
def fast_eval_ml(df, n_test=200, min_train=300, window_matches=600, alpha=0.008, rho=-0.13):
    """NOTA: Solo para diagnóstico rápido, no para métricas reales"""
    st.warning("⚠️ fast_eval_ml: Solo para diagnóstico rápido. Use 'strict_walkforward_eval_ml_blocks' para evaluación real.")
    
    df_sorted = df.dropna(subset=["date","home","away","home_goals","away_goals"]).sort_values("date").reset_index(drop=True)
    
    if len(df_sorted) < (n_test + min_train):
        return None

    train_df = df_sorted.iloc[:-n_test].copy()
    test_df  = df_sorted.iloc[-n_test:].copy()

    team_stats, avg_h, avg_a, _ = calculate_strengths(
        train_df, ref_date=train_df["date"].max(), 
        window_matches=window_matches, alpha=alpha
    )

    X_train, y_train = [], []
    for _, r in train_df.tail(window_matches).iterrows():
        if r["home"] not in team_stats or r["away"] not in team_stats:
            continue
        X_train.append(build_features_for_match(r, team_stats, avg_h, avg_a))
        y_train.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

    if len(y_train) < 200:
        return None

    X_train = np.vstack(X_train)
    y_train = np.array(y_train, dtype=int)
    model = fit_ml_multiclass(X_train, y_train)

    preds, y_true = [], []
    for _, r in test_df.iterrows():
        if r["home"] not in team_stats or r["away"] not in team_stats:
            continue
        x = build_features_for_match(r, team_stats, avg_h, avg_a).reshape(1, -1)
        preds.append(model.predict_proba(x)[0])
        y_true.append(outcome_1x2_label(r["home_goals"], r["away_goals"]))

    if len(y_true) == 0:
        return None

    P = np.vstack(preds)
    y = np.array(y_true, dtype=int)

    ll = float(log_loss(y, P, labels=[0,1,2]))
    br = brier_multiclass(P, y)
    
    pred_classes = np.argmax(P, axis=1)
    accuracy = np.mean(pred_classes == y)
    
    return {
        "mode": "rápido (DIAGNÓSTICO ONLY - puede tener data leakage)",
        "n": int(len(y)), 
        "logloss": ll, 
        "brier": br, 
        "accuracy": accuracy
    }

# ======================================================
# 6. TAB 5: LABORATORIO CORREGIDO
# ======================================================
with t5:
    st.markdown("## 🧪 Laboratorio - Backtesting Avanzado")
    
    col_lab1, col_lab2 = st.columns(2)
    with col_lab1:
        n_test = st.slider("Partidos a evaluar", 20, 300, 100, step=10)
        min_train = st.slider("Mínimo entrenamiento", 50, 500, 150, step=25)
    with col_lab2:
        stake_unit = st.number_input("Stake unit", 0.5, 10.0, 1.0, step=0.5)
        min_ev_backtest = st.slider("EV mínimo", 0.0, 0.2, 0.05, step=0.01)  # CORREGIDO: Este parámetro SÍ se usa
    
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
                min_ev_threshold=min_ev_backtest  # CORREGIDO: Usando el parámetro
            )
        
        if test_df.empty or n_bets == 0:
            st.warning(f"No se encontraron apuestas con EV > {min_ev_backtest*100:.1f}%")
        else:
            st.markdown("### 📊 Resultados Backtesting (CORREGIDO: Elegir por EV)")
            
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            with col_res1:
                st.metric("Apuestas", n_bets)
            with col_res2:
                win_rate = (ok / n_bets) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col_res3:
                st.metric("Profit", f"{profit:.2f} U")
            with col_res4:
                st.metric("ROI", f"{roi_bt:.2f}%")
            
            # Métricas adicionales
            avg_ev = test_df["EV"].str.rstrip('%').astype(float).mean()
            expectancy = profit / n_bets if n_bets > 0 else 0
            
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("Avg EV", f"{avg_ev:.1f}%")
            with col_met2:
                st.metric("Expectancy", f"{expectancy:.3f}")
            with col_met3:
                st.metric("Stake Total", f"{tot_stake:.2f} U")
            
            st.info(f"**Estrategia:** Apostar cuando EV > {min_ev_backtest*100:.1f}%, eligiendo la opción con máximo EV")
            
            st.divider()
            st.markdown("#### 📋 Detalle de Apuestas")
            st.dataframe(test_df, use_container_width=True, height=300)

# ======================================================
# 7. TAB 4: ESCÁNER CORREGIDO
# ======================================================
with t4:
    st.markdown("## 💎 Escáner Seguro (CORREGIDO)")
    
    # ... (código previo del escáner) ...
    
    if code in st.session_state.market_storage:
        stored = st.session_state.market_storage[code]
        data_to_display = stored.get("data", [])
        st.info(f"📂 Datos en memoria ({len(data_to_display)} partidos). Actualizado: {stored['timestamp'].strftime('%H:%M:%S')}")
        
        if data_to_display:
            # Usar la función corregida
            live_rows_data = run_scanner_analysis()
            
            if live_rows_data:
                # Convertir a DataFrame
                df_live = pd.DataFrame(live_rows_data)
                
                # CORRECCIÓN: Asegurar ordenamiento numérico
                if "Value" in df_live.columns:
                    # Extraer valor numérico del string "X.X%"
                    df_live["Value_num"] = df_live["Value"].str.rstrip('%').astype(float)
                    df_live = df_live.sort_values("Value_num", ascending=False)
                    df_live = df_live.drop("Value_num", axis=1)
                
                st.markdown(f"### 🎯 Oportunidades ({len(df_live)} partidos) - Ordenado por Value")
                
                # Color por value
                def highlight_value(val):
                    try:
                        # Manejar tanto números como strings
                        if isinstance(val, str):
                            value_pct = float(val.strip('%'))
                        else:
                            value_pct = float(val)
                        
                        if value_pct > 10:
                            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                        elif value_pct > 5:
                            return 'background-color: #fff3cd; color: #856404;'
                        else:
                            return ''
                    except:
                        return ''
                
                st.dataframe(
                    df_live.style.applymap(highlight_value, subset=["Value"]),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Botón de descarga
                csv_live = df_live.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar oportunidades",
                    data=csv_live,
                    file_name=f"oportunidades_{code}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No se encontraron partidos con valor positivo en el rango de tiempo seleccionado.")

# ======================================================
# 8. TAB 7: EVALUACIÓN ML (con advertencias)
# ======================================================
with t7:
    st.markdown("## 🤖 ML 1X2: Ensemble Avanzado")
    
    st.markdown("### ⚠️ Importante sobre evaluación ML")
    st.warning("""
    **fast_eval_ml()** tiene potential data leakage (usa team_stats calculados una sola vez).  
    **Solo úsalo para diagnóstico rápido.**  
    
    Para métricas reales, usa **strict_walkforward_eval_ml_blocks()** que reentrena en cada bloque.
    """)
    
    # ... (resto del código de la pestaña 7) ...
    
    if st.button("▶️ Evaluar Modelo ML (usar solo 'Estricto' para métricas reales)"):
        # Mostrar ambas evaluaciones pero con advertencias claras
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
                st.metric("LogLoss (fast)", f"{out_fast['logloss']:.4f}")
                st.metric("Accuracy (fast)", f"{out_fast.get('accuracy', 0)*100:.1f}%")
                st.caption("⚠️ Puede estar optimista por data leakage")
        
        with col_eval_strict:
            st.markdown("#### 🧪 Estricto (Métrica real)")
            with st.spinner("Evaluación walkforward..."):
                out_strict = strict_walkforward_eval_ml_blocks(
                    df, 
                    n_test=n_test_ml, 
                    min_train=min_train_ml, 
                    window_matches=window_ml,
                    retrain_every=retrain_every,
                    train_step=train_step,
                    alpha=st.session_state.model_params["alpha"],
                    rho=st.session_state.model_params["rho"]
                )
            
            if out_strict:
                st.metric("LogLoss (strict)", f"{out_strict['logloss']:.4f}")
                st.metric("Accuracy (strict)", f"{out_strict.get('accuracy', 0)*100:.1f}%")
                st.caption("✅ Métrica real sin data leakage")
