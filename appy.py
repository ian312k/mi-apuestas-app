# --- TAB 6: RENDIMIENTO (BI) ---
with t6:
    st.markdown("## 📈 Estadísticas de Rendimiento")
    if os.path.exists(CSV_FILE):
        df_hist = pd.read_csv(CSV_FILE)
        # Filtramos solo las finalizadas
        df_finished = df_hist[df_hist["Estado"].isin(["Ganada", "Perdida", "Push"])].copy()

        if not df_finished.empty:
            # Ordenamos por ID para simular la secuencia real de apuestas
            df_finished = df_finished.sort_values("ID")

            # --- CÁLCULOS FINANCIEROS ---
            tot_inv = df_finished["Stake"].sum()
            tot_prof = df_finished["Ganancia"].sum()
            roi = (tot_prof / tot_inv * 100) if tot_inv > 0 else 0

            # --- CÁLCULO DE DRAWDOWN ---
            # 1. Equity Curve (Acumulado)
            df_finished["Equity"] = df_finished["Ganancia"].cumsum()
            
            # 2. High Water Mark (El punto más alto alcanzado hasta el momento)
            df_finished["Peak"] = df_finished["Equity"].cummax()
            
            # 3. Drawdown (Distancia actual respecto al pico)
            df_finished["Drawdown"] = df_finished["Equity"] - df_finished["Peak"]
            
            # 4. Max Drawdown (La peor caída registrada)
            max_dd = df_finished["Drawdown"].min()

            # --- METRICAS ---
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Beneficio Neto", f"${tot_prof:,.2f}")
            k2.metric("ROI", f"{roi:.2f}%")
            # Delta inverse: si es muy negativo, sale rojo (alerta)
            k3.metric("Max Drawdown", f"{max_dd:.2f} U", help="Máxima caída acumulada desde el punto más alto de ganancias.", delta="Riesgo", delta_color="off") 
            k4.metric("Apuestas", len(df_finished))

            # --- GRÁFICAS ---
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🌊 Curva de Drawdown (Riesgo)")
                # Gráfico de área roja para ver el riesgo
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=pd.to_datetime(df_finished["Fecha"]), 
                    y=df_finished["Drawdown"],
                    fill='tozeroy',
                    mode='lines',
                    line=dict(color='#FF5252', width=2),
                    name='Drawdown'
                ))
                fig_dd.update_layout(
                    height=250, 
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis_title="Unidades por debajo del pico"
                )
                st.plotly_chart(fig_dd, use_container_width=True)

            with c2:
                st.markdown("##### 📊 Distribución por Liga")
                prof_league = df_finished.groupby("Liga")["Ganancia"].sum().sort_values()
                # Colores dinámicos: Verde si gana, Rojo si pierde
                colors = ['#FF5252' if x < 0 else '#4CAF50' for x in prof_league.values]
                
                fig_l = go.Figure(go.Bar(
                    x=prof_league.values, 
                    y=prof_league.index, 
                    orientation="h",
                    marker_color=colors
                ))
                fig_l.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_l, use_container_width=True)

        else:
            st.info("No hay apuestas finalizadas para analizar.")
    else:
        st.warning("Aún no hay historial.")
