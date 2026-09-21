"""
Dashboard Interactivo de Negocio para Retail Intelligence Platform.
Construido con Streamlit y Plotly para exploracion de forecast y simulacion de promociones.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Configuracion de pagina
st.set_page_config(
    page_title="Retail Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visual CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Carga los datos precalculados de features y benchmark."""
    features_path = Path("data/processed/features_master.parquet")
    benchmark_path = Path("data/processed/model_benchmark_results.csv")
    submission_path = Path("data/processed/submission.csv")

    df = pd.read_parquet(features_path)
    benchmark_df = pd.read_csv(benchmark_path) if benchmark_path.exists() else pd.DataFrame()
    sub_df = pd.read_csv(submission_path) if submission_path.exists() else pd.DataFrame()

    return df, benchmark_df, sub_df


def main():
    st.markdown('<div class="main-header">🛒 Retail Demand Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Plataforma de Forecasting Multi-Serie y Simulador de Demanda | Corporación Favorita (Ecuador)</div>', unsafe_allow_html=True)

    try:
        df, benchmark_df, sub_df = load_data()
    except Exception as e:
        st.error(f"Error cargando datos: {e}. Asegúrese de haber ejecutado los pipelines de datos.")
        return

    # Sidebar - Filtros
    st.sidebar.header("🔍 Filtros de Selección")

    stores = sorted(df["store_nbr"].unique().tolist())
    selected_store = st.sidebar.selectbox("Seleccione Tienda:", stores, index=0)

    # Info de la tienda seleccionada
    store_meta = df[df["store_nbr"] == selected_store].iloc[0]
    st.sidebar.markdown(f"""
    **Detalles de Tienda:**
    - **Ciudad:** {store_meta.get('city', 'N/A')}
    - **Provincia:** {store_meta.get('state', 'N/A')}
    - **Tipo:** {store_meta.get('type', 'N/A')} | **Cluster:** {store_meta.get('cluster', 'N/A')}
    """)

    families = sorted(df["family"].unique().tolist())
    selected_family = st.sidebar.selectbox("Seleccione Familia de Producto:", families, index=families.index("GROCERY I") if "GROCERY I" in families else 0)

    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["📈 Pronóstico de Demanda", "🏷️ Simulador de Promociones", "🏆 Benchmark de Modelos"])

    # Filtrar datos de la serie seleccionada
    series_df = df[(df["store_nbr"] == selected_store) & (df["family"] == selected_family)].sort_values("date")
    hist_df = series_df[~series_df["is_test"]].tail(60)
    test_df = series_df[series_df["is_test"]]

    # ==================== TAB 1: FORECAST ====================
    with tab1:
        st.subheader(f"Pronóstico de Demanda — Tienda {selected_store} | {selected_family}")

        # Unir predicciones del test
        if not sub_df.empty and not test_df.empty:
            test_merged = test_df.merge(sub_df, on="id", suffixes=("", "_pred"))
            forecast_sales = test_merged["sales_pred"].values
            total_forecast = forecast_sales.sum()
            avg_forecast = forecast_sales.mean()
            peak_day = test_merged.loc[test_merged["sales_pred"].idxmax(), "date"].strftime("%Y-%m-%d")
        else:
            total_forecast, avg_forecast, peak_day = 0, 0, "N/A"

        # KPIs superiores
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Venta Total Pronosticada (16d)", f"{total_forecast:,.0f} unds")
        col2.metric("Promedio Diario Esperado", f"{avg_forecast:,.1f} unds/día")
        col3.metric("Día Pico de Demanda", peak_day)
        col4.metric("Horizonte de Pronóstico", f"{len(test_df)} días")

        # Gráfico interactivo con Plotly
        fig = go.Figure()

        # Histórico reciente
        fig.add_trace(go.Scatter(
            x=hist_df["date"],
            y=hist_df["sales"],
            mode="lines+markers",
            name="Ventas Reales Históricas",
            line=dict(color="#1E293B", width=2),
            marker=dict(size=4)
        ))

        # Pronóstico futuro
        if not sub_df.empty and not test_df.empty:
            fig.add_trace(go.Scatter(
                x=test_merged["date"],
                y=test_merged["sales_pred"],
                mode="lines+markers",
                name="Pronóstico XGBoost Champion (Test)",
                line=dict(color="#2563EB", width=2.5, dash="dash"),
                marker=dict(size=6, symbol="diamond")
            ))

        fig.update_layout(
            title="Evolución Histórica y Pronóstico Futuro de Ventas Diarias",
            xaxis_title="Fecha",
            yaxis_title="Ventas (Unidades)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla detallada de forecast
        with st.expander("Ver tabla diaria de predicciones detallada"):
            if not sub_df.empty and not test_df.empty:
                display_table = test_merged[["date", "onpromotion", "sales_pred"]].copy()
                display_table.columns = ["Fecha", "Promociones Activas", "Venta Pronosticada (unds)"]
                display_table["Fecha"] = display_table["Fecha"].dt.strftime("%Y-%m-%d")
                display_table["Venta Pronosticada (unds)"] = display_table["Venta Pronosticada (unds)"].round(2)
                st.dataframe(display_table, use_container_width=True)

    # ==================== TAB 2: SIMULADOR DE PROMOCIONES ====================
    with tab2:
        st.subheader("Simulador de Elasticidad Promocional")
        st.markdown("Ajuste el volumen de promociones planificadas para estimar el incremento proyectado de ventas (Uplift).")

        col_sim1, col_sim2 = st.columns([1, 2])

        with col_sim1:
            current_promos = int(test_df["onpromotion"].mean()) if not test_df.empty else 0
            simulated_promo_boost = st.slider(
                "Variación en Promociones (%):",
                min_value=-50,
                max_value=100,
                value=20,
                step=5
            )
            promo_multiplier = 1 + (simulated_promo_boost / 100.0)
            st.info(f"Promociones base promedio: **{current_promos}** items $\rightarrow$ Simulado: **{int(current_promos * promo_multiplier)}** items.")

        with col_sim2:
            if not sub_df.empty and not test_df.empty:
                # Elasticidad estimada
                elasticity_factor = 0.35  # Factor de sensibilidad promedio
                simulated_sales = test_merged["sales_pred"] * (1 + (simulated_promo_boost / 100.0) * elasticity_factor)

                sim_fig = go.Figure()
                sim_fig.add_trace(go.Bar(
                    x=test_merged["date"].dt.strftime("%Y-%m-%d"),
                    y=test_merged["sales_pred"],
                    name="Pronóstico Base",
                    marker_color="#94A3B8"
                ))
                sim_fig.add_trace(go.Bar(
                    x=test_merged["date"].dt.strftime("%Y-%m-%d"),
                    y=simulated_sales,
                    name=f"Pronóstico Simulado ({simulated_promo_boost:+d}%)",
                    marker_color="#10B981"
                ))
                sim_fig.update_layout(
                    title="Impacto Proyectado de Variación en Promociones",
                    barmode="group",
                    xaxis_title="Fecha",
                    yaxis_title="Ventas Proyectadas",
                    template="plotly_white",
                    height=380
                )
                st.plotly_chart(sim_fig, use_container_width=True)

    # ==================== TAB 3: BENCHMARK ====================
    with tab3:
        st.subheader("Benchmark de Modelos y Evaluación Técnica")
        st.markdown("Comparativa de modelos en la ventana de validación temporal (`2017-07-31` a `2017-08-15`, 28,512 observaciones):")

        if not benchmark_df.empty:
            st.dataframe(benchmark_df, use_container_width=True)

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_rmsle = px.bar(
                    benchmark_df,
                    x="Modelo",
                    y="RMSLE",
                    color="Modelo",
                    title="Comparativa de RMSLE (Menor es Mejor)",
                    template="plotly_white"
                )
                fig_rmsle.update_layout(showlegend=False)
                st.plotly_chart(fig_rmsle, use_container_width=True)

            with col_b2:
                fig_wape = px.bar(
                    benchmark_df,
                    x="Modelo",
                    y="WAPE (%)",
                    color="Modelo",
                    title="Error Porcentual de Volumen Retail (WAPE %)",
                    template="plotly_white"
                )
                fig_wape.update_layout(showlegend=False)
                st.plotly_chart(fig_wape, use_container_width=True)
        else:
            st.warning("No se encontró el archivo de benchmark.")


if __name__ == "__main__":
    main()
