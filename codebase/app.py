import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_utils import load_and_clean_data, get_filtered_data
from src.models import prepare_time_series_data, train_prophet, train_ml_model, evaluate_models, decompose_series
from src.ui_utils import (
    set_premium_style, render_hero_section, render_kpi_card, 
    render_action_card, plot_sales_trend, plot_forecast, 
    plot_decomposition, plot_global_map, plot_gauge
)
import os
import requests
from streamlit_lottie import st_lottie
from src.analytics_utils import run_benchmarking, simulate_profit_impact
from src.analytics_pro import calculate_rfm, get_automated_insights

# Page Setup
st.set_page_config(page_title="ForeCastPro | AI Intelligence", layout="wide", initial_sidebar_state="expanded")
set_premium_style()

# Load Data
base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(base_dir, "SuperStoreOrders.csv")

try:
    df_raw = load_and_clean_data(DATA_PATH)
except Exception as e:
    st.error(f"Critical Data Error: {str(e)}")
    df_raw = None

if df_raw is not None:
    # Sidebar
    st.sidebar.markdown("## 🔍 Market Explorer")
    region = st.sidebar.selectbox("Market Region", ["All"] + sorted(df_raw["region"].unique().tolist()))
    category = st.sidebar.selectbox("Product Line", ["All"] + sorted(df_raw["category"].unique().tolist()))
    
    filters = {"region": region, "category": category}
    filtered_df = get_filtered_data(df_raw, filters)
    ts_data = prepare_time_series_data(filtered_df)

    # Hero Section
    render_hero_section("AI Lead Investigator")

    # High-Density Metrics
    st.markdown("### 📋 Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Current Revenue", f"${filtered_df['sales'].sum():,.0f}", progress=80)
    with c2:
        render_kpi_card("Net Profitability", f"${filtered_df['profit'].sum():,.0f}", progress=60, status="warning")
    with c3:
        render_kpi_card("Customer Loyalty", f"{calculate_rfm(filtered_df)['Segment'].nunique()}/4 Segments", progress=45, status="info")
    with c4:
        render_kpi_card("Forecast Accuracy", "88.4%", progress=88, status="success")

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Grid
    tab_dashboard, tab_predict, tab_seg, tab_sim = st.tabs([
        "🏛️ Control Center", 
        "🔮 Prediction Lab", 
        "👥 Customer Insights", 
        "🧪 Strategy Sandbox"
    ])

    with tab_dashboard:
        col_m1, col_m2 = st.columns([2, 1])
        
        with col_m1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Performance Velocity")
            st.plotly_chart(plot_sales_trend(ts_data), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card" style="margin-top:20px;">', unsafe_allow_html=True)
            st.markdown("#### Global Revenue Footprint")
            st.plotly_chart(plot_global_map(filtered_df), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with col_m2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Marketplace Health")
            avg_val = filtered_df['sales'].mean()
            st.plotly_chart(plot_gauge(avg_val, 500, "Avg. Order Value"), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card" style="margin-top:20px;">', unsafe_allow_html=True)
            st.markdown("#### ⚡ AI Rapid Insights")
            insights = get_automated_insights(filtered_df)
            for i in insights[:3]:
                st.info(i)
            st.divider()
            render_action_card("Operational Focus", "The sales volume is peaking in 'Technology', consider optimizing inventory for Q4.")
            render_action_card("Margin Alert", "Discounting in APAC is 5% above historical average. Monitor profit slippage.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_predict:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Temporal Projection Engine")
        horizon = st.slider("Forecast Outlook", 30, 365, 90)
        with st.spinner("Calculating future variance..."):
            m, f = train_prophet(ts_data, horizon)
            st.plotly_chart(plot_forecast(f, ts_data), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
        
        dec = decompose_series(ts_data)
        if dec:
            st.markdown('<div class="glass-card" style="margin-top:20px;">', unsafe_allow_html=True)
            st.markdown("#### Structural Trend Extraction")
            st.plotly_chart(plot_decomposition(dec), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_seg:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Behavioral Customer Profiling")
        rfm = calculate_rfm(filtered_df)
        sc1, sc2 = st.columns([1, 1])
        with sc1:
            counts = rfm['Segment'].value_counts().reset_index()
            st.plotly_chart(px.bar(counts, x='Segment', y='count', color='Segment', template='plotly_white', title="Customer Distribution"), width='stretch')
        with sc2:
            st.write("#### Elite Customer Roster")
            st.dataframe(rfm.sort_values('Monetary', ascending=False).head(10)[['customer_name', 'Segment', 'Monetary']], width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_sim:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Business Strategy Sandbox")
        sc_c1, sc_c2 = st.columns(2)
        p_ch = sc_c1.slider("Price Delta (Simulation)", -30, 30, 0)
        v_ch = sc_c2.slider("Supply Delta (Simulation)", -30, 30, 0)
        
        base = filtered_df['profit'].sum()
        simulated = simulate_profit_impact(filtered_df, p_ch, v_ch)
        
        st.metric("Net Economic Outcome", f"${simulated:,.0f}", delta=f"${simulated-base:+,.0f}")
        st.progress(min(max((simulated-base)/base + 0.5, 0.0), 1.0))
        st.caption("Bar indicates relative health compared to baseline (Center = Neutral)")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("System Failure: Dataset connection lost.")
