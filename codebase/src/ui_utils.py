import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

def set_premium_style():
    """
    Applies an even more refined 'Executive BI' theme with glassmorphism and soft depth.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');
        
        :root {
            --bg-color: #f7f8f9;
            --card-bg: #ffffff;
            --primary: #121212;
            --secondary: #64748b;
            --accent: #f59e0b;
            --success: #10b981;
            --gradient-1: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            --radius-xl: 30px;
            --radius-lg: 20px;
        }

        .stApp {
            background-color: var(--bg-color);
            color: var(--primary);
            font-family: 'Outfit', sans-serif;
        }

        /* Hero Container */
        .hero-banner {
            background: var(--gradient-1);
            color: white;
            padding: 40px;
            border-radius: var(--radius-xl);
            margin-bottom: 30px;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.1);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::after {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 50%;
        }

        /* Card System */
        .glass-card {
            background-color: var(--card-bg);
            border-radius: var(--radius-lg);
            padding: 24px;
            border: 1px solid rgba(0,0,0,0.03);
            box-shadow: 0 4px 20px rgba(0,0,0,0.02);
            height: 100%;
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.05);
            border-color: rgba(245, 158, 11, 0.2);
        }

        /* Status Badges */
        .status-badge {
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 12px;
            font-weight: 500;
            display: inline-block;
            margin-bottom: 15px;
        }
        .badge-success { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef9c3; color: #854d0e; }
        .badge-info { background: #e0f2fe; color: #075985; }

        /* Action Card (For Suggestions) */
        .action-card {
            background: #f8fafc;
            border-left: 4px solid var(--accent);
            padding: 15px;
            border-radius: 0 12px 12px 0;
            margin-bottom: 10px;
            font-size: 14px;
        }

        /* Navigation Style */
        .stTabs [data-baseweb="tab-list"] { background-color: transparent; padding: 10px 0; }
        .stTabs [data-baseweb="tab"] {
            background: #e2e8f0; border-radius: 12px; padding: 10px 25px; border: none;
            color: #475569; margin-right: 10px; font-weight: 500;
        }
        .stTabs [aria-selected="true"] { background: var(--primary) !important; color: white !important; }

        /* KPI Bars */
        .kpi-title { color: var(--secondary); font-size: 14px; margin-bottom: 5px; }
        .kpi-value { font-size: 28px; font-weight: 600; color: var(--primary); }
        .progress-bar-bg { width: 100%; background: #f1f5f9; height: 8px; border-radius: 4px; margin-top: 10px; }
        .progress-bar-fill { height: 100%; background: var(--accent); border-radius: 4px; }
        
        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
        </style>
    """, unsafe_allow_html=True)

def render_hero_section(name, version="3.0"):
    st.markdown(f"""
        <div class="hero-banner">
            <h1 style="color:white; margin:0;">Welcome Back, {name}</h1>
            <p style="color: rgba(255,255,255,0.7); margin:10px 0 0 0;">
                AI Forecaster v{version} is monitoring {datetime.now().strftime('%B %Y')} trends. 
                System Status: <span style="color:#4ade80;">Active & Optimized</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_kpi_card(label, value, progress, status="success"):
    status_class = f"badge-{status}"
    st.markdown(f"""
        <div class="glass-card">
            <div class="status-badge {status_class}">LIVE UPDATE</div>
            <div class="kpi-title">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {progress}%"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_action_card(title, description):
    st.markdown(f"""
        <div class="action-card">
            <b style="color:#f59e0b;">Action Recommendation:</b> {title}<br>
            <span style="color:#64748b;">{description}</span>
        </div>
    """, unsafe_allow_html=True)

def plot_sales_trend(df):
    fig = px.line(df, x='ds', y='y', title=None, template='plotly_white')
    fig.update_traces(line_color='#1e293b', line_width=4)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False)
    )
    return fig

def plot_gauge(value, target, label):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': label, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, target*1.5]},
            'bar': {'color': "#1e293b"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#f1f5f9",
            'steps': [
                {'range': [0, target*0.8], 'color': '#fef2f2'},
                {'range': [target*0.8, target], 'color': '#fef9c3'}
            ],
            'threshold': {
                'line': {'color': "#f59e0b", 'width': 4},
                'thickness': 0.75,
                'value': target
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_forecast(forecast, actual=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Forecast', line=dict(color='#f59e0b', width=4)))
    if actual is not None:
        fig.add_trace(go.Scatter(x=actual['ds'], y=actual['y'], name='Actual', mode='lines', line=dict(color='#cbd5e1', width=1, dash='dot')))
    fig.update_layout(template='plotly_white', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
    return fig

def plot_decomposition(dec):
    if dec is None: return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dec.trend.index, y=dec.trend, name='Trend', line_color='#1a1a1a'))
    fig.add_trace(go.Scatter(x=dec.seasonal.index, y=dec.seasonal, name='Seasonality', line_color='#f59e0b'))
    
    fig.update_layout(
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def plot_global_map(df):
    country_sales = df.groupby('country').agg({'sales': 'sum', 'profit': 'sum'}).reset_index()
    fig = px.scatter_geo(country_sales, locations="country", locationmode='country names',
                         size="sales", color="profit", hover_name="country",
                         projection="natural earth",
                         template='plotly_white', color_continuous_scale='Greys')
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig
