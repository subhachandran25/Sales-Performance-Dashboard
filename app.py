import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="Sales Intelligence Dashboard")

# --- CUSTOM CSS FOR MILD YELLOW/WHITE THEME ---
st.markdown("""
    <style>
    .stApp {background-color: #FFFBED;}
    .stMetric {background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #F4D03F;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("sales_performance_data.csv")

df = load_data()

# --- DRILL DOWN FILTERS ---
st.title("📊 Sales Intelligence Dashboard - India Operations")
col1, col2, col3, col4 = st.columns(4)
region = col1.selectbox("Region", df['Region'].unique())
auh = col2.selectbox("Area Head", df[df['Region'] == region]['AUH_Name'].unique())
sm = col3.selectbox("Senior Manager", df[df['AUH_Name'] == auh]['Senior_Manager_Name'].unique())
mgr = col4.selectbox("Sales Manager", df[df['Senior_Manager_Name'] == sm]['Sales_Manager_Name'].unique())

filtered_df = df[df['Sales_Manager_Name'] == mgr]

# --- HOME PAGE CONSOLIDATION ---
st.subheader("Executive Summary")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Revenue", f"₹{filtered_df['Total_Revenue'].sum():,.0f}")
kpi2.metric("Total Leads", filtered_df['New_Leads'].sum())
kpi3.metric("Deals Closed", filtered_df['Deals_Closed'].sum())
kpi4.metric("Avg Talk Time", f"{filtered_df['Call_Time_Mins'].mean():.1f} min")

# --- ANALYSIS TABS ---
tab1, tab2, tab3 = st.tabs(["Descriptive Analytics", "Diagnostic & Correlation", "Predictive & Prescriptive"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_funnel = px.funnel(filtered_df, x=['New_Leads', 'Qualified', 'Converted', 'Deals_Closed'], title="Lead Utilization Funnel")
        st.plotly_chart(fig_funnel, use_container_width=True)
    with col_b:
        fig_rev = px.bar(df.groupby('Region')['Total_Revenue'].sum().reset_index(), x='Region', y='Total_Revenue', title="Revenue Across Regions", color='Region')
        st.plotly_chart(fig_rev, use_container_width=True)

with tab2:
    st.subheader("Correlation Heatmap")
    corr = df[['Calls_Dialed', 'Call_Time_Mins', 'Total_Revenue', 'Converted', 'New_Leads']].corr()
    fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='YlOrBr')
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.subheader("Talk Time vs Revenue")
    fig_scatter = px.scatter(filtered_df, x='Call_Time_Mins', y='Total_Revenue', color='Deals_Closed', trendline="ols")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Predictive Model (Revenue Estimation)")
    X = df[['Call_Time_Mins', 'Converted']]
    y = df['Total_Revenue']
    model = LinearRegression().fit(X, y)
    
    talk_val = st.slider("Adjust Talk Time for Prediction", 100, 2000, 500)
    pred_rev = model.predict([[talk_val, df['Converted'].mean()]])
    st.info(f"Predicted Revenue based on {talk_val} mins: ₹{pred_rev[0]:,.2f}")
    
    st.subheader("Prescriptive Recommendations")
    if filtered_df['Call_Time_Mins'].mean() < 400:
        st.warning("⚠️ Training Required: Average talk time is low. Suggestion: Focus on 'Discovery Phase' training.")
    else:
        st.success("✅ Talk time is healthy. Focus on lead conversion optimization.")
