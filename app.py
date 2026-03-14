import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Sales Dashboard")

# Load Data with Error Handling
try:
    df = pd.read_csv("sales_performance_data.csv")
except FileNotFoundError:
    st.error("File 'sales_performance_data.csv' not found. Please run the generator script first.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Drill-Down Filters")
region = st.sidebar.selectbox("Region", df['Region'].unique())
auh = st.sidebar.selectbox("Area Head", df[df['Region'] == region]['AUH_Name'].unique())
sm = st.sidebar.selectbox("Senior Manager", df[df['AUH_Name'] == auh]['Senior_Manager_Name'].unique())
mgr = st.sidebar.selectbox("Sales Manager", df[df['Senior_Manager_Name'] == sm]['Sales_Manager_Name'].unique())

# Filter Data
filtered_df = df[df['Sales_Manager_Name'] == mgr]

# Dashboard Layout
st.title("📊 Sales Performance Dashboard")
st.markdown("---")

# KPI Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${filtered_df['Total_Revenue'].sum():,.0f}")
c2.metric("Deals Closed", filtered_df['Deals_Closed'].sum())
c3.metric("Avg Talk Time", f"{filtered_df['Call_Time_Mins'].mean():.1f}m")
c4.metric("Lead Conversion", f"{(filtered_df['Converted'].sum()/filtered_df['New_Leads'].sum())*100:.1f}%")

# Visualizations
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(filtered_df, x='Sales_Rep_Name', y='Total_Revenue', title="Revenue by Sales Rep")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.pie(filtered_df, names='Sales_Rep_Name', values='Deals_Closed', title="Deal Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# Data Table
st.subheader("Detailed Performance Table")
st.dataframe(filtered_df)
