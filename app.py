import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
st.set_page_config(layout="wide", page_title="Sales Intelligence Pro")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("sales_performance_data.csv")
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
region = st.sidebar.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())
data = df[df['Region'].isin(region)]

# Tabs
tabs = st.tabs(["🏠 Home", "📊 Descriptive", "🔍 Diagnostic", "🎯 Perspective", "🔮 Predictive"])

# 1. HOME PAGE
with tabs[0]:
    st.title("Executive Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dialed", data['Calls_Dialed'].sum())
    c2.metric("Connected", data['Converted'].sum())
    c3.metric("Avg Talk", f"{data['Call_Time_Mins'].mean():.1f}m")
    c4.metric("Deals", data['Deals_Closed'].sum())
    c5.metric("Revenue", f"₹{data['Total_Revenue'].sum():,.0f}")

    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(px.bar(data, x='Sales_Rep_Name', y=['Calls_Dialed', 'Converted'], barmode='group', title="Dialed vs Connected"), use_container_width=True)
    col2.plotly_chart(px.histogram(data, x='Call_Time_Mins', title="Talk Time Histogram"), use_container_width=True)
    col3.plotly_chart(px.funnel(data, x=['Calls_Dialed', 'Converted', 'Deals_Closed'], y='Sales_Rep_Name', title="Conversion Funnel"), use_container_width=True)
    
    col4, col5, col6 = st.columns(3)
    col4.plotly_chart(px.pie(data, names='Sales_Rep_Name', values='Total_Revenue', title="Revenue Share"), use_container_width=True)
    col5.plotly_chart(px.bar(data, x='Sales_Rep_Name', y='Deals_Closed', title="Deals Closed"), use_container_width=True)
    col6.plotly_chart(px.line(data, x='Sales_Rep_Name', y='Total_Revenue', title="Revenue Trend"), use_container_width=True)

# 2. DESCRIPTIVE PAGE
with tabs[1]:
    st.header("Descriptive Analytics")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(data, x='Sales_Rep_Name', y='Call_Time_Mins', title="Talk Time per Rep"), use_container_width=True)
    c2.plotly_chart(px.histogram(data, x='Call_Time_Mins', title="Talk Time Distribution"), use_container_width=True)
    st.dataframe(data.describe())

# 3. DIAGNOSTIC PAGE
with tabs[2]:
    st.header("Diagnostic Analysis")
    c1, c2 = st.columns(2)
    # Pareto
    pareto = data.sort_values('Total_Revenue', ascending=False)
    c1.plotly_chart(px.bar(pareto, x='Sales_Rep_Name', y='Total_Revenue', title="Pareto: Revenue Contribution"), use_container_width=True)
    c2.plotly_chart(px.box(data, y='Call_Time_Mins', title="Talk Time Outliers"), use_container_width=True)
    st.plotly_chart(px.imshow(data.select_dtypes(include=np.number).corr(), title="Performance Correlation Heatmap"), use_container_width=True)

# 4. PERSPECTIVE PAGE
with tabs[3]:
    st.header("Perspective & Benchmarks")
    # Benchmark
    avg_rev = data['Total_Revenue'].mean()
    fig = px.bar(data, x='Sales_Rep_Name', y='Total_Revenue', title="Rep vs Team Average")
    fig.add_hline(y=avg_rev, line_dash="dash", annotation_text="Team Avg")
    st.plotly_chart(fig, use_container_width=True)
    
    # Waterfall
    fig_w = go.Figure(go.Waterfall(x=["New", "Qualified", "Converted", "Revenue"], y=[100, 80, 40, data['Total_Revenue'].sum()]))
    st.plotly_chart(fig_w, use_container_width=True)

# 5. PREDICTIVE PAGE
with tabs[4]:
    st.header("Predictive & Prescriptive")
    # Regression
    X = data[['Call_Time_Mins', 'Calls_Dialed']]
    y = data['Total_Revenue']
    model = LinearRegression().fit(X, y)
    st.plotly_chart(px.scatter(data, x='Call_Time_Mins', y='Total_Revenue', trendline="ols", title="Revenue Prediction Model"), use_container_width=True)
    
    # Decision Tree
    dt = DecisionTreeRegressor().fit(X, y)
    st.write("### Key Drivers of Revenue")
    st.bar_chart(pd.DataFrame({'Importance': dt.feature_importances_}, index=X.columns))
    
    # Scenario
    inc = st.slider("What-if: Increase Calls by %", 0, 100, 10)
    st.metric("Projected Revenue", f"₹{data['Total_Revenue'].sum() * (1 + inc/100):,.0f}")
