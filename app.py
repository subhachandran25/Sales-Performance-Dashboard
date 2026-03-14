import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(layout="wide", page_title="India Sales Intelligence")

@st.cache_data
def load_data():
    return pd.read_csv("sales_performance_data.csv")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")
region_list = ['All'] + list(df['Region'].unique())
selected_region = st.sidebar.selectbox("Select Region", region_list, key="reg_filter")

# Filter logic
data = df.copy()
if selected_region != 'All':
    data = data[data['Region'] == selected_region]

auh_list = ['All'] + list(data['AUH_Name'].unique())
selected_auh = st.sidebar.selectbox("Area Head", auh_list, key="auh_filter")
if selected_auh != 'All':
    data = data[data['AUH_Name'] == selected_auh]

# --- MAIN DASHBOARD ---
st.title("📊 Sales Diagnostic & Predictive Dashboard")

# Summary Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"₹{data['Total_Revenue'].sum():,.0f}")
c2.metric("Total Deals", data['Deals_Closed'].sum())
c3.metric("Avg Talk Time", f"{data['Call_Time_Mins'].mean():.1f}m")
c4.metric("Conversion Rate", f"{(data['Converted'].sum()/data['New_Leads'].sum())*100:.1f}%")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Descriptive & Diagnostic", "Perspective & Benchmarks", "Predictive & Prescriptive"])

with tab1:
    st.header("Descriptive & Diagnostic Analysis")
    col1, col2 = st.columns(2)
    
    # Pareto Chart
    rep_rev = data.groupby('Sales_Rep_Name')['Total_Revenue'].sum().sort_values(ascending=False).reset_index()
    col1.plotly_chart(px.bar(rep_rev.head(10), x='Sales_Rep_Name', y='Total_Revenue', title="Pareto: Top 10 Revenue Drivers"), use_container_width=True)
    
    # Box Plot
    col2.plotly_chart(px.box(data, y="Call_Time_Mins", title="Talk Time Outlier Detection"), use_container_width=True)
    
    # Funnel
    col1.plotly_chart(px.funnel(data.melt(value_vars=['Calls_Dialed', 'Converted', 'Deals_Closed']), x='value', y='variable', title="Conversion Funnel"), use_container_width=True)
    
    # Heatmap
    col2.plotly_chart(px.imshow(data[['Calls_Dialed', 'Call_Time_Mins', 'Deals_Closed', 'Total_Revenue']].corr(), text_auto=True, title="Correlation Heatmap"), use_container_width=True)

with tab2:
    st.header("Perspective & Benchmarks")
    
    # Radar Chart
    rep = st.selectbox("Select Rep for Radar", data['Sales_Rep_Name'].unique(), key="radar_persp")
    r_data = data[data['Sales_Rep_Name'] == rep].iloc[0]
    st.plotly_chart(px.line_polar(r=[r_data['Calls_Dialed'], r_data['Call_Time_Mins'], r_data['Deals_Closed']], theta=['Calls', 'TalkTime', 'Closures'], line_close=True))
    
    # Waterfall
    fig_water = go.Figure(go.Waterfall(x=["New", "Qualified", "Converted", "Revenue"], y=[100, 80, 40, data['Total_Revenue'].sum()]))
    st.plotly_chart(fig_water)

with tab3:
    st.header("Predictive & Prescriptive")
    
    # Regression
    st.plotly_chart(px.scatter(data, x='Call_Time_Mins', y='Total_Revenue', trendline="ols", title="Revenue vs Talk Time Regression"))
    
    # Decision Tree
    X = data[['Calls_Dialed', 'Call_Time_Mins', 'Converted']]
    y = data['Total_Revenue']
    model = DecisionTreeRegressor().fit(X, y)
    imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    st.bar_chart(imp.set_index('Feature'))
    
    # Scenario
    inc = st.slider("Increase Dialed Calls by %", 0, 50, 10)
    st.write(f"Projected Revenue Increase: ₹{(data['Total_Revenue'].sum() * (inc/100)):,.0f}")

    # Lead Utilization
    fig_lead = px.bar(data.groupby('Sales_Manager_Name')['Qualified'].sum().reset_index(), x='Sales_Manager_Name', y='Qualified', title="Leads Qualified per Manager")
    st.plotly_chart(fig_lead)
