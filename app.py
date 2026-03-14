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
st.sidebar.header("Hierarchy Drill-Down")
region = st.sidebar.selectbox("Region", ["All"] + list(df['Region'].unique()))
data = df if region == "All" else df[df['Region'] == region]

auh = st.sidebar.selectbox("Area Head", ["All"] + list(data['AUH_Name'].unique()))
if auh != "All": data = data[data['AUH_Name'] == auh]

sm = st.sidebar.selectbox("Senior Manager", ["All"] + list(data['Senior_Manager_Name'].unique()))
if sm != "All": data = data[data['Senior_Manager_Name'] == sm]

mgr = st.sidebar.selectbox("Sales Manager", ["All"] + list(data['Sales_Manager_Name'].unique()))
if mgr != "All": data = data[data['Sales_Manager_Name'] == mgr]

# --- HOME PAGE: SUMMARY ---
st.title("📊 India Sales Performance Dashboard")
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
    
    # Pareto Chart (Revenue by Rep)
    rep_rev = data.groupby('Sales_Rep_Name')['Total_Revenue'].sum().sort_values(ascending=False).reset_index()
    fig_pareto = px.bar(rep_rev.head(10), x='Sales_Rep_Name', y='Total_Revenue', title="Pareto: Top 10 Revenue Drivers")
    col1.plotly_chart(fig_pareto, use_container_width=True)
    
    # Box Plot (Talk Time)
    fig_box = px.box(data, y="Call_Time_Mins", title="Talk Time Outlier Detection")
    col2.plotly_chart(fig_box, use_container_width=True)
    
    # Funnel
    fig_funnel = px.funnel(data.melt(value_vars=['Calls_Dialed', 'Converted', 'Deals_Closed']), x='value', y='variable', title="Conversion Funnel")
    col1.plotly_chart(fig_funnel, use_container_width=True)
    
    # Heatmap
    fig_heat = px.imshow(data[['Calls_Dialed', 'Call_Time_Mins', 'Deals_Closed', 'Total_Revenue']].corr(), text_auto=True, title="Correlation Heatmap")
    col2.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    st.header("Perspective & Benchmarks")
    # Radar Chart
    rep = st.selectbox("Select Rep for Radar", data['Sales_Rep_Name'].unique())
    r_data = data[data['Sales_Rep_Name'] == rep].iloc[0]
    fig_radar = px.line_polar(r=[r_data['Calls_Dialed'], r_data['Call_Time_Mins'], r_data['Deals_Closed']], 
                              theta=['Calls', 'TalkTime', 'Closures'], line_close=True)
    st.plotly_chart(fig_radar)
    
    # Waterfall
    fig_water = go.Figure(go.Waterfall(x=["New", "Qualified", "Converted", "Revenue"], y=[100, 80, 40, data['Total_Revenue'].sum()]))
    st.plotly_chart(fig_water)

with tab3:
    st.header("Predictive & Prescriptive")
    # Regression
    fig_reg = px.scatter(data, x='Call_Time_Mins', y='Total_Revenue', trendline="ols", title="Revenue vs Talk Time Regression")
    st.plotly_chart(fig_reg)
    
    # Decision Tree Importance
    X = data[['Calls_Dialed', 'Call_Time_Mins', 'Converted']]
    y = data['Total_Revenue']
    model = DecisionTreeRegressor().fit(X, y)
    imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    st.bar_chart(imp.set_index('Feature'))
    
    # Scenario Simulation
    st.subheader("What-If Analysis")
    inc = st.slider("Increase Dialed Calls by %", 0, 50, 10)
    st.write(f"Projected Revenue Increase: ₹{(data['Total_Revenue'].sum() * (inc/100)):,.0f}")

# --- ADVANCED FILTERS ---
st.sidebar.header("Dashboard Filters")
region_list = ['All'] + list(df['Region'].unique())
selected_region = st.sidebar.selectbox("Select Region", region_list)

# Filter logic
if selected_region == 'All':
    data = df
else:
    data = df[df['Region'] == selected_region]

# Hierarchy Drill-Down
# Add key="auh_unique_key" to the end of the line
auh = st.sidebar.selectbox("Area Head", ['All'] + list(data['AUH_Name'].unique()), key="auh_unique_key")
if auh != 'All': data = data[data['AUH_Name'] == auh]

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Performance Overview", "Diagnostic & Radar", "Predictive & Prescriptive"])

with tab1:
    st.header("Revenue Waterfall & Pareto")
    # Waterfall Chart: Revenue Steps
    fig_waterfall = go.Figure(go.Waterfall(
        name="Revenue", orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["New Leads", "Qualified", "Converted", "Total Revenue"],
        y=[0, 5000, 10000, data['Total_Revenue'].sum()],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig_waterfall.update_layout(template="plotly_dark", title="Revenue Contribution Waterfall")
    st.plotly_chart(fig_waterfall, use_container_width=True)

with tab2:
    st.header("Diagnostic: Radar & Heatmap")
    # Radar Chart for a selected Rep
    rep_name = st.selectbox("Select Rep for Radar Analysis", data['Sales_Rep_Name'].unique())
    rep_data = data[data['Sales_Rep_Name'] == rep_name].iloc[0]
    
    categories = ['Calls_Dialed', 'Call_Time_Mins', 'Deals_Closed', 'Converted']
    fig_radar = px.line_polar(r=[rep_data[c] for c in categories], theta=categories, line_close=True)
    fig_radar.update_traces(fill='toself')
    fig_radar.update_layout(template="plotly_dark", title=f"Performance Radar: {rep_name}")
    st.plotly_chart(fig_radar)

with tab3:
    st.header("Predictive: Decision Tree Importance")
    # Decision Tree Feature Importance
    X = data[['Calls_Dialed', 'Call_Time_Mins', 'Converted', 'Followup_Leads']]
    y = data['Total_Revenue']
    model = DecisionTreeRegressor().fit(X, y)
    
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    fig_tree = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Revenue Drivers (Decision Tree)")
    fig_tree.update_layout(template="plotly_dark")
    st.plotly_chart(fig_tree)
    
    st.write("### Prescriptive Insight")
    top_feature = importance_df.sort_values(by='Importance', ascending=False).iloc[0]['Feature']
    st.success(f"Prescription: The most significant driver of revenue is **{top_feature}**. Focus training efforts here.")
