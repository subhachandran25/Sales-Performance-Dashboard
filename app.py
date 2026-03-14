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
auh = st.sidebar.selectbox("Area Head", ['All'] + list(data['AUH_Name'].unique()))
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
