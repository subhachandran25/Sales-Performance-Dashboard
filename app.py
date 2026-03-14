import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor

# Page Configuration
st.set_page_config(layout="wide", page_title="India Sales Intelligence")

# Load Data
@st.cache_data
def load_data():
    # Ensure your CSV file is in the same directory
    return pd.read_csv("sales_performance_data.csv")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")

# Region Filter
region_list = ['All'] + list(df['Region'].unique())
selected_region = st.sidebar.selectbox("Select Region", region_list, key="reg_filter")

# Apply Region Filter
data = df.copy()
if selected_region != 'All':
    data = data[data['Region'] == selected_region]

# Area Head Filter
auh_list = ['All'] + list(data['AUH_Name'].unique())
selected_auh = st.sidebar.selectbox("Area Head", auh_list, key="auh_filter")
if selected_auh != 'All':
    data = data[data['AUH_Name'] == selected_auh]

# --- MAIN DASHBOARD ---
st.title("📊 India Sales Intelligence Dashboard")

# Summary Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"₹{data['Total_Revenue'].sum():,.0f}")
c2.metric("Total Deals", data['Deals_Closed'].sum())
c3.metric("Avg Talk Time", f"{data['Call_Time_Mins'].mean():.1f}m")
c4.metric("Conversion Rate", f"{(data['Converted'].sum()/data['New_Leads'].sum())*100:.1f}%")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Performance Overview", "Diagnostic & Radar", "Predictive & Prescriptive"])

with tab1:
    st.header("Revenue Waterfall & Pareto Analysis")
    col1, col2 = st.columns(2)
    
    # Waterfall Chart
    fig_waterfall = go.Figure(go.Waterfall(
        name="Revenue", orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["New Leads", "Qualified", "Converted", "Total Revenue"],
        y=[0, data['Qualified'].sum(), data['Converted'].sum(), data['Total_Revenue'].sum()],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig_waterfall.update_layout(title="Revenue Contribution Waterfall")
    col1.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Pareto Chart
    rep_rev = data.groupby('Sales_Rep_Name')['Total_Revenue'].sum().sort_values(ascending=False).reset_index()
    fig_pareto = px.bar(rep_rev.head(10), x='Sales_Rep_Name', y='Total_Revenue', title="Top 10 Revenue Drivers")
    col2.plotly_chart(fig_pareto, use_container_width=True)

with tab2:
    st.header("Diagnostic: Radar & Lead Utilization")
    col1, col2 = st.columns(2)
    
    # Radar Chart
    rep_name = st.selectbox("Select Rep for Radar Analysis", data['Sales_Rep_Name'].unique(), key="radar_rep")
    rep_data = data[data['Sales_Rep_Name'] == rep_name].iloc[0]
    categories = ['Calls_Dialed', 'Call_Time_Mins', 'Deals_Closed', 'Converted']
    fig_radar = px.line_polar(r=[rep_data[c] for c in categories], theta=categories, line_close=True)
    fig_radar.update_traces(fill='toself')
    col1.plotly_chart(fig_radar)
    
    # Lead Utilization
    fig_lead = px.bar(data.groupby('Sales_Manager')['Qualified'].sum().reset_index(), 
                     x='Sales_Manager', y='Qualified', title="Leads Qualified per Manager")
    col2.plotly_chart(fig_lead, use_container_width=True)

with tab3:
    st.header("Predictive & Prescriptive Insights")
    
    # Decision Tree Feature Importance
    X = data[['Calls_Dialed', 'Call_Time_Mins', 'Converted', 'Followup_Leads']]
    y = data['Total_Revenue']
    model = DecisionTreeRegressor().fit(X, y)
    
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    fig_tree = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Revenue Drivers (Decision Tree)")
    st.plotly_chart(fig_tree, use_container_width=True)
    
    # Prescriptive Insight
    top_feature = importance_df.sort_values(by='Importance', ascending=False).iloc[0]['Feature']
    st.success(f"Prescription: The most significant driver of revenue is **{top_feature}**. Focus training efforts here.")
    
    # What-If Analysis
    st.subheader("What-If Analysis")
    inc = st.slider("Increase Dialed Calls by %", 0, 50, 10, key="what_if_slider")
    st.write(f"Projected Revenue Increase: ₹{(data['Total_Revenue'].sum() * (inc/100)):,.0f}")
