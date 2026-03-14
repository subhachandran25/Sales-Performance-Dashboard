import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("sales_performance_data.csv")

df = load_data()

# --- DEBUGGING: PRINT COLUMN NAMES ---
st.write("### Debug: Your CSV Column Names are:")
st.write(df.columns.tolist())
# -------------------------------------

# Sidebar Filters
region_list = ['All'] + list(df['Region'].unique())
selected_region = st.sidebar.selectbox("Select Region", region_list, key="reg_filter")

data = df.copy()
if selected_region != 'All':
    data = data[data['Region'] == selected_region]


# --- TABS NAVIGATION ---
tab_home, tab_desc, tab_diag, tab_persp, tab_pred = st.tabs([
    "🏠 Home", "📊 Descriptive", "🔍 Diagnostic", "🎯 Perspective", "🔮 Predictive"
])

# --- HOME PAGE ---
with tab_home:
    st.title("Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"₹{data['Total_Revenue'].sum():,.0f}")
    c2.metric("Deals", data['Deals_Closed'].sum())
    c3.metric("Avg Talk", f"{data['Call_Time_Mins'].mean():.1f}m")
    c4.metric("Conversion", f"{(data['Converted'].sum()/data['New_Leads'].sum())*100:.1f}%")
    
    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(px.bar(data, x='Sales_Rep_Name', y=['Calls_Dialed', 'Converted'], title="Dialed vs Connected"), use_container_width=True)
    col2.plotly_chart(px.histogram(data, x='Call_Time_Mins', title="Talk Time Histogram"), use_container_width=True)
    col3.plotly_chart(px.funnel(data, x='New_Leads', y='Deals_Closed', title="Conversion Funnel"), use_container_width=True)
    
    col4, col5, col6 = st.columns(3)
    col4.plotly_chart(px.pie(data, names='Region', values='Total_Revenue', title="Revenue Share"), use_container_width=True)
        # Pre-aggregate the data to ensure one row per Sales_Manager
   df_stacked = data.groupby(['Sales_Manager_Name', 'Region'])['Deals_Closed'].sum().reset_index()
    
    # Now use the aggregated dataframe for the chart
    fig_stacked = px.bar(df_stacked, x='Sales_Manager', y='Deals_Closed', 
                         color='Region', title="Deals Closed (Stacked)", barmode='stack')
    col5.plotly_chart(fig_stacked, use_container_width=True)
    col6.plotly_chart(px.line(data, x='Sales_Rep_Name', y='Total_Revenue', title="Revenue Trend"), use_container_width=True)

# --- DESCRIPTIVE PAGE ---
with tab_desc:
    st.header("Descriptive Analytics")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(data, x='Sales_Rep_Name', y='Call_Time_Mins', title="Talk Time per Rep"), use_container_width=True)
    c2.plotly_chart(px.bar(data, x='Sales_Rep_Name', y=['Calls_Dialed', 'Converted'], title="Dialed vs Connected"), use_container_width=True)
    st.dataframe(data.describe()) # KPI Table

# --- DIAGNOSTIC PAGE ---
with tab_diag:
    st.header("Diagnostic Analysis")
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(data.sort_values('Total_Revenue', ascending=False), x='Sales_Rep_Name', y='Total_Revenue', title="Pareto Analysis"), use_container_width=True)
    col2.plotly_chart(px.box(data, y='Call_Time_Mins', title="Box Plot: Talk Time"), use_container_width=True)
    st.plotly_chart(px.imshow(data.select_dtypes(include='number').corr(), title="Performance Heatmap"), use_container_width=True)

# --- PERSPECTIVE PAGE ---
with tab_persp:
    st.header("Perspective & Benchmarks")
    rep = st.selectbox("Select Rep", data['Sales_Rep_Name'].unique(), key="persp_rep")
    r_data = data[data['Sales_Rep_Name'] == rep].iloc[0]
    st.plotly_chart(px.line_polar(r=[r_data['Calls_Dialed'], r_data['Call_Time_Mins'], r_data['Deals_Closed']], theta=['Calls', 'TalkTime', 'Closures'], line_close=True), use_container_width=True)
    st.plotly_chart(px.area(data, x='Sales_Rep_Name', y='Total_Revenue', title="Stacked Area Revenue"), use_container_width=True)

# --- PREDICTIVE PAGE ---
with tab_pred:
    st.header("Predictive & Prescriptive")
    X = data[['Calls_Dialed', 'Call_Time_Mins', 'Converted']]
    y = data['Total_Revenue']
    model = DecisionTreeRegressor().fit(X, y)
    st.plotly_chart(px.scatter(data, x='Call_Time_Mins', y='Total_Revenue', trendline="ols", title="Regression Analysis"), use_container_width=True)
    st.bar_chart(pd.DataFrame({'Importance': model.feature_importances_}, index=X.columns))
    inc = st.slider("Scenario: Increase Calls by %", 0, 50, 10)
    st.metric("Projected Revenue", f"₹{data['Total_Revenue'].sum() * (1 + inc/100):,.0f}")
