import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="Sales Intelligence Dashboard")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("sales_performance_data.csv")
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Hierarchy Drill-Down")
region = st.sidebar.selectbox("Region", df['Region'].unique())
auh = st.sidebar.selectbox("Area Head", df[df['Region'] == region]['AUH_Name'].unique())
sm = st.sidebar.selectbox("Senior Manager", df[df['AUH_Name'] == auh]['Senior_Manager_Name'].unique())
mgr = st.sidebar.selectbox("Sales Manager", df[df['Senior_Manager_Name'] == sm]['Sales_Manager_Name'].unique())

filtered_df = df[df['Sales_Manager_Name'] == mgr]

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])

with tab1:
    st.title("Descriptive Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${filtered_df['Total_Revenue'].sum():,.0f}")
    col2.metric("Avg Talk Time", f"{filtered_df['Call_Time_Mins'].mean():.1f} min")
    col3.metric("Lead Utilization", f"{(filtered_df['Converted'].sum()/filtered_df['New_Leads'].sum())*100:.1f}%")
    
    fig = px.funnel(filtered_df, x=['New_Leads', 'Qualified', 'Converted', 'Deals_Closed'], title="Lead Funnel")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.title("Diagnostic Analysis (Correlation)")
    corr = df[['Calls_Dialed', 'Call_Time_Mins', 'Total_Revenue', 'Converted']].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)
    st.write("Insight: High correlation between Call Time and Revenue suggests talk-time is a key driver.")

with tab3:
    st.title("Predictive Analysis")
    X = df[['Call_Time_Mins', 'Converted']]
    y = df['Total_Revenue']
    model = LinearRegression().fit(X, y)
    st.write("Model Prediction: Revenue is strongly dependent on Talk Time and Conversion.")
    # Simple prediction input
    talk_input = st.slider("Predict Revenue based on Talk Time", 100, 2000, 500)
    pred = model.predict([[talk_input, df['Converted'].mean()]])
    st.success(f"Estimated Revenue for {talk_input} mins: ${pred[0]:,.2f}")

with tab4:
    st.title("Prescriptive Recommendations")
    if filtered_df['Call_Time_Mins'].mean() < 500:
        st.warning("Action Required: Average talk time is low. Suggestion: Implement 'Value-Based Selling' training.")
    else:
        st.success("Talk time is optimal. Focus on lead quality.")
    
    st.write("Personalized Offer Strategy: Target high-followup leads with a 10% discount to close deals.")
