import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="India Sales Intelligence")

@st.cache_data
def load_data():
    df = pd.read_csv("sales_performance_data.csv")
    return df

df = load_data()

# --- DRILL DOWN FILTERS ---
st.sidebar.header("Hierarchy Filters")
region = st.sidebar.selectbox("Region", df['Region'].unique())
auh = st.sidebar.selectbox("Area Head", df[df['Region'] == region]['AUH_Name'].unique())
sm = st.sidebar.selectbox("Senior Manager", df[df['AUH_Name'] == auh]['Senior_Manager_Name'].unique())
mgr = st.sidebar.selectbox("Sales Manager", df[df['Senior_Manager_Name'] == sm]['Sales_Manager_Name'].unique())

filtered_df = df[df['Sales_Manager_Name'] == mgr]

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])

with tab1: # Descriptive
    st.header("Descriptive Analytics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"₹{filtered_df['Total_Revenue'].sum():,.0f}")
    col2.metric("Avg Talk Time", f"{filtered_df['Call_Time_Mins'].mean():.1f}m")
    col3.metric("Lead Utilization", f"{(filtered_df['Converted'].sum()/filtered_df['New_Leads'].sum())*100:.1f}%")
    
    fig_pie = px.pie(filtered_df, values='Total_Revenue', names='Sales_Rep_Name', title="Revenue Contribution by Rep")
    st.plotly_chart(fig_pie, use_container_width=True)

with tab2: # Diagnostic
    st.header("Diagnostic Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_heat = px.imshow(df[['Calls_Dialed', 'Call_Time_Mins', 'Deals_Closed', 'Total_Revenue']].corr(), text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig_heat)
    with col_b:
        fig_box = px.box(filtered_df, y="Call_Time_Mins", title="Talk Time Distribution (Outlier Detection)")
        st.plotly_chart(fig_box)

with tab3: # Predictive
    st.header("Predictive Analysis")
    X = df[['Call_Time_Mins', 'Converted', 'Calls_Dialed']]
    y = df['Total_Revenue']
    model = LinearRegression().fit(X, y)
    
    st.write("### Revenue Forecast Simulation")
    sim_calls = st.slider("Simulate Dialed Calls", 100, 1000, 500)
    pred = model.predict([[df['Call_Time_Mins'].mean(), df['Converted'].mean(), sim_calls]])
    st.success(f"Projected Revenue: ₹{pred[0]:,.2f}")

with tab4: # Prescriptive
    st.header("Prescriptive Actions")
    if filtered_df['Call_Time_Mins'].mean() < 300:
        st.warning("Prescription: Increase talk time via 'Value-Based Selling' training.")
    else:
        st.success("Prescription: Focus on lead conversion rate optimization.")
    
    # Personalized Offer Logic
    st.write("### Personalized Offers")
    high_potential = filtered_df[filtered_df['Converted'] > filtered_df['Converted'].mean()]
    st.table(high_potential[['Sales_Rep_Name', 'Total_Revenue']].head(5))
