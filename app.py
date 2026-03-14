import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Sales Intelligence Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("sales_performance_data.csv")
    # Ensure we have a date column for the line chart
    df['Date'] = pd.to_datetime('2023-01-01') # Placeholder
    return df

df = load_data()

# --- DRILL DOWN FILTERS ---
st.sidebar.header("Hierarchy Drill-Down")
region = st.sidebar.selectbox("Region", df['Region'].unique())
auh = st.sidebar.selectbox("Area Head", df[df['Region'] == region]['AUH_Name'].unique())
sm = st.sidebar.selectbox("Senior Manager", df[df['AUH_Name'] == auh]['Senior_Manager_Name'].unique())
mgr = st.sidebar.selectbox("Sales Manager", df[df['Senior_Manager_Name'] == sm]['Sales_Manager_Name'].unique())

filtered_df = df[df['Sales_Manager_Name'] == mgr]

st.title("🚀 Sales Performance Matrix")

# --- CHARTS ---
col1, col2 = st.columns(2)

with col1:
    # 1. Bar Chart: Talk Time vs Connected Calls
    fig1 = px.bar(filtered_df, x='Sales_Rep_Name', y='Call_Time_Mins', color='Converted', 
                  title="Talk Time vs Connected Calls", template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # 2. Stacked Column: Dialed vs Connected
    fig2 = px.bar(filtered_df, x='Sales_Rep_Name', y=['Calls_Dialed', 'Converted'], 
                  title="Efficiency: Dialed vs Connected", barmode='stack', template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # 3. Bubble Chart: Talk Time vs Deal Closures
    fig3 = px.scatter(filtered_df, x='Call_Time_Mins', y='Deals_Closed', size='Total_Revenue', 
                      color='Total_Revenue', title="Talk Time vs Deal Closures (Bubble=Revenue)", template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # 4. Funnel Chart
    fig4 = px.funnel(filtered_df.melt(value_vars=['Calls_Dialed', 'Converted', 'Deals_Closed']), 
                     x='value', y='variable', title="Conversion Funnel", template="plotly_dark")
    st.plotly_chart(fig4, use_container_width=True)

# 5. Heatmap
st.subheader("Performance Matrix Heatmap")
matrix_cols = ['Call_Time_Mins', 'Calls_Dialed', 'Deals_Closed', 'Total_Revenue']
corr = filtered_df[matrix_cols].corr()
fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='YlOrBr', template="plotly_dark")
st.plotly_chart(fig_heat, use_container_width=True)

# 6. Pie Chart
fig_pie = px.pie(filtered_df, values='Total_Revenue', names='Sales_Rep_Name', title="Revenue Contribution by Rep", template="plotly_dark")
st.plotly_chart(fig_pie, use_container_width=True)
