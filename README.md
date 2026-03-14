# 📊 Sales Intelligence & Performance Dashboard

An enterprise-grade sales analytics platform designed to provide deep insights into sales performance across a multi-tier organizational hierarchy (Area Head -> Senior Manager -> Sales Manager -> Sales Rep).

## 🚀 Overview
This dashboard provides a 360-degree view of sales operations in India, covering four key analytical pillars:
- **Descriptive:** Visualizing lead funnels, regional performance, and KPI tracking.
- **Diagnostic:** Correlation heatmaps to identify drivers of revenue and deal closure.
- **Predictive:** Linear regression models to estimate revenue based on talk-time and lead utilization.
- **Prescriptive:** Automated recommendations for sales training and personalized offer strategies.

## 🛠️ Technical Stack
- **Framework:** [Streamlit](https://streamlit.io/)
- **Data Processing:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (for predictive revenue modeling)
- **Visualization:** `plotly` (interactive charts)
- **Styling:** Custom `.toml` configuration for a professional, mild-yellow/white UI.

## 📂 Project Structure
```text
├── .streamlit/
│   └── config.toml        # UI Theme configuration
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── sales_performance_data.csv # Dataset (1000+ records)
└── README.md              # Project documentation
