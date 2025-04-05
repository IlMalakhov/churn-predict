import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")
    
    model = load_model()

st.set_page_config(page_title="Customer Churn Demo", layout="wide")

st.title("🔍 Customer Churn Predictor Demo")

st.header("🧾 Prediction Form")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Customer Age", 18, 80, 30)
    with col2:
        orders = st.number_input("Total Orders", min_value=0, max_value=100, value=10)
    with col3:
        last_order_days = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=45)

    submitted = st.form_submit_button("Predict")

    if submitted:
        st.success("✅ Prediction: Customer **left** the service.")
        st.info("Probability: 85.2% (mock data)")

if st.button("Send balloons to celebrate the best team ever!"):
    st.balloons()

st.markdown("---")
st.header("📊 Dashboard Analytics")

np.random.seed(42)
df = pd.DataFrame({
    "Region": np.random.choice(["North", "South", "East", "West"], size=100),
    "Orders": np.random.poisson(10, 100),
    "Churned": np.random.choice(["Yes", "No"], size=100, p=[0.3, 0.7]),
    "Revenue": np.random.normal(100, 20, 100),
})

embed_url = "https://app.powerbi.com/reportEmbed?reportId=5d5e6245-c983-496c-b458-c06d2f1113f3&autoAuth=true&ctid=dfe014b9-885d-4e4a-8eb4-597464b165c5"

components.iframe(embed_url, width=1400, height=800)

col1, col2 = st.columns(2)

with col1:
    churn_pie = df["Churned"].value_counts().reset_index()
    churn_pie.columns = ["Churned", "Count"]
    fig = px.pie(churn_pie, values="Count", names="Churned", title="Churn Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    revenue_hist = px.histogram(df, x="Revenue", nbins=20, title="Revenue Distribution")
    st.plotly_chart(revenue_hist, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    region_orders = df.groupby("Region")["Orders"].sum().reset_index()
    fig2 = px.bar(region_orders, x="Region", y="Orders", title="Total Orders by Region")
    st.plotly_chart(fig2, use_container_width=True)

with col4:
    region_rev = df.groupby("Region")["Revenue"].mean().reset_index()
    fig3 = px.box(df, x="Region", y="Revenue", title="Revenue per Region")
    st.plotly_chart(fig3, use_container_width=True)

st.caption("All data is randomly generated for demonstration.")
