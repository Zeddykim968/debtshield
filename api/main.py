import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from ..predict import generate_risk_report
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="DebtShield AI", layout="wide")

# Title UI
st.title("💰 DebtShield 2.0 – AI Financial Risk Engine")
st.markdown("ML + Monte Carlo + Hybrid Risk Intelligence System")

# -------------------------
# INPUT SECTION
# -------------------------
st.sidebar.header("📊 Enter Financial Data")

income = st.sidebar.number_input("Monthly Income", value=50000)
expenses = st.sidebar.number_input("Monthly Expenses", value=30000)
debt = st.sidebar.number_input("Total Debt", value=100000)

# Simulated feature vector for ML model (replace with real engineered features)
X = [[income, expenses, debt]]

# -------------------------
# RUN ANALYSIS
# -------------------------
if st.button("Run Risk Analysis 🚀"):

    # Fake model placeholder (replace with trained model)
    model = st.session_state.get("model", None)

    # Run full pipeline
    result = generate_risk_report(model, X, income, expenses, debt)

    # -------------------------
    # METRICS DISPLAY
    # -------------------------
    st.subheader("📌 Financial Risk Summary")

    st.metric("ML Risk Probability", round(result["ml_probability"], 3))
    st.metric("Simulation Risk", round(result["simulation_probability"], 3))
    st.metric("Hybrid Score", result["hybrid_score"])
    st.metric("Risk Level", result["risk_level"])

    # -------------------------
    # PIE CHART (RISK SPLIT)
    # -------------------------
    fig = px.pie(
        values=[
            result["simulation_probability"],
            1 - result["simulation_probability"]
        ],
        names=["Risk", "Stable"],
        title="Financial Stability Distribution"
    )

    st.plotly_chart(fig)

    # -------------------------
    # BAR CHART (SIMPLE VIEW)
    # -------------------------
    fig2, ax = plt.subplots()

    ax.bar(
        ["ML Risk", "Simulation Risk"],
        [result["ml_probability"], result["simulation_probability"]]
    )

    ax.set_title("Risk Component Comparison")

    st.pyplot(fig2)

    # -------------------------
    # FINAL INTERPRETATION
    # -------------------------
    st.subheader("🧠 Insight")

    if result["hybrid_score"] > 70:
        st.success("Strong Financial Health 🟢")
    elif result["hybrid_score"] > 40:
        st.warning("Moderate Financial Risk 🟡")
    else:
        st.error("High Financial Risk 🔴")