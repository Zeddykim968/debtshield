import sys
import os
import io
import tempfile
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ingestion.mpesa_parser import extract_mpesa_data, load_mpesa_csv
from processing.cleaning import clean_mpesa_data, classify_the_data
from processing.feature_enineering import generate_features
from predict import (
    generate_risk_report_from_features,
    generate_risk_report,
    load_model,
)


st.set_page_config(page_title="DebtShield AI", layout="wide", page_icon="💰")

st.title("💰 DebtShield 2.0 – AI Financial Risk Engine")
st.caption("Upload an M-Pesa statement (PDF or CSV) to extract behaviour features and run hybrid ML + Monte Carlo risk analysis.")

model = load_model()

# ---------------------------------------------------------------
# SIDEBAR: data source
# ---------------------------------------------------------------
st.sidebar.header("📥 Data Source")
mode = st.sidebar.radio(
    "Choose input mode",
    ["Upload M-Pesa Statement", "Manual Entry"],
)

df_clean = None
features = None

if mode == "Upload M-Pesa Statement":
    uploaded = st.sidebar.file_uploader(
        "Upload statement (PDF or CSV)",
        type=["pdf", "csv"],
        help="M-Pesa statement export. PDFs are parsed with pdfplumber; CSVs need columns date, details, amount, balance.",
    )
    pdf_password = st.sidebar.text_input(
        "PDF password (if any)", type="password",
        help="Safaricom often emails password-protected PDFs.",
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                raw = extract_mpesa_data(tmp_path, password=pdf_password or None)
                os.unlink(tmp_path)
            else:
                raw = load_mpesa_csv(uploaded)

            if raw is None or raw.empty:
                st.error("No tabular data could be extracted from this file. If your PDF is password-protected, enter the password in the sidebar.")
            else:
                with st.expander("Raw parsed rows (debug)"):
                    st.dataframe(raw.head(20), use_container_width=True)
                df_clean = clean_mpesa_data(raw)
                if df_clean.empty:
                    st.error("File parsed, but no valid transaction rows were found after cleaning. Check the column layout.")
                else:
                    df_clean = classify_the_data(df_clean)
                    features = generate_features(df_clean)
                    st.sidebar.success(f"Parsed {len(df_clean)} transactions")
        except Exception as e:
            st.error(f"Failed to parse statement: {e}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    else:
        st.info("👈 Upload an M-Pesa PDF or CSV in the sidebar to begin, or switch to Manual Entry.")

else:
    st.sidebar.subheader("Enter financial summary")
    income = st.sidebar.number_input("Monthly Income", value=50000, step=1000)
    expenses = st.sidebar.number_input("Monthly Expenses", value=30000, step=1000)
    debt = st.sidebar.number_input("Total Debt", value=100000, step=1000)
    if st.sidebar.button("Run Manual Analysis 🚀"):
        result = generate_risk_report(model, [[income, expenses, debt]], income, expenses, debt)
        st.session_state["manual_result"] = result
        st.session_state["manual_inputs"] = (income, expenses, debt)


# ---------------------------------------------------------------
# RISK GAUGE HELPER
# ---------------------------------------------------------------
def risk_gauge(score, title="Hybrid Financial Health Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, 40], "color": "#f8d7da"},
                {"range": [40, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
    return fig


def render_results(result, features=None, df_clean=None):
    st.subheader("📌 Risk Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ML Risk Probability", f"{result['ml_probability']*100:.1f}%")
    c2.metric("Simulation Risk", f"{result['simulation_probability']*100:.1f}%")
    c3.metric("Hybrid Health Score", result["hybrid_score"])
    c4.metric("Risk Level", result["risk_level"])

    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(risk_gauge(result["hybrid_score"]), use_container_width=True)
    with g2:
        comp = pd.DataFrame({
            "Component": ["ML Model", "Monte Carlo", "Hybrid"],
            "Risk Probability": [
                result["ml_probability"],
                result["simulation_probability"],
                result["hybrid_probability"],
            ],
        })
        fig = px.bar(
            comp, x="Component", y="Risk Probability",
            color="Component", text=comp["Risk Probability"].apply(lambda v: f"{v*100:.1f}%"),
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Risk Component Comparison",
        )
        fig.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Behaviour breakdown only available with parsed statement
    if features is not None and df_clean is not None and not df_clean.empty:
        st.subheader("🔍 Spending Behaviour")

        b1, b2 = st.columns(2)
        with b1:
            cat_totals = (
                df_clean.groupby("category")["amount"].sum().abs().reset_index()
            )
            fig = px.pie(
                cat_totals, names="category", values="amount", hole=0.45,
                title="Transaction Volume by Category",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig, use_container_width=True)

        with b2:
            inflow = features["total_income"]
            outflow = features["total_expenses"]
            cf = pd.DataFrame({
                "Flow": ["Inflow", "Outflow", "Net"],
                "Amount": [inflow, outflow, inflow - outflow],
            })
            fig = px.bar(
                cf, x="Flow", y="Amount", color="Flow",
                text=cf["Amount"].apply(lambda v: f"{v:,.0f}"),
                title="Cashflow Snapshot",
                color_discrete_map={"Inflow": "#2ca02c", "Outflow": "#d62728", "Net": "#1f77b4"},
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Daily balance trend
        if df_clean["balance"].notna().any():
            ts = df_clean.sort_values("date").dropna(subset=["balance"])
            fig = px.area(
                ts, x="date", y="balance",
                title="Account Balance Over Time",
                color_discrete_sequence=["#1f77b4"],
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Inflow vs outflow over time
        ts2 = df_clean.copy()
        ts2["day"] = ts2["date"].dt.date
        flow_daily = (
            ts2.assign(signed=lambda d: np.where(d["direction"] == "inflow", d["amount"].abs(), -d["amount"].abs()))
               .groupby(["day", "direction"])["amount"].sum().reset_index()
        )
        if not flow_daily.empty:
            fig = px.bar(
                flow_daily, x="day", y="amount", color="direction", barmode="group",
                title="Daily Inflow vs Outflow",
                color_discrete_map={"inflow": "#2ca02c", "outflow": "#d62728", "neutral": "#999999"},
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧮 Extracted Behaviour Features")
        feat_df = pd.DataFrame(
            [(k, v) for k, v in features.items()],
            columns=["Feature", "Value"],
        )
        feat_df["Value"] = feat_df["Value"].apply(
            lambda v: f"{v:,.4f}" if isinstance(v, float) else f"{v:,}"
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        with st.expander("View parsed transactions"):
            st.dataframe(df_clean, use_container_width=True, hide_index=True)

    # Insight banner
    st.subheader("🧠 Insight")
    score = result["hybrid_score"]
    if score > 70:
        st.success(f"Strong financial health (score {score}). Risk of distress is low.")
    elif score > 40:
        st.warning(f"Moderate financial risk (score {score}). Watch volatility and savings ratio.")
    else:
        st.error(f"High financial risk (score {score}). Cashflow stress is likely without intervention.")


# ---------------------------------------------------------------
# MAIN ACTION
# ---------------------------------------------------------------
if mode == "Upload M-Pesa Statement" and features is not None:
    if st.button("Run Risk Analysis 🚀", type="primary"):
        result = generate_risk_report_from_features(features, model=model)
        render_results(result, features=features, df_clean=df_clean)

if mode == "Manual Entry" and "manual_result" in st.session_state:
    render_results(st.session_state["manual_result"])
