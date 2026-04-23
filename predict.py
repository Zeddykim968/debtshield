import os
import joblib
import numpy as np

from models.hybrid_score import (
    compute_hybrid_risk,
    hybrid_risk_score,
    interpret_hybrid_risk,
)
from risk_simulation.monte_carlo import MonteCarloRiskSimulator


FEATURE_ORDER = [
    "total_income",
    "total_expenses",
    "income_mean",
    "expense_mean",
    "income_instability",
    "expense_volatility",
    "savings_ratio",
    "coverage_ratio",
    "transaction_freq",
    "days_without_income",
    "balance_trend",
    "largest_expense",
    "largest_expense_ratio",
    "income_diversity",
]


def load_model(path="risk_pipeline.pkl"):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def features_to_vector(features):
    return np.array([[features.get(f, 0) for f in FEATURE_ORDER]])


def _heuristic_probability(features):
    sav = features.get("savings_ratio", 0)
    vol = features.get("expense_volatility", 0)
    days_no_inc = features.get("days_without_income", 0)
    cov = features.get("coverage_ratio", 0)
    largest_ratio = features.get("largest_expense_ratio", 0)

    score = 0.0
    score += 0.30 * (1 - max(0, min(sav, 1)))
    score += 0.20 * min(vol, 1.5) / 1.5
    score += 0.20 * min(days_no_inc, 30) / 30
    score += 0.15 * (1 - min(cov, 2) / 2)
    score += 0.15 * min(largest_ratio, 1)
    return float(max(0.0, min(1.0, score)))


def ml_probability_from_features(model, features):
    if model is not None:
        try:
            X = features_to_vector(features)
            return float(model.predict_proba(X)[0][1])
        except Exception:
            pass
    return _heuristic_probability(features)


def generate_risk_report_from_features(features, model=None):
    income = features.get("income_mean", 0) or features.get("total_income", 0)
    expenses = features.get("expense_mean", 0) or features.get("total_expenses", 0)
    debt = features.get("largest_expense", 0)

    ml_prob = ml_probability_from_features(model, features)

    sim = MonteCarloRiskSimulator()
    sim_result = sim.run(income, expenses, debt, ml_prob)
    sim_prob = sim_result["risk_probability"]

    hybrid_prob = compute_hybrid_risk(ml_prob, sim_prob)

    return {
        "ml_probability": ml_prob,
        "simulation_probability": sim_prob,
        "hybrid_probability": hybrid_prob,
        "hybrid_score": hybrid_risk_score(hybrid_prob),
        "risk_level": interpret_hybrid_risk(hybrid_prob),
        "mean_financial_outcome": sim_result["mean_financial_outcome"],
    }


def generate_risk_report(model, X, income, expenses, debt):
    if income <= 0:
        ml_prob = 1.0
    else:
        debt_to_income = debt / max(income * 12, 1)
        expense_ratio = expenses / income
        ml_prob = float(max(0.0, min(1.0,
            0.5 * min(expense_ratio, 1.5) / 1.5 + 0.5 * min(debt_to_income, 2.0) / 2.0
        )))
    if model is not None:
        try:
            ml_prob = float(model.predict_proba(X)[0][1])
        except Exception:
            pass

    sim = MonteCarloRiskSimulator()
    sim_result = sim.run(income, expenses, debt, ml_prob)
    sim_prob = sim_result["risk_probability"]
    hybrid_prob = compute_hybrid_risk(ml_prob, sim_prob)

    return {
        "ml_probability": ml_prob,
        "simulation_probability": sim_prob,
        "hybrid_probability": hybrid_prob,
        "hybrid_score": hybrid_risk_score(hybrid_prob),
        "risk_level": interpret_hybrid_risk(hybrid_prob),
        "mean_financial_outcome": sim_result["mean_financial_outcome"],
    }
