from models.hybrid_score import (
    compute_hybrid_risk,
    hybrid_risk_score,
    interpret_hybrid_risk,
)
from risk_simulation.monte_carlo import MonteCarloRiskSimulator


def _ml_probability(model, X, income, expenses, debt):
    if model is not None:
        try:
            return float(model.predict_proba(X)[0][1])
        except Exception:
            pass
    # Heuristic fallback when no trained model is available
    if income <= 0:
        return 1.0
    debt_to_income = debt / max(income * 12, 1)
    expense_ratio = expenses / income
    prob = 0.5 * min(expense_ratio, 1.5) / 1.5 + 0.5 * min(debt_to_income, 2.0) / 2.0
    return float(max(0.0, min(1.0, prob)))


def generate_risk_report(model, X, income, expenses, debt):
    ml_prob = _ml_probability(model, X, income, expenses, debt)

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
