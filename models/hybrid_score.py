def compute_hybrid_risk(ml_prob, simulation_prob):
    """
    Combine ML + Monte Carlo risk estimates

    ML = learned behavior patterns
    Simulation = uncertainty + future risk

    Output = balanced financial risk probability
    """

    # Weighted fusion (baseline model)
    hybrid_prob = 0.6 * ml_prob + 0.4 * simulation_prob

    return hybrid_prob


def hybrid_risk_score(prob):
    """
    Convert probability (0–1) → score (0–100)
    Higher score = better financial health
    """

    return round((1 - prob) * 100, 2)


def interpret_hybrid_risk(prob):
    """
    Convert risk probability into human-readable category
    """

    if prob < 0.2:
        return "Low Risk 🟢"
    elif prob < 0.5:
        return "Moderate Risk 🟡"
    else:
        return "High Risk 🔴"