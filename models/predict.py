# Here the cleaned and structured data is now used to extract the necessary features that are needed for the model to make a prediction on the risk of default. The features are extracted and then passed through the model to get a probability of default which is then converted to a risk score and interpreted for better understanding.

import joblib
import numpy as np

# Load pipeline
pipeline = joblib.load("risk_pipeline.pkl")


def prepare_features(features_dict):
    """
    Convert dictionary → model input format
    
    WHY:
    Models expect ordered arrays, not dictionaries
    """

    feature_order = [
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
        "income_diversity"
    ]

    return np.array([[features_dict[f] for f in feature_order]])


def predict_risk(features_dict):
    """
    Predict probability of default
    """

    X = prepare_features(features_dict)

    probability = pipeline.predict_proba(X)[0][1]

    return float(probability)


def risk_score_from_probability(prob):
    """
    Convert probability → score (0–100)

    LOWER probability = BETTER score
    """

    score = (1 - prob) * 100

    return round(score, 2)


def interpret_risk(prob):
    """
    Human-readable risk interpretation
    """

    if prob < 0.2:
        return "Low Risk"
    elif prob < 0.5:
        return "Moderate Risk"
    else:
        return "High Risk"