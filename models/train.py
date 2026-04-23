import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_model():
    """
    Train advanced risk model for DebtShield

    WHY THIS SETUP:
    - Gradient Boosting = strong for tabular financial data
    - Scaling = stabilizes feature distribution
    - Pipeline = production-safe
    """

    # --------------------------------------------------
    # 🔥 SAMPLE TRAINING DATA (REPLACE WITH REAL DATA)
    # --------------------------------------------------

    np.random.seed(42)

    data = pd.DataFrame({
        "total_income": np.random.randint(5000, 50000, 200),
        "total_expenses": np.random.randint(3000, 40000, 200),
        "income_mean": np.random.randint(1000, 10000, 200),
        "expense_mean": np.random.randint(1000, 10000, 200),
        "income_instability": np.random.rand(200),
        "expense_volatility": np.random.rand(200),
        "savings_ratio": np.random.uniform(-0.5, 0.7, 200),
        "coverage_ratio": np.random.uniform(0.5, 3, 200),
        "transaction_freq": np.random.randint(10, 200, 200),
        "days_without_income": np.random.randint(0, 30, 200),
        "balance_trend": np.random.randint(-10000, 10000, 200),
        "largest_expense": np.random.randint(500, 20000, 200),
        "largest_expense_ratio": np.random.rand(200),
        "income_diversity": np.random.randint(1, 10, 200),
    })

    # Simulated target (default risk)
    data["default"] = (
        (data["savings_ratio"] < 0.1) &
        (data["expense_volatility"] > 0.5) &
        (data["days_without_income"] > 10)
    ).astype(int)

    # --------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------

    X = data.drop("default", axis=1)
    y = data["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------------------------------
    # 🔥 PIPELINE
    # --------------------------------------------------

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    pipeline.fit(X_train, y_train)

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n📈 ROC-AUC Score:")
    print(roc_auc_score(y_test, y_prob))

    # --------------------------------------------------
    # CROSS VALIDATION (STABILITY CHECK)
    # --------------------------------------------------

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    print("\n🔁 CV ROC-AUC:", cv_scores.mean())

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    model = pipeline.named_steps["model"]

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    print("\n🔥 Feature Importance:")
    print(importance)

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    joblib.dump(pipeline, "risk_pipeline.pkl")

    print("\n✅ Model saved as risk_pipeline.pkl")


if __name__ == "__main__":
    train_model()