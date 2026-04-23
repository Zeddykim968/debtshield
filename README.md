# DebtShield 🛡️

AI-powered financial risk assessment that combines **Machine Learning** with **Monte Carlo simulation** to predict default probability.

## What is DebtShield?

DebtShield predicts the likelihood of financial default by analyzing transaction data. It uses a hybrid approach:
- **ML Model** – Predicts default risk from 14 financial features
- **Monte Carlo Simulation** – Models financial scenarios probabilistically
- **Hybrid Score** – Combines both for more accurate risk assessment

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run api/main.py --server.port 5000 --server.address 0.0.0.0
```

## Key Functions

### `generate_risk_report_from_features()`
Creates a complete risk assessment report:
```python
from predict import generate_risk_report_from_features

features = {
    "total_income": 50000,
    "total_expenses": 35000,
    "savings_ratio": 0.30,
    "coverage_ratio": 1.43,
    "days_without_income": 5,
    # ... 9 more features
}

report = generate_risk_report_from_features(features)
# Returns: ml_probability, simulation_probability, hybrid_probability, risk_level
```

### `ml_probability_from_features()`
Gets ML-based default probability with fallback heuristics if model unavailable.

### `hybrid_risk_score()`
Converts probability (0-1) to readable score (0-100).

## Core Modules

| Module | Purpose |
|--------|---------|
| `predict.py` | Orchestrates ML + Monte Carlo + hybrid scoring |
| `models/hybrid_score.py` | Combines ML and simulation probabilities |
| `risk_simulation/monte_carlo.py` | Probabilistic financial scenario modeling |
| `ingestion/` | Parses financial documents (M-Pesa PDFs) |

## Risk Levels

| Probability | Level | Status |
|------------|-------|--------|
| < 20% | 🟢 Low | Financially stable |
| 20-50% | 🟡 Moderate | Some stress indicators |
| > 50% | 🔴 High | High default risk |

## Features Analyzed

- Income/expense stability and volatility
- Savings and coverage ratios
- Balance trends
- Payment history (transaction frequency, gaps)
- Debt-to-income ratios

## Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn
- **Data**: pandas, numpy
- **Visualization**: Plotly, Matplotlib
- **Document Parsing**: pdfplumber

## What Makes DebtShield Unique

✨ **Hybrid Approach** – Combines ML predictions with stochastic simulations for robust risk scoring  
✨ **Financial Scenario Modeling** – Monte Carlo simulations project outcomes under various conditions  
✨ **Intelligent Fallbacks** – Heuristic-based probabilities when ML model unavailable  
✨ **Multi-source Support** – Processes PDFs and transaction exports  

---

**DebtShield** – Intelligent debt default prediction 🛡️
