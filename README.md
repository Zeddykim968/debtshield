# DebtShield 🛡️

An intelligent financial risk assessment engine that combines machine learning with Monte Carlo simulation to predict and analyze the probability of financial default.

## What is DebtShield?

DebtShield is a comprehensive financial risk analysis platform designed to help individuals and organizations assess their debt repayment capacity and financial stability. It uses a hybrid approach combining:

- **Machine Learning Models** – Trained classifiers that predict default probability from financial features
- **Monte Carlo Simulation** – Probabilistic modeling that simulates various financial scenarios
- **Hybrid Risk Scoring** – Intelligent combination of ML predictions and simulation results for robust risk assessment

The platform provides actionable insights through an intuitive Streamlit web interface that visualizes financial risk levels, trends, and outcomes.

---

## Key Features

✅ **Multi-source Financial Analysis** – Process transactions from various sources (e.g., M-Pesa PDFs)  
✅ **Advanced Risk Modeling** – Combines ML predictions with stochastic simulations  
✅ **Interactive Dashboard** – Real-time visualization of financial metrics and risk assessments  
✅ **Hybrid Risk Scoring** – Intelligently weights ML and simulation-based probabilities  
✅ **Financial Outcome Projections** – Simulates mean financial outcomes under various conditions  
✅ **Scalable Architecture** – Designed for deployment with auto-scaling capabilities  

---

## Project Structure

```
debtshield/
├── api/
│   └── main.py                    # Streamlit UI entry point
├── models/
│   ├── hybrid_score.py            # Hybrid risk scoring logic
│   ├── predict.py                 # Feature preparation & risk prediction
│   └── [training utilities]       # Model training scripts
├── risk_simulation/
│   └── monte_carlo.py             # Monte Carlo risk simulator
├── processing/
│   └── [data processing]          # Financial data cleaning & transformation
├── ingestion/
│   └── [data ingestion]           # Parser for financial documents (M-Pesa, etc.)
├── predict.py                     # Main orchestration module
├── requirements.txt               # Python dependencies
├── .replit                        # Replit configuration
└── replit.md                      # Replit setup guide
```

---

## How It Works

### 1. **Data Ingestion**
- Parse financial documents (PDFs, transaction exports)
- Extract transaction records, income, and expense data

### 2. **Data Processing**
- Clean and normalize financial data
- Calculate financial metrics:
  - Income/expense stability
  - Savings ratio
  - Coverage ratios
  - Balance trends
  - Transaction frequency

### 3. **Risk Prediction**
- **ML Component**: Trained classifier predicts default probability from features
- **Simulation Component**: Monte Carlo simulator models financial scenarios
- **Hybrid Scoring**: Combines both approaches for robust risk assessment

### 4. **Reporting**
- Generate risk scores (0–100 scale)
- Classify risk levels (Low, Moderate, High)
- Project mean financial outcomes

---

## Core Modules

### `predict.py` – Risk Report Orchestration
Generates comprehensive risk reports by:
1. Computing ML-based default probability
2. Running Monte Carlo simulations
3. Computing hybrid probabilities
4. Interpreting risk levels

**Key Functions:**
- `generate_risk_report_from_features()` – Create reports from feature dictionaries
- `ml_probability_from_features()` – Get ML prediction with fallback heuristics
- `hybrid_risk_score()` – Convert probability to 0–100 score

### `models/predict.py` – Feature Processing
Prepares financial features for model input:
- Standardizes feature order
- Converts dictionaries to model-ready arrays
- Interprets probabilities as risk categories

### `risk_simulation/monte_carlo.py` – Stochastic Modeling
Runs probabilistic simulations to:
- Model financial volatility
- Estimate default risk under various scenarios
- Project financial outcomes

---

## Installation

### Prerequisites
- Python 3.12+
- pip

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zeddykim968/debtshield.git
   cd debtshield
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the Streamlit App

```bash
streamlit run api/main.py --server.port 5000 --server.address 0.0.0.0
```

The application will be available at:
- **Local**: `http://localhost:5000`
- **Network**: `http://<your-ip>:5000`

### Generate a Risk Report Programmatically

```python
from predict import generate_risk_report_from_features

# Prepare financial features
features = {
    "total_income": 50000,
    "total_expenses": 35000,
    "income_mean": 4166.67,
    "expense_mean": 2916.67,
    "income_instability": 0.15,
    "expense_volatility": 0.25,
    "savings_ratio": 0.30,
    "coverage_ratio": 1.43,
    "transaction_freq": 25,
    "days_without_income": 5,
    "balance_trend": 100,
    "largest_expense": 5000,
    "largest_expense_ratio": 0.10,
    "income_diversity": 0.8,
}

# Generate report
report = generate_risk_report_from_features(features)

print(f"Risk Level: {report['risk_level']}")
print(f"Hybrid Probability: {report['hybrid_probability']:.2%}")
print(f"Hybrid Score: {report['hybrid_score']:.1f}/100")
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.12 |
| **ML/Data** | scikit-learn, pandas, numpy |
| **Simulation** | Monte Carlo engine (custom) |
| **Visualization** | Plotly, Matplotlib |
| **Document Parsing** | pdfplumber |
| **Serialization** | joblib |

---

## Features Explained

### Financial Features Used in Risk Assessment

- **total_income** – Sum of all income
- **total_expenses** – Sum of all expenses
- **income_mean** – Average monthly income
- **expense_mean** – Average monthly expenses
- **income_instability** – Variance in income (higher = less stable)
- **expense_volatility** – Variance in expenses (higher = more unpredictable)
- **savings_ratio** – Proportion of income saved (higher = better)
- **coverage_ratio** – Income-to-expense ratio (higher = better)
- **transaction_freq** – Number of transactions (activity indicator)
- **days_without_income** – Maximum consecutive days without income (higher = riskier)
- **balance_trend** – Direction of account balance over time
- **largest_expense** – Highest single expense amount
- **largest_expense_ratio** – Largest expense as % of income
- **income_diversity** – Variety of income sources (0–1 scale)

### Risk Levels

| Probability | Risk Level | Interpretation |
|------------|-----------|-----------------|
| < 20% | 🟢 Low Risk | Strong financial stability |
| 20–50% | 🟡 Moderate Risk | Some financial stress indicators |
| > 50% | 🔴 High Risk | Significant default probability |

---

## Deployment

DebtShield is configured for cloud deployment with auto-scaling:

```bash
streamlit run api/main.py \
  --server.port 5000 \
  --server.address 0.0.0.0 \
  --server.headless true
```

See `.streamlit/config.toml` for detailed configuration options.

---

## Dependencies

```
streamlit              # Interactive web framework
pandas                 # Data manipulation
numpy                  # Numerical computing
scikit-learn          # Machine learning
matplotlib            # Static visualizations
plotly                # Interactive charts
pdfplumber            # PDF parsing
joblib                # Model serialization
```

---

## Roadmap

- [ ] Integration with real banking APIs
- [ ] Multi-currency support
- [ ] Advanced portfolio analysis
- [ ] Custom risk thresholds
- [ ] API endpoints for third-party integration
- [ ] Historical trend analysis
- [ ] Predictive alerts

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is open source. See `LICENSE` file for details.

---

## Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the `replit.md` file for Replit-specific setup

---

## Acknowledgments

Built as a comprehensive financial risk assessment solution combining modern machine learning with Monte Carlo simulation techniques.

---

**DebtShield** – Empower informed financial decisions 🛡️
