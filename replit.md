# DebtShield 2.0

Streamlit-based AI financial risk engine combining a (placeholder) ML model with Monte Carlo simulation.

## Stack
- Python 3.12
- Streamlit (frontend on port 5000, host 0.0.0.0)
- pandas, numpy, scikit-learn, matplotlib, plotly, pdfplumber, joblib

## Layout
- `api/main.py` – Streamlit UI entry point
- `predict.py` – `generate_risk_report` orchestrating ML + Monte Carlo + hybrid scoring
- `models/` – training, prediction, hybrid scoring
- `risk_simulation/monte_carlo.py` – Monte Carlo simulator
- `processing/`, `ingestion/` – data cleaning / M-Pesa PDF parsing helpers

## Run
Workflow `Streamlit App` runs:
`streamlit run api/main.py --server.port 5000 --server.address 0.0.0.0`

Streamlit config in `.streamlit/config.toml` allows iframe proxying (CORS/XSRF disabled, headless mode).

## Deployment
Configured for autoscale with the same streamlit run command.
