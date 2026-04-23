import numpy as np


def generate_features(df):
    income_df = df[df["category"].isin(["income", "loan_inflow"])]
    expense_df = df[df["category"].isin(["expenses", "loan_repayment", "fees", "savings"])]

    income = income_df["amount"]
    expenses = expense_df["amount"]

    total_income = float(income.sum())
    total_expenses = float(expenses.sum())

    income_mean = float(income.mean()) if not income.empty else 0.0
    expense_mean = float(expenses.mean()) if not expenses.empty else 0.0

    income_std = float(income.std()) if len(income) > 1 else 0.0
    expense_std = float(expenses.std()) if len(expenses) > 1 else 0.0

    income_instability = income_std / income_mean if income_mean else 0.0
    expense_volatility = expense_std / expense_mean if expense_mean else 0.0

    savings_ratio = (total_income - total_expenses) / total_income if total_income > 0 else 0.0
    coverage_ratio = total_income / total_expenses if total_expenses > 0 else 0.0

    transaction_freq = int(len(df))

    income_days = set(income_df["date"].dt.date.unique())
    all_days = set(df["date"].dt.date.unique())
    days_without_income = len(all_days - income_days)

    df_sorted = df.sort_values("date")
    if not df_sorted.empty and df_sorted["balance"].notna().any():
        bal = df_sorted["balance"].dropna()
        balance_trend = float(bal.iloc[-1] - bal.iloc[0]) if len(bal) >= 2 else 0.0
    else:
        balance_trend = 0.0

    largest_expense = float(expenses.max()) if not expenses.empty else 0.0
    largest_expense_ratio = largest_expense / total_income if total_income > 0 else 0.0

    income_diversity = int(income_df["details"].nunique())

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "income_mean": income_mean,
        "expense_mean": expense_mean,
        "income_instability": income_instability,
        "expense_volatility": expense_volatility,
        "savings_ratio": savings_ratio,
        "coverage_ratio": coverage_ratio,
        "transaction_freq": transaction_freq,
        "days_without_income": days_without_income,
        "balance_trend": balance_trend,
        "largest_expense": largest_expense,
        "largest_expense_ratio": largest_expense_ratio,
        "income_diversity": income_diversity,
    }
