

import numpy as np

def generete_features(df):
    income_df = df[df["category"].isin(["income", "loan_inflow"])]
    expense_df = df[df["category"].isin(["expense", "loan_repayment", "fees", "savings"])]

    income = income_df["amount"]
    expenses = expense_df["amount"]

    # --------------------------------------
    # Basic
    # --------------------------------------
    total_income = income.sum()
    total_expenses = expenses.sum()
    
    # --------------------------------------
    # Means
    # --------------------------------------
    income_maen = income.mean() if not income.empty else 0
    expense_mean = expenses.mean() if not expenses.empty else 0

    # --------------------------------------
    # STABILITY (Coefficient of Varaiation)
    # --------------------------------------
    income_std = income.std() if not income.empty else 0
    expense_std = expense.std() if not expense.empty else 0 

    income_stability = income_std / income_mean if income_mean != 0 else 0
    expense_volatility = expense_std / expense_mean if  expense_mean != 0 else 0

    # --------------------------------------
    # SAVINGS
    # --------------------------------------
    savings_ratio = (total_income - total_expenses) / total_income if total_income > 0 else 0

    # --------------------------------------
    # COVERAGE
    # --------------------------------------
    coverage_ratio = total_income / total_expenses if total_expenses > 0 else 0

    # --------------------------------------
    # TRANSACTION FREQUENCY
    # --------------------------------------
    tansaction_freq = len(df)

    # --------------------------------------
    # DAYS WITHOUT INCOME
    # --------------------------------------
    date_only = df["date"].dt.date
    income_days = income_df["date"].dt.date.unique()

    all_days = df["date"].dt.date.unique()
    days_without_income = len(set(all_days) - set(income_days))

    # --------------------------------------
    # BALANCE TREND
    # --------------------------------------
    df_sorted = df.sort_values("date")

    balance_trend = (df_sorted["balance"].iloc[-1] - df_sorted["balance"].iloc[0] if not df_sorted.empty else 0)

    # --------------------------------------
    # LARGE EXPENSES SHOCK
    # --------------------------------------
    largest_expense = expenses.max() if not expenses.empty else 0
    largest_expense_ratio = largest_expense / total_income if total_income > 0 else 0 

    # --------------------------------------
    # INCOME DIVERSITY (count of unique descriptions)
    # --------------------------------------
     income_diversity = income_df["details"].nunique()

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "income_mean": income_mean,
        "expense_mean": expense_mean,
        "income_stability": income_stability,
        "expense_volatility": expense_volatility,
        "savings_ratio": savings_ratio,
        "coverage_ratio": coverage_ratio,
        "transaction_freq": transaction_freq,
        "days_without_income": days_without_income,
        "balance_trend": balance_trend,
        "largest_expense": largest_expense,
        "largest_expense_ratio": largest_expense_ratio,
        "income_diversity": income_diversity
    }


