import pandas as pd
import re


def clean_mpesa_data(df):
    df = df.iloc[:, :4].copy()
    df.columns = ["date", "details", "amount", "balance"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["amount"] = (
        df["amount"].astype(str).str.replace(",", "", regex=False)
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df["balance"] = (
        df["balance"].astype(str).str.replace(",", "", regex=False)
    )
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")

    df = df.dropna(subset=["date", "amount"]).reset_index(drop=True)
    return df


def classify_the_data(df):
    def classify_transaction(text):
        text = str(text).lower()
        if re.search(r"received|deposit|credited|salary", text):
            return "income"
        elif re.search(r"loan repayment|mshwari repayment|kcb mpesa repayment", text):
            return "loan_repayment"
        elif re.search(r"mshwari|kcb mpesa|savings", text):
            return "savings"
        elif re.search(r"sent|withdraw|paid|purchase|lipa na mpesa|pay bill|buy goods", text):
            return "expenses"
        elif re.search(r"charge|fee|cost", text):
            return "fees"
        elif re.search(r"transfer", text):
            return "transfer"
        else:
            return "other"

    df = df.copy()
    df["category"] = df["details"].apply(classify_transaction)

    def get_direction(cat):
        if cat in ["income", "loan_inflow"]:
            return "inflow"
        elif cat in ["expenses", "loan_repayment", "fees", "savings"]:
            return "outflow"
        else:
            return "neutral"

    df["direction"] = df["category"].apply(get_direction)
    return df
