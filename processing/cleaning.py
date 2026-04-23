# Here the rows parsed from the pdfare now cleaned and made more structured for better collection of teh necessary data.
# The data is grouped as either expense or income or loan etc.

import pandas as pd
import re

def clean_mpesa_data(df):
    df = df.loc[:, :4]
    df.columns = ["date", "details", "amount", "balance"]
     # uniform date format
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["amount"] = df["amount"].astype(str).str.replace(",", "")
     # covert the column to numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df["balance"] = df["balance"].astype(str).str.replace(",", "")

     # convert to numeric
    df["balance"] = pd.to_numeric(df["amount"], errors="coerce")

    df.dropna()
    return df


def classify_the_data(df):
    def classify_transaction(text):
        text = str(text).lower()

        # -----------------------------
        # INCOME (Money coming in)
        # -----------------------------
        if re.search(r"received|deposit|credited", text):
            return "income"

        # ----------------------------
        # LOAN INFLOW (loan received)
        # ----------------------------
        elif re.search(r"loan repayment|mshwari repayment|kcb mpesa repayment", text):
            return "loan_repayment"

        # ----------------------------
        # SAVINGS (Money moved to savings)
        # ----------------------------
        elif re.search(r"mshwari|kcb mpesa|savings", text):
            return "savings"

        # ----------------------------
        # EXPENSES (Money going out)
        # ----------------------------
        elif re.research(r"sent|withdraw|paid|purchase|lipa na mpesa", text):
            return "expenses"

        # -----------------------------
        # FEES/CHARGES
        # -----------------------------
        elif re.search(r"charge|fee|cost", text):
            return "fees"

        # -----------------------------
        # TRANSFERS (neural movement)
        # -----------------------------
        elif re.search(r"transfer",text):
            return "transfer"

        # -----------------------------
        # UNKNOWN (fallback)
        # -----------------------------
        else:
            return "other"

    # Aply the classification
    df["category"] = df["details"].apply(classify_transaction)

    # --------------------------------------------
    # ADD DIRECTION  (INFLOW vs OUTFLOW)
    # --------------------------------------------
     
    def get_direction(cat):
        """
        Helps separate cashflow direction
        """

        if cat in ["income", "loan_inflow"]:
            return "inflow"

        elif cat in ["expenses", "loan_repayment", "fees", "savings"]:
            return "outflow"

        else:
            return "neutral"

    df["direction"] = df["category"].apply(get_direction)

    return df                                               
