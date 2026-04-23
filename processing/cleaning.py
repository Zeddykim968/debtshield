import pandas as pd
import re


def _to_number(series):
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": None, "nan": None, "None": None, "-": None})
    return pd.to_numeric(s, errors="coerce")


def clean_mpesa_data(df):
    """Normalise an M-Pesa statement DataFrame into:
        date, details, amount, balance
    Handles both the official Safaricom layout (Receipt No, Completion Time,
    Details, Transaction Status, Paid In, Withdrawn, Balance) and the simpler
    4-column layout (date, details, amount, balance).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "details", "amount", "balance"])

    df = df.copy()
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    def find(*keys):
        for k in keys:
            for c in cols:
                if k in c:
                    return c
        return None

    date_col = find("completion time", "date", "time")
    details_col = find("details", "description", "narrative")
    paid_in_col = find("paid in", "paidin", "credit", "money in")
    withdrawn_col = find("withdrawn", "debit", "money out")
    amount_col = find("amount")
    balance_col = find("balance")

    out = pd.DataFrame()

    if date_col is not None:
        raw_date = df[date_col]
    elif df.shape[1] >= 1:
        raw_date = df.iloc[:, 0]
    else:
        raw_date = pd.Series([None] * len(df))

    parsed = pd.to_datetime(raw_date, errors="coerce")
    if parsed.notna().sum() < max(1, len(parsed) // 4):
        parsed = pd.to_datetime(raw_date, errors="coerce", dayfirst=True)
    out["date"] = parsed.ffill()

    if details_col is not None:
        out["details"] = df[details_col].astype(str)
    elif df.shape[1] >= 2:
        out["details"] = df.iloc[:, 1].astype(str)
    else:
        out["details"] = ""

    if paid_in_col is not None or withdrawn_col is not None:
        paid_in = _to_number(df[paid_in_col]) if paid_in_col else 0
        withdrawn = _to_number(df[withdrawn_col]) if withdrawn_col else 0
        paid_in = paid_in.fillna(0) if hasattr(paid_in, "fillna") else paid_in
        withdrawn = withdrawn.fillna(0) if hasattr(withdrawn, "fillna") else withdrawn
        out["amount"] = paid_in + withdrawn.abs()
        out["_signed_amount"] = paid_in - withdrawn.abs()
    elif amount_col is not None:
        out["amount"] = _to_number(df[amount_col]).abs()
        out["_signed_amount"] = _to_number(df[amount_col])
    elif df.shape[1] >= 3:
        signed = _to_number(df.iloc[:, 2])
        out["amount"] = signed.abs()
        out["_signed_amount"] = signed
    else:
        out["amount"] = pd.NA
        out["_signed_amount"] = pd.NA

    if balance_col is not None:
        out["balance"] = _to_number(df[balance_col])
    elif df.shape[1] >= 4:
        out["balance"] = _to_number(df.iloc[:, 3])
    else:
        out["balance"] = pd.NA

    out = out.dropna(subset=["amount"]).reset_index(drop=True)
    if out["date"].isna().all():
        out["date"] = pd.Timestamp.today().normalize()
    else:
        out["date"] = out["date"].ffill().bfill()
    return out


def classify_the_data(df):
    df = df.copy()

    def classify_transaction(row):
        text = str(row.get("details", "")).lower()
        signed = row.get("_signed_amount", None)

        if re.search(r"loan repayment|mshwari repayment|kcb mpesa repayment|fuliza repayment", text):
            return "loan_repayment"
        if re.search(r"mshwari deposit|kcb mpesa deposit|lock savings|saving", text):
            return "savings"
        if re.search(r"charge|fee|cost|excise|tariff", text):
            return "fees"
        if re.search(r"received|deposit|credited|salary|reversal", text):
            return "income"
        if re.search(r"sent|withdraw|paid|purchase|lipa na mpesa|pay bill|buy goods|airtime|merchant", text):
            return "expenses"
        if re.search(r"transfer", text):
            return "transfer"

        # Fall back to the sign of the amount when wording is unfamiliar
        try:
            if signed is not None and float(signed) > 0:
                return "income"
            if signed is not None and float(signed) < 0:
                return "expenses"
        except (TypeError, ValueError):
            pass
        return "other"

    df["category"] = df.apply(classify_transaction, axis=1)

    def get_direction(cat):
        if cat in ["income", "loan_inflow"]:
            return "inflow"
        if cat in ["expenses", "loan_repayment", "fees", "savings"]:
            return "outflow"
        return "neutral"

    df["direction"] = df["category"].apply(get_direction)
    return df
