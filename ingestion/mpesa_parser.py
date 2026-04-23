import pdfplumber
import pandas as pd


def extract_mpesa_data(file_or_path):
    rows = []
    with pdfplumber.open(file_or_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table[1:]:
                    rows.append(row)
    return pd.DataFrame(rows)


def load_mpesa_csv(file_or_path):
    df = pd.read_csv(file_or_path)
    return df
