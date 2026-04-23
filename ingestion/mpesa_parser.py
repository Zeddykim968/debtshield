# This file should be able to read and extract data from mpesa statements in pdf format.
# It should be able to handle different formats of statements and extract the relevant data such as date, amount, transaction type, and description. 
# The extracted data should be stored in a structured format such as a CSV file or a database for further analysis.  

import pdfplumber
import pandas as pd

def extract_mpesa_data(file_path):
    rows = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()

            if table:
                for row in table[1:]:
                    rows.append(row)

    return pd.DataFrame(rows)                