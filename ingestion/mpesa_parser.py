import pdfplumber
import pandas as pd


def extract_mpesa_data(file_or_path, password=None):
    """Extract all table rows from an M-Pesa PDF statement.

    Returns a DataFrame whose first row contains the header (column names
    as found in the PDF). The cleaning step is responsible for normalising
    the columns.
    """
    rows = []
    header = None

    open_kwargs = {}
    if password:
        open_kwargs["password"] = password

    with pdfplumber.open(file_or_path, **open_kwargs) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                if not table:
                    continue
                if header is None:
                    header = [
                        (c or "").strip().lower().replace("\n", " ")
                        for c in table[0]
                    ]
                    body = table[1:]
                else:
                    first = [(c or "").strip().lower() for c in table[0]]
                    if first == header:
                        body = table[1:]
                    else:
                        body = table
                for row in body:
                    if row and any((c or "").strip() for c in row):
                        rows.append(row)

    if not rows:
        return pd.DataFrame()

    width = max(len(r) for r in rows)
    rows = [list(r) + [None] * (width - len(r)) for r in rows]

    if header and len(header) == width:
        return pd.DataFrame(rows, columns=header)
    return pd.DataFrame(rows)


def load_mpesa_csv(file_or_path):
    return pd.read_csv(file_or_path)
