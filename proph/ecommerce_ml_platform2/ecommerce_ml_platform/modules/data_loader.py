# =============================================================
#  data_loader.py
#  Handles file upload, parsing, and column auto-detection
# =============================================================

import streamlit as st
import pandas as pd

MAX_FILE_MB = 50


def load_raw(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel file, normalise column names to lowercase."""
    if uploaded_file.size > MAX_FILE_MB * 1024 * 1024:
        st.error(f"❌ File too large. Maximum allowed size is {MAX_FILE_MB} MB.")
        st.stop()

    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "csv":
        df = pd.read_csv(uploaded_file, low_memory=False)
    elif ext in ("xls", "xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported format — use CSV or Excel.")

    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def guess_columns(df: pd.DataFrame):
    """
    Auto-detect date, numeric (sales), and customer ID columns.
    Returns (date_col, num_col, cust_col)
    """
    # Date column
    date_col = next(
        (c for c in df.columns if any(k in c for k in ["date", "time", "order_date"])),
        None,
    )

    # Numeric / sales column
    preferred_num = ["total_price", "total", "amount", "sales", "revenue", "totalprice"]
    numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_col = next((c for c in preferred_num if c in df.columns), None)
    if num_col is None and numeric_candidates:
        num_col = numeric_candidates[0]

    # Customer ID column — prefer exact match first
    preferred_cust = ["user_id", "customer_id", "userid", "customerid"]
    cust_col = next((c for c in preferred_cust if c in df.columns), None)
    if cust_col is None:
        cust_col = next(
            (c for c in df.columns if any(
                k in c for k in ["customer", "cust", "buyer", "client", "user"]
            )),
            None,
        )

    return date_col, num_col, cust_col
