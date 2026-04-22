import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    """
    Loads the SuperStore dataset and performs comprehensive cleaning.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

    # Handle Date Columns
    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], dayfirst=True, errors="coerce")
    
    # Drop rows with invalid dates
    df = df.dropna(subset=["order_date"])

    # Robust Sales & Profit Cleaning (always force numeric)
    for col in ["sales", "profit"]:
        if col in df.columns:
            # Remove currency symbols and commas
            df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Numeric Columns
    numeric_cols = ["profit", "discount", "quantity", "shipping_cost"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle Missing Values
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Safely calculate median for sales
    if not df["sales"].dropna().empty:
        df["sales"] = df["sales"].fillna(df["sales"].median())
    else:
        df["sales"] = df["sales"].fillna(0)
    
    categorical_cols = ["region", "category", "sub_category", "market", "segment", "country", "state"]
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Add Time-based features
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["day"] = df["order_date"].dt.day
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df

def get_filtered_data(df, filters):
    """
    Applies filters to the dataframe based on user selection.
    filters: dict with keys matching column names and values being the selection (or "All")
    """
    filtered_df = df.copy()
    for col, val in filters.items():
        if val != "All":
            filtered_df = filtered_df[filtered_df[col] == val]
    return filtered_df
