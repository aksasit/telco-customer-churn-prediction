import os, sys
import pandas as pd

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/telco_customer_churn.csv"
OUT = "data/processed/telco_churn_processed.csv"

# Load raw file
df = pd.read_csv(RAW)

# Preprocess the data (drops id, fixes TotalCharges, etc.)
df = preprocess_data(df, target_col="Churn")

# Ensure target is 0/1 if still object
if "Churn" in df.columns and (df["Churn"].dtype == "object" or df["Churn"].dtype == "string"):
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype('Int64')

# Sanity Checks
assert df["Churn"].isna().sum() == 0, "Churn has NaN after Preprocess"
assert set(df["Churn"].unique()) <= {0,1}, "Churn have value other than 0/1 after preprocess"

# Features
df_processed = build_features(df, target_col="Churn")

# Save 
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)

print(f"✅ Processed dataset saved to {OUT} | Shape: {df_processed.shape}")