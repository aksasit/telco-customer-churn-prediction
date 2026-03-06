import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Cleaning for IBM Telco churn.
    - Trim column names
    - Drop obvious ID cols
    - Fix TotalCharges to numeric
    - Map target Churn to 0/1 if needed
    - Simple NA handling
    """
    # Tidy Headers, Remove leading/trailing whitespace
    df.columns = df.columns.str.strip()
    
    # Drop ids if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # Map target column to 0/1 if it is Yes/No
    if target_col in df.columns and (df["Churn"].dtype == "object" or df["Churn"].dtype == "string"):
        df[target_col] = df[target_col].str.strip().map({"No":0, "Yes": 1})
        
    # Total Charges often has blanks in the dataset
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    # SeniorCitizen should be 0/1 integer if present
    
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)
     
    # Simple NA strategy:   
    # - Numeric: fill with 0
    # - Others: leave for encoders to handle (get_dummies ignores NaN safely)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df