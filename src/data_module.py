import pandas as pd
import os
from src.preprocessing import clean_and_prepare_data

def make_dataset(url: str, raw_file_name: str, final_file_name: str):
    """
    Downloads, cleans, and prepares the data, saving the result.
    This acts as the single governance point for the entire pipeline.
    """
    
    # 1. Sourcing Data
    print(f"--- 1. Sourcing data from: {url} ---")
    df = pd.read_csv(url)
    
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    raw_file_path = os.path.join('data', raw_file_name)
    df.to_csv(raw_file_path, index=False)
    print(f"   -> Raw data loaded and saved to: {raw_file_path}")
    
    # 2. Applying stable preprocessing pipeline
    print("\n--- 2. Applying stable preprocessing pipeline ---")
    
    # The preprocessing function returns features (X) and target (y)
    X_cleaned, y_target = clean_and_prepare_data(df)
    
    # Combine back for final saving
    final_df = X_cleaned.copy()
    final_df['HomeTeamResult'] = y_target
    
    # 3. Saving final data
    final_file_path = os.path.join('data', final_file_name)
    
    # CRITICAL: Drop rows where the target is NaN before saving (e.g., if result was corrupted)
    initial_rows = final_df.shape[0]
    final_df.dropna(subset=['HomeTeamResult'], inplace=True)
    
    final_df.to_csv(final_file_path, index=False)
    
    print(f"\n Pipeline Complete. Final data saved to: {final_file_path}")
    if initial_rows != final_df.shape[0]:
        print(f"WARNING: Dropped {initial_rows - final_df.shape[0]} rows due to missing results.")
        
    return final_df
