import pandas as pd
import os
from preprocessing import clean_and_prepare_data # Imports cleaning logic from the same folder

def make_dataset(raw_data_url: str, raw_file_name: str, final_file_name: str) -> pd.DataFrame:
    """
    Sourcing and Cleaning Pipeline (Addresses Feedback 3: single make_dataset function).

    1. Loads raw data from URL.
    2. Saves the raw data.
    3. Calls the stable 'clean_and_prepare_data' pipeline.
    4. Saves the final cleaned, feature-engineered, and leakage-fixed dataset.

    Returns:
        The final cleaned DataFrame.
    """

    # 1. Data Sourcing and Initial Load
    print(f"--- 1. Sourcing data from: {raw_data_url} ---")
    raw_save_path = os.path.join("data", raw_file_name)

    try:
        raw_df = pd.read_csv(raw_data_url, encoding='latin1', low_memory=False)
        raw_df.to_csv(raw_save_path, index=False)
        print(f"   -> Raw data loaded and saved to: {raw_save_path}")
    except Exception as e:
        print(f"   -> ERROR: Could not load data. Details: {e}")
        return pd.DataFrame()

    # 2. Cleaning and Preparation (Using stable module)
    print("\n--- 2. Applying stable preprocessing pipeline ---")
    df_cleaned_features, df_target_text = clean_and_prepare_data(raw_df.copy())

    # Re-attach the text target for saving the full dataset
    df_cleaned_features['HomeTeamResult'] = df_target_text

    # 3. Final Audit and Save
    final_file_path = os.path.join("data", final_file_name)
    df_cleaned_features.to_csv(final_file_path, index=False)

    print(f"\n Pipeline Complete. Final data saved to: {final_file_path}")
    return df_cleaned_features
