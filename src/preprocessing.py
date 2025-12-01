import pandas as pd
import numpy as np

def clean_and_prepare_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Consolidates all schema definitions, type conversions, target creation,
    and data leakage removal into a single, stable preprocessing pipeline.

    Args:
        df: The raw DataFrame containing EPL match data.

    Returns:
        A tuple of (cleaned feature DataFrame, text target Series).
    """
    print("--- Running Consolidated Preprocessing Module ---")

    # 1. Type Conversions / Schema Definition
    # Ensure key columns are numeric/datetime (coerce errors to NaN for later handling)
    numeric_cols = [col for col in df.columns if 'score' in col or 'goals' in col or 'yellows' in col or 'reds' in col or 'offsides' in col or 'fouls' in col or 'shots' in col or 'corners' in col or 'posse' in col]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill NaNs with 0 for count features

    # Convert Date column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')

    # Sort data chronologically (Crucial for time-series modeling)
    # Ensure 'PL_id' and 'kickoff' are handled for reliable sorting if dates are the same
    sort_cols = [col for col in ['date', 'kickoff', 'PL_id'] if col in df.columns]
    df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
    print("Data types cleaned and data sorted chronologically.")

    # 2. Target Variable Creation (HomeTeamResult)
    try:
        result_map = {'H': 'W', 'D': 'D', 'A': 'L'}
        df['HomeTeamResult'] = df['result'].map(result_map)
        print("Target variable 'HomeTeamResult' created successfully.")
    except KeyError:
        print("ERROR: Could not find or map the original 'result' column.")
        # Return empty frames if target cannot be created
        return pd.DataFrame(), pd.Series() 

    # 3. Data Leakage Rules & Feature Filtering (Addressing Feedback 2)
    LEAKAGE_COLS = [
        'home_goals', 'away_goals', # Raw goal counts (leakage)
        'result', # Original target variable
        'home_score', 'away_score', # If created from goals, they are leakage
        'home_ht_score', 'away_ht_score', # Half-time scores (leakage)
        'home_goals_clean', 'away_goals_clean', # Clean goal counts (leakage)
        'home_first_goal', 'away_first_goal', 'home_last_goal', 'away_last_goal', # Goal-related time/order (leakage)
        'home_yellow_pl', 'away_yellow_pl', 'home_red_pl', 'away_red_pl', # Player-list based card counts (potential leakage/redundant)
    ]

    # Drop all leakage and non-numeric columns
    all_cols_to_exclude = LEAKAGE_COLS + ['referee', 'stadium', 'city', 'home_team', 'away_team', 'home_lineup', 'away_lineup', 'date', 'kickoff']
    
    # Drop columns that exist in the DataFrame
    cols_to_drop = [col for col in all_cols_to_exclude if col in df.columns]
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')

    # Separate the target before filtering out remaining text
    target_series = df['HomeTeamResult']

    # Drop any remaining non-numeric columns that slipped through (e.g., text columns)
    df_cleaned = df_cleaned.select_dtypes(include=np.number)

    print(f"Formally dropped {len(cols_to_drop)} leakage and text columns. Features remaining: {len(df_cleaned.columns)}")

    # Add back non-leakage identifiers if needed (e.g., PL_id)
    if 'PL_id' in df.columns:
        df_cleaned['PL_id'] = df['PL_id']

    # Final result is the feature set and the text target
    return df_cleaned, target_series

# The second function is no longer needed as the first returns both X and y.
