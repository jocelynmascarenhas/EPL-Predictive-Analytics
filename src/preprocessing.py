import pandas as pd
import numpy as np

# --- HELPER FUNCTION FOR ROLLING FORM (Addressing Feedback 1) ---
def calculate_rolling_team_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Calculates rolling average form statistics for each team (window=5) using past data only."""
    print(f"   -> Calculating rolling form over {window} matches...")

    def transform_match_to_team_view(df, team_type):
        is_home = (team_type == 'home')
        prefix = 'home_' if is_home else 'away_'
        
        stats = {
            'goals_scored': f'{prefix}goals_clean',
            'goals_conceded': f'away_goals_clean' if is_home else f'home_goals_clean',
            'shots': f'{prefix}shots',
            'posse': f'{prefix}posse'
        }
        
        team_df = df[[f'{team_type}_team', 'date', 'PL_id'] + list(stats.values())].copy()
        team_df.rename(columns={
            f'{team_type}_team': 'Team',
            stats['goals_scored']: 'Goals_Scored',
            stats['goals_conceded']: 'Goals_Conceded',
            stats['shots']: 'Shots',
            stats['posse']: 'Possession'
        }, inplace=True)
        
        if is_home:
            team_df['Points'] = df['HomeTeamResult'].map({'W': 3, 'D': 1, 'L': 0})
        else:
            team_df['Points'] = df['HomeTeamResult'].map({'W': 0, 'D': 1, 'L': 3}) 

        return team_df

    home_view = transform_match_to_team_view(df, 'home')
    away_view = transform_match_to_team_view(df, 'away')
    all_team_matches = pd.concat([home_view, away_view]).sort_values(by=['date', 'PL_id'])
    
    form_cols = ['Goals_Scored', 'Goals_Conceded', 'Shots', 'Possession', 'Points']
    
    # Crucial: closed='left' ensures no data leakage from the current match
    rolling_stats = all_team_matches.groupby('Team')[form_cols].apply(
        lambda x: x.rolling(window=window, min_periods=1, closed='left').mean()
    ).reset_index()
    
    df = df.merge(
        rolling_stats.rename(columns=lambda x: f'Form_Home_{x}' if x in form_cols else x),
        left_on=['PL_id', 'home_team'],
        right_on=['PL_id', 'Team'],
        how='left'
    ).drop(columns=['Team'], errors='ignore')

    df = df.merge(
        rolling_stats.rename(columns=lambda x: f'Form_Away_{x}' if x in form_cols else x),
        left_on=['PL_id', 'away_team'],
        right_on=['PL_id', 'Team'],
        how='left'
    ).drop(columns=['Team'], errors='ignore')
    
    rolling_cols = [col for col in df.columns if 'Form_' in col]
    df[rolling_cols] = df[rolling_cols].fillna(0)
    
    return df
    
# --- UPDATED clean_and_prepare_data FUNCTION (WITH BUG FIX) ---
def clean_and_prepare_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Consolidates all schema definitions, type conversions, target creation,
    and data leakage removal into a single, stable preprocessing pipeline.
    """
    print("--- Running Consolidated Preprocessing Module ---")

    # 1. Type Conversions / Schema Definition
    numeric_cols = [col for col in df.columns if 'score' in col or 'goals' in col or 'yellows' in col or 'reds' in col or 'offsides' in col or 'fouls' in col or 'shots' in col or 'corners' in col or 'posse' in col]
    for col in numeric_cols:
        # **CRITICAL BUG FIX**: Correctly use df[col] to access the column data
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 

    df['home_goals_clean'] = df['home_goals'].apply(lambda x: 0 if pd.isna(x) else x.count('(') if isinstance(x, str) else x)
    df['away_goals_clean'] = df['away_goals'].apply(lambda x: 0 if pd.isna(x) else x.count('(') if isinstance(x, str) else x)
    
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')

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
        return pd.DataFrame(), pd.Series() 
    
    df = calculate_rolling_team_form(df, window=5)
    
    # 3. Data Leakage Rules & Feature Filtering
    LEAKAGE_COLS = [
        'home_goals', 'away_goals', 'home_goals_clean', 'away_goals_clean', 
        'result', 'home_score', 'away_score', 'home_ht_score', 'away_ht_score', 
        'home_first_goal', 'away_first_goal', 'home_last_goal', 'away_last_goal', 
        'home_yellow_pl', 'away_yellow_pl', 'home_red_pl', 'away_red_pl', 
    ]
    
    TEXT_COLS = ['referee', 'stadium', 'city', 'home_team', 'away_team', 'home_lineup', 'away_lineup', 'date', 'kickoff']

    all_cols_to_exclude = LEAKAGE_COLS + TEXT_COLS
    cols_to_drop = [col for col in all_cols_to_exclude if col in df.columns]
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')

    target_series = df['HomeTeamResult']
    df_cleaned = df_cleaned.select_dtypes(include=np.number)

    print(f"Formally dropped {len(cols_to_drop)} leakage and text columns. Features remaining: {len(df_cleaned.columns)}")

    if 'PL_id' in df.columns:
        df_cleaned['PL_id'] = df['PL_id']

    df_cleaned['HomeTeamResult'] = target_series

    return df_cleaned, target_series
