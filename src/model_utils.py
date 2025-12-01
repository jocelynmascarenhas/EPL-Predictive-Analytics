import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, matthews_corrcoef, f1_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any

# Dictionary mapping model names to their estimators
MODEL_BENCHMARKS = {
    'LogisticRegression': LogisticRegression(solver='liblinear', multi_class='ovr', random_state=42, max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, max_depth=5)
}

def evaluate_model(X: pd.DataFrame, y: pd.Series, model_name: str, le: LabelEncoder, n_splits: int = 5) -> Dict[str, Any]:
    """
    Evaluates a single model using TimeSeriesSplit (Addressing Feedback 5).

    Returns:
        A dictionary containing mean and standard deviation of key metrics across folds.
    """
    print(f"\n--- Evaluating {model_name} using TimeSeriesSplit (k={n_splits}) ---")
    
    model = MODEL_BENCHMARKS[model_name]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Lists to store metrics from each fold
    macro_f1_scores = []
    log_losses = []
    mcc_scores = []
    
    # Key indices used for feature importance stability check
    fold_indices = []

    # Prepare data (encoding must be done outside the loop for stability)
    y_encoded = le.transform(y)
    X_features = X.drop(columns=['PL_id'], errors='ignore').values
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X_features)):
        X_train, X_test = X_features[train_index], X_features[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict probabilities for log_loss
        y_pred_proba = model.predict_proba(X_test)
        
        # Predict class labels for F1 and MCC
        y_pred = model.predict(X_test)
        
        # Calculate Metrics
        macro_f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        log_losses.append(log_loss(y_test, y_pred_proba))
        mcc_scores.append(matthews_corrcoef(y_test, y_pred))
        
        # Save indices for potential feature importance checks later
        fold_indices.append(test_index)
        
        print(f"Fold {fold+1}: Macro F1={macro_f1_scores[-1]:.4f}, Log Loss={log_losses[-1]:.4f}")

    # Summarize results
    results = {
        'model': model_name,
        'f1_macro_mean': np.mean(macro_f1_scores),
        'f1_macro_std': np.std(macro_f1_scores),
        'log_loss_mean': np.mean(log_losses),
        'log_loss_std': np.std(log_losses),
        'mcc_mean': np.mean(mcc_scores),
        'mcc_std': np.std(mcc_scores),
        'feature_names': list(X.drop(columns=['PL_id'], errors='ignore').columns),
        'fold_indices': fold_indices
    }
    
    # If the model supports feature importance (RF/XGB), store it for later analysis (Feedback 6)
    if hasattr(model, 'feature_importances_'):
        # For simplicity, calculate feature importance on the final fold's model
        results['feature_importances'] = model.feature_importances_
    
    return results

def run_benchmarking_and_validation(X: pd.DataFrame, y: pd.Series, le: LabelEncoder, n_splits: int = 5) -> List[Dict[str, Any]]:
    """Runs all model evaluations and returns a list of result dictionaries."""
    
    all_results = []
    
    for model_name in MODEL_BENCHMARKS.keys():
        results = evaluate_model(X, y, model_name, le, n_splits)
        all_results.append(results)
        
    return all_results
