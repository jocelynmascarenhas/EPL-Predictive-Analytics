# EPL-Predictive-Analytics

## Project Goal
To develop a robust and time-aware machine learning model for predicting English Premier League (EPL) match outcomes (Home Win, Draw, Away Win) from the perspective of the Home Team. The prediction is framed as a multi-class classification problem.

## Methodology
The pipeline incorporates rigorous data governance, chronological data integrity checks, advanced feature engineering (e.g., rolling form statistics), and time-series cross-validation to prevent data leakage.

## Key Features
1. **Time-Aware Validation:** Uses `TimeSeriesSplit` to ensure models are only trained on past data.
2. **Rolling Form Features:** Calculates pre-match performance metrics (goals, shots, points) over the last 5 matches.
3. **Benchmarking:** Evaluates Logistic Regression, Random Forest, and XGBoost based on Macro F1, Log Loss, and MCC.

## Project Structure
- `data/`: Raw and final cleaned datasets.
- `src/`: Reusable Python modules (data_module, preprocessing, model_utils).
- `notebooks/`: Analytical and execution notebooks.
- `reports/`: Final visualizations (Confusion Matrix, Feature Importance).

## Getting Started
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
