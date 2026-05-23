"""
Per-Company Model Training with Conformal Calibration
======================================================

Pipeline:
  Phase 2: Train XGBoost model per company (60% train, 20% validation)
  Phase 3: Compute conformal q-values (20% calibration data)

Output:
  - models/company_models.pkl: Trained models per company
  - models/q_values.pkl: Conformal q-values per company
  - models/training_stats.csv: Training statistics (incl. validation MAE)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib
import json
from collections import defaultdict
from datetime import datetime


# ==============================================================================
# CONFIGURATION
# ==============================================================================

MIN_SAMPLES_PER_COMPANY = 30   # Minimum bids to train a model (lowered from 100)
MIN_CALIBRATION_SAMPLES = 30  # Minimum samples for stable conformal calibration
VALIDATION_RATIO = 0.2        # 20% for validation
CALIBRATION_RATIO = 0.2       # 20% for calibration  
# --> Train: 60%, Validation: 20%, Calibration: 20%
CONFORMAL_ALPHA = 0.1         # 90% coverage (1 - alpha)

XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'gpu_id': 0
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_project_root():
    return Path(__file__).parent


def load_features():
    """Load pre-computed features and merge with train_clean.csv for company_code."""
    root = get_project_root()
    features_path = root / 'analysis_results' / 'feature_sample.csv'
    train_path = root / 'data' / 'processed' / 'train_clean.csv'  # Use cleaned data
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}\n"
            "Run feature_engineering.py first to generate features."
        )
    
    # Load features
    print("Loading features...")
    features_df = pd.read_csv(features_path)
    print(f"  Features: {len(features_df):,} rows × {len(features_df.columns)} columns")
    
    # CRITICAL: Validate it's the full dataset, not a sample
    EXPECTED_MIN_ROWS = 100000
    if len(features_df) < EXPECTED_MIN_ROWS:
        raise ValueError(
            f"Feature file has only {len(features_df):,} rows (expected ≥{EXPECTED_MIN_ROWS:,})\n"
            "Did you run feature_engineering.py with sample_size=None?"
        )
    
    # Load train.csv for company_code and announce_date
    print("Loading train.csv for company metadata...")
    train_df = pd.read_csv(train_path, usecols=['record_id', 'company_code', 'announce_date'])
    print(f"  Train: {len(train_df):,} rows")
    
    # Merge company_code and announce_date into features
    df = features_df.merge(train_df, on='record_id', how='left')
    
    # Verify merge
    missing_company = df['company_code'].isna().sum()
    if missing_company > 0:
        print(f"  ⚠️ {missing_company} records missing company_code after merge")
        df = df.dropna(subset=['company_code'])
    
    print(f"Final dataset: {len(df):,} rows with company_code")
    return df


def prepare_company_data(df, company_code):
    """Extract and prepare data for a specific company."""
    company_df = df[df['company_code'] == company_code].copy()
    
    # Sort by date for temporal split
    if 'announce_date' in company_df.columns:
        company_df = company_df.sort_values('announce_date')
    
    return company_df


def get_feature_columns(df):
    """Get list of feature columns (exclude metadata and target)."""
    exclude_cols = [
        'record_id', 'company_code', 'notice_id', 'institution_code',
        'target', 'normalized_bid_rate', 'announce_date', 'is_winner',
        'bid_amt', 'base_amt', 'choose_avg', 'yega_rate', 'margin_from_min'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def encode_full_dataset(df):
    """
    One-hot encode entire dataset ONCE to ensure consistent dimensions.
    
    Returns:
        X: Feature matrix (np.array)
        y: Target array (np.array)
        feature_names: List of feature column names
        company_codes: Series of company codes (for indexing)
        announce_dates: Series of announce dates (for sorting)
    """
    print("Encoding full dataset...")
    
    # Identify categorical columns that need encoding
    categorical_cols = [
        'behavioral_type', 'position_category', 'competition_intensity', 
        'amt_group', 'inst_yega_consistency', 'inst_bidding_frequency', 
        'inst_yega_bias', 'data_window_used'
    ]
    
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
    
    # Columns to exclude from features
    exclude_from_features = [
        'record_id', 'company_code', 'notice_id', 'institution_code', 
        'target', 'announce_date'
    ]
    
    feature_cols = [col for col in df_encoded.columns if col not in exclude_from_features]
    
    # Fill NaN with -999 (XGBoost handles this)
    X = df_encoded[feature_cols].fillna(-999).values
    y = df['target'].values if 'target' in df.columns else None
    
    print(f"  Encoded X: {X.shape}")
    print(f"  Feature count: {len(feature_cols)}")
    
    return X, y, feature_cols, df['company_code'].values, df['announce_date'].values


# ==============================================================================
# MAIN TRAINING FUNCTIONS
# ==============================================================================

def train_company_model(X_train, y_train, company_code):
    """Train XGBoost model for a single company."""
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def compute_conformal_qvalue(model, X_cal, y_cal, alpha=0.1):
    """
    Compute conformal prediction q-value.
    
    q-value is the (1-alpha) quantile of absolute residuals,
    providing a prediction interval: ŷ ± q
    
    Args:
        model: Trained XGBoost model
        X_cal: Calibration features
        y_cal: Calibration targets
        alpha: Coverage level (default 0.1 = 90% coverage)
    
    Returns:
        dict with q_value and calibration statistics
    """
    # Get predictions on calibration set
    y_pred_cal = model.predict(X_cal)
    
    # Compute nonconformity scores (absolute residuals)
    scores = np.abs(y_cal - y_pred_cal)
    
    # Conformal quantile with finite-sample correction
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # Cap at 1.0
    
    q_value = np.quantile(scores, q_level)
    
    # Verify coverage on calibration set
    actual_coverage = np.mean(np.abs(y_cal - y_pred_cal) <= q_value)
    
    return {
        'q_value': float(q_value),
        'alpha': alpha,
        'target_coverage': 1 - alpha,
        'actual_coverage': float(actual_coverage),
        'n_calibration': n,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'max_score': float(np.max(scores))
    }


def train_all_companies(X, y, company_codes, announce_dates, feature_names):
    """
    Train models for all eligible companies.
    
    Pipeline:
      1. Filter to company indices and sort by date
      2. Split into 80% train / 20% calibration (temporal)
      3. Train XGBoost on train set
      4. Compute conformal q-value on calibration set
    
    Args:
        X: Pre-encoded feature matrix (N x D)
        y: Target array (N,)
        company_codes: Company code for each row (N,)
        announce_dates: Announce date for each row (N,)
        feature_names: List of feature column names
    """
    models = {}
    q_values = {}
    training_stats = []
    
    # Get unique companies with sufficient data
    unique_companies, counts = np.unique(company_codes, return_counts=True)
    company_counts = dict(zip(unique_companies, counts))
    eligible_companies = [c for c, n in company_counts.items() if n >= MIN_SAMPLES_PER_COMPANY]
    
    print(f"\n{'='*60}")
    print(f"PER-COMPANY MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Total companies: {len(unique_companies):,}")
    print(f"Eligible (≥{MIN_SAMPLES_PER_COMPANY} samples): {len(eligible_companies):,}")
    print(f"{'='*60}\n")
    
    for i, company_code in enumerate(eligible_companies):
        # Get indices for this company
        company_mask = (company_codes == company_code)
        company_indices = np.where(company_mask)[0]
        
        # Get company data
        X_company = X[company_indices]
        y_company = y[company_indices]
        dates_company = announce_dates[company_indices]
        
        # Sort by date for temporal split
        date_order = np.argsort(dates_company)
        X_company = X_company[date_order]
        y_company = y_company[date_order]
        
        n_samples = len(y_company)
        
        # 3-way temporal split: 60% train, 20% val, 20% cal
        n_cal = max(int(n_samples * CALIBRATION_RATIO), 1)
        n_val = max(int(n_samples * VALIDATION_RATIO), 1)
        n_train = n_samples - n_cal - n_val
        
        # Check minimum requirements
        if n_train < 10:
            continue  # Too few training samples
        
        if n_cal < MIN_CALIBRATION_SAMPLES:
            # Weak calibration warning but proceed
            pass
        
        # Temporal split (oldest -> newest)
        X_train = X_company[:n_train]
        y_train = y_company[:n_train]
        
        X_val = X_company[n_train:n_train+n_val]
        y_val = y_company[n_train:n_train+n_val]
        
        X_cal = X_company[-n_cal:]
        y_cal = y_company[-n_cal:]
        
        # Check for valid data
        if np.isnan(y_train).any() or np.isnan(y_val).any() or np.isnan(y_cal).any():
            continue
        
        # Train model with early stopping on validation set
        try:
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Compute conformal q-value on calibration set
            conformal_result = compute_conformal_qvalue(model, X_cal, y_cal, CONFORMAL_ALPHA)
            
            # Store results
            models[company_code] = model
            q_values[company_code] = conformal_result['q_value']
            
            # Compute metrics
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            train_mae = np.mean(np.abs(y_train - y_pred_train))
            train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
            val_mae = np.mean(np.abs(y_val - y_pred_val))
            val_rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
            
            stats = {
                'company_code': company_code,
                'n_total': n_samples,
                'n_train': n_train,
                'n_validation': n_val,
                'n_calibration': n_cal,
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'q_value': conformal_result['q_value'],
                'actual_coverage': conformal_result['actual_coverage'],
                'mean_cal_score': conformal_result['mean_score'],
                'weak_calibration': n_cal < MIN_CALIBRATION_SAMPLES
            }
            training_stats.append(stats)
            
            # Progress update
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Processed {i+1:,}/{len(eligible_companies):,} companies")
                print(f"    Last: {company_code} | n={n_samples} | val_mae={val_mae:.4f} | q={conformal_result['q_value']:.4f}")
        
        except Exception as e:
            print(f"  ❌ {company_code}: Error - {str(e)[:50]}")
            continue
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models trained: {len(models):,}")
    if len(q_values) > 0:
        print(f"Average q-value: {np.mean(list(q_values.values())):.4f}")
        weak_cal = sum(1 for s in training_stats if s.get('weak_calibration', False))
        if weak_cal > 0:
            print(f"⚠️ Companies with weak calibration (<{MIN_CALIBRATION_SAMPLES} samples): {weak_cal}")
    print(f"{'='*60}\n")
    
    return models, q_values, training_stats


def save_results(models, q_values, training_stats):
    """Save trained models and q-values."""
    root = get_project_root()
    models_dir = root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    models_path = models_dir / 'company_models.pkl'
    joblib.dump(models, models_path)
    print(f"Saved models to {models_path}")
    
    # Save q-values
    qvalues_path = models_dir / 'q_values.pkl'
    joblib.dump(q_values, qvalues_path)
    print(f"Saved q-values to {qvalues_path}")
    
    # Also save q-values as JSON for easy inspection
    qvalues_json_path = models_dir / 'q_values.json'
    with open(qvalues_json_path, 'w') as f:
        json.dump(q_values, f, indent=2)
    print(f"Saved q-values (JSON) to {qvalues_json_path}")
    
    # Save training statistics
    stats_df = pd.DataFrame(training_stats)
    stats_path = models_dir / 'training_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved training stats to {stats_path}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("Q-VALUE SUMMARY")
    print(f"{'='*60}")
    q_arr = np.array(list(q_values.values()))
    print(f"  Count: {len(q_arr)}")
    print(f"  Mean:  {np.mean(q_arr):.4f} (±{np.mean(q_arr)*100:.2f}%)")
    print(f"  Std:   {np.std(q_arr):.4f}")
    print(f"  Min:   {np.min(q_arr):.4f} (most predictable)")
    print(f"  Max:   {np.max(q_arr):.4f} (most volatile)")
    print(f"  25%:   {np.percentile(q_arr, 25):.4f}")
    print(f"  50%:   {np.percentile(q_arr, 50):.4f}")
    print(f"  75%:   {np.percentile(q_arr, 75):.4f}")
    
    return models_path, qvalues_path, stats_path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("Company Model Training with Conformal Calibration")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load pre-computed features with company metadata
    df = load_features()
    
    # Verify company_code is present
    if 'company_code' not in df.columns:
        raise ValueError("company_code column not found in features")
    
    # Encode entire dataset ONCE for consistent dimensions
    X, y, feature_names, company_codes, announce_dates = encode_full_dataset(df)
    
    # Train all eligible companies
    models, q_values, training_stats = train_all_companies(
        X, y, company_codes, announce_dates, feature_names
    )
    
    # Save results
    if len(models) > 0:
        save_results(models, q_values, training_stats)
    else:
        print("⚠️ No models trained - check data requirements")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
