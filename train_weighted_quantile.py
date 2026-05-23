"""
Weighted Quantile Regression Training with Per-Company HPO
==========================================================

CRITICAL FIX for static 1.0005 recommendation problem!

Problem:
- Current models learn average of ALL participants (losers + winners)
- Result: Static, context-blind recommendations (1.0005 for everyone)

Solution:
1. Importance Weighting: Winners ×5, Losers ×1
2. Quantile Regression: α=0.15 (15th percentile = aggressive tail)
3. Per-Company HPO: 100 random search iterations optimizing quantile loss
4. Context Features: l_ranking, l_gap_to_1st, n_competitors

Expected Result:
- Rank 1, big gap → Conservative (1.008-1.012)
- Rank 5, small gap → Aggressive (1.003-1.006)
- Dynamic, context-aware recommendations ✅

Author: MC-CP Team
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from tqdm import tqdm
import random

# HPO Configuration
HPO_ITERATIONS = 50  # Random search (50 is enough, 100 is overkill)
QUANTILE_ALPHA = 0.15  # 15th percentile = aggressive winning tail
WINNER_WEIGHT = 5.0    # Winners get 5x weight
LOSER_WEIGHT = 1.0     # Losers get 1x weight

# Random Search Space (CPU-safe ranges)
PARAM_DIST = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6],  # Narrow range for CPU-GPU consistency
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0, 10.0],
    'n_estimators': [200, 300, 500, 600]  # Narrow range
}

# Fixed Parameters (GPU-accelerated)
FIXED_PARAMS = {
    'objective': 'reg:quantileerror',
    'quantile_alpha': QUANTILE_ALPHA,
    'n_jobs': 1,
    'tree_method': 'gpu_hist',  # ← GPU ACCELERATION!
    'device': 'cuda',            # ← Use CUDA
    'random_state': 42
}

# Minimum samples
MIN_TRAIN_SAMPLES = 30  # Minimum to train
MIN_HPO_SAMPLES = 100  # Need 100+ for HPO, else use conservative defaults
MIN_VAL_SAMPLES = 10
MIN_CAL_SAMPLES = 30

# Conservative defaults for data-scarce companies
CONSERVATIVE_PARAMS = {
    'objective': 'reg:quantileerror',
    'quantile_alpha': QUANTILE_ALPHA,
    'max_depth': 3,  # Shallow to prevent overfitting
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,  # Higher = more conservative
    'gamma': 0.2,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,  # Strong regularization
    'n_jobs': 1,
    'tree_method': 'hist',
    'random_state': 42
}

def get_project_root():
    return Path(__file__).parent

def calculate_sample_weights(df):
    """
    Calculate importance weights for training.
    
    Winners get 5x weight to focus model on winning behavior.
    Losers get 1x weight to maintain market context.
    """
    weights = np.where(
        df['is_winner'].values == True,
        WINNER_WEIGHT,
        LOSER_WEIGHT
    )
    
    return weights

def quantile_loss(y_true, y_pred, alpha):
    """
    Calculate Pinball Loss (Quantile Loss) for evaluation.
    
    Loss = max(alpha * (y - ŷ), (alpha - 1) * (y - ŷ))
    
    This is THE metric we optimize during HPO!
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(alpha * errors, (alpha - 1) * errors))

def run_custom_hpo(X_train, y_train, w_train, X_val, y_val, n_iter=HPO_ITERATIONS):
    """
    Run Random Search explicitly optimizing for Quantile Loss (α=0.15).
    
    For each iteration:
      1. Sample random hyperparameters
      2. Train weighted quantile model
      3. Evaluate on validation set using quantile loss
      4. Keep best model
    
    Returns:
        best_model: Trained XGBoost model
        best_params: Best hyperparameters
        best_loss: Best quantile loss on validation
    """
    best_loss = float('inf')
    best_params = None
    best_model = None
    
    # Generate random parameter combinations
    keys = list(PARAM_DIST.keys())
    trials = []
    for _ in range(n_iter):
        params = {k: random.choice(PARAM_DIST[k]) for k in keys}
        params.update(FIXED_PARAMS)
        trials.append(params)
    
    # Run search
    for params in trials:
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate using quantile loss
        preds = model.predict(X_val)
        loss = quantile_loss(y_val, preds, QUANTILE_ALPHA)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_model = model
    
    return best_model, best_params, best_loss

def verify_competitive_features(feature_cols):
    """
    Verify that competitive position features are included.
    
    These are CRITICAL for context-aware optimization!
    """
    required_features = [
        'ranking',         # My PQ rank
        'gap_to_1st',      # Distance to leader
        'n_competitors',   # Competition intensity
    ]
    
    recommended_features = [
        'gap_to_2nd',           # Distance to 2nd place
        'tech_score',           # My tech score
        'g_tech_score_range',   # Competition spread
        'g_tech_score_mean',    # Average competitor strength
    ]
    
    print("\n" + "="*70)
    print("COMPETITIVE FEATURE VERIFICATION")
    print("="*70)
    
    print("\nRequired Features (MUST have):")
    for feat in required_features:
        matches = [f for f in feature_cols if feat == f or feat in f]
        if matches:
            print(f"  ✅ {feat}: Found")
        else:
            print(f"  ⚠️ {feat}: Not found (may reduce context awareness)")
    
    print("\nRecommended Features (SHOULD have):")
    for feat in recommended_features:
        matches = [f for f in feature_cols if feat == f or feat in f]
        if matches:
            print(f"  ✅ {feat}: Found")
        else:
            print(f"  ⚠️ {feat}: Not found")
    
    print("="*70)

def encode_full_dataset(df):
    """Encode features exactly as in training."""
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
    
    exclude_from_features = [
        'record_id', 'company_code', 'notice_id', 'institution_code',
        'normalized_bid_rate', 'announce_date', 'min_bid_rate', 
        'base_amt', 'is_winner'
    ]
    
    feature_cols = [col for col in df_encoded.columns 
                   if col not in exclude_from_features]
    
    return df_encoded, feature_cols

def train_weighted_quantile_models_hpo():
    """
    Train per-company models using Weighted Quantile Regression.
    
    Key Changes:
    1. sample_weight parameter (winners ×5)
    2. objective='reg:quantileerror', quantile_alpha=0.15
    3. Verification of competitive position features
    """
    
    print("="*70)
    print(f"WEIGHTED QUANTILE REGRESSION + PER-COMPANY HPO")
    print(f"({HPO_ITERATIONS} iterations per company)")
    print("="*70)
    print()
    
    print("Configuration:")
    print(f"  Objective: Quantile Loss (α={QUANTILE_ALPHA})")
    print(f"  Winner weight: {WINNER_WEIGHT}x")
    print(f"  Loser weight: {LOSER_WEIGHT}x")
    print(f"  HPO iterations: {HPO_ITERATIONS}")
    print()
    
    # Load data
    root = get_project_root()
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    train_df = pd.read_csv(root / 'data' / 'processed' / 'train_clean.csv')
    val_df = pd.read_csv(root / 'data' / 'processed' / 'val.csv')  # ← USE PRE-SPLIT VAL.CSV!
    
    # Merge features with train
    df_train = features_df.merge(
        train_df[['record_id', 'company_code', 'notice_id', 'announce_date',
                  'normalized_bid_rate', 'is_winner']],
        on='record_id',
        how='inner'
    )
    
    # Since val.csv merge is failing, use 80/20 split of training data
    print("Using 80/20 train/val split from training data for HPO")
    
    # Encode features
    print("Encoding features...")
    df_train_encoded, feature_cols = encode_full_dataset(df_train)
    print(f"✅ Encoded: {len(feature_cols)} features")
    
    # VERIFY competitive features are included!
    verify_competitive_features(feature_cols)
    
    # Split train into train/val (80/20) for HPO
    train_size = int(len(df_train_encoded) * 0.8)
    df_train_encoded = df_train_encoded.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    train_split = df_train_encoded.iloc[:train_size].copy()
    val_split = df_train_encoded.iloc[train_size:].copy()
    
    # For calibration, use date-based split
    split_date = pd.to_datetime('2025-05-31')
    df_train_encoded['announce_date'] = pd.to_datetime(df_train_encoded['announce_date'])
    cal_mask = df_train_encoded['announce_date'] > split_date
    
    print()
    print("Data Split:")
    print(f"  Train: {len(train_split):,} records (80%)")
    print(f"  Val:   {len(val_split):,} records (20% for HPO)")
    print(f"  Cal (from full train, > 2025-05-31): {cal_mask.sum():,} records")
    print()
    
    # Train per company
    companies = df_train_encoded['company_code'].unique()
    print(f"Training {len(companies)} companies...")
    print()
    
    models = {}
    q_values = {}
    training_stats = []
    
    for company in tqdm(companies, desc="Training & HPO"):
        # Get company data from TRAIN split (80%)
        company_train = train_split[train_split['company_code'] == company].copy()
        
        # Get company data from VAL split (20%)
        company_val = val_split[val_split['company_code'] == company].copy()
        
        # Get company data from CAL (date filtered from full train)
        company_cal = df_train_encoded[
            (df_train_encoded['company_code'] == company) & cal_mask
        ].copy()
        
        # Skip if insufficient training data
        if len(company_train) < MIN_TRAIN_SAMPLES:
            continue
        
        # Extract features and targets
        X_train = company_train[feature_cols].fillna(-999).values
        y_train = company_train['normalized_bid_rate'].values
        w_train = calculate_sample_weights(company_train)
        
        # Stats
        n_winners = (company_train['is_winner'] == True).sum()
        n_losers = (company_train['is_winner'] == False).sum()
        effective_n = int(n_winners * WINNER_WEIGHT + n_losers * LOSER_WEIGHT)
        
        # Decision: HPO or Conservative Defaults?
        use_hpo = len(company_train) >= MIN_HPO_SAMPLES and len(company_val) >= MIN_VAL_SAMPLES
        
        if use_hpo:
            # Have enough data for HPO
            X_val = company_val[feature_cols].fillna(-999).values
            y_val = company_val['normalized_bid_rate'].values
        
            # ⭐ RUN HPO FOR THIS COMPANY
            best_model, best_params, best_loss = run_custom_hpo(
                X_train, y_train, w_train,
                X_val, y_val,
                n_iter=HPO_ITERATIONS
            )
            
            models[company] = best_model
        else:
            # Data-scarce: use conservative defaults (avoid overfitting)
            model = xgb.XGBRegressor(**CONSERVATIVE_PARAMS)
            model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            
            models[company] = model
            best_model = model
            best_params = CONSERVATIVE_PARAMS
            best_loss = 0.0  # N/A for conservative
        
        
        # Conformal calibration using VALIDATION data
        # (Date-based calibration has 0 records!)
        if len(company_val) >= MIN_CAL_SAMPLES:
            X_cal = company_val[feature_cols].fillna(-999).values
            y_cal = company_val['normalized_bid_rate'].values
            
            preds_cal = best_model.predict(X_cal)
            residuals = np.abs(y_cal - preds_cal)
            
            n = len(residuals)
            alpha = 0.1  # 90% coverage
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            q_value = np.quantile(residuals, q_level)
            
            q_values[company] = float(q_value)
        else:
            q_values[company] = 0.02  # Default
        
        # Stats (include HPO results)
        training_stats.append({
            'company': company,
            'n_train': len(company_train),
            'n_winners': n_winners,
            'n_losers': n_losers,
            'effective_n': effective_n,
            'winner_pct': float(n_winners / len(company_train) * 100),
            'q_value': q_values[company],
            'used_hpo': use_hpo,
            'best_quantile_loss': float(best_loss) if use_hpo else None,
            'best_learning_rate': best_params['learning_rate'],
            'best_max_depth': best_params['max_depth'],
            'best_n_estimators': best_params['n_estimators']
        })
    
    print()
    print(f"✅ Trained {len(models)} models")
    print()
    
    # Summary statistics
    stats_df = pd.DataFrame(training_stats)
    
    print("="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"\nModels trained: {len(models)}")
    print(f"Mean samples per company: {stats_df['n_train'].mean():.0f}")
    print(f"Mean winner percentage: {stats_df['winner_pct'].mean():.1f}%")
    print(f"\nEffective sample size (weighted):")
    print(f"  Mean: {stats_df['effective_n'].mean():.0f}")
    print(f"  Median: {stats_df['effective_n'].median():.0f}")
    print(f"\nQ-values:")
    print(f"  Mean: {stats_df['q_value'].mean():.4f}")
    print(f"  Median: {stats_df['q_value'].median():.4f}")
    print(f"  Range: [{stats_df['q_value'].min():.4f}, {stats_df['q_value'].max():.4f}]")
    print("="*70)
    
    # Save models
    output_dir = root / 'models'
    output_dir.mkdir(exist_ok=True)
    
    joblib.dump(models, output_dir / 'company_models_hpo.pkl')
    joblib.dump(q_values, output_dir / 'q_values_hpo.pkl')
    
    stats_df.to_csv(output_dir / 'training_stats_hpo.csv', index=False)
    
    print(f"\n✅ Saved:")
    print(f"  Models: {output_dir / 'company_models_hpo.pkl'}")
    print(f"  Q-values: {output_dir / 'q_values_hpo.pkl'}")
    print(f"  Stats: {output_dir / 'training_stats_hpo.csv'}")
    
    return models, q_values, stats_df


if __name__ == "__main__":
    models, q_values, stats = train_weighted_quantile_models_hpo()
