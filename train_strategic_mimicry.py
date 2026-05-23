"""
Weighted MSE Training with Per-Company HPO (Strategic Mimicry)
==============================================================

FINAL RECIPE for Context-Aware Bidding Strategy.

Philosophy:
- "Don't just bid low (Quantile). Mimic the Winner's Strategy (Weighted MSE)."
- If Rank 1 wins at 1.005, learn 1.005.
- If Rank 10 wins at 0.990, learn 0.990.

Solution:
1. Objective: reg:squarederror (Targeting the exact winning price)
2. Importance Weighting: Winners ×10 (Signal), Losers ×1 (Context)
3. Per-Company HPO: Optimizing Weighted RMSE
4. Context Features: l_ranking, l_gap_to_1st, n_competitors

Expected Result:
- Leader (rank=1, gap>5) → Conservative bid (1.008-1.012)
- Underdog (rank=10, gap<2) → Aggressive bid (0.985-0.995)
- Context-aware, strategic recommendations!

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

# =============================================================================
# CONFIGURATION - The Recipe for Strategic Mimicry
# =============================================================================

HPO_ITERATIONS = 50       # 50 iterations for Random Search
WINNER_WEIGHT = 10.0      # ← Winners get 10x weight (SIGNAL BOOST!)
LOSER_WEIGHT = 1.0        # ← Losers get 1x weight (context only)

# Random Search Space (Expanded for Context Learning)
PARAM_DIST = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6, 7, 8],  # ← Deeper trees for complex context
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0, 10.0],
    'n_estimators': [300, 500, 700, 1000]  # More estimators, let model decide
}

# Fixed Parameters (GPU Optimized)
FIXED_PARAMS = {
    'objective': 'reg:squarederror',  # ← MSE: Mimic exact winner value!
    'n_jobs': 1,
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'random_state': 42
}

# Minimum Samples
MIN_TRAIN_SAMPLES = 30
MIN_HPO_SAMPLES = 100   # Need 100+ for meaningful HPO
MIN_VAL_SAMPLES = 10
MIN_CAL_SAMPLES = 30

# Conservative Defaults (for data-scarce companies)
CONSERVATIVE_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_lambda': 5.0,
    'n_jobs': 1,
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'random_state': 42
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_project_root():
    return Path(__file__).parent

def calculate_sample_weights(df):
    """
    Calculate importance weights.
    Winners are 10x more important - they are the SIGNAL!
    Losers provide context but are mostly noise.
    """
    weights = np.where(
        df['is_winner'].values == True,
        WINNER_WEIGHT,
        LOSER_WEIGHT
    )
    return weights

def calculate_weighted_rmse(y_true, y_pred, weights):
    """
    Custom Metric: Weighted RMSE.
    
    We care MUCH more about errors on Winner samples.
    This is what we optimize during HPO.
    """
    mse = np.average((y_true - y_pred) ** 2, weights=weights)
    return np.sqrt(mse)

def run_custom_hpo(X_train, y_train, w_train, X_val, y_val, w_val, n_iter=HPO_ITERATIONS):
    """
    Run Random Search optimizing Weighted RMSE.
    
    For each iteration:
      1. Sample random hyperparameters
      2. Train weighted MSE model
      3. Evaluate on validation using Weighted RMSE
      4. Keep best model
    
    Returns:
        best_model, best_params, best_weighted_rmse
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
    
    # Search loop
    for params in trials:
        model = xgb.XGBRegressor(**params)
        
        # Fit with importance weights
        # XGBoost will minimize WEIGHTED MSE automatically!
        model.fit(
            X_train, y_train,
            sample_weight=w_train,  # ← Winners ×10 during training
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate: Calculate Weighted RMSE on Validation
        preds = model.predict(X_val)
        loss = calculate_weighted_rmse(y_val, preds, w_val)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_model = model
            
    return best_model, best_params, best_loss

def verify_context_features(feature_cols):
    """
    Verify that context features exist.
    These are CRITICAL for strategic mimicry!
    """
    required = [
        ('ranking', 'l_ranking'),      # My rank
        ('gap_to_1st', 'l_gap_to_1st'),  # Distance to leader
        ('n_competitors', 'n_competitors'),  # Competition size
    ]
    
    print("\n" + "="*70)
    print("CONTEXT FEATURE VERIFICATION")
    print("="*70)
    print("\nRequired for Strategic Mimicry:")
    
    for display_name, search_pattern in required:
        matches = [f for f in feature_cols if search_pattern in f]
        if matches:
            print(f"  ✅ {display_name}: {matches[0]}")
        else:
            print(f"  ⚠️ {display_name}: NOT FOUND (context awareness reduced)")
    
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

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_strategic_models():
    """
    Train models using Strategic Mimicry approach.
    
    Core Philosophy:
    - Learn EXACTLY what winners bid in each context
    - Winners are 10x more important than losers
    - HPO optimizes "how well we mimic winners"
    """
    
    print("="*70)
    print("STRATEGIC MIMICRY TRAINING (Weighted MSE)")
    print("="*70)
    print()
    
    print("Configuration:")
    print(f"  Objective: MSE (Exact Winner Mimicry)")
    print(f"  Winner weight: {WINNER_WEIGHT}x")
    print(f"  Loser weight: {LOSER_WEIGHT}x")
    print(f"  HPO metric: Weighted RMSE")
    print(f"  HPO iterations: {HPO_ITERATIONS}")
    print()
    
    # Load Data
    root = get_project_root()
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    train_df = pd.read_csv(root / 'data' / 'processed' / 'train_clean.csv')
    
    # Merge
    df_train = features_df.merge(
        train_df[['record_id', 'company_code', 'notice_id', 'announce_date',
                  'normalized_bid_rate', 'is_winner']],
        on='record_id',
        how='inner'
    )
    
    print(f"Loaded {len(df_train):,} training records")
    print(f"  Winners: {(df_train['is_winner']==True).sum():,} ({(df_train['is_winner']==True).mean()*100:.1f}%)")
    print(f"  Losers: {(df_train['is_winner']==False).sum():,} ({(df_train['is_winner']==False).mean()*100:.1f}%)")
    print()
    
    # Encode features
    print("Encoding features...")
    df_encoded, feature_cols = encode_full_dataset(df_train)
    print(f"✅ Encoded: {len(feature_cols)} features")
    
    # Verify context features
    verify_context_features(feature_cols)
    
    # 80/20 Split for HPO
    print("\nUsing 80/20 train/val split for HPO")
    df_encoded = df_encoded.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_encoded) * 0.8)
    train_split = df_encoded.iloc[:split_idx].copy()
    val_split = df_encoded.iloc[split_idx:].copy()
    
    # Calibration data (date-based)
    df_encoded['announce_date'] = pd.to_datetime(df_encoded['announce_date'])
    cal_mask = df_encoded['announce_date'] > pd.to_datetime('2025-05-31')
    
    print()
    print("Data Split:")
    print(f"  Train: {len(train_split):,} records (80%)")
    print(f"  Val:   {len(val_split):,} records (20%)")
    print(f"  Cal:   {cal_mask.sum():,} records (for q-values)")
    print()
    
    # Train per company
    companies = df_encoded['company_code'].unique()
    print(f"Training {len(companies)} companies...")
    print()
    
    models = {}
    q_values = {}
    training_stats = []
    
    for company in tqdm(companies, desc="Training & HPO"):
        # Get company data
        c_train = train_split[train_split['company_code'] == company].copy()
        c_val = val_split[val_split['company_code'] == company].copy()
        c_cal = df_encoded[(df_encoded['company_code'] == company) & cal_mask].copy()
        
        if len(c_train) < MIN_TRAIN_SAMPLES:
            continue
        
        # Prepare training data
        X_train = c_train[feature_cols].fillna(-999).values
        y_train = c_train['normalized_bid_rate'].values
        w_train = calculate_sample_weights(c_train)
        
        # Stats
        n_winners = (c_train['is_winner'] == True).sum()
        n_losers = (c_train['is_winner'] == False).sum()
        effective_n = int(n_winners * WINNER_WEIGHT + n_losers * LOSER_WEIGHT)
        
        # Decide: HPO or Conservative
        use_hpo = len(c_train) >= MIN_HPO_SAMPLES and len(c_val) >= MIN_VAL_SAMPLES
        
        if use_hpo:
            # Enough data for HPO
            X_val = c_val[feature_cols].fillna(-999).values
            y_val = c_val['normalized_bid_rate'].values
            w_val = calculate_sample_weights(c_val)
            
            # RUN HPO (Optimizing Weighted RMSE)
            best_model, best_params, best_loss = run_custom_hpo(
                X_train, y_train, w_train,
                X_val, y_val, w_val,
                n_iter=HPO_ITERATIONS
            )
            models[company] = best_model
        else:
            # Data-scarce: use conservative defaults
            model = xgb.XGBRegressor(**CONSERVATIVE_PARAMS)
            model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            
            models[company] = model
            best_model = model
            best_params = CONSERVATIVE_PARAMS
            best_loss = 0.0
        
        # Calculate Q-Value for Simulation Noise
        # Even with MSE, we need q-value for conformal prediction
        if len(c_cal) >= MIN_CAL_SAMPLES:
            X_cal = c_cal[feature_cols].fillna(-999).values
            y_cal = c_cal['normalized_bid_rate'].values
            preds = best_model.predict(X_cal)
            residuals = np.abs(y_cal - preds)
            q_values[company] = float(np.quantile(residuals, 0.90))
        else:
            q_values[company] = 0.015
        
        # Stats
        training_stats.append({
            'company': company,
            'n_train': len(c_train),
            'n_winners': n_winners,
            'n_losers': n_losers,
            'effective_n': effective_n,
            'winner_pct': float(n_winners / len(c_train) * 100),
            'q_value': q_values[company],
            'used_hpo': use_hpo,
            'best_weighted_rmse': float(best_loss) if use_hpo else None,
            'best_learning_rate': best_params['learning_rate'],
            'best_max_depth': best_params['max_depth'],
            'best_n_estimators': best_params['n_estimators']
        })
    
    print()
    print(f"✅ Trained {len(models)} models")
    print()
    
    # Summary
    stats_df = pd.DataFrame(training_stats)
    
    print("="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"\nModels trained: {len(models)}")
    print(f"Used HPO: {stats_df['used_hpo'].sum()}")
    print(f"Conservative: {(~stats_df['used_hpo']).sum()}")
    
    if stats_df['used_hpo'].sum() > 0:
        hpo_stats = stats_df[stats_df['used_hpo'] == True]
        print(f"\nHPO Hyperparameter Diversity:")
        print(f"  Learning rates: {hpo_stats['best_learning_rate'].nunique()} unique")
        print(f"    Range: [{hpo_stats['best_learning_rate'].min():.3f}, {hpo_stats['best_learning_rate'].max():.3f}]")
        print(f"  Max depths: {hpo_stats['best_max_depth'].nunique()} unique")
        print(f"    Range: [{int(hpo_stats['best_max_depth'].min())}, {int(hpo_stats['best_max_depth'].max())}]")
        print(f"  Weighted RMSE:")
        print(f"    Mean: {hpo_stats['best_weighted_rmse'].mean():.6f}")
        print(f"    Range: [{hpo_stats['best_weighted_rmse'].min():.6f}, {hpo_stats['best_weighted_rmse'].max():.6f}]")
    
    print("="*70)
    
    # Save
    out_dir = root / 'models'
    out_dir.mkdir(exist_ok=True)
    
    joblib.dump(models, out_dir / 'company_models_strategic.pkl')
    joblib.dump(q_values, out_dir / 'q_values_strategic.pkl')
    stats_df.to_csv(out_dir / 'training_stats_strategic.csv', index=False)
    
    print(f"\n✅ Saved:")
    print(f"  Models: {out_dir / 'company_models_strategic.pkl'}")
    print(f"  Q-values: {out_dir / 'q_values_strategic.pkl'}")
    print(f"  Stats: {out_dir / 'training_stats_strategic.csv'}")
    
    return models, q_values, stats_df


if __name__ == "__main__":
    models, q_values, stats = train_strategic_models()
