"""
Validation Backtest: Apply model to val.csv (Out-of-Sample)
============================================================

Uses val.csv instead of train_clean.csv for true out-of-sample testing.
Features are generated on-the-fly since val doesn't have pre-computed features.

Author: Harim Choi
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SIMULATION_ITERATIONS = 500_000  # MC iterations
GRID_START = 0.975
GRID_END = 1.025
GRID_STEP = 0.0005

# =============================================================================
# PATHS
# =============================================================================

def get_project_root():
    return Path(__file__).parent

ROOT = get_project_root()
MODEL_PATH = ROOT / 'models' / 'company_models_hpo.pkl'
Q_VALUE_PATH = ROOT / 'models' / 'q_values_hpo.pkl'
VAL_DATA = ROOT / 'data' / 'processed' / 'val.csv'  # ← Use val.csv!
VAL_FEATURES = ROOT / 'analysis_results' / 'feature_sample_val.csv'  # ← Full features!
TRAIN_DATA = ROOT / 'data' / 'processed' / 'train_clean.csv'  # For feature reference
FEATURE_DATA = ROOT / 'analysis_results' / 'feature_sample.csv'
INST_PROFILE_PATH = ROOT / 'data' / 'processed' / 'inst_profile_static.csv'

# =============================================================================
# LOAD FUNCTIONS
# =============================================================================

def load_institution_profiles():
    """Load institution yega profiles."""
    try:
        df = pd.read_csv(INST_PROFILE_PATH)
        profiles = {}
        for _, row in df.iterrows():
            inst = row['inst_code']
            profiles[inst] = {
                'mu': row['mean_yega_rate_all'],
                'sigma': row['std_yega_rate_all']
            }
        return profiles
    except Exception as e:
        print(f"Warning: Could not load inst profiles: {e}")
        return {}

def sample_yega_korean_style(mu, sigma, n_samples=4):
    """Korean PQ style: pick 4 random, average them."""
    samples = np.random.normal(mu, sigma / 2, n_samples)
    samples = np.clip(samples, 0.97, 1.03)
    return np.mean(samples)

# =============================================================================
# SIMPLE FEATURE ENGINEERING (For val.csv)
# =============================================================================

def generate_simple_features(df):
    """
    Generate minimal features for validation.
    Uses only columns available in val.csv.
    """
    df = df.copy()
    
    # Basic numeric features (already in val.csv)
    feature_cols = [
        'ranking', 'tech_score', 'gap_to_1st', 'gap_to_2nd',
        'base_amt', 'min_bid_rate', 'n_competitors'
    ]
    
    # Derived features
    df['log_base_amt'] = np.log1p(df['base_amt'])
    df['tech_score_normalized'] = df['tech_score'] / 100
    df['is_leader'] = (df['ranking'] == 1).astype(int)
    df['is_underdog'] = (df['ranking'] >= 3).astype(int)
    
    # Competition features
    df['competition_intensity'] = pd.cut(df['n_competitors'], 
        bins=[-1, 3, 5, 8, 100], labels=['low', 'medium', 'high', 'very_high'])
    
    # Amount group
    df['amt_group'] = pd.cut(df['base_amt'],
        bins=[-1, 100_000_000, 500_000_000, 1_000_000_000, float('inf')],
        labels=['small', 'medium', 'large', 'xlarge'])
    
    return df

def encode_features_simple(df):
    """Simple feature encoding for val data."""
    categorical_cols = ['competition_intensity', 'amt_group']
    
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
    
    exclude = [
        'record_id', 'company_code', 'notice_id', 'institution_code',
        'normalized_bid_rate', 'announce_date', 'min_bid_rate',
        'base_amt', 'is_winner', 'choose_avg', 'bid_id', 'pq_date',
        'bid_date', 'winner_code', 'winner_normalized_rate', 'winner_margin',
        'bid_amt', 'bid_rate_diff', 'norm_bid_margin', 'margin_from_min',
        'normal_bid_line'
    ]
    
    feature_cols = [col for col in df_encoded.columns 
                    if col not in exclude and df_encoded[col].dtype in ['int64', 'float64', 'uint8']]
    
    return df_encoded, feature_cols

# =============================================================================
# VECTORIZED SIMULATION (Same as backtest_simulation.py)
# =============================================================================

def simulate_auction_vectorized(
    competitors_df, target_company, inst_profile,
    models, q_values, feature_cols, base_amt, my_min_bid_rate
):
    """Vectorized MC simulation."""
    mu = inst_profile.get('mu', 1.00)
    sigma = inst_profile.get('sigma', 0.02)
    n_iters = SIMULATION_ITERATIONS
    
    # 1. Yega Sampling
    sim_yegas = np.random.normal(mu, sigma / 2, n_iters)
    sim_yegas = np.clip(sim_yegas, 0.97, 1.03)
    
    # 2. Competitor Bid Prediction
    comp_bids_norm = []
    comp_min_rates = []
    
    for _, row in competitors_df.iterrows():
        comp_code = row['company_code']
        min_rate = row['min_bid_rate']
        
        if comp_code in models:
            model = models[comp_code]
            q = q_values.get(comp_code, 0.02)
            feat = row[feature_cols].fillna(-999).values.reshape(1, -1)
            try:
                pred = model.predict(feat)[0]
            except:
                pred = 1.000
                q = 0.03
        else:
            pred = 1.000
            q = 0.03
        
        noise = np.random.uniform(-q, q, n_iters)
        sim_norm = pred + noise
        sim_norm = np.clip(sim_norm, 0.90, 1.10)
        
        comp_bids_norm.append(sim_norm)
        comp_min_rates.append(min_rate)
    
    if not comp_bids_norm:
        return GRID_START, 1.0
    
    comp_bids_norm = np.array(comp_bids_norm)
    comp_min_rates = np.array(comp_min_rates).reshape(-1, 1)
    
    # 3. Vectorized Korean PQ Logic
    sim_yejeong_prices = base_amt * sim_yegas
    comp_bid_amts = base_amt * comp_min_rates * comp_bids_norm
    comp_thresholds = sim_yejeong_prices * comp_min_rates
    
    is_valid_comp = comp_bid_amts >= comp_thresholds
    valid_comp_amts = np.where(is_valid_comp, comp_bid_amts, np.inf)
    best_comp_prices = np.min(valid_comp_amts, axis=0)
    
    # 4. Grid Search
    candidate_bids = np.arange(GRID_START, GRID_END + GRID_STEP, GRID_STEP)
    win_probs = []
    
    my_thresholds = sim_yejeong_prices * my_min_bid_rate
    
    for my_norm_bid in candidate_bids:
        my_bid_amt = base_amt * my_min_bid_rate * my_norm_bid
        am_i_valid = my_bid_amt >= my_thresholds
        beat_competitors = my_bid_amt < best_comp_prices
        wins = am_i_valid & beat_competitors
        win_probs.append(np.mean(wins))
    
    best_idx = np.argmax(win_probs)
    return candidate_bids[best_idx], win_probs[best_idx]

# =============================================================================
# WINNER DETERMINATION
# =============================================================================

def determine_winner_actual(competitors_df, base_amt, yega_rate, my_bid_info=None):
    """Determine winner using ACTUAL data."""
    yejeong_price = base_amt * yega_rate
    candidates = []
    
    for _, row in competitors_df.iterrows():
        threshold = yejeong_price * row['min_bid_rate']
        bid_amt = base_amt * row['min_bid_rate'] * row['normalized_bid_rate']
        if bid_amt >= threshold:
            candidates.append((row['company_code'], bid_amt))
    
    if my_bid_info:
        my_norm, my_min = my_bid_info
        my_thresh = yejeong_price * my_min
        my_amt = base_amt * my_min * my_norm
        if my_amt >= my_thresh:
            candidates.append(('TARGET', my_amt))
    
    if not candidates:
        return None
    
    winner_code, _ = min(candidates, key=lambda x: x[1])
    return winner_code

# =============================================================================
# MAIN VALIDATION BACKTEST
# =============================================================================

def run_validation_backtest(target_company, max_auctions=None):
    """Run backtest on val.csv (out-of-sample)."""
    print("="*70)
    print(f"VALIDATION BACKTEST (OUT-OF-SAMPLE): {target_company}")
    print("="*70)
    print()
    
    # Load models
    print("Loading models...")
    models = joblib.load(MODEL_PATH)
    q_values = joblib.load(Q_VALUE_PATH)
    print(f"  Models: {len(models)}")
    
    # Load inst profiles
    inst_profiles = load_institution_profiles()
    print(f"  Inst profiles: {len(inst_profiles)}")
    
    # Load val.csv
    print("Loading val.csv (out-of-sample)...")
    val_df = pd.read_csv(VAL_DATA)
    print(f"  Val records: {len(val_df):,}")
    
    # Load precomputed features (78 features!)
    print("Loading precomputed features...")
    feature_df = pd.read_csv(VAL_FEATURES)
    print(f"  Feature records: {len(feature_df):,}")
    
    # Merge val_df with features on record_id
    val_encoded = feature_df.merge(
        val_df[['record_id', 'company_code', 'notice_id', 'institution_code',
                'normalized_bid_rate', 'is_winner', 'min_bid_rate', 'base_amt',
                'choose_avg', 'announce_date']],
        on='record_id',
        how='inner'
    )
    print(f"  Merged records: {len(val_encoded):,}")
    
    # Encode categorical features
    categorical_cols = [
        'behavioral_type', 'position_category', 'competition_intensity',
        'amt_group', 'inst_yega_consistency', 'inst_bidding_frequency',
        'inst_yega_bias', 'data_window_used'
    ]
    
    for col in categorical_cols:
        if col in val_encoded.columns:
            dummies = pd.get_dummies(val_encoded[col], prefix=col, dummy_na=True)
            val_encoded = pd.concat([val_encoded, dummies], axis=1)
            val_encoded = val_encoded.drop(columns=[col])
    
    # Get feature columns
    exclude = [
        'record_id', 'company_code', 'notice_id', 'institution_code',
        'normalized_bid_rate', 'announce_date', 'min_bid_rate',
        'base_amt', 'is_winner', 'choose_avg', 'target'
    ]
    feature_cols = [col for col in val_encoded.columns 
                    if col not in exclude and val_encoded[col].dtype in ['int64', 'float64', 'uint8', 'bool']]
    print(f"  Features: {len(feature_cols)}")
    print()
    
    # Filter target company
    my_auctions = val_encoded[val_encoded['company_code'] == target_company].copy()
    my_auctions = my_auctions.sort_values('announce_date')
    
    if max_auctions:
        my_auctions = my_auctions.head(max_auctions)
    
    print(f"Target auctions: {len(my_auctions)}")
    print()
    
    # Backtest
    results = []
    
    print("Running validation backtests...")
    for idx, row in tqdm(my_auctions.iterrows(), total=len(my_auctions), desc="Auctions"):
        notice_id = row['notice_id']
        base_amt = row['base_amt']
        my_min_bid = row['min_bid_rate']
        actual_my_bid = row['normalized_bid_rate']
        actual_win = row['is_winner']
        choose_avg = row['choose_avg']
        
        # ACTUAL yega
        if choose_avg > 0 and base_amt > 0:
            actual_yega_rate = choose_avg / base_amt
        else:
            actual_yega_rate = 1.00
        
        # Competitors
        competitors = val_encoded[
            (val_encoded['notice_id'] == notice_id) &
            (val_encoded['company_code'] != target_company)
        ].copy()
        
        if len(competitors) == 0:
            continue
        
        # Inst profile
        inst_code = row['institution_code']
        profile = inst_profiles.get(inst_code, {'mu': 1.0, 'sigma': 0.02})
        
        # 1. RUN SIMULATION
        try:
            optimal_bid, optimal_prob = simulate_auction_vectorized(
                competitors, target_company, profile,
                models, q_values, feature_cols, base_amt, my_min_bid
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        # 2. VALIDATION
        cf_winner = determine_winner_actual(
            competitors, base_amt, actual_yega_rate,
            my_bid_info=(optimal_bid, my_min_bid)
        )
        counterfactual_win = (cf_winner == 'TARGET')
        
        results.append({
            'notice_id': notice_id,
            'optimal_bid': optimal_bid,
            'optimal_prob': optimal_prob,
            'actual_bid': actual_my_bid,
            'actual_win': actual_win,
            'counterfactual_win': counterfactual_win,
            'actual_yega': actual_yega_rate,
            'n_competitors': len(competitors)
        })
    
    # Analysis
    results_df = pd.DataFrame(results)
    
    print()
    print("="*70)
    print("VALIDATION BACKTEST RESULTS (OUT-OF-SAMPLE)")
    print("="*70)
    print()
    
    print(f"Auctions tested: {len(results_df)}")
    print()
    
    act_rate = results_df['actual_win'].mean() * 100
    mod_rate = results_df['counterfactual_win'].mean() * 100
    lift = mod_rate - act_rate
    
    print(f"Actual Win Rate: {act_rate:.2f}%")
    print(f"Model Win Rate:  {mod_rate:.2f}%")
    print(f"Improvement:     {lift:+.2f}%")
    print()
    
    avg_adj = (results_df['optimal_bid'] - results_df['actual_bid']).mean()
    print(f"Avg Bid Adjustment: {avg_adj:.5f}")
    print(f"Avg Optimal Prob: {results_df['optimal_prob'].mean():.2%}")
    print()
    
    # Save
    out_dir = ROOT / 'analysis_results'
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / f'validation_backtest_{target_company}.csv', index=False)
    print(f"Saved: {out_dir / f'validation_backtest_{target_company}.csv'}")
    print("="*70)
    
    return results_df

# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    results = run_validation_backtest('C0027', max_auctions=400)
