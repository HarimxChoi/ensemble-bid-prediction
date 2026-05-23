"""
Walk-Forward Backtesting Simulation (TOLERANCE VERSION)
========================================================

Simple modification: Instead of just picking argmax,
pick the bid FURTHEST from 1.0 among those within TOLERANCE of max probability.

Based on: backtest_simulation.py (original)
Author: Harim Choi
Date: 2025-12-20
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

SIMULATION_ITERATIONS = 500_000 
GRID_START = 0.97
GRID_END = 1.03
GRID_STEP = 0.0005
TOLERANCE = 0.02  # 2% probability tolerance

# =============================================================================
# PATHS
# =============================================================================

def get_project_root():
    return Path(__file__).parent

ROOT = get_project_root()
MODEL_PATH = ROOT / 'models' / 'company_models_hpo.pkl'
Q_VALUE_PATH = ROOT / 'models' / 'q_values_hpo.pkl'
TRAIN_DATA = ROOT / 'data' / 'processed' / 'train_clean.csv'
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

# =============================================================================
# TOLERANCE-BASED BID SELECTION
# =============================================================================

def select_bid_with_tolerance(candidate_bids, win_probs):
    """
    Select bid using TOLERANCE approach:
    Among bids within TOLERANCE of max prob, pick the one furthest from 1.0
    """
    win_probs = np.array(win_probs)
    max_prob = np.max(win_probs)
    
    # Filter candidates within tolerance of max
    threshold = max_prob - TOLERANCE
    valid_mask = win_probs >= threshold
    valid_bids = candidate_bids[valid_mask]
    valid_probs = win_probs[valid_mask]
    
    if len(valid_bids) == 0:
        best_idx = np.argmax(win_probs)
        return candidate_bids[best_idx], max_prob
    
    # Pick the one furthest from 1.0 (avoid crowd)
    distances = np.abs(valid_bids - 1.0)
    strategic_idx = np.argmax(distances)
    
    return valid_bids[strategic_idx], valid_probs[strategic_idx]

# =============================================================================
# VECTORIZED SIMULATION ENGINE
# =============================================================================

def simulate_auction_vectorized(
    competitors_df, target_company, inst_profile,
    models, q_values, feature_cols, base_amt, my_min_bid_rate
):
    """
    Vectorized simulation with TOLERANCE-based bid selection.
    """
    mu = inst_profile.get('mu', 1.00)
    sigma = inst_profile.get('sigma', 0.02)
    n_iters = SIMULATION_ITERATIONS
    
    # 1. Yega Sampling (Original: Normal distribution)
    #sim_yegas = np.random.normal(mu, sigma / 2, n_iters)
    sim_yegas = np.random.normal(0.9990, 0.0087, n_iters)
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
    
    # 5. TOLERANCE-based bid selection (avoid crowd)
    return select_bid_with_tolerance(candidate_bids, np.array(win_probs))

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
# ENCODING
# =============================================================================

def encode_features_for_inference(df):
    """Encode features matching training format."""
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
    
    exclude = [
        'record_id', 'company_code', 'notice_id', 'institution_code',
        'normalized_bid_rate', 'announce_date', 'min_bid_rate',
        'base_amt', 'is_winner', 'choose_avg'
    ]
    
    feature_cols = [col for col in df_encoded.columns if col not in exclude]
    
    return df_encoded, feature_cols

# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_backtest(target_company, max_auctions=100):
    """Run backtest with TOLERANCE-based bid selection."""
    print("="*70)
    print(f"BACKTEST (TOLERANCE={TOLERANCE}): {target_company}")
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
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_DATA)
    feature_df = pd.read_csv(FEATURE_DATA)
    
    df_all = feature_df.merge(
        train_df[['record_id', 'company_code', 'notice_id', 'announce_date',
                  'institution_code', 'normalized_bid_rate', 'is_winner',
                  'min_bid_rate', 'base_amt', 'choose_avg']],
        on='record_id',
        how='inner'
    )
    
    print(f"  Records: {len(df_all):,}")
    
    # Filter target
    my_auctions = df_all[df_all['company_code'] == target_company].copy()
    my_auctions = my_auctions.sort_values('announce_date')
    
    if max_auctions:
        my_auctions = my_auctions.head(max_auctions)
    
    print(f"  Target auctions: {len(my_auctions)}")
    print()
    
    # Encode
    df_encoded, feature_cols = encode_features_for_inference(df_all)
    print(f"Features: {len(feature_cols)}")
    print()
    
    # Backtest
    results = []
    
    print("Running backtests...")
    for idx, row in tqdm(my_auctions.iterrows(), total=len(my_auctions), desc="Auctions"):
        notice_id = row['notice_id']
        base_amt = row['base_amt']
        my_min_bid = row['min_bid_rate']
        actual_my_bid = row['normalized_bid_rate']
        actual_win = row['is_winner']
        choose_avg = row['choose_avg']
        
        if choose_avg > 0 and base_amt > 0:
            actual_yega_rate = choose_avg / base_amt
        else:
            actual_yega_rate = 1.00
        
        competitors = df_encoded[
            (df_encoded['notice_id'] == notice_id) &
            (df_encoded['company_code'] != target_company)
        ].copy()
        
        if len(competitors) == 0:
            continue
        
        inst_code = row['institution_code']
        profile = inst_profiles.get(inst_code, {'mu': 1.0, 'sigma': 0.02})
        
        try:
            optimal_bid, optimal_prob = simulate_auction_vectorized(
                competitors, target_company, profile,
                models, q_values, feature_cols, base_amt, my_min_bid
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        
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
    print(f"BACKTEST RESULTS (TOLERANCE={TOLERANCE})")
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
    results_df.to_csv(out_dir / f'backtest_tolerance_{target_company}.csv', index=False)
    print(f"Saved: {out_dir / f'backtest_tolerance_{target_company}.csv'}")
    print("="*70)
    
    return results_df

# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    results = run_backtest('C0027', max_auctions=1000)
