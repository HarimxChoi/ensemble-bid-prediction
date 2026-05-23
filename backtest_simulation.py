"""
Walk-Forward Backtesting Simulation (VECTORIZED + CORRECT)
===========================================================

CORRECT Logic:
- Vectorized MC: 300K iterations in ~1 second
- Uses model predictions (not actual values) for competitors
- Validates with ACTUAL yega = choose_avg / base_amt
- Korean PQ threshold: bid_amt >= estimated_price × min_bid_rate

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

SIMULATION_ITERATIONS = 500_000 
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
# VECTORIZED SIMULATION ENGINE
# =============================================================================

def simulate_auction_vectorized(
    competitors_df, target_company, inst_profile,
    models, q_values, feature_cols, base_amt, my_min_bid_rate
):
    """
    Vectorized simulation: 1000x faster with NumPy broadcasting.
    Uses MODEL PREDICTIONS (not actual values) for competitors.
    """
    mu = inst_profile.get('mu', 1.00)
    sigma = inst_profile.get('sigma', 0.02)
    n_iters = SIMULATION_ITERATIONS
    
    # 1. Yega Sampling (Vectorized) - Korean style (4 samples averaged ≈ sigma/2)
    sim_yegas = np.random.normal(mu, sigma / 2, n_iters)
    sim_yegas = np.clip(sim_yegas, 0.97, 1.03)
    
    # 2. Competitor Bid Prediction (using MODELS, not actual values!)
    comp_bids_norm = []
    comp_min_rates = []
    
    for _, row in competitors_df.iterrows():
        comp_code = row['company_code']
        min_rate = row['min_bid_rate']
        
        # Use MODEL prediction (not actual value - that would be cheating!)
        if comp_code in models:
            model = models[comp_code]
            q = q_values.get(comp_code, 0.02)
            # Extract features
            feat = row[feature_cols].fillna(-999).values.reshape(1, -1)
            try:
                pred = model.predict(feat)[0]
            except:
                pred = 1.000
                q = 0.03
        else:
            # No model - conservative assumption
            pred = 1.000
            q = 0.03
        
        # Add MC noise
        noise = np.random.uniform(-q, q, n_iters)
        sim_norm = pred + noise
        sim_norm = np.clip(sim_norm, 0.90, 1.10)
        
        comp_bids_norm.append(sim_norm)
        comp_min_rates.append(min_rate)
    
    # No competitors = easy win
    if not comp_bids_norm:
        return GRID_START, 1.0
    
    comp_bids_norm = np.array(comp_bids_norm)  # (N_comps, N_iters)
    comp_min_rates = np.array(comp_min_rates).reshape(-1, 1)  # (N_comps, 1)
    
    # 3. Vectorized Korean PQ Logic
    # estimated_price = base_amt × yega
    sim_yejeong_prices = base_amt * sim_yegas  # (N_iters,)
    
    # Competitor bid amounts and thresholds
    comp_bid_amts = base_amt * comp_min_rates * comp_bids_norm  # (N_comps, N_iters)
    comp_thresholds = sim_yejeong_prices * comp_min_rates  # (N_comps, N_iters)
    
    # Validity: bid_amt >= threshold
    is_valid_comp = comp_bid_amts >= comp_thresholds
    
    # Best valid competitor price (invalid = infinity)
    valid_comp_amts = np.where(is_valid_comp, comp_bid_amts, np.inf)
    best_comp_prices = np.min(valid_comp_amts, axis=0)  # (N_iters,)
    
    # 4. Grid Search (My Bids)
    candidate_bids = np.arange(GRID_START, GRID_END + GRID_STEP, GRID_STEP)
    win_probs = []
    
    # My thresholds
    my_thresholds = sim_yejeong_prices * my_min_bid_rate  # (N_iters,)
    
    for my_norm_bid in candidate_bids:
        my_bid_amt = base_amt * my_min_bid_rate * my_norm_bid
        
        # Am I valid?
        am_i_valid = my_bid_amt >= my_thresholds
        
        # Am I lowest? (tie = lose, conservative)
        beat_competitors = my_bid_amt < best_comp_prices
        
        wins = am_i_valid & beat_competitors
        win_probs.append(np.mean(wins))
    
    best_idx = np.argmax(win_probs)
    return candidate_bids[best_idx], win_probs[best_idx]

# =============================================================================
# WINNER DETERMINATION (For Validation)
# =============================================================================

def determine_winner_actual(competitors_df, base_amt, yega_rate, my_bid_info=None):
    """
    Determine winner using ACTUAL data for validation.
    """
    yejeong_price = base_amt * yega_rate
    candidates = []
    
    # Competitors (using their ACTUAL bids)
    for _, row in competitors_df.iterrows():
        threshold = yejeong_price * row['min_bid_rate']
        bid_amt = base_amt * row['min_bid_rate'] * row['normalized_bid_rate']
        if bid_amt >= threshold:
            candidates.append((row['company_code'], bid_amt))
    
    # My counterfactual bid
    if my_bid_info:
        my_norm, my_min = my_bid_info
        my_thresh = yejeong_price * my_min
        my_amt = base_amt * my_min * my_norm
        if my_amt >= my_thresh:
            candidates.append(('TARGET', my_amt))
    
    if not candidates:
        return None
    
    # Winner = lowest valid bid
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
    """
    Run walk-forward backtest with vectorized MC simulation.
    """
    print("="*70)
    print(f"WALK-FORWARD BACKTESTING (VECTORIZED): {target_company}")
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
        
        # ACTUAL yega (from data!)
        if choose_avg > 0 and base_amt > 0:
            actual_yega_rate = choose_avg / base_amt
        else:
            actual_yega_rate = 1.00
        
        # Competitors
        competitors = df_encoded[
            (df_encoded['notice_id'] == notice_id) &
            (df_encoded['company_code'] != target_company)
        ].copy()
        
        if len(competitors) == 0:
            continue
        
        # Inst profile
        inst_code = row['institution_code']
        profile = inst_profiles.get(inst_code, {'mu': 1.0, 'sigma': 0.02})
        
        # 1. RUN SIMULATION (Model prediction + MC)
        try:
            optimal_bid, optimal_prob = simulate_auction_vectorized(
                competitors, target_company, profile,
                models, q_values, feature_cols, base_amt, my_min_bid
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        # 2. VALIDATION (Counterfactual with ACTUAL data)
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
    print("BACKTEST RESULTS")
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
    results_df.to_csv(out_dir / f'backtest_vectorized_{target_company}.csv', index=False)
    print(f"Saved: {out_dir / f'backtest_vectorized_{target_company}.csv'}")
    print("="*70)
    
    return results_df

# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    results = run_backtest('C0005', max_auctions=100)
