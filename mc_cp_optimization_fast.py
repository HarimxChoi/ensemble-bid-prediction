"""
MC-CP Bid Optimization - FAST VERSION
=====================================

Reduced parameters for faster testing:
- 20 candidates (step 0.25%)
- 2K iterations per candidate
- 5 test projects
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

# Configuration - REDUCED FOR SPEED
SIMULATION_COMPANY = 'COMPANY_A'  # example identifier
N_ITERATIONS = 2000  # Reduced from 10000
CANDIDATE_BIDS = np.arange(0.98, 1.021, 0.0025)  # 98%-102%, step 0.25% (17 candidates)
N_TEST_PROJECTS = 5  # Reduced from 10

YEGA_MEAN = 100.0
YEGA_STD = 0.8
YEGA_MIN = 97.0
YEGA_MAX = 103.0

def get_project_root():
    return Path(__file__).parent

def sample_yega_rate_for_inst(inst_code=None):
    """Sample yega_rate from distribution."""
    return np.clip(
        np.random.normal(YEGA_MEAN, YEGA_STD),
        YEGA_MIN,
        YEGA_MAX
    )

def determine_winner_fast(bids_df, yega_rate, base_amt):
    """Fast winner determination."""
    estimated_price = base_amt * (yega_rate / 100)
    
    bids_df['bid_amount'] = estimated_price * bids_df['min_bid_rate'] * bids_df['normalized_bid_rate']
    bids_df['threshold'] = estimated_price * bids_df['min_bid_rate']
    
    valid_bids = bids_df[bids_df['bid_amount'] >= bids_df['threshold']]
    
    if len(valid_bids) == 0:
        return None
    
    winner_idx = valid_bids['bid_amount'].idxmin()
    return valid_bids.loc[winner_idx, 'company_code']

def run_mc_for_candidate(project_df, candidate_bid, models, q_values, base_amt, inst_code, n_iter=N_ITERATIONS):
    """Run MC simulation with FIXED bid for simulation company."""
    win_count = 0
    valid_iterations = 0
    
    # Get simulation company's min_bid_rate
    sim_row = project_df[project_df['company_code'] == SIMULATION_COMPANY].iloc[0]
    sim_min_bid_rate = sim_row['min_bid_rate']
    
    # Prepare competitors (exclude simulation company)
    competitors = []
    for idx, row in project_df.iterrows():
        company = row['company_code']
        
        if company == SIMULATION_COMPANY:
            continue
        
        if company not in models:
            continue
        
        competitors.append({
            'company_code': company,
            'min_bid_rate': row['min_bid_rate'],
            'q': q_values[company]
        })
    
    for iteration in range(n_iter):
        yega = sample_yega_rate_for_inst(inst_code)
        
        bids = []
        
        # Simulation company's bid (FIXED)
        bids.append({
            'company_code': SIMULATION_COMPANY,
            'normalized_bid_rate': candidate_bid,
            'min_bid_rate': sim_min_bid_rate
        })
        
        # Competitors' bids (SAMPLED)
        for comp in competitors:
            sampled_normalized = 1.0 + np.random.uniform(-comp['q'], comp['q'])
            sampled_normalized = np.clip(sampled_normalized, 0.75, 1.10)
            
            bids.append({
                'company_code': comp['company_code'],
                'normalized_bid_rate': sampled_normalized,
                'min_bid_rate': comp['min_bid_rate']
            })
        
        bids_df = pd.DataFrame(bids)
        winner = determine_winner_fast(bids_df, yega, base_amt)
        
        if winner is not None:
            valid_iterations += 1
            if winner == SIMULATION_COMPANY:
                win_count += 1
    
    if valid_iterations == 0:
        return 0.0
    
    return win_count / valid_iterations

def optimize_bid_for_project(project_df, models, q_values):
    """Find optimal bid using grid search."""
    base_amt = project_df.iloc[0]['base_amt']
    inst_code = project_df.iloc[0].get('institution_code', None)
    
    best_bid = None
    best_p_win = 0.0
    p_win_curve = []
    
    for candidate in tqdm(CANDIDATE_BIDS, desc="  Grid", leave=False):
        p_win = run_mc_for_candidate(
            project_df, candidate, models, q_values, base_amt, inst_code, n_iter=N_ITERATIONS
        )
        
        p_win_curve.append({
            'candidate_bid': float(candidate),
            'p_win': float(p_win)
        })
        
        if p_win > best_p_win:
            best_p_win = p_win
            best_bid = candidate
    
    return best_bid, best_p_win, p_win_curve

def main():
    print("=" * 70)
    print("MC-CP BID OPTIMIZATION - FAST VERSION")
    print("=" * 70)
    print(f"Candidates: {len(CANDIDATE_BIDS)} (98%-102%, step 0.25%)")
    print(f"Iterations: {N_ITERATIONS} per candidate")
    print(f"Projects: {N_TEST_PROJECTS}")
    print()
    
    np.random.seed(42)
    root = get_project_root()
    
    # Load models
    print("Loading models...")
    models = joblib.load(root / 'models' / 'company_models.pkl')
    q_values = joblib.load(root / 'models' / 'q_values.pkl')
    print(f"✅ Loaded {len(models)} models")
    print()
    
    # Load data
    print("Loading data...")
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    train_df = pd.read_csv(root / 'data' / 'processed' / 'train_clean.csv')
    
    df = features_df.merge(
        train_df[['record_id', 'company_code', 'notice_id', 'base_amt', 'min_bid_rate', 
                  'institution_code', 'is_winner', 'normalized_bid_rate']],
        on='record_id',
        how='inner'
    )
    print(f"✅ Loaded {len(df):,} records")
    print()
    
    # Filter to simulation company projects
    sim_projects = df[df['company_code'] == SIMULATION_COMPANY]['notice_id'].unique()
    print(f"Found {len(sim_projects)} projects with {SIMULATION_COMPANY}")
    
    if len(sim_projects) > N_TEST_PROJECTS:
        test_projects = np.random.choice(sim_projects, N_TEST_PROJECTS, replace=False)
    else:
        test_projects = sim_projects[:N_TEST_PROJECTS]
    
    print(f"Testing {len(test_projects)} projects")
    print()
    
    # Run optimization
    results = []
    
    for notice_id in tqdm(test_projects, desc="Projects"):
        project_df = df[df['notice_id'] == notice_id]
        
        actual_winner_rows = project_df[project_df['is_winner'] == True]
        if len(actual_winner_rows) == 0:
            continue
        
        actual_winner = actual_winner_rows.iloc[0]['company_code']
        sim_actual_bid = project_df[project_df['company_code'] == SIMULATION_COMPANY].iloc[0]['normalized_bid_rate']
        
        # Optimize
        optimal_bid, max_p_win, p_win_curve = optimize_bid_for_project(
            project_df, models, q_values
        )
        
        results.append({
            'notice_id': str(notice_id),  # Convert to string for JSON
            'actual_winner': actual_winner,
            'sim_won': (actual_winner == SIMULATION_COMPANY),
            'actual_bid': float(sim_actual_bid),
            'optimal_bid': float(optimal_bid),
            'p_win_at_optimal': float(max_p_win),
            'p_win_curve': p_win_curve
        })
        
        print(f"  {notice_id}: optimal={optimal_bid:.4f} (P(win)={max_p_win:.1%}), actual={sim_actual_bid:.4f}, winner={actual_winner}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_sim_won = sum([r['sim_won'] for r in results])
    avg_p_win = np.mean([r['p_win_at_optimal'] for r in results])
    
    print(f"Projects: {len(results)}")
    print(f"{SIMULATION_COMPANY} won: {n_sim_won}/{len(results)} ({n_sim_won/len(results):.1%})")
    print(f"Avg P(win) at optimal: {avg_p_win:.1%}")
    
    # Save
    output_path = root / 'analysis_results' / 'mc_optimization_results_fast.json'
    with open(output_path, 'w') as f:
        json.dump({
            'simulation_company': SIMULATION_COMPANY,
            'n_projects': len(results),
            'n_iterations': N_ITERATIONS,
            'n_candidates': len(CANDIDATE_BIDS),
            'results': results
        }, f, indent=2)
    
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
