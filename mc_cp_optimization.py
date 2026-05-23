"""
MC-CP Bid Optimization for target (C0022)
======================================

Strategy:
  1. For each validation project where target participated
  2. Try different candidate normalized_bid_rates (grid search)
  3. For each candidate, run MC simulation to estimate P(win)
  4. Choose candidate with highest P(win) as "optimal bid"
  5. Validate: Would this optimal bid have won in reality?

Note: MC-CP is NOT a learning/convergence model - we use grid search optimization.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

# Configuration
SIMULATION_COMPANY = 'COMPANY_A'  # target (can be changed to test other companies)
N_ITERATIONS = 10000  # MC iterations per candidate
CANDIDATE_BIDS = np.arange(0.975, 1.025, 0.0005)  # 97.5% to 102.5%, step 0.05% (101 candidates)

YEGA_MEAN = 100.0
YEGA_STD = 0.8
YEGA_MIN = 97.0
YEGA_MAX = 103.0

def get_project_root():
    return Path(__file__).parent

def sample_yega_rate_for_inst(inst_code=None):
    """
    Sample yega_rate from distribution.
    
    TODO: Add institution-specific tendency if inst_code provided.
    For now, use global distribution.
    """
    if inst_code is not None:
        # Future: Load institution profile and adjust mean/std
        # inst_profile = load_inst_profile(inst_code)
        # mean = inst_profile['mean_yega']
        # std = inst_profile['std_yega']
        pass
    
    return np.clip(
        np.random.normal(YEGA_MEAN, YEGA_STD),
        YEGA_MIN,
        YEGA_MAX
    )

def determine_winner_fast(bids_df, yega_rate, base_amt):
    """Fast winner determination (vectorized)."""
    estimated_price = base_amt * (yega_rate / 100)
    
    bids_df['bid_amount'] = estimated_price * bids_df['min_bid_rate'] * bids_df['normalized_bid_rate']
    bids_df['threshold'] = estimated_price * bids_df['min_bid_rate']
    
    valid_bids = bids_df[bids_df['bid_amount'] >= bids_df['threshold']]
    
    if len(valid_bids) == 0:
        return None
    
    winner_idx = valid_bids['bid_amount'].idxmin()
    return valid_bids.loc[winner_idx, 'company_code']

def run_mc_for_candidate(project_df, candidate_bid_target, models, q_values, base_amt, inst_code, n_iter=N_ITERATIONS):
    """
    Run MC simulation with FIXED bid for simulation company, sampled bids for competitors.
    
    Args:
        inst_code: Institution code for yega sampling (institution-specific tendency)
    
    Returns: P(win) for simulation company with this candidate bid
    """
    win_count = 0
    valid_iterations = 0
    
    # Prepare competitor data (exclude simulation company)
    competitors = []
    for idx, row in project_df.iterrows():
        company = row['company_code']
        
        if company == SIMULATION_COMPANY:
            continue  # Skip simulation company - we'll use candidate_bid
        
        if company not in models:
            continue
        
        competitors.append({
            'company_code': company,
            'min_bid_rate': row['min_bid_rate'],
            'model': models[company],
            'q': q_values[company],
            'X': None  # Will need features
        })
    
    # Get simulation company's min_bid_rate
    sim_row = project_df[project_df['company_code'] == SIMULATION_COMPANY].iloc[0]
    sim_min_bid_rate = sim_row['min_bid_rate']
    
    for iteration in range(n_iter):
        # Sample yega (institution-specific)
        yega = sample_yega_rate_for_inst(inst_code)
        
        # Build bids list
        bids = []
        
        # Simulation company's bid (FIXED candidate)
        bids.append({
            'company_code': SIMULATION_COMPANY,
            'normalized_bid_rate': candidate_bid_target,
            'min_bid_rate': sim_min_bid_rate
        })
        
        # Competitors' bids (SAMPLED from models)
        for comp in competitors:
            # Sample from conformal interval
            # Note: Ideally use features, but for now use q-value range
            sampled_normalized = 1.0 + np.random.uniform(-comp['q'], comp['q'])
            sampled_normalized = np.clip(sampled_normalized, 0.75, 1.10)
            
            bids.append({
                'company_code': comp['company_code'],
                'normalized_bid_rate': sampled_normalized,
                'min_bid_rate': comp['min_bid_rate']
            })
        
        bids_df = pd.DataFrame(bids)
        
        # Determine winner
        winner = determine_winner_fast(bids_df, yega, base_amt)
        
        if winner is not None:
            valid_iterations += 1
            if winner == SIMULATION_COMPANY:
                win_count += 1
    
    if valid_iterations == 0:
        return 0.0
    
    return win_count / valid_iterations

def optimize_bid_for_project(project_df, models, q_values):
    """
    Find optimal bid for simulation company using grid search.
    
    Returns: (optimal_bid, max_p_win, p_win_per_candidate)
    """
    base_amt = project_df.iloc[0]['base_amt']
    inst_code = project_df.iloc[0].get('institution_code', None)
    
    best_bid = None
    best_p_win = 0.0
    p_win_curve = []
    
    print(f"  Testing {len(CANDIDATE_BIDS)} candidates (97.5%-102.5%, step 0.05%)...")
    
    for candidate in tqdm(CANDIDATE_BIDS, desc="  Grid search", leave=False):
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
    print("MC-CP BID OPTIMIZATION FOR target (C0022)")
    print("=" * 70)
    print()
    
    np.random.seed(42)
    root = get_project_root()
    
    # Load models and q-values
    print("Loading models...")
    models = joblib.load(root / 'models' / 'company_models.pkl')
    q_values = joblib.load(root / 'models' / 'q_values.pkl')
    
    print(f"✅ Loaded {len(models)} company models (will use for competitors)")
    print(f"   Note: Simulation company ({SIMULATION_COMPANY}) bid is FIXED (grid search)")
    print()
    
    # Load feature data
    print("Loading feature data...")
    feature_path = root / 'analysis_results' / 'feature_sample.csv'
    features_df = pd.read_csv(feature_path)
    
    # Load company metadata (company_code, min_bid_rate, etc.)
    train_path = root / 'data' / 'processed' / 'train_clean.csv'
    train_df = pd.read_csv(train_path)
    
    # Merge to get company_code and metadata
    df = features_df.merge(
        train_df[['record_id', 'company_code', 'notice_id', 'base_amt', 'min_bid_rate', 
                  'institution_code', 'is_winner', 'normalized_bid_rate']],
        on='record_id',
        how='inner'
    )
    
    print(f"Loaded {len(df):,} records with features + metadata")
    
    # Filter to target projects only
    target_projects = df[df['company_code'] == SIMULATION_COMPANY]['notice_id'].unique()
    print(f"Found {len(target_projects)} projects where target participated")
    
    # Sample N projects for testing
    N_TEST_PROJECTS = 10
    if len(target_projects) > N_TEST_PROJECTS:
        test_projects = np.random.choice(target_projects, N_TEST_PROJECTS, replace=False)
    else:
        test_projects = target_projects
    
    print(f"Testing on {len(test_projects)} projects")
    print()
    
    # Run optimization
    results = []
    
    for notice_id in tqdm(test_projects, desc="Optimizing"):
        project_df = df[df['notice_id'] == notice_id]
        
        # Get actual outcome
        actual_winner_rows = project_df[project_df['is_winner'] == True]
        if len(actual_winner_rows) == 0:
            continue
        
        actual_winner = actual_winner_rows.iloc[0]['company_code']
        target_actual_bid = project_df[project_df['company_code'] == SIMULATION_COMPANY].iloc[0]['normalized_bid_rate']
        
        # Optimize
        print(f"\n  Project {notice_id}:")
        optimal_bid, max_p_win, p_win_curve = optimize_bid_for_project(
            project_df, models, q_values
        )
        
        # Would optimal bid have won?
        # Run one final high-precision simulation with optimal bid
        p_win_optimal = run_mc_for_candidate(
            project_df, optimal_bid, models, q_values,
            project_df.iloc[0]['base_amt'],
            project_df.iloc[0].get('institution_code', None),
            n_iter=20000  # Double iterations for final validation
        )
        
        results.append({
            'notice_id': notice_id,
            'actual_winner': actual_winner,
            'target_actually_won': (actual_winner == SIMULATION_COMPANY),
            'actual_bid_target': float(target_actual_bid),
            'optimal_bid': float(optimal_bid),
            'p_win_at_optimal': float(p_win_optimal),
            'p_win_curve': p_win_curve
        })
        
        print(f"    Actual bid: {target_actual_bid:.4f}")
        print(f"    Optimal bid: {optimal_bid:.4f}")
        print(f"    P(win) at optimal: {p_win_optimal:.2%}")
        print(f"    Actual winner: {actual_winner} ({'target' if actual_winner == SIMULATION_COMPANY else 'Other'})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_target_won = sum([r['target_actually_won'] for r in results])
    avg_p_win_optimal = np.mean([r['p_win_at_optimal'] for r in results])
    
    print(f"Projects tested: {len(results)}")
    print(f"target actually won: {n_target_won} ({n_target_won/len(results):.1%})")
    print(f"Average P(win) at optimal: {avg_p_win_optimal:.1%}")
    print()
    
    # Check if optimal would have increased win rate
    high_p_win_projects = [r for r in results if r['p_win_at_optimal'] > 0.5]
    print(f"Projects with P(win) > 50% at optimal: {len(high_p_win_projects)}")
    
    # Save results
    output_path = root / 'analysis_results' / 'mc_optimization_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'simulation_company': SIMULATION_COMPANY,
            'n_projects': len(results),
            'n_iterations_per_candidate': N_ITERATIONS,
            'candidate_range': [float(CANDIDATE_BIDS[0]), float(CANDIDATE_BIDS[-1])],
            'results': results
        }, f, indent=2)
    
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
