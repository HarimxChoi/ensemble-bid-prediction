"""
Strategic Bidding Optimizer - Validation Script
===============================================

Validate optimizer on 50 actual auction winners.

Test Questions:
1. Does optimizer recommend bids close to actual winners?
2. Do actual winners have high P(win) with their bids?
3. Which yega method performs best?

Metrics:
- Bid error (should be <0.5%)
- % near-optimal bids
- P(win) calibration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

from strategic_bidding_optimizer import (
    StrategicBiddingOptimizer,
    encode_features_for_inference,
    N_VALIDATION_PROJECTS
)

def validate_optimizer():
    """
    Main validation function.
    
    For each project:
      1. Get actual winner + their bid
      2. Run optimizer for that company
      3. Measure: bid_error, P(win) at actual bid,P(win) at optimal bid
    """
    print("=" * 70)
    print("STRATEGIC BIDDING OPTIMIZER - VALIDATION")
    print("=" * 70)
    print(f"Testing on {N_VALIDATION_PROJECTS} projects")
    print()
    
    # Initialize
    optimizer = StrategicBiddingOptimizer()
    
    # Load data
    root = Path(__file__).parent
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    train_df = pd.read_csv(root / 'data' / 'processed' / 'train_clean.csv')
    
    df = features_df.merge(
        train_df[['record_id', 'company_code', 'notice_id', 'base_amt',
                  'min_bid_rate', 'institution_code', 'is_winner', 
                  'normalized_bid_rate']],
        on='record_id',
        how='inner'
    )
    
    print(f"Loaded {len(df):,} records")
    
    # Encode features
    print("Encoding features...")
    X_all, feature_cols = encode_features_for_inference(df)
    print(f"✅ Features: {len(feature_cols)} (X shape: {X_all.shape})")
    print()
    
    # Get projects with winners
    projects_with_winners = df[df['is_winner'] == True]['notice_id'].unique()
    print(f"Found {len(projects_with_winners)} projects with winners")
    
    # Sample N_VALIDATION_PROJECTS
    if len(projects_with_winners) > N_VALIDATION_PROJECTS:
        np.random.seed(42)
        test_projects = np.random.choice(
            projects_with_winners, 
            N_VALIDATION_PROJECTS, 
            replace=False
        )
    else:
        test_projects = projects_with_winners[:N_VALIDATION_PROJECTS]
    
    print(f"Testing on {len(test_projects)} projects")
    print()
    
    # Run validation
    validation_results = []
    
    for project_idx, notice_id in enumerate(tqdm(test_projects, desc="Validating")):
        project_df = df[df['notice_id'] == notice_id]
        project_indices = df[df['notice_id'] == notice_id].index
        
        # Get actual winner
        winner_rows = project_df[project_df['is_winner'] == True]
        if len(winner_rows) == 0:
            continue
        
        actual_winner = winner_rows.iloc[0]['company_code']
        actual_winner_bid = winner_rows.iloc[0]['normalized_bid_rate']
        
        # Extract project data
        companies = project_df['company_code'].tolist()
        X_features = X_all[project_indices]
        
        project_data = {
            'companies': companies,
            'X_features': X_features,
            'base_amt': project_df.iloc[0]['base_amt'],
            'inst_code': project_df.iloc[0].get('institution_code', None),
            'min_bid_rates': project_df['min_bid_rate'].values
        }
        
        # Skip if actual winner not in companies list
        if actual_winner not in companies:
            continue
        
        print(f"\n--- Project {project_idx+1}/{len(test_projects)}: {notice_id} ---")
        print(f"Actual winner: {actual_winner}, bid: {actual_winner_bid:.4f}")
        
        # Run optimizer for actual winner
        optimal, curve = optimizer.optimize_bid(
            company_code=actual_winner,
            project_data=project_data,
            n_iterations=50000,  # 50K for validation (faster)
            yega_method='auto',
            show_progress=False
        )
        
        # Get P(win) for actual bid (interpolate from curve)
        curve_df = pd.DataFrame(curve)
        actual_p_win = np.interp(
            actual_winner_bid,
            curve_df['candidate_bid'].values,
            curve_df['p_win'].values
        )
        
        # Calculate metrics
        bid_error = abs(optimal['candidate_bid'] - actual_winner_bid)
        is_near_optimal = bid_error < 0.005  # Within 0.5%
        
        validation_results.append({
            'notice_id': str(notice_id),
            'winner': actual_winner,
            'actual_bid': float(actual_winner_bid),
            'recommended_bid': float(optimal['candidate_bid']),
            'bid_error': float(bid_error),
            'p_win_recommended': float(optimal['p_win']),
            'p_win_actual': float(actual_p_win),
            'is_near_optimal': bool(is_near_optimal),  # Convert numpy bool to Python bool
            'n_companies': int(len(companies))
        })
        
        status = "✅" if is_near_optimal else "⚠️"
        print(f"{status} Recommended: {optimal['candidate_bid']:.4f} (P(win)={optimal['p_win']:.1%})")
        print(f"   Actual P(win): {actual_p_win:.1%}")
        print(f"   Error: {bid_error:.4f} ({bid_error*100:.2f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    mean_bid_error = np.mean([r['bid_error'] for r in validation_results])
    pct_near_optimal = np.mean([r['is_near_optimal'] for r in validation_results])
    mean_p_win_actual = np.mean([r['p_win_actual'] for r in validation_results])
    mean_p_win_recommended = np.mean([r['p_win_recommended'] for r in validation_results])
    
    print(f"Projects tested: {len(validation_results)}")
    print(f"Mean bid error: {mean_bid_error:.4f} ({mean_bid_error*100:.2f}%)")
    print(f"Near-optimal bids (<0.5% error): {pct_near_optimal:.1%}")
    print(f"Mean P(win) for actual bids: {mean_p_win_actual:.1%}")
    print(f"Mean P(win) for recommended bids: {mean_p_win_recommended:.1%}")
    print()
    
    # Detailed results
    print("Individual Results:")
    print("-" * 70)
    for r in validation_results[:10]:  # Show first 10
        status = "✅" if r['is_near_optimal'] else "⚠️"
        print(f"{status} {r['notice_id']}: {r['winner']}")
        print(f"   Actual: {r['actual_bid']:.4f} (P(win)={r['p_win_actual']:.1%})")
        print(f"   Recommended: {r['recommended_bid']:.4f} (P(win)={r['p_win_recommended']:.1%})")
        print(f"   Error: {r['bid_error']:.4f}")
    
    if len(validation_results) > 10:
        print(f"... and {len(validation_results)-10} more")
    print()
    
    # Save results
    output_path = root / 'analysis_results' / 'optimizer_validation.json'
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'n_projects': len(validation_results),
                'mean_bid_error': float(mean_bid_error),
                'pct_near_optimal': float(pct_near_optimal),
                'mean_p_win_actual': float(mean_p_win_actual),
                'mean_p_win_recommended': float(mean_p_win_recommended)
            },
            'results': validation_results
        }, f, indent=2)
    
    print(f"✅ Saved: {output_path}")
    
    # Visualization
    create_validation_plots(validation_results, root / 'analysis_results')
    
    return validation_results


def create_validation_plots(results, output_dir):
    """Create visualization plots for validation results."""
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Bid error distribution
    ax = axes[0, 0]
    ax.hist(df['bid_error'] * 100, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', label='0.5% threshold')
    ax.set_xlabel('Bid Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Bid Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. P(win) comparison
    ax = axes[0, 1]
    ax.scatter(df['p_win_actual'], df['p_win_recommended'], alpha=0.6)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect match')
    ax.set_xlabel('P(win) at Actual Bid')
    ax.set_ylabel('P(win) at Recommended Bid')
    ax.set_title('P(win) Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Bid error vs P(win)
    ax = axes[1, 0]
    ax.scatter(df['p_win_actual'], df['bid_error'] * 100, alpha=0.6)
    ax.axhline(0.5, color='red', linestyle='--', label='0.5% threshold')
    ax.set_xlabel('P(win) at Actual Bid')
    ax.set_ylabel('Bid Error (%)')
    ax.set_title('Bid Error vs P(win)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    VALIDATION SUMMARY
    
    Projects Tested: {len(results)}
    
    Bid Accuracy:
      Mean Error: {df['bid_error'].mean()*100:.2f}%
      Median Error: {df['bid_error'].median()*100:.2f}%
      Near-Optimal (<0.5%): {(df['bid_error'] < 0.005).mean()*100:.1f}%
    
    P(win) Calibration:
      Actual Mean: {df['p_win_actual'].mean()*100:.1f}%
      Recommended Mean: {df['p_win_recommended'].mean()*100:.1f}%
      
    Performance:
      Best Case: {df['bid_error'].min()*100:.3f}%
      Worst Case: {df['bid_error'].max()*100:.2f}%
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_validation_plots.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'optimizer_validation_plots.png'}")
    
    plt.close()


def run_ablation_study():
    """
    Run ablation study on 5 sample projects.
    
    Compare yega sampling methods:
    - global
    - all (static)
    - 1y
    - 6m
    - 3m
    - auto (adaptive)
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY: Yega Sampling Methods")
    print("=" * 70)
    print()
    
    optimizer = StrategicBiddingOptimizer()
    
    # Load data
    root = Path(__file__).parent
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    train_df = pd.read_csv(root / 'data' / 'processed' / 'train_clean.csv')
    
    df = features_df.merge(
        train_df[['record_id', 'company_code', 'notice_id', 'base_amt',
                  'min_bid_rate', 'institution_code', 'is_winner', 
                  'normalized_bid_rate']],
        on='record_id',
        how='inner'
    )
    
    # Encode features
    X_all, feature_cols = encode_features_for_inference(df)
    
    # Test on 5 projects
    test_projects = df['notice_id'].unique()[:5]
    
    all_ablation_results = []
    
    for notice_id in test_projects:
        project_df = df[df['notice_id'] == notice_id]
        project_indices = df[df['notice_id'] == notice_id].index
        
        winner_rows = project_df[project_df['is_winner'] == True]
        if len(winner_rows) == 0:
            continue
        
        actual_winner = winner_rows.iloc[0]['company_code']
        
        # Extract project data
        companies = project_df['company_code'].tolist()
        X_features = X_all[project_indices]
        
        project_data = {
            'companies': companies,
            'X_features': X_features,
            'base_amt': project_df.iloc[0]['base_amt'],
            'inst_code': project_df.iloc[0].get('institution_code', None),
            'min_bid_rates': project_df['min_bid_rate'].values
        }
        
        if actual_winner not in companies:
            continue
        
        print(f"\n--- Project {notice_id}: {actual_winner} ---")
        
        # Run ablation
        ablation_results = optimizer.ablation_study_yega(
            company_code=actual_winner,
            project_data=project_data,
            methods=['global', 'all', '1y', '6m', '3m', 'auto'],
            n_iterations=30000  # 30K for ablation (faster)
        )
        
        for r in ablation_results:
            r['notice_id'] = str(notice_id)
            r['winner'] = actual_winner
        
        all_ablation_results.extend(ablation_results)
    
    # Save
    output_path = root / 'analysis_results' / 'yega_ablation_study.json'
    with open(output_path, 'w') as f:
        json.dump(all_ablation_results, f, indent=2)
    
    print(f"\n✅ Saved: {output_path}")
    
    return all_ablation_results


if __name__ == "__main__":
    # Run validation
    validate_optimizer()
    
    # Run ablation study
    # run_ablation_study()
