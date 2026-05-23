"""
Copula Hypothesis Test (CORRECTED)
===================================

CORRECT METHODOLOGY: Across-project correlation
  - For each company pair (A, B)
  - Find all projects where BOTH participated
  - Correlate their raw residuals across these shared projects
  
WRONG (previous): Within-project standardization
  - Artificially forces negative correlation (sum of z-scores = 0)

Hypothesis:
  H₀: ρ < 0.15 → Skip copula
  H₁: ρ > 0.25 → Implement copula
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from itertools import combinations
from collections import defaultdict


# ==============================================================================
# CONFIGURATION
# ==============================================================================

RHO_SKIP_THRESHOLD = 0.15
RHO_IMPLEMENT_THRESHOLD = 0.25

MIN_SHARED_PROJECTS = 10  # Minimum shared projects for valid pair correlation
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_project_root():
    return Path(__file__).parent


def load_models_and_data():
    """Load trained models and data."""
    root = get_project_root()
    
    print("Loading models and data...")
    
    models = joblib.load(root / 'models' / 'company_models.pkl')
    print(f"  Loaded {len(models)} company models")
    
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    print(f"  Features: {len(features_df):,} rows")
    
    train_df = pd.read_csv(
        root / 'data' / 'processed' / 'train_clean.csv',
        usecols=['record_id', 'company_code', 'notice_id', 'announce_date']
    )
    
    df = features_df.merge(train_df, on='record_id', how='left')
    print(f"  Merged: {len(df):,} rows")
    
    return models, df


def encode_features(df):
    """One-hot encode categorical features."""
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
    
    exclude_cols = ['record_id', 'company_code', 'notice_id', 'target', 'announce_date']
    feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]
    
    X = df_encoded[feature_cols].fillna(-999).values
    
    return X, feature_cols


# ==============================================================================
# RESIDUAL COMPUTATION
# ==============================================================================

def compute_residuals(models, df, X):
    """Compute raw residuals (no standardization!)."""
    print("\nComputing residuals...")
    
    company_codes = df['company_code'].values
    y_actual = df['target'].values
    
    residuals = np.full(len(df), np.nan)
    predictions = np.full(len(df), np.nan)
    
    for company_code, model in models.items():
        mask = (company_codes == company_code)
        if mask.sum() == 0:
            continue
        
        X_company = X[mask]
        y_pred = model.predict(X_company)
        
        predictions[mask] = y_pred
        residuals[mask] = y_actual[mask] - y_pred  # Raw residuals!
    
    print(f"  Valid residuals: {(~np.isnan(residuals)).sum():,}")
    
    df['prediction'] = predictions
    df['residual'] = residuals
    
    return df


# ==============================================================================
# CORRECT: ACROSS-PROJECT CORRELATION
# ==============================================================================

def compute_pairwise_correlations_correct(df):
    """
    CORRECT methodology: Across-project correlation.
    
    For each company pair (A, B):
      1. Find projects where BOTH participated
      2. Get their raw residuals across these projects
      3. Compute Pearson correlation of these sequences
    
    This tests: Do companies have correlated DEVIATIONS from expected?
    """
    print("\nComputing pairwise correlations (ACROSS projects - CORRECT)...")
    
    df_valid = df[~df['residual'].isna()].copy()
    
    # Build lookup: company -> {notice_id: residual}
    company_residuals = {}
    for company in df_valid['company_code'].unique():
        company_df = df_valid[df_valid['company_code'] == company]
        company_residuals[company] = dict(zip(
            company_df['notice_id'].values,
            company_df['residual'].values
        ))
    
    companies = list(company_residuals.keys())
    print(f"  Companies with residuals: {len(companies)}")
    
    pair_correlations = {}
    n_pairs_tested = 0
    n_pairs_significant = 0
    
    for company_a, company_b in combinations(companies, 2):
        # Find shared projects
        projects_a = set(company_residuals[company_a].keys())
        projects_b = set(company_residuals[company_b].keys())
        shared_projects = projects_a & projects_b
        
        if len(shared_projects) < MIN_SHARED_PROJECTS:
            continue
        
        # Build parallel residual sequences
        residuals_a = []
        residuals_b = []
        
        for project_id in shared_projects:
            residuals_a.append(company_residuals[company_a][project_id])
            residuals_b.append(company_residuals[company_b][project_id])
        
        # Compute Pearson correlation ACROSS projects
        rho, p_value = stats.pearsonr(residuals_a, residuals_b)
        
        pair_correlations[(company_a, company_b)] = {
            'rho': rho,
            'p_value': p_value,
            'n_shared': len(shared_projects)
        }
        
        n_pairs_tested += 1
        if p_value < 0.05:
            n_pairs_significant += 1
    
    print(f"  Pairs tested (≥{MIN_SHARED_PROJECTS} shared projects): {n_pairs_tested:,}")
    print(f"  Significant pairs (p < 0.05): {n_pairs_significant:,}")
    
    return pair_correlations


def aggregate_correlations(pair_correlations):
    """Aggregate across-project correlations."""
    print("\nAggregating correlations...")
    
    # Extract all correlations (not just significant)
    all_rho = [v['rho'] for v in pair_correlations.values()]
    all_rho = np.array(all_rho)
    
    # Filter to significant only for subset analysis
    sig_rho = [v['rho'] for v in pair_correlations.values() if v['p_value'] < 0.05]
    sig_rho = np.array(sig_rho) if sig_rho else np.array([])
    
    mean_rho = np.mean(all_rho)
    std_rho = np.std(all_rho)
    
    # Bootstrap CI on all correlations
    np.random.seed(RANDOM_SEED)
    bootstrap_means = []
    for _ in range(N_BOOTSTRAP):
        sample = np.random.choice(all_rho, size=len(all_rho), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    results = {
        'n_pairs_all': len(all_rho),
        'n_pairs_significant': len(sig_rho),
        'mean_correlation': float(mean_rho),
        'std_correlation': float(std_rho),
        'median_correlation': float(np.median(all_rho)),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'all_correlations': all_rho.tolist(),
        'significant_correlations': sig_rho.tolist() if len(sig_rho) > 0 else [],
        'sig_mean': float(np.mean(sig_rho)) if len(sig_rho) > 0 else None,
        'sig_std': float(np.std(sig_rho)) if len(sig_rho) > 0 else None,
    }
    
    # Add pair-level details
    pair_details = {}
    for (a, b), v in pair_correlations.items():
        pair_details[f"{a}_{b}"] = v
    results['pair_details'] = pair_details
    
    return results


# ==============================================================================
# HYPOTHESIS TEST
# ==============================================================================

def test_hypothesis(results):
    """Test copula skip hypothesis."""
    mean_rho = results['mean_correlation']
    ci_lower = results['ci_95_lower']
    ci_upper = results['ci_95_upper']
    
    correlations = np.array(results['all_correlations'])
    
    # One-sample t-test against 0
    t_stat_zero, p_value_zero = stats.ttest_1samp(correlations, 0)
    
    # One-sample t-test against skip threshold
    t_stat_skip, p_value_skip = stats.ttest_1samp(correlations, RHO_SKIP_THRESHOLD)
    
    # Determine decision
    if ci_upper < RHO_SKIP_THRESHOLD:
        decision = 'SKIP_COPULA'
        confidence = 'HIGH'
        rationale = f"95% CI upper ({ci_upper:.3f}) < skip threshold ({RHO_SKIP_THRESHOLD})"
    elif mean_rho < RHO_SKIP_THRESHOLD and ci_upper < RHO_IMPLEMENT_THRESHOLD:
        decision = 'SKIP_COPULA'
        confidence = 'MEDIUM'
        rationale = f"Mean ρ ({mean_rho:.3f}) < skip threshold, CI within bounds"
    elif mean_rho > RHO_IMPLEMENT_THRESHOLD:
        decision = 'IMPLEMENT_COPULA'
        confidence = 'HIGH' if ci_lower > RHO_IMPLEMENT_THRESHOLD else 'MEDIUM'
        rationale = f"Mean ρ ({mean_rho:.3f}) > implement threshold ({RHO_IMPLEMENT_THRESHOLD})"
    else:
        decision = 'GRAY_ZONE'
        confidence = 'LOW'
        rationale = f"Mean ρ ({mean_rho:.3f}) between thresholds - manual review needed"
    
    # Sanity check for negative mean
    if mean_rho < -0.10:
        sanity_warning = "⚠️ Strong negative correlation detected - unusual for sealed bids"
    elif mean_rho < 0:
        sanity_warning = "Note: Slight negative correlation (competitive displacement)"
    else:
        sanity_warning = None
    
    return {
        'decision': decision,
        'confidence': confidence,
        'rationale': rationale,
        'sanity_warning': sanity_warning,
        'mean_rho': float(mean_rho),
        'ci_95': [float(ci_lower), float(ci_upper)],
        't_statistic_vs_zero': float(t_stat_zero),
        'p_value_vs_zero': float(p_value_zero),
        't_statistic_vs_threshold': float(t_stat_skip),
        'p_value_vs_threshold': float(p_value_skip),
        'skip_threshold': RHO_SKIP_THRESHOLD,
        'implement_threshold': RHO_IMPLEMENT_THRESHOLD
    }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_visualizations(results, hypothesis_result, output_dir):
    """Create diagnostic visualizations."""
    print("\nCreating visualizations...")
    
    correlations = np.array(results['all_correlations'])
    
    # 1. Correlation distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(correlations, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, label='Zero (independence)')
    ax.axvline(x=results['mean_correlation'], color='red', linestyle='-', linewidth=2, 
               label=f'Mean ρ = {results["mean_correlation"]:.3f}')
    ax.axvline(x=RHO_SKIP_THRESHOLD, color='green', linestyle='--', linewidth=2,
               label=f'Skip threshold = {RHO_SKIP_THRESHOLD}')
    ax.axvline(x=RHO_IMPLEMENT_THRESHOLD, color='orange', linestyle='--', linewidth=2,
               label=f'Implement threshold = {RHO_IMPLEMENT_THRESHOLD}')
    
    ax.axvspan(results['ci_95_lower'], results['ci_95_upper'], alpha=0.2, color='red',
               label=f'95% CI: [{results["ci_95_lower"]:.3f}, {results["ci_95_upper"]:.3f}]')
    
    ax.set_xlabel('Across-Project Residual Correlation (ρ)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'CORRECTED Copula Test: {hypothesis_result["decision"]}\n' +
                 f'Mean ρ = {results["mean_correlation"]:.3f} (across {results["n_pairs_all"]:,} company pairs)',
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'copula_correlation_dist_CORRECTED.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: copula_correlation_dist_CORRECTED.png")
    
    # 2. QQ-plot
    fig, ax = plt.subplots(figsize=(8, 8))
    stats.probplot(correlations, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Across-Project Correlations', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'copula_qq_plot_CORRECTED.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: copula_qq_plot_CORRECTED.png")
    
    # 3. Top pairs (extreme correlations)
    pair_details = results.get('pair_details', {})
    if len(pair_details) >= 20:
        sorted_pairs = sorted(pair_details.items(), key=lambda x: abs(x[1]['rho']), reverse=True)[:20]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pair_names = [p[0] for p in sorted_pairs]
        pair_rhos = [p[1]['rho'] for p in sorted_pairs]
        pair_ns = [p[1]['n_shared'] for p in sorted_pairs]
        
        colors = ['red' if r > 0 else 'green' for r in pair_rhos]
        bars = ax.barh(range(len(pair_names)), pair_rhos, color=colors, alpha=0.7)
        
        # Add n_shared annotations
        for i, (n, bar) in enumerate(zip(pair_ns, bars)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'n={n}', va='center', fontsize=8)
        
        ax.set_yticks(range(len(pair_names)))
        ax.set_yticklabels(pair_names, fontsize=8)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=RHO_SKIP_THRESHOLD, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=-RHO_SKIP_THRESHOLD, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Across-Project Correlation (ρ)', fontsize=12)
        ax.set_title('Top 20 Company Pairs by |ρ| (CORRECTED)', fontsize=14)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'copula_top_pairs_CORRECTED.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: copula_top_pairs_CORRECTED.png")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("COPULA HYPOTHESIS TEST (CORRECTED METHODOLOGY)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("Methodology: ACROSS-PROJECT correlation (correct)")
    print("  - Correlate company residuals across shared projects")
    print("  - NOT within-project standardization (wrong)")
    print()
    
    print(f"Hypothesis:")
    print(f"  H₀: ρ < {RHO_SKIP_THRESHOLD} → SKIP copula")
    print(f"  H₁: ρ > {RHO_IMPLEMENT_THRESHOLD} → IMPLEMENT copula")
    print()
    
    root = get_project_root()
    output_dir = root / 'analysis_results'
    
    # 1. Load data
    models, df = load_models_and_data()
    
    # 2. Encode features
    X, feature_cols = encode_features(df)
    
    # 3. Compute raw residuals
    df = compute_residuals(models, df, X)
    
    # 4. Compute ACROSS-PROJECT correlations (CORRECT!)
    pair_correlations = compute_pairwise_correlations_correct(df)
    
    # 5. Aggregate
    results = aggregate_correlations(pair_correlations)
    
    # 6. Test hypothesis
    hypothesis_result = test_hypothesis(results)
    
    # 7. Visualize
    create_visualizations(results, hypothesis_result, output_dir)
    
    # 8. Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY (CORRECTED)")
    print("=" * 70)
    print(f"  Company pairs tested: {results['n_pairs_all']:,}")
    print(f"  Significant (p<0.05): {results['n_pairs_significant']:,}")
    print()
    print(f"  Mean residual correlation (ρ): {results['mean_correlation']:.4f}")
    print(f"  Std: {results['std_correlation']:.4f}")
    print(f"  95% CI: [{results['ci_95_lower']:.4f}, {results['ci_95_upper']:.4f}]")
    print()
    print(f"  Decision: {hypothesis_result['decision']}")
    print(f"  Confidence: {hypothesis_result['confidence']}")
    print(f"  Rationale: {hypothesis_result['rationale']}")
    if hypothesis_result.get('sanity_warning'):
        print(f"  {hypothesis_result['sanity_warning']}")
    print("=" * 70)
    
    # 9. Save
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'across_project_correlation',
        'min_shared_projects': MIN_SHARED_PROJECTS,
        'hypothesis_result': hypothesis_result,
        'statistics': {
            'n_pairs_all': results['n_pairs_all'],
            'n_pairs_significant': results['n_pairs_significant'],
            'mean_correlation': results['mean_correlation'],
            'std_correlation': results['std_correlation'],
            'median_correlation': results['median_correlation'],
            'ci_95_lower': results['ci_95_lower'],
            'ci_95_upper': results['ci_95_upper'],
        }
    }
    
    results_path = output_dir / 'copula_test_results_CORRECTED.json'
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved: {results_path}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return full_results


if __name__ == "__main__":
    main()
