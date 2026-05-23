"""
Rank-Based Copula Analysis
==========================

Tests residual correlation between specific ranks across projects.
Does Rank 1's deviation correlate with Rank 2's deviation?

Methodology:
  1. Compute residuals for all bids
  2. Pivot table: Rows=Projects, Columns=Ranks (1..10)
  3. Compute correlation matrix of columns

Output:
  - analysis_results/rank_correlation_matrix.csv
  - analysis_results/rank_correlation_heatmap.png
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats

def get_project_root():
    return Path(__file__).parent

def main():
    print("=" * 70)
    print("RANK-BASED COPULA ANALYSIS")
    print("=" * 70)
    
    root = get_project_root()
    output_dir = root / 'analysis_results'
    
    # 1. Load data
    print("Loading data...")
    try:
        models = joblib.load(root / 'models' / 'company_models.pkl')
    except:
        # Fallback if running relative to script location
        models = joblib.load('models/company_models.pkl')
        
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    
    # Drop target-related columns from features if they exist (to avoid conflicts)
    cols_to_drop = [c for c in features_df.columns if c in ['target', 'normalized_bid_rate']]
    if cols_to_drop:
        features_df = features_df.drop(columns=cols_to_drop)
        
    train_df = pd.read_csv(
        root / 'data' / 'processed' / 'train_clean.csv',
        usecols=['record_id', 'company_code', 'notice_id', 'normalized_bid_rate']
    )
    # Rename normalized_bid_rate to target
    train_df = train_df.rename(columns={'normalized_bid_rate': 'target'})
    
    # Merge
    df = features_df.merge(train_df, on='record_id', how='left')
    
    print(f"  Columns available: {list(df.columns)}")
    if 'target' not in df.columns:
        raise ValueError("Target column missing after merge!")
    
    # 2. Encode and Predict (to get residuals)
    print("Computing residuals...")
    
    # Simple encoding (copy from copula_test.py logic)
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
    
    # Predict
    model_map = models  # Alias
    company_codes = df['company_code'].values
    y_actual = df['target'].values
    
    residuals = np.full(len(df), np.nan)
    
    count = 0
    for code, model in model_map.items():
        mask = (company_codes == code)
        if mask.sum() > 0:
            y_pred = model.predict(X[mask])
            residuals[mask] = y_actual[mask] - y_pred
            count += 1
            
    df['residual'] = residuals
    
    print(f"  Computed residuals for {count} companies")
    print(f"  Valid residuals: {df['residual'].notna().sum():,}")
    
    # 3. Pivot by Rank
    print("\nPivoting by Rank...")
    
    # Ensure l_ranking is integer
    df['rank'] = df['l_ranking'].astype(int)
    
    # Filter to valid residuals and Top 10 ranks
    df_pivot = df[
        (df['residual'].notna()) & 
        (df['rank'] <= 10)
    ][['notice_id', 'rank', 'residual']]
    
    # Aggregate duplicates (ties in rank) by taking mean of residuals
    # This handles cases where multiple companies share the same rank in a project
    df_agg = df_pivot.groupby(['notice_id', 'rank'])['residual'].mean().reset_index()
    
    # Pivot: Index=notice_id, Columns=rank, Values=residual
    rank_matrix = df_agg.pivot(index='notice_id', columns='rank', values='residual')
    
    print(f"  Projects with data: {len(rank_matrix):,}")
    print("  Bidder counts per rank:")
    print(rank_matrix.count().to_string())
    
    # 4. Compute Correlation Matrix
    print("\nComputing Correlation Matrix...")
    # min_periods=30 to ensure statistical relevance
    corr_matrix = rank_matrix.corr(min_periods=100)
    
    print("\nMean Correlation per Rank Pair:")
    print(corr_matrix.round(4))
    
    # 5. Summary Stats
    # Get upper triangle values (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    upper_values = corr_matrix.where(mask).stack().values
    
    print("\nSTATISTICS across all rank pairs:")
    print(f"  Mean ρ: {np.mean(upper_values):.4f}")
    print(f"  Median ρ: {np.median(upper_values):.4f}")
    print(f"  Max ρ: {np.max(upper_values):.4f} (Pair: {corr_matrix.where(mask).stack().idxmax()})")
    print(f"  Min ρ: {np.min(upper_values):.4f}")
    
    # 6. Visualization
    print("\nGenerating heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                vmin=-0.15, vmax=0.15, cbar_kws={'label': 'Residual Correlation'})
    plt.title("Rank-Based Residual Correlation Matrix\n(Rank i vs Rank j across projects)")
    plt.tight_layout()
    plt.savefig(output_dir / 'rank_correlation_heatmap.png', dpi=300)
    print(f"  Saved: rank_correlation_heatmap.png")
    
    # Save CSV
    corr_matrix.to_csv(output_dir / 'rank_correlation_matrix.csv')
    print(f"  Saved: rank_correlation_matrix.csv")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    max_rho = np.max(upper_values)
    mean_rho = np.mean(upper_values)
    
    if max_rho < 0.15:
        print(f"✅ SKIP COPULA confirmed.")
        print(f"  Max rank correlation ({max_rho:.3f}) < 0.15")
    else:
        print(f"⚠️ POTENTIAL CORRELATION DETECTED.")
        print(f"  Max rank correlation ({max_rho:.3f}) approaches threshold")

    # Check specifically Rank 1 vs Rank 2
    r1_r2 = corr_matrix.loc[1, 2]
    print(f"  Rank 1 vs Rank 2: {r1_r2:.4f}")
    
    print("\nCompleted.")

if __name__ == "__main__":
    main()
