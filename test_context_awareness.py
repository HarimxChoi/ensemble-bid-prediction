"""
Quick Context Awareness Test
============================

Test if new HPO models provide different recommendations for different contexts.

Expected:
- Static 1.0005 → FAIL ❌
- Bid variation 0.5-1.0% → SUCCESS ✅
"""

import pandas as pd
import numpy as np
from pathlib import Path
from strategic_bidding_optimizer import StrategicBiddingOptimizer, encode_features_for_inference

def test_context_awareness():
    print("="*70)
    print("CONTEXT AWARENESS TEST")
    print("="*70)
    print()
    
    optimizer = StrategicBiddingOptimizer()
    
    # Load data
    root = Path(__file__).parent
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    train_df = pd.read_csv(root / 'data' / 'processed' / 'train_clean.csv')
    
    df = features_df.merge(train_df, on='record_id')
    X_all, feature_cols = encode_features_for_inference(df)
    
    # Test on 10 diverse projects
    results = []
    
    for notice_id in df['notice_id'].unique()[:20]:
        project_df = df[df['notice_id'] == notice_id]
        
        if len(project_df) < 3:
            continue
        
        winner = project_df[project_df['is_winner'] == True]
        if len(winner) == 0:
            continue
        
        winner_row = winner.iloc[0]
        company = winner_row['company_code']
        
        # Get project data
        project_indices = df[df['notice_id'] == notice_id].index
        companies = project_df['company_code'].tolist()
        X_features = X_all[project_indices]
        
        project_data = {
            'companies': companies,
            'X_features': X_features,
            'base_amt': project_df.iloc[0]['base_amt'],
            'inst_code': project_df.iloc[0].get('institution_code', None),
            'min_bid_rates': project_df['min_bid_rate'].values
        }
        
        # Optimize
        try:
            optimal, _ = optimizer.optimize_bid(
                company_code=company,
                project_data=project_data,
                n_iterations=10000,  # Quick test
                show_progress=False
            )
            
            results.append({
                'notice_id': notice_id,
                'company': company,
                'n_competitors': len(project_df),
                'recommended_bid': optimal['candidate_bid'],
                'p_win': optimal['p_win']
            })
        except:
            continue
        
        if len(results) >= 10:
            break
    
    # Analyze
    if len(results) == 0:
        print("⚠️ No results collected - all optimizations may have failed")
        return None
    
    results_df = pd.DataFrame(results)
    
    print("\nRESULTS:")
    print("="*70)
    print(results_df.to_string(index=False))
    print()
    
    # Check variation
    bid_std = results_df['recommended_bid'].std()
    bid_range = results_df['recommended_bid'].max() - results_df['recommended_bid'].min()
    unique_bids = results_df['recommended_bid'].nunique()
    
    print("="*70)
    print("CONTEXT AWARENESS METRICS")
    print("="*70)
    print(f"Unique recommendations: {unique_bids}/{len(results_df)}")
    print(f"Bid std: {bid_std:.5f}")
    print(f"Bid range: {bid_range:.5f} ({bid_range*100:.2f}%)")
    print()
    
    if unique_bids == 1:
        print("🚨 FAILED: All bids identical (static problem persists!)")
        print(f"   All bids = {results_df['recommended_bid'].iloc[0]:.6f}")
    elif bid_std < 0.001:
        print("⚠️ WARNING: Very low variation")
        print(f"   Bids vary by only {bid_range*100:.3f}%")
    else:
        print("✅ SUCCESS: Models ARE context-aware!")
        print(f"   Bids vary by {bid_range*100:.2f}%")
        print(f"   {unique_bids} unique recommendations")
    
    print("="*70)
    
    return results_df

if __name__ == "__main__":
    test_context_awareness()
