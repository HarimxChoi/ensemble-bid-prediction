"""
Simple Prediction Variation Test
================================

Test if models give different predictions for same vs different contexts.

Expected:
- Same context → Similar predictions
- Different contexts → Different predictions (NOT all 1.0005!)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def test_prediction_variation():
    print("="*70)
    print("PREDICTION VARIATION TEST")
    print("="*70)
    print()
    
    root = Path(__file__).parent
    
    # Load HPO models
    models = joblib.load(root / 'models' / 'company_models_hpo.pkl')
    print(f"Loaded {len(models)} HPO models")
    print()
    
    # Load data
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    train_df = pd.read_csv(root / 'data' / 'processed' / 'train_clean.csv')
    
    df = features_df.merge(train_df[['record_id', 'company_code']], on='record_id')
    
    # Encode features
    from strategic_bidding_optimizer import encode_features_for_inference
    X_all, feature_cols = encode_features_for_inference(df)
    
    # Test: Pick one company, get 20 random samples, check prediction variation
    test_company = 'C0022'
    
    if test_company not in models:
        print(f"⚠️ {test_company} not in models, using first available")
        test_company = list(models.keys())[0]
    
    company_df = df[df['company_code'] == test_company]
    company_indices = df[df['company_code'] == test_company].index
    
    if len(company_indices) < 20:
        print(f"⚠️ Only {len(company_indices)} samples for {test_company}")
        sample_indices = company_indices
    else:
        sample_indices = np.random.choice(company_indices, 20, replace=False)
    
    X_sample = X_all[sample_indices]
    
    # Get predictions
    model = models[test_company]
    predictions = model.predict(X_sample)
    
    print(f"Testing company: {test_company}")
    print(f"Samples: {len(predictions)}")
    print()
    
    print("Predictions:")
    print("-" * 70)
    for i, pred in enumerate(predictions[:10]):
        print(f"  Sample {i+1}: {pred:.6f}")
    if len(predictions) > 10:
        print(f"  ... and {len(predictions)-10} more")
    print()
    
    # Statistics
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    pred_min = predictions.min()
    pred_max = predictions.max()
    pred_range = pred_max - pred_min
    unique_preds = len(np.unique(predictions.round(6)))
    
    print("="*70)
    print("VARIATION STATISTICS")
    print("="*70)
    print(f"Mean: {pred_mean:.6f}")
    print(f"Std:  {pred_std:.6f}")
    print(f"Min:  {pred_min:.6f}")
    print(f"Max:  {pred_max:.6f}")
    print(f"Range: {pred_range:.6f} ({pred_range*100:.3f}%)")
    print(f"Unique: {unique_preds}/{len(predictions)}")
    print()
    
    # Verdict
    if unique_preds == 1:
        print("🚨 FAILED: All predictions identical!")
        print(f"   Model outputs constant: {pred_mean:.6f}")
    elif pred_std < 0.001:
        print("⚠️ WARNING: Very low variation")
        print(f"   Predictions vary by only {pred_range*100:.4f}%")
        print("   Model may not be context-aware")
    else:
        print("✅ SUCCESS: Predictions vary!")
        print(f"   Range: {pred_range*100:.3f}%")
        print(f"   Std: {pred_std*100:.3f}%")
        print("   Model IS context-aware!")
    
    print("="*70)
    
    # Test multiple companies
    print("\nTesting 5 companies:")
    print("-" * 70)
    
    test_companies = list(models.keys())[:5]
    all_preds = []
    
    for comp in test_companies:
        comp_df = df[df['company_code'] == comp]
        if len(comp_df) < 5:
            continue
        
        comp_indices = df[df['company_code'] == comp].index[:5]
        X_comp = X_all[comp_indices]
        
        preds = models[comp].predict(X_comp)
        pred_std_comp = preds.std()
        
        all_preds.extend(preds.tolist())
        
        print(f"{comp}: mean={preds.mean():.4f}, std={pred_std_comp:.4f}, range={preds.max()-preds.min():.4f}")
    
    # Overall diversity
    all_preds = np.array(all_preds)
    print()
    print("Overall diversity:")
    print(f"  All predictions std: {all_preds.std():.4f}")
    print(f"  All predictions range: {all_preds.max() - all_preds.min():.4f}")
    print("="*70)

if __name__ == "__main__":
    test_prediction_variation()
