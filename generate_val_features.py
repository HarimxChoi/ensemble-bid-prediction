"""
Generate Features for Validation Data
======================================

Uses the same feature engineering as training data.
Outputs to analysis_results/feature_sample_val.csv

Author: Harim Choi
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from feature_engineering.py
from feature_engineering import (
    extract_features_batch,
    DATA_DIR,
    OUTPUT_DIR
)

def main():
    print("="*60)
    print("Generating Features for Validation Data")
    print("="*60)
    
    # Load val.csv instead of train_clean.csv
    val_path = DATA_DIR / 'val.csv'
    val_df = pd.read_csv(val_path)
    
    # Convert dates
    val_df['announce_date'] = pd.to_datetime(val_df['announce_date'])
    val_df['bid_date'] = pd.to_datetime(val_df['bid_date'].astype(str), format='%Y%m%d', errors='coerce')
    
    print(f"Loaded {len(val_df):,} records from val.csv")
    print(f"Unique companies: {val_df['company_code'].nunique()}")
    print(f"Unique notices: {val_df['notice_id'].nunique()}")
    print()
    
    # Load pre-computed institution features
    inst_features_path = DATA_DIR / 'inst_features.csv'
    if inst_features_path.exists():
        inst_features_df = pd.read_csv(inst_features_path)
        print(f"Loaded {len(inst_features_df):,} inst_features records")
    else:
        inst_features_df = None
        print("WARNING: inst_features.csv not found")
    
    # Extract features
    print("\n=== Feature Extraction ===")
    features_df, targets = extract_features_batch(val_df, inst_features_df=inst_features_df, sample_size=None)
    
    # Save
    features_df['target'] = targets
    output_path = OUTPUT_DIR / 'feature_sample_val.csv'
    features_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved features to {output_path}")
    print(f"Features: {len(features_df.columns)} columns")
    print(f"Records: {len(features_df):,}")
    
    return features_df

if __name__ == '__main__':
    features = main()
