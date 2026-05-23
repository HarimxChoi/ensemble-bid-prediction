"""
Verification Script: Normalized Bid Rate Formula
================================================

Verify if stored normalized_bid_rate matches expected formula:
  normalized_bid_rate = bid_amt / (base_amt × min_bid_rate)

This checks if yega_rate is excluded from normalization (as expected).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def verify_normalized_formula():
    print("=" * 70)
    print("NORMALIZED BID RATE FORMULA VERIFICATION")
    print("=" * 70)
    print()
    
    # Load data
    data_path = Path('data/processed/train_clean.csv')
    print(f"Loading: {data_path}")
    
    df = pd.read_csv(data_path, nrows=100)  # Check first 100 rows
    print(f"Loaded {len(df)} rows")
    print()
    
    # Check required columns exist
    required_cols = ['base_amt', 'min_bid_rate', 'normalized_bid_rate']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        print(f"❌ Missing columns: {missing}")
        print(f"\nAvailable columns: {list(df.columns)}")
        return
    
    # Check if we have bid_amt or need to reconstruct it
    if 'bid_amt' not in df.columns:
        print("⚠️ 'bid_amt' not in data")
        print("Looking for alternative columns to reconstruct bid_amt...")
        print(f"\nColumns with 'bid' or 'amount': {[c for c in df.columns if 'bid' in c.lower() or 'amount' in c.lower()]}")
        return
    
    print("Testing Formula: normalized_bid_rate = bid_amt / (base_amt × min_bid_rate)")
    print()
    
    # Verify formula for each row
    matches = 0
    mismatches = 0
    errors = []
    
    for idx, row in df.iterrows():
        try:
            bid_amt = row['bid_amt']
            base_amt = row['base_amt']
            min_bid_rate = row['min_bid_rate']
            stored_normalized = row['normalized_bid_rate']
            
            # Expected formula (user's hypothesis)
            # min_bid_rate might be percentage (85.0) or decimal (0.85)
            # Let's try both
            
            # Option 1: min_bid_rate as decimal
            calc_normalized_v1 = bid_amt / (base_amt * min_bid_rate)
            
            # Option 2: min_bid_rate as percentage
            calc_normalized_v2 = bid_amt / (base_amt * (min_bid_rate / 100))
            
            # Check which matches better
            diff_v1 = abs(calc_normalized_v1 - stored_normalized)
            diff_v2 = abs(calc_normalized_v2 - stored_normalized)
            
            if diff_v1 < 0.001:
                matches += 1
                if idx < 5:  # Print first 5 matches
                    print(f"✅ Row {idx}: MATCH (decimal)")
                    print(f"   bid_amt: {bid_amt:,.0f}")
                    print(f"   base_amt: {base_amt:,.0f}")
                    print(f"   min_bid_rate: {min_bid_rate:.4f}")
                    print(f"   Calculated: {calc_normalized_v1:.6f}")
                    print(f"   Stored:     {stored_normalized:.6f}")
                    print()
            elif diff_v2 < 0.001:
                matches += 1
                if idx < 5:
                    print(f"✅ Row {idx}: MATCH (percentage)")
                    print(f"   bid_amt: {bid_amt:,.0f}")
                    print(f"   base_amt: {base_amt:,.0f}")
                    print(f"   min_bid_rate: {min_bid_rate:.4f}")
                    print(f"   Calculated: {calc_normalized_v2:.6f}")
                    print(f"   Stored:     {stored_normalized:.6f}")
                    print()
            else:
                mismatches += 1
                if len(errors) < 5:  # Store first 5 mismatches
                    errors.append({
                        'idx': idx,
                        'bid_amt': bid_amt,
                        'base_amt': base_amt,
                        'min_bid_rate': min_bid_rate,
                        'calc_v1': calc_normalized_v1,
                        'calc_v2': calc_normalized_v2,
                        'stored': stored_normalized,
                        'diff_v1': diff_v1,
                        'diff_v2': diff_v2
                    })
        except Exception as e:
            print(f"❌ Error at row {idx}: {e}")
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows checked: {len(df)}")
    print(f"Matches: {matches} ({matches/len(df)*100:.1f}%)")
    print(f"Mismatches: {mismatches} ({mismatches/len(df)*100:.1f}%)")
    print()
    
    if matches == len(df):
        print("✅ FORMULA VERIFIED!")
        print("   normalized_bid_rate = bid_amt / (base_amt × min_bid_rate)")
        print("   No changes needed to feature engineering.")
    elif matches > len(df) * 0.9:
        print("⚠️ MOSTLY MATCHES (>90%)")
        print("   Small discrepancies might be rounding errors.")
    else:
        print("❌ FORMULA MISMATCH!")
        print("   Current formula is DIFFERENT from expected.")
        print()
        print("First few mismatches:")
        for err in errors[:3]:
            print(f"\n  Row {err['idx']}:")
            print(f"    bid_amt: {err['bid_amt']:,.0f}")
            print(f"    base_amt: {err['base_amt']:,.0f}")
            print(f"    min_bid_rate: {err['min_bid_rate']:.4f}")
            print(f"    Calc (decimal):  {err['calc_v1']:.6f} (diff: {err['diff_v1']:.6f})")
            print(f"    Calc (percent):  {err['calc_v2']:.6f} (diff: {err['diff_v2']:.6f})")
            print(f"    Stored:          {err['stored']:.6f}")
        
        print()
        print("🔍 NEED TO INVESTIGATE:")
        print("   - What IS the current formula?")
        print("   - Does normalized_bid_rate include yega_rate?")
        print()
        print("   Possible current formula:")
        print("   normalized_bid_rate = bid_amt / (base_amt × yega_rate × min_bid_rate)")
        print("   or")
        print("   normalized_bid_rate = bid_amt / (base_amt × yega_rate)")
    
    print("=" * 70)

if __name__ == "__main__":
    verify_normalized_formula()
