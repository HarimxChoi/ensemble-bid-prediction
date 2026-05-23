"""
Validate judge inference using base_score from CS_BIDS_RESULT columns.
"""
import pandas as pd
import numpy as np
import math

def calculate_min_bid_rate(tech_score, bid_value1, bid_value2, bid_value3, base_score, price_comp=5):
    """Calculate min_bid_rate using the verified formula."""
    if pd.isna(tech_score) or pd.isna(bid_value1) or pd.isna(bid_value2) or pd.isna(bid_value3):
        return None
    if bid_value2 == 0:
        bid_value2 = 1
    try:
        tech_points = (tech_score / 100) * bid_value3 + base_score
        min_bid_rate_raw = (bid_value1 - tech_points) / bid_value2
        if not math.isfinite(min_bid_rate_raw):
            return None
        multiplier = 10 ** price_comp
        rounded = math.ceil(min_bid_rate_raw * multiplier) / multiplier
        return round(rounded - 0.005, 3)
    except:
        return None

def main():
    print("=" * 60)
    print("JUDGE INFERENCE VALIDATION (Using base_score from RESULT)")
    print("=" * 60)
    
    # Load EXPR formulas
    expr_df = pd.read_csv("mccp_simulation/dataset/raw_dataset/CS_BIDS_EXPR_202601061507.csv", encoding='cp949')
    expr_df = expr_df.dropna(subset=['BID_VALUE1', 'BID_VALUE2', 'BID_VALUE3'])
    
    candidates = []
    for _, row in expr_df.iterrows():
        candidates.append({
            'id': f"{int(row['JUDGE_BASE'])}-{int(row['JUDGE_SEQ'])}",
            'v1': row['BID_VALUE1'],
            'v2': row['BID_VALUE2'],
            'v3': row['BID_VALUE3'],
        })
    print(f"Loaded {len(candidates)} judge formulas.")
    
    # Load RESULT with score columns
    result_df = pd.read_csv("mccp_simulation/dataset/raw_dataset/CS_BIDS_RESULT_202601061459.csv", encoding='cp949')
    
    # Filter records with all required columns
    required = ['TECH_SCORE', 'BID_RATE', 'CAREER_SCORE', 'AREA_SCORE', 'MANAGE_SCORE']
    clean = result_df.dropna(subset=required)
    print(f"Records with base_score columns: {len(clean)} / {len(result_df)} ({len(clean)/len(result_df)*100:.1f}%)")
    
    # Calculate base_score per record
    clean = clean.copy()
    clean['base_score'] = clean['CAREER_SCORE'] + clean['AREA_SCORE'] + clean['MANAGE_SCORE']
    
    # Sample for testing
    sample = clean.sample(n=min(300, len(clean)))
    
    matches_found = 0
    multi_matches = 0
    no_matches = 0
    
    print(f"\nTesting {len(sample)} random samples:")
    print(f"{'TECH':<7} {'RATE':<8} {'BASE':<5} {'MATCHED':<15} {'CALC_RATE'}")
    print("-" * 55)
    
    details = []
    
    for idx, row in sample.iterrows():
        tech = row['TECH_SCORE']
        actual_rate = row['BID_RATE']
        base_score = row['base_score']
        
        matched = []
        for cand in candidates:
            calc = calculate_min_bid_rate(tech, cand['v1'], cand['v2'], cand['v3'], base_score)
            if calc is not None and abs(calc - actual_rate) < 0.002:
                matched.append((cand['id'], calc))
        
        if len(matched) == 1:
            matches_found += 1
            status = matched[0][0]
            calc_rate = matched[0][1]
        elif len(matched) > 1:
            multi_matches += 1
            matches_found += 1  # Still a match, just ambiguous
            status = f"MULTI({len(matched)})"
            calc_rate = matched[0][1]
        else:
            no_matches += 1
            status = "NONE"
            calc_rate = "-"
        
        details.append({'tech': tech, 'rate': actual_rate, 'base': base_score, 'status': status, 'calc': calc_rate})
    
    # Print first 25
    for d in details[:25]:
        print(f"{d['tech']:<7} {d['rate']:<8} {d['base']:<5} {str(d['status']):<15} {d['calc']}")
    
    print("-" * 55)
    print(f"Total Samples: {len(sample)}")
    print(f"✓ Matches:     {matches_found} ({matches_found/len(sample)*100:.1f}%)")
    print(f"  - Unique:    {matches_found - multi_matches}")
    print(f"  - Ambiguous: {multi_matches}")
    print(f"✗ No Match:    {no_matches} ({no_matches/len(sample)*100:.1f}%)")

if __name__ == "__main__":
    main()
