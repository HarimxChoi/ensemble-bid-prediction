"""
Quick test: Verify winner determination with real data
"""

import pandas as pd
import numpy as np

# Recreate the exact scenario from notice_id 2925
data = {
    'company_code': ['C0009', 'C0024', 'C0021', 'C0050', 'C0003', 'C0004'],
    'min_bid_rate': [0.80715, 0.80705, 0.80665, 0.80655, 0.80655, 0.80265],
    'normalized_bid_rate': [1.009983497, 0.994003744, 1.008700487, 0.993542347, 0.99509839, 1.008883761],
    'actual_bid_amt': [2260450000, 2224410000, 2256180000, 2222000000, 2225480000, 2245400000],
    'is_winner': [False, False, False, False, False, True]
}

bids_df = pd.DataFrame(data)

# Known values from data
base_amt = 2772850000
yega_rate = 100.3322529  # Calculated from estimated_price / base_amt * 100

# Apply our logic
estimated_price = base_amt * (yega_rate / 100)
print(f"estimated_price: {estimated_price:,.0f}")
print()

# Calculate each company's bid and threshold
bids_df['bid_amount'] = estimated_price * bids_df['min_bid_rate'] * bids_df['normalized_bid_rate']
bids_df['threshold'] = estimated_price * bids_df['min_bid_rate']
bids_df['valid'] = bids_df['bid_amount'] >= bids_df['threshold']

print("Company Analysis:")
print("="*100)
for idx, row in bids_df.iterrows():
    valid_mark = "✅" if row['valid'] else "❌"
    winner_mark = "🏆" if row['is_winner'] else "  "
    print(f"{winner_mark} {row['company_code']}: "
          f"bid={row['bid_amount']:>14,.0f}  "
          f"threshold={row['threshold']:>14,.0f}  "
          f"valid={valid_mark}  "
          f"(actual={row['actual_bid_amt']:>14,.0f})")

print()

# Filter valid and find winner
valid_bids = bids_df[bids_df['valid']]
if len(valid_bids) > 0:
    winner_idx = valid_bids['bid_amount'].idxmin()
    predicted_winner = valid_bids.loc[winner_idx, 'company_code']
    actual_winner = bids_df[bids_df['is_winner']==True]['company_code'].values[0]
    
    print(f"Predicted winner: {predicted_winner}")
    print(f"Actual winner:    {actual_winner}")
    print(f"Match: {'✅ YES!' if predicted_winner == actual_winner else '❌ NO'}")
    
    if predicted_winner == actual_winner:
        print("\n🎯 WINNER LOGIC VERIFIED!")
else:
    print("❌ No valid bids found")
