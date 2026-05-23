# -*- coding: utf-8 -*-
"""
Institution Feature Engineering Pipeline
=========================================

Generates 3 tables for institutional analysis:
1. inst_yega_timeseries.csv - Monthly aggregation for frontend + trends
2. inst_features.csv - Rolling window features for ML training
3. inst_profile_static.csv - Static profile per institution

All rates in DECIMAL format (e.g., 0.998 not 99.8%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent / 'data' / 'processed'
OUTPUT_DIR = DATA_DIR  # Output to same directory

# Minimum BID COUNT requirements for statistical significance
# If a window has fewer bids than this, fall back to longer window
MIN_BIDS_THRESHOLD = 20  # Minimum bids needed for meaningful stats

# Minimum months of data required (secondary check)
MIN_MONTHS_3M = 1  # At least 1 month of data
MIN_MONTHS_6M = 2  # At least 2 months of data
MIN_MONTHS_1Y = 3  # At least 3 months of data

# Total minimum bids for an institution to be included
MIN_TOTAL_BIDS_FOR_INCLUSION = 20  # Institutions with fewer total bids are excluded

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_trend(values):
    """Calculate linear regression slope."""
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    try:
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)
    except:
        return 0.0


def calculate_yega_rate(row):
    """Calculate yega_rate in DECIMAL format."""
    if row['base_amt'] > 0 and pd.notna(row['choose_avg']):
        return row['choose_avg'] / row['base_amt']
    return np.nan


# =============================================================================
# TABLE 1: INST_YEGA_TIMESERIES
# =============================================================================

def generate_inst_timeseries(df, inst_master=None):
    """
    Generate monthly time series for each institution.
    
    Output columns:
    - inst_code, announce_month, n_bids
    - mean_yega_rate, std_yega_rate, median_yega_rate
    - min_yega_rate, max_yega_rate, yega_deviation
    """
    print("\n=== Generating inst_yega_timeseries.csv ===")
    
    # Calculate yega_rate (DECIMAL)
    df = df.copy()
    df['yega_rate'] = df.apply(calculate_yega_rate, axis=1)
    
    # Extract year-month
    df['announce_month'] = pd.to_datetime(df['announce_date']).dt.to_period('M').astype(str)
    
    # Group by institution and month
    grouped = df.groupby(['institution_code', 'announce_month'])
    
    timeseries_data = []
    
    for (inst_code, month), group in grouped:
        yega_values = group['yega_rate'].dropna()
        
        if len(yega_values) == 0:
            continue
        
        row = {
            'inst_code': inst_code,
            'announce_month': month,
            'n_bids': len(group),
            'mean_yega_rate': yega_values.mean(),
            'std_yega_rate': yega_values.std() if len(yega_values) > 1 else 0.0,
            'median_yega_rate': yega_values.median(),
            'min_yega_rate': yega_values.min(),
            'max_yega_rate': yega_values.max(),
            'yega_deviation': yega_values.mean() - 1.0,  # Deviation from 1.0 (100%)
        }
        
        # Winner statistics
        winners = group[group['is_winner'] == True]
        if len(winners) > 0:
            row['mean_winner_margin'] = winners['winner_margin'].mean()
            row['mean_winner_normalized_rate'] = winners['winner_normalized_rate'].mean()
        else:
            row['mean_winner_margin'] = np.nan
            row['mean_winner_normalized_rate'] = np.nan
        
        timeseries_data.append(row)
    
    timeseries_df = pd.DataFrame(timeseries_data)
    
    # Add institution name if master available
    if inst_master is not None:
        timeseries_df = timeseries_df.merge(
            inst_master[['institution_code', 'institution_name']],
            left_on='inst_code',
            right_on='institution_code',
            how='left'
        )
        timeseries_df = timeseries_df.drop(columns=['institution_code'], errors='ignore')
        timeseries_df = timeseries_df.rename(columns={'institution_name': 'inst_name'})
    
    print(f"  Generated {len(timeseries_df):,} rows for {timeseries_df['inst_code'].nunique()} institutions")
    
    return timeseries_df


# =============================================================================
# TABLE 2: INST_FEATURES (Rolling Window Features)
# =============================================================================

def generate_inst_features(df, timeseries_df):
    """
    Generate rolling window features for each record.
    
    Uses 3-month primary window with fallback to 6m/1y.
    """
    print("\n=== Generating inst_features.csv ===")
    
    df = df.copy()
    df['yega_rate'] = df.apply(calculate_yega_rate, axis=1)
    df['announce_date'] = pd.to_datetime(df['announce_date'])
    df['announce_month'] = df['announce_date'].dt.to_period('M').astype(str)
    
    # Pre-compute timeseries lookup
    timeseries_df = timeseries_df.copy()
    timeseries_df['announce_month_dt'] = pd.to_datetime(timeseries_df['announce_month'])
    
    # Group timeseries by institution
    ts_by_inst = {}
    for inst_code, group in timeseries_df.groupby('inst_code'):
        ts_by_inst[inst_code] = group.sort_values('announce_month_dt')
    
    features_list = []
    total = len(df)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if i > 0 and i % 10000 == 0:
            print(f"  Processed {i:,} / {total:,} ({100*i/total:.1f}%)")
        
        inst_code = row['institution_code']
        announce_date = row['announce_date']
        announce_month = pd.to_datetime(row['announce_month'])
        current_yega = row['yega_rate']
        
        # Get historical timeseries for this institution
        if inst_code in ts_by_inst:
            hist = ts_by_inst[inst_code]
            hist = hist[hist['announce_month_dt'] < announce_month]
        else:
            hist = pd.DataFrame()
        
        # Initialize feature dict
        feat = {
            'record_id': row['record_id'],
            'inst_code': inst_code,
            'announce_date': announce_date,
        }
        
        # Calculate all windows first
        hist_3m = hist.tail(3)
        hist_6m = hist.tail(6)
        hist_1y = hist.tail(12)
        
        # Compute bid counts for each window
        n_bids_3m = hist_3m['n_bids'].sum() if len(hist_3m) >= MIN_MONTHS_3M else 0
        n_bids_6m = hist_6m['n_bids'].sum() if len(hist_6m) >= MIN_MONTHS_6M else 0
        n_bids_1y = hist_1y['n_bids'].sum() if len(hist_1y) >= MIN_MONTHS_1Y else 0
        
        # 3-month window features (always compute if data exists)
        if len(hist_3m) >= MIN_MONTHS_3M:
            feat['inst_n_bids_3m'] = n_bids_3m
            feat['inst_mean_yega_3m'] = hist_3m['mean_yega_rate'].mean()
            feat['inst_std_yega_3m'] = hist_3m['mean_yega_rate'].std()
            feat['inst_trend_3m'] = calculate_trend(hist_3m['mean_yega_rate'].values)
            feat['inst_mean_winner_margin_3m'] = hist_3m['mean_winner_margin'].mean()
            
            if feat['inst_mean_yega_3m'] > 0:
                feat['inst_yega_volatility_3m'] = feat['inst_std_yega_3m'] / feat['inst_mean_yega_3m']
            else:
                feat['inst_yega_volatility_3m'] = np.nan
        else:
            feat['inst_n_bids_3m'] = np.nan
            feat['inst_mean_yega_3m'] = np.nan
            feat['inst_std_yega_3m'] = np.nan
            feat['inst_trend_3m'] = np.nan
            feat['inst_mean_winner_margin_3m'] = np.nan
            feat['inst_yega_volatility_3m'] = np.nan
        
        # 6-month window features
        if len(hist_6m) >= MIN_MONTHS_6M:
            feat['inst_n_bids_6m'] = n_bids_6m
            feat['inst_mean_yega_6m'] = hist_6m['mean_yega_rate'].mean()
            feat['inst_std_yega_6m'] = hist_6m['mean_yega_rate'].std()
            feat['inst_trend_6m'] = calculate_trend(hist_6m['mean_yega_rate'].values)
            feat['inst_mean_winner_margin_6m'] = hist_6m['mean_winner_margin'].mean()
        else:
            feat['inst_n_bids_6m'] = np.nan
            feat['inst_mean_yega_6m'] = np.nan
            feat['inst_std_yega_6m'] = np.nan
            feat['inst_trend_6m'] = np.nan
            feat['inst_mean_winner_margin_6m'] = np.nan
        
        # 1-year window features
        if len(hist_1y) >= MIN_MONTHS_1Y:
            feat['inst_n_bids_1y'] = n_bids_1y
            feat['inst_mean_yega_1y'] = hist_1y['mean_yega_rate'].mean()
            feat['inst_std_yega_1y'] = hist_1y['mean_yega_rate'].std()
            feat['inst_trend_1y'] = calculate_trend(hist_1y['mean_yega_rate'].values)
            feat['inst_mean_winner_margin_1y'] = hist_1y['mean_winner_margin'].mean()
        else:
            feat['inst_n_bids_1y'] = np.nan
            feat['inst_mean_yega_1y'] = np.nan
            feat['inst_std_yega_1y'] = np.nan
            feat['inst_trend_1y'] = np.nan
            feat['inst_mean_winner_margin_1y'] = np.nan
        
        # === ADAPTIVE WINDOW SELECTION BASED ON BID COUNT ===
        # Use the shortest window that has >= MIN_BIDS_THRESHOLD bids
        # If 3m has >= 20 bids -> use 3m
        # Else if 6m has >= 20 bids -> use 6m
        # Else if 1y has >= 20 bids -> use 1y
        # Else -> insufficient data
        
        if n_bids_3m >= MIN_BIDS_THRESHOLD:
            active_mean = feat['inst_mean_yega_3m']
            active_std = feat['inst_std_yega_3m']
            active_n = n_bids_3m
            active_window = '3m'
        elif n_bids_6m >= MIN_BIDS_THRESHOLD:
            active_mean = feat['inst_mean_yega_6m']
            active_std = feat['inst_std_yega_6m']
            active_n = n_bids_6m
            active_window = '6m'
        elif n_bids_1y >= MIN_BIDS_THRESHOLD:
            active_mean = feat['inst_mean_yega_1y']
            active_std = feat['inst_std_yega_1y']
            active_n = n_bids_1y
            active_window = '1y'
        else:
            # Even with 1y, not enough bids - mark as insufficient
            active_mean = np.nan
            active_std = np.nan
            active_n = max(n_bids_3m, n_bids_6m, n_bids_1y)  # Store max available
            active_window = 'insufficient'
        
        feat['data_window_used'] = active_window
        feat['active_n_bids'] = active_n  # Store the actual bid count used
        
        # Consistency classification (std < 0.01 = stable)
        if pd.notna(active_std):
            if active_std < 0.008:  # 0.8% std
                feat['inst_yega_consistency'] = 'stable'
            elif active_std < 0.015:  # 1.5% std
                feat['inst_yega_consistency'] = 'moderate'
            else:
                feat['inst_yega_consistency'] = 'volatile'
        else:
            feat['inst_yega_consistency'] = 'unknown'
        
        # Frequency classification (based on bid count in active window)
        if active_n >= MIN_BIDS_THRESHOLD:
            if active_n >= 50:
                feat['inst_bidding_frequency'] = 'high'
            elif active_n >= 30:
                feat['inst_bidding_frequency'] = 'medium'
            else:
                feat['inst_bidding_frequency'] = 'low'
        else:
            feat['inst_bidding_frequency'] = 'insufficient'
        
        # Bias classification (deviation from 1.0)
        if pd.notna(active_mean):
            if active_mean < 0.995:  # Below 99.5%
                feat['inst_yega_bias'] = 'conservative'
            elif active_mean > 1.005:  # Above 100.5%
                feat['inst_yega_bias'] = 'aggressive'
            else:
                feat['inst_yega_bias'] = 'neutral'
        else:
            feat['inst_yega_bias'] = 'unknown'
        
        # Project vs Institution interaction
        if pd.notna(current_yega) and pd.notna(active_mean):
            feat['project_yega_vs_inst_mean'] = current_yega - active_mean
        else:
            feat['project_yega_vs_inst_mean'] = np.nan
        
        features_list.append(feat)
    
    features_df = pd.DataFrame(features_list)
    
    print(f"  Generated {len(features_df):,} feature rows")
    
    return features_df


# =============================================================================
# TABLE 3: INST_PROFILE_STATIC
# =============================================================================

def generate_inst_profile(df, timeseries_df, inst_master=None):
    """
    Generate static profile for each institution.
    """
    print("\n=== Generating inst_profile_static.csv ===")
    
    df = df.copy()
    df['yega_rate'] = df.apply(calculate_yega_rate, axis=1)
    df['announce_date'] = pd.to_datetime(df['announce_date'])
    
    profiles = []
    
    for inst_code in df['institution_code'].unique():
        inst_df = df[df['institution_code'] == inst_code]
        inst_ts = timeseries_df[timeseries_df['inst_code'] == inst_code]
        
        yega_values = inst_df['yega_rate'].dropna()
        
        profile = {
            'inst_code': inst_code,
            'total_bids': len(inst_df),
            'total_notices': inst_df['notice_id'].nunique(),
            'first_bid_date': inst_df['announce_date'].min(),
            'last_bid_date': inst_df['announce_date'].max(),
        }
        
        # Yega statistics
        if len(yega_values) > 0:
            profile['mean_yega_rate_all'] = yega_values.mean()
            profile['std_yega_rate_all'] = yega_values.std() if len(yega_values) > 1 else 0.0
            profile['median_yega_rate_all'] = yega_values.median()
            profile['min_yega_rate_all'] = yega_values.min()
            profile['max_yega_rate_all'] = yega_values.max()
            profile['yega_q25'] = yega_values.quantile(0.25)
            profile['yega_q75'] = yega_values.quantile(0.75)
            profile['yega_iqr'] = profile['yega_q75'] - profile['yega_q25']
        else:
            for col in ['mean_yega_rate_all', 'std_yega_rate_all', 'median_yega_rate_all',
                        'min_yega_rate_all', 'max_yega_rate_all', 'yega_q25', 'yega_q75', 'yega_iqr']:
                profile[col] = np.nan
        
        # Winner statistics
        winners = inst_df[inst_df['is_winner'] == True]
        if len(winners) > 0:
            profile['mean_winner_margin_all'] = winners['winner_margin'].mean()
            profile['std_winner_margin_all'] = winners['winner_margin'].std()
            profile['mean_n_competitors'] = inst_df.groupby('notice_id')['company_code'].nunique().mean()
        else:
            profile['mean_winner_margin_all'] = np.nan
            profile['std_winner_margin_all'] = np.nan
            profile['mean_n_competitors'] = np.nan
        
        # Trend
        if len(inst_ts) >= 3:
            profile['overall_trend'] = calculate_trend(inst_ts['mean_yega_rate'].values)
        else:
            profile['overall_trend'] = np.nan
        
        # Scores
        if profile['mean_yega_rate_all'] > 0:
            profile['yega_consistency_score'] = 1 - min(profile['std_yega_rate_all'] / profile['mean_yega_rate_all'], 1.0)
        else:
            profile['yega_consistency_score'] = np.nan
        
        # Bidding frequency (bids per month)
        date_range = (profile['last_bid_date'] - profile['first_bid_date']).days
        if date_range > 0:
            profile['bidding_frequency_score'] = profile['total_bids'] / (date_range / 30)
        else:
            profile['bidding_frequency_score'] = profile['total_bids']
        
        # Classification
        if profile['total_bids'] >= 50:
            profile['inst_tier'] = 'frequent'
        elif profile['total_bids'] >= 20:
            profile['inst_tier'] = 'moderate'
        else:
            profile['inst_tier'] = 'rare'
        
        if pd.notna(profile['std_yega_rate_all']):
            if profile['std_yega_rate_all'] < 0.008:
                profile['yega_pattern'] = 'stable'
            elif pd.notna(profile['overall_trend']) and profile['overall_trend'] > 0.001:
                profile['yega_pattern'] = 'trending_up'
            elif pd.notna(profile['overall_trend']) and profile['overall_trend'] < -0.001:
                profile['yega_pattern'] = 'trending_down'
            else:
                profile['yega_pattern'] = 'volatile'
        else:
            profile['yega_pattern'] = 'unknown'
        
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    
    # Add institution name if master available
    if inst_master is not None:
        profiles_df = profiles_df.merge(
            inst_master[['institution_code', 'institution_name']],
            left_on='inst_code',
            right_on='institution_code',
            how='left'
        )
        profiles_df = profiles_df.drop(columns=['institution_code'], errors='ignore')
        profiles_df = profiles_df.rename(columns={'institution_name': 'inst_name'})
        
        # Reorder columns to put inst_name after inst_code
        cols = profiles_df.columns.tolist()
        if 'inst_name' in cols:
            cols.remove('inst_name')
            cols.insert(1, 'inst_name')
            profiles_df = profiles_df[cols]
    
    print(f"  Generated {len(profiles_df):,} institution profiles")
    
    return profiles_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Institution Feature Engineering Pipeline")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Load data
    print("\n=== Loading Data ===")
    
    train_df = pd.read_csv(DATA_DIR / 'train_clean.csv')  # Use cleaned data
    val_df = pd.read_csv(DATA_DIR / 'val.csv')
    
    # Load institution master if available
    inst_master_path = DATA_DIR / 'institution_master.csv'
    if inst_master_path.exists():
        inst_master = pd.read_csv(inst_master_path)
        print(f"  Loaded institution_master.csv: {len(inst_master)} institutions")
    else:
        inst_master = None
        print("  institution_master.csv not found, proceeding without names")
    
    print(f"  Loaded train.csv: {len(train_df):,} records")
    print(f"  Loaded val.csv: {len(val_df):,} records")
    print(f"  Unique institutions in train: {train_df['institution_code'].nunique()}")
    
    # Combined for frontend (train + val)
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"  Combined (train+val): {len(combined_df):,} records")
    
    # ===========================================
    # TABLE 1: Time Series (use combined for frontend)
    # ===========================================
    timeseries_df = generate_inst_timeseries(combined_df, inst_master)
    timeseries_df.to_csv(OUTPUT_DIR / 'inst_yega_timeseries.csv', index=False)
    print(f"  Saved to {OUTPUT_DIR / 'inst_yega_timeseries.csv'}")
    
    # ===========================================
    # TABLE 2: Features (train only for ML)
    # ===========================================
    # First generate train-based timeseries for feature computation
    train_timeseries_df = generate_inst_timeseries(train_df, inst_master)
    
    # Generate features for TRAIN
    train_features_df = generate_inst_features(train_df, train_timeseries_df)
    print(f"  Generated {len(train_features_df):,} feature rows for TRAIN")
    
    # Generate features for VAL (using same timeseries for consistency)
    val_features_df = generate_inst_features(val_df, train_timeseries_df)
    print(f"  Generated {len(val_features_df):,} feature rows for VAL")
    
    # Combine train + val features
    features_df = pd.concat([train_features_df, val_features_df], ignore_index=True)
    print(f"  Total features: {len(features_df):,} rows (train + val)")
    
    features_df.to_csv(OUTPUT_DIR / 'inst_features.csv', index=False)
    print(f"  Saved to {OUTPUT_DIR / 'inst_features.csv'}")
    
    # ===========================================
    # TABLE 3: Static Profile (use combined for completeness)
    # ===========================================
    profile_df = generate_inst_profile(combined_df, timeseries_df, inst_master)
    profile_df.to_csv(OUTPUT_DIR / 'inst_profile_static.csv', index=False)
    print(f"  Saved to {OUTPUT_DIR / 'inst_profile_static.csv'}")
    
    # ===========================================
    # SUMMARY
    # ===========================================
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"\nOutput files:")
    print(f"  1. inst_yega_timeseries.csv: {len(timeseries_df):,} rows")
    print(f"  2. inst_features.csv: {len(features_df):,} rows")
    print(f"  3. inst_profile_static.csv: {len(profile_df):,} rows")
    
    # Quick stats on features
    print(f"\nFeature coverage (data availability):")
    for window in ['3m', '6m', '1y']:
        col = f'inst_mean_yega_{window}'
        coverage = features_df[col].notna().mean() * 100
        print(f"  {window} window: {coverage:.1f}% records have data")
    
    print(f"\nAdaptive window selection (MIN_BIDS_THRESHOLD = {MIN_BIDS_THRESHOLD}):")
    print(features_df['data_window_used'].value_counts().to_string())
    
    # Bid count distribution
    print(f"\nBid count statistics in active windows:")
    sufficient = features_df[features_df['data_window_used'] != 'insufficient']
    if len(sufficient) > 0:
        print(f"  Mean bids: {sufficient['active_n_bids'].mean():.1f}")
        print(f"  Median bids: {sufficient['active_n_bids'].median():.1f}")
        print(f"  Min bids: {sufficient['active_n_bids'].min():.0f}")
        print(f"  Max bids: {sufficient['active_n_bids'].max():.0f}")
    
    insufficient = features_df[features_df['data_window_used'] == 'insufficient']
    print(f"\nRecords with insufficient data (<{MIN_BIDS_THRESHOLD} bids in 1yr): {len(insufficient):,} ({100*len(insufficient)/len(features_df):.1f}%)")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return {
        'timeseries': timeseries_df,
        'features': features_df,
        'profile': profile_df
    }


if __name__ == '__main__':
    results = main()
