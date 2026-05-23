# -*- coding: utf-8 -*-
"""
Feature Engineering for Company Model
=====================================

Comprehensive feature engineering with SHAP analysis for company-level bid prediction.
Features are organized into:
1. Company Historical Profiling (from past bids)
2. Notice-Level Competitive Features (current project)
3. Institution-Related Features (agency patterns)
4. Project Features (size, timing)

Target: normalized_bid_rate (bid_amt / min_bid_rate / base_amt * 100)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import json

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent / 'data' / 'processed'
OUTPUT_DIR = Path(__file__).parent / 'analysis_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# Target variable
TARGET = 'normalized_bid_rate'

# POST-HOC COLUMNS (CORRECTED!)
# Only variables unknowable before actual bidding
POST_HOC_COLS = [
    # Actual bid amounts and derivatives (unknowable before bidding)
    'bid_amt',
    'normalized_bid_rate',  # Our target - actual value unknown
    'bid_rate_diff',
    'norm_bid_margin',
    'margin_from_min',
    
    # Winner data (only known after auction)
    'is_winner',
    'winner_code',
    'winner_normalized_rate',
    'winner_margin',
    
    # Actual yega (we sample it during simulation)
    'yega_rate',
    'bidding_price_ratio',
]

# NOTE: Ranking features (l_ranking, l_gap_to_1st, etc.) are NOT post-hoc!
# These can be calculated from predicted bids during simulation.

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load training data and pre-computed institution features."""
    train_path = DATA_DIR / 'train_clean.csv'  # Use cleaned data
    train_df = pd.read_csv(train_path)
    
    # Convert dates
    train_df['announce_date'] = pd.to_datetime(train_df['announce_date'])
    train_df['bid_date'] = pd.to_datetime(train_df['bid_date'].astype(str), format='%Y%m%d', errors='coerce')
    
    print(f"Loaded {len(train_df):,} records from train_clean.csv")
    print(f"Unique companies: {train_df['company_code'].nunique()}")
    print(f"Unique notices: {train_df['notice_id'].nunique()}")
    print(f"Unique institutions: {train_df['institution_code'].nunique()}")
    
    # Load pre-computed institution features
    inst_features_path = DATA_DIR / 'inst_features.csv'
    if inst_features_path.exists():
        inst_features_df = pd.read_csv(inst_features_path)
        print(f"Loaded {len(inst_features_df):,} records from inst_features.csv")
        print(f"  Window distribution: {inst_features_df['data_window_used'].value_counts().to_dict()}")
    else:
        inst_features_df = None
        print("WARNING: inst_features.csv not found, will compute institution features on-the-fly")
    
    return train_df, inst_features_df


# =============================================================================
# COMPANY HISTORICAL FEATURES
# =============================================================================

def create_company_historical_features(df, company_code, current_date, windows=[30, 100, 300]):
    """
    Create historical profiling features for a company.
    
    Uses only data BEFORE current_date to avoid leakage.
    
    Args:
        df: Full training dataframe
        company_code: Company to create features for
        current_date: Current project date (exclude this and future)
        windows: List of window sizes (number of past bids)
    """
    # Filter to company's historical bids only
    company_df = df[
        (df['company_code'] == company_code) & 
        (df['announce_date'] < current_date)
    ].sort_values('announce_date')
    
    n_hist = len(company_df)
    features = {
        'n_hist_bids': n_hist,
    }
    
    if n_hist == 0:
        # Cold start - no history
        for w in windows:
            features[f'hist_mean_margin_{w}'] = np.nan
            features[f'hist_std_margin_{w}'] = np.nan
            features[f'hist_mean_normalized_rate_{w}'] = np.nan
            features[f'hist_win_rate_{w}'] = np.nan
        features['behavioral_type'] = 'unknown'
        return features
    
    # Adaptive window based on data availability
    for w in windows:
        actual_w = min(w, n_hist)
        window_df = company_df.tail(actual_w)
        
        # Margin statistics (from margin_from_min)
        if 'margin_from_min' in window_df.columns:
            features[f'hist_mean_margin_{w}'] = window_df['margin_from_min'].mean()
            features[f'hist_std_margin_{w}'] = window_df['margin_from_min'].std()
            features[f'hist_median_margin_{w}'] = window_df['margin_from_min'].median()
        
        # Normalized bid rate statistics
        if 'normalized_bid_rate' in window_df.columns:
            features[f'hist_mean_normalized_rate_{w}'] = window_df['normalized_bid_rate'].mean()
            features[f'hist_std_normalized_rate_{w}'] = window_df['normalized_bid_rate'].std()
        
        # Win rate
        if 'is_winner' in window_df.columns:
            features[f'hist_win_rate_{w}'] = (window_df['is_winner'] == True).mean()
        
        # Ranking statistics
        if 'ranking' in window_df.columns:
            features[f'hist_mean_ranking_{w}'] = window_df['ranking'].mean()
            features[f'hist_rankings_1or2_{w}'] = (window_df['ranking'] <= 2).mean()
    
    # Behavioral classification (using longest window)
    longest_w = windows[-1]
    mean_margin = features.get(f'hist_mean_margin_{longest_w}', np.nan)
    std_margin = features.get(f'hist_std_margin_{longest_w}', np.nan)
    
    if pd.notna(mean_margin):
        if mean_margin < 0.015:  # < 1.5% margin
            features['behavioral_type'] = 'aggressive'
        elif mean_margin < 0.025:  # 1.5-2.5%
            features['behavioral_type'] = 'balanced'
        else:
            features['behavioral_type'] = 'conservative'
    else:
        features['behavioral_type'] = 'unknown'
    
    # Trend (recent vs long-term)
    if n_hist >= 30:
        recent_margin = features.get(f'hist_mean_margin_{windows[0]}', np.nan)
        long_margin = features.get(f'hist_mean_margin_{windows[-1]}', np.nan)
        if pd.notna(recent_margin) and pd.notna(long_margin):
            features['margin_trend'] = recent_margin - long_margin  # Positive = becoming more conservative
    
    return features


# =============================================================================
# INSTITUTION FEATURES
# =============================================================================

def create_institution_features(df, institution_code, current_date, company_code=None):
    """
    Create institution-related features (FALLBACK when pre-computed not available).
    
    ALIGNED with generate_institution_tables.py:
    - Uses MONTH-BASED aggregation (not time-based 90 days)
    - Average of monthly means (not mean of all individual bids)
    - BID COUNT threshold (20 bids) for adaptive windowing
    
    Args:
        df: Full training dataframe
        institution_code: Institution to analyze
        current_date: Current project date
        company_code: If provided, also compute company-institution interactions
    """
    MIN_BIDS_THRESHOLD = 20  # Same as generate_institution_tables.py
    MIN_MONTHS_3M = 1
    MIN_MONTHS_6M = 2
    MIN_MONTHS_1Y = 3
    
    def calculate_trend(values):
        """Calculate linear regression slope (same as generate_institution_tables.py)."""
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
    
    # Filter to institution's historical bids
    inst_df = df[
        (df['institution_code'] == institution_code) & 
        (df['announce_date'] < current_date)
    ].copy()
    
    features = {
        'inst_n_hist_bids': len(inst_df),
    }
    
    if len(inst_df) == 0:
        # No history - set all to NaN
        features['inst_mean_yega_3m'] = np.nan
        features['inst_std_yega_3m'] = np.nan
        features['inst_trend_3m'] = np.nan
        features['inst_mean_winner_margin_3m'] = np.nan
        features['inst_yega_volatility_3m'] = np.nan
        features['inst_n_bids_3m'] = np.nan
        features['inst_mean_yega_6m'] = np.nan
        features['inst_std_yega_6m'] = np.nan
        features['inst_trend_6m'] = np.nan
        features['inst_mean_winner_margin_6m'] = np.nan
        features['inst_n_bids_6m'] = np.nan
        features['inst_mean_yega_1y'] = np.nan
        features['inst_std_yega_1y'] = np.nan
        features['inst_trend_1y'] = np.nan
        features['inst_mean_winner_margin_1y'] = np.nan
        features['inst_n_bids_1y'] = np.nan
        features['data_window_used'] = 'insufficient'
        features['active_n_bids'] = 0
        features['inst_yega_consistency'] = 'unknown'
        features['inst_bidding_frequency'] = 'insufficient'
        features['inst_yega_bias'] = 'unknown'
        features['project_yega_vs_inst_mean'] = np.nan
        return features
    
    # === ALIGNED: Group by month (same as generate_institution_tables.py) ===
    inst_df['yega_rate'] = inst_df.apply(
        lambda row: row['choose_avg'] / row['base_amt'] if row['base_amt'] > 0 and pd.notna(row['choose_avg']) else np.nan,
        axis=1
    )
    inst_df['announce_month'] = pd.to_datetime(inst_df['announce_date']).dt.to_period('M')
    
    # Monthly aggregation (same methodology as generate_institution_tables.py)
    monthly = inst_df.groupby('announce_month').agg({
        'yega_rate': ['mean', 'count'],
        'winner_margin': 'mean'
    })
    monthly.columns = ['mean_yega_rate', 'n_bids', 'mean_winner_margin']
    monthly = monthly.sort_index()
    
    # === ALIGNED: Same window logic (tail N months) ===
    hist_3m = monthly.tail(3)
    hist_6m = monthly.tail(6)
    hist_1y = monthly.tail(12)
    
    # Compute bid counts (sum of monthly totals)
    n_bids_3m = int(hist_3m['n_bids'].sum()) if len(hist_3m) >= MIN_MONTHS_3M else 0
    n_bids_6m = int(hist_6m['n_bids'].sum()) if len(hist_6m) >= MIN_MONTHS_6M else 0
    n_bids_1y = int(hist_1y['n_bids'].sum()) if len(hist_1y) >= MIN_MONTHS_1Y else 0
    
    features['inst_n_bids_3m'] = n_bids_3m
    features['inst_n_bids_6m'] = n_bids_6m
    features['inst_n_bids_1y'] = n_bids_1y
    
    # === ALIGNED: Average of monthly means ===
    if len(hist_3m) >= MIN_MONTHS_3M:
        features['inst_mean_yega_3m'] = hist_3m['mean_yega_rate'].mean()  # ✅ Average of monthly means
        features['inst_std_yega_3m'] = hist_3m['mean_yega_rate'].std() if len(hist_3m) > 1 else 0.0
        features['inst_trend_3m'] = calculate_trend(hist_3m['mean_yega_rate'].values)  # ✅ Actual trend
        features['inst_mean_winner_margin_3m'] = hist_3m['mean_winner_margin'].mean()
        if features['inst_mean_yega_3m'] > 0:
            features['inst_yega_volatility_3m'] = features['inst_std_yega_3m'] / features['inst_mean_yega_3m']
        else:
            features['inst_yega_volatility_3m'] = np.nan
    else:
        features['inst_mean_yega_3m'] = np.nan
        features['inst_std_yega_3m'] = np.nan
        features['inst_trend_3m'] = np.nan
        features['inst_mean_winner_margin_3m'] = np.nan
        features['inst_yega_volatility_3m'] = np.nan
    
    if len(hist_6m) >= MIN_MONTHS_6M:
        features['inst_mean_yega_6m'] = hist_6m['mean_yega_rate'].mean()
        features['inst_std_yega_6m'] = hist_6m['mean_yega_rate'].std() if len(hist_6m) > 1 else 0.0
        features['inst_trend_6m'] = calculate_trend(hist_6m['mean_yega_rate'].values)
        features['inst_mean_winner_margin_6m'] = hist_6m['mean_winner_margin'].mean()
    else:
        features['inst_mean_yega_6m'] = np.nan
        features['inst_std_yega_6m'] = np.nan
        features['inst_trend_6m'] = np.nan
        features['inst_mean_winner_margin_6m'] = np.nan
    
    if len(hist_1y) >= MIN_MONTHS_1Y:
        features['inst_mean_yega_1y'] = hist_1y['mean_yega_rate'].mean()
        features['inst_std_yega_1y'] = hist_1y['mean_yega_rate'].std() if len(hist_1y) > 1 else 0.0
        features['inst_trend_1y'] = calculate_trend(hist_1y['mean_yega_rate'].values)
        features['inst_mean_winner_margin_1y'] = hist_1y['mean_winner_margin'].mean()
    else:
        features['inst_mean_yega_1y'] = np.nan
        features['inst_std_yega_1y'] = np.nan
        features['inst_trend_1y'] = np.nan
        features['inst_mean_winner_margin_1y'] = np.nan
    
    # === ADAPTIVE WINDOW SELECTION BASED ON BID COUNT ===
    if n_bids_3m >= MIN_BIDS_THRESHOLD:
        active_mean = features['inst_mean_yega_3m']
        active_std = features['inst_std_yega_3m']
        active_n = n_bids_3m
        active_window = '3m'
    elif n_bids_6m >= MIN_BIDS_THRESHOLD:
        active_mean = features['inst_mean_yega_6m']
        active_std = features['inst_std_yega_6m']
        active_n = n_bids_6m
        active_window = '6m'
    elif n_bids_1y >= MIN_BIDS_THRESHOLD:
        active_mean = features['inst_mean_yega_1y']
        active_std = features['inst_std_yega_1y']
        active_n = n_bids_1y
        active_window = '1y'
    else:
        active_mean = np.nan
        active_std = np.nan
        active_n = max(n_bids_3m, n_bids_6m, n_bids_1y)
        active_window = 'insufficient'
    
    features['data_window_used'] = active_window
    features['active_n_bids'] = active_n
    
    # Categorical classifications
    if pd.notna(active_std):
        if active_std < 0.008:
            features['inst_yega_consistency'] = 'stable'
        elif active_std < 0.015:
            features['inst_yega_consistency'] = 'moderate'
        else:
            features['inst_yega_consistency'] = 'volatile'
    else:
        features['inst_yega_consistency'] = 'unknown'
    
    if active_n >= MIN_BIDS_THRESHOLD:
        if active_n >= 50:
            features['inst_bidding_frequency'] = 'high'
        elif active_n >= 30:
            features['inst_bidding_frequency'] = 'medium'
        else:
            features['inst_bidding_frequency'] = 'low'
    else:
        features['inst_bidding_frequency'] = 'insufficient'
    
    if pd.notna(active_mean):
        if active_mean < 0.995:
            features['inst_yega_bias'] = 'conservative'
        elif active_mean > 1.005:
            features['inst_yega_bias'] = 'aggressive'
        else:
            features['inst_yega_bias'] = 'neutral'
    else:
        features['inst_yega_bias'] = 'unknown'
    
    # Placeholder (computed in extract_features_for_row if needed)
    features['project_yega_vs_inst_mean'] = np.nan
    
    # Company-Institution interaction
    if company_code is not None:
        comp_inst_df = inst_df[inst_df['company_code'] == company_code]
        features['comp_inst_n_prev_bids'] = len(comp_inst_df)
        if len(comp_inst_df) > 0:
            features['comp_inst_win_rate'] = (comp_inst_df['is_winner'] == True).mean()
            features['comp_inst_mean_margin'] = comp_inst_df['margin_from_min'].mean()
        else:
            features['comp_inst_win_rate'] = np.nan
            features['comp_inst_mean_margin'] = np.nan
    
    return features


# =============================================================================
# NOTICE-LEVEL COMPETITIVE FEATURES
# =============================================================================

def create_notice_level_features(notice_df, company_row):
    """
    Create notice-level competitive features.
    
    All PRE-BID information:
    - Tech score distribution
    - Company's position
    - Competition structure
    
    Args:
        notice_df: All bidders in this notice
        company_row: Current company's row
    """
    features = {}
    
    tech_scores = notice_df['tech_score'].values
    n_competitors = len(tech_scores)
    company_tech = company_row['tech_score']
    company_ranking = company_row['ranking']
    
    # Competition size
    features['n_competitors'] = n_competitors
    
    # Tech score distribution
    features['g_tech_score_mean'] = np.mean(tech_scores)
    features['g_tech_score_std'] = np.std(tech_scores)
    features['g_tech_score_max'] = np.max(tech_scores)
    features['g_tech_score_min'] = np.min(tech_scores)
    features['g_tech_score_range'] = np.max(tech_scores) - np.min(tech_scores)
    
    # Rank-based gaps
    sorted_scores = np.sort(tech_scores)[::-1]  # Descending
    if n_competitors >= 2:
        features['g_gap_1st_to_2nd'] = sorted_scores[0] - sorted_scores[1]
    else:
        features['g_gap_1st_to_2nd'] = 0
    
    if n_competitors >= 3:
        features['g_gap_2nd_to_3rd'] = sorted_scores[1] - sorted_scores[2]
    else:
        features['g_gap_2nd_to_3rd'] = 0
    
    # Company position (from data)
    features['l_ranking'] = company_ranking
    features['l_tech_score'] = company_tech
    features['l_gap_to_1st'] = company_row.get('gap_to_1st', sorted_scores[0] - company_tech)
    features['l_gap_to_2nd'] = company_row.get('gap_to_2nd', 0)
    
    # Standardized position
    if np.std(tech_scores) > 0:
        features['l_tech_zscore'] = (company_tech - np.mean(tech_scores)) / np.std(tech_scores)
    else:
        features['l_tech_zscore'] = 0
    
    # Percentile position (0-100, higher = better)
    features['l_tech_percentile'] = (n_competitors - company_ranking) / n_competitors * 100
    
    # Min bid rate (PRE-BID known)
    features['l_min_bid_rate'] = company_row['min_bid_rate']
    
    # Normal bid line (PRE-BID known)
    features['l_normal_bid_line'] = company_row['normal_bid_line']
    
    # Position categories
    if company_ranking == 1 and features['g_gap_1st_to_2nd'] > 1.5:
        features['position_category'] = 'runaway_leader'
    elif company_ranking <= 2:
        features['position_category'] = 'top_tier'
    elif company_ranking <= 5:
        features['position_category'] = 'mid_tier'
    else:
        features['position_category'] = 'trailing'
    
    # Competition intensity
    if n_competitors >= 8 and features['g_tech_score_std'] < 2.0:
        features['competition_intensity'] = 'high'
    elif n_competitors <= 4 or features['g_tech_score_std'] > 3.5:
        features['competition_intensity'] = 'low'
    else:
        features['competition_intensity'] = 'medium'
    
    return features


# =============================================================================
# PROJECT FEATURES
# =============================================================================

def create_project_features(row):
    """
    Create project-level features.
    
    Args:
        row: Single row of data
    """
    features = {}
    
    # Amount features
    base_amt = row['base_amt']
    features['base_amt_log'] = np.log10(base_amt) if base_amt > 0 else 0
    
    # Amount groups
    if base_amt < 500_000_000:  # < 500M_KRW
        features['amt_group'] = 'G1'
    elif base_amt < 1_000_000_000:  # 500M_to_1B
        features['amt_group'] = 'G2'
    else:
        features['amt_group'] = 'G3'
    
    # Timing features (if date available)
    if pd.notna(row.get('announce_date')):
        date = pd.to_datetime(row['announce_date'])
        features['month'] = date.month
        features['quarter'] = (date.month - 1) // 3 + 1
        features['is_year_end'] = 1 if date.month in [11, 12] else 0
        features['is_year_start'] = 1 if date.month in [1, 2] else 0
    
    return features


# =============================================================================
# FULL FEATURE EXTRACTION
# =============================================================================

def extract_features_for_row(df, idx, row, precomputed_company_hist=None, precomputed_inst_features=None):
    """
    Extract all features for a single row.
    
    Args:
        df: Full dataframe
        idx: Index of current row
        row: Row data
        precomputed_company_hist: Optional precomputed company history dict
        precomputed_inst_features: Optional dict of record_id -> inst features
    """
    features = {}
    
    current_date = pd.to_datetime(row['announce_date'])
    company_code = row['company_code']
    institution_code = row['institution_code']
    notice_id = row['notice_id']
    record_id = row['record_id']
    
    # 1. Company historical features
    if precomputed_company_hist and company_code in precomputed_company_hist:
        company_features = precomputed_company_hist[company_code].get(current_date, {})
        if not company_features:
            company_features = create_company_historical_features(df, company_code, current_date)
    else:
        company_features = create_company_historical_features(df, company_code, current_date)
    features.update(company_features)
    
    # 2. Institution features (use pre-computed if available)
    if precomputed_inst_features is not None and record_id in precomputed_inst_features:
        inst_features = precomputed_inst_features[record_id]
        features.update(inst_features)
        
        # Also get the dynamic company-institution interaction
        inst_df = df[
            (df['institution_code'] == institution_code) & 
            (df['announce_date'] < current_date)
        ]
        if len(inst_df) > 0:
            comp_inst_df = inst_df[inst_df['company_code'] == company_code]
            features['comp_inst_n_prev_bids'] = len(comp_inst_df)
            if len(comp_inst_df) > 0:
                features['comp_inst_win_rate'] = (comp_inst_df['is_winner'] == True).mean()
                features['comp_inst_mean_margin'] = comp_inst_df['margin_from_min'].mean()
            else:
                features['comp_inst_win_rate'] = np.nan
                features['comp_inst_mean_margin'] = np.nan
        else:
            features['comp_inst_n_prev_bids'] = 0
            features['comp_inst_win_rate'] = np.nan
            features['comp_inst_mean_margin'] = np.nan
    else:
        # Fallback to computing on-the-fly
        inst_features = create_institution_features(df, institution_code, current_date, company_code)
        features.update(inst_features)
    
    # 3. Notice-level features
    notice_df = df[df['notice_id'] == notice_id]
    notice_features = create_notice_level_features(notice_df, row)
    features.update(notice_features)
    
    # 4. Project features
    project_features = create_project_features(row)
    features.update(project_features)
    
    return features


def extract_features_batch(df, inst_features_df=None, sample_size=None):
    """
    Extract features for all rows (or sample).
    
    Args:
        df: Full dataframe
        inst_features_df: Pre-computed institution features dataframe (REQUIRED)
        sample_size: If provided, process only a sample
    """
    # === CRITICAL: Validate inst_features.csv ===
    if inst_features_df is None:
        raise ValueError(
            "inst_features.csv is REQUIRED!\n"
            "Please run generate_institution_tables.py first to create pre-computed features.\n"
            "This ensures consistent month-based aggregation across all training records."
        )
    
    # Validate coverage
    train_ids = set(df['record_id'])
    inst_ids = set(inst_features_df['record_id'])
    missing = train_ids - inst_ids
    
    if missing:
        raise ValueError(
            f"{len(missing)} training records missing from inst_features.csv!\n"
            f"First 10 missing: {list(missing)[:10]}\n"
            "Re-run generate_institution_tables.py to regenerate."
        )
    
    print(f"✅ inst_features.csv covers all {len(train_ids):,} training records")
    
    if sample_size:
        # Stratified sample by company to ensure diversity
        df_sample = df.groupby('company_code').apply(
            lambda x: x.sample(min(len(x), max(1, sample_size // df['company_code'].nunique())))
        ).reset_index(drop=True)
        if len(df_sample) > sample_size:
            df_sample = df_sample.sample(sample_size)
    else:
        df_sample = df
    
    print(f"Extracting features for {len(df_sample):,} rows...")
    print("Using pre-computed institution features (month-based aggregation, bid-count threshold)")
    
    # Columns to use from inst_features
    inst_feature_cols = [
        'inst_n_bids_3m', 'inst_mean_yega_3m', 'inst_std_yega_3m', 'inst_trend_3m',
        'inst_mean_winner_margin_3m', 'inst_yega_volatility_3m',
        'inst_n_bids_6m', 'inst_mean_yega_6m', 'inst_std_yega_6m', 'inst_trend_6m',
        'inst_mean_winner_margin_6m',
        'inst_n_bids_1y', 'inst_mean_yega_1y', 'inst_std_yega_1y', 'inst_trend_1y',
        'inst_mean_winner_margin_1y',
        'data_window_used', 'active_n_bids',
        'inst_yega_consistency', 'inst_bidding_frequency', 'inst_yega_bias',
        'project_yega_vs_inst_mean'
    ]
    
    # Build fast lookup dict
    precomputed_inst = {}
    for _, irow in inst_features_df.iterrows():
        record_id = irow['record_id']
        precomputed_inst[record_id] = {col: irow[col] for col in inst_feature_cols if col in irow.index}
    
    all_features = []
    targets = []
    
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        if i > 0 and i % 1000 == 0:
            print(f"  Processed {i:,} / {len(df_sample):,}")
        
        features = extract_features_for_row(df, idx, row, precomputed_inst_features=precomputed_inst)
        features['record_id'] = row['record_id']  # For joining back
        all_features.append(features)
        targets.append(row[TARGET])
    
    features_df = pd.DataFrame(all_features)
    
    return features_df, np.array(targets)


# =============================================================================
# PREPARE FOR MODELING
# =============================================================================

def prepare_for_modeling(features_df, target):
    """
    Prepare features and target for XGBoost.
    
    1. Handle categoricals (one-hot encode)
    2. Handle missing values
    3. Return X, y, feature_names
    """
    # Columns to exclude
    exclude_cols = ['record_id', 'target', 'announce_date']
    
    # Identify categorical columns (including new institution features)
    categorical_cols = [
        'behavioral_type', 'position_category', 'competition_intensity', 'amt_group',
        'inst_yega_consistency', 'inst_bidding_frequency', 'inst_yega_bias', 'data_window_used'
    ]
    
    # One-hot encode categoricals
    df = features_df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
    
    # Drop exclude columns
    for col in exclude_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Fill NaN with -999 (XGBoost can handle missing natively but explicit is clearer)
    df = df.fillna(-999)
    
    feature_names = df.columns.tolist()
    X = df.values
    y = target
    
    print(f"Prepared X: {X.shape}, y: {len(y)}")
    print(f"Feature count: {len(feature_names)}")
    
    return X, y, feature_names


# =============================================================================
# TRAIN MODEL & SHAP ANALYSIS
# =============================================================================

def train_and_analyze(X, y, feature_names):
    """
    Train XGBoost and analyze with SHAP.
    Uses GPU acceleration for faster training and SHAP computation.
    """
    print("\n=== Training XGBoost (GPU Accelerated) ===")
    
    model = xgb.XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=200,  # Increased for better performance
        random_state=42,
        tree_method='gpu_hist',  # GPU acceleration
        gpu_id=0,
        predictor='gpu_predictor'
    )
    
    model.fit(X, y)
    
    # Basic evaluation
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    print(f"Training MAE: {mae:.6f}")
    print(f"Training RMSE: {rmse:.6f}")
    
    # SHAP Analysis
    print("\n=== SHAP Analysis ===")
    
    # Sample for SHAP (use larger sample with GPU for better estimates)
    shap_sample_size = min(10000, len(X))  # Increased from 5K to 10K
    indices = np.random.choice(len(X), shap_sample_size, replace=False)
    X_shap = X[indices]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # Mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'std_shap': np.std(shap_values, axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Aggregate categorical feature importance
    importance_df = aggregate_categorical_importance(importance_df)
    
    return model, importance_df, shap_values, X_shap


def aggregate_categorical_importance(importance_df):
    """
    Aggregate one-hot encoded categorical features back together.
    
    For example:
    - behavioral_type_aggressive, behavioral_type_balanced, behavioral_type_conservative
    -> behavioral_type (sum of their SHAP)
    """
    # Find one-hot encoded columns (contain underscore after prefix)
    categorical_prefixes = [
        'behavioral_type', 'position_category', 'competition_intensity', 'amt_group',
        'inst_yega_consistency', 'inst_bidding_frequency', 'inst_yega_bias', 'data_window_used'
    ]
    
    aggregated_rows = []
    remaining_df = importance_df.copy()
    
    for prefix in categorical_prefixes:
        # Find all columns starting with this prefix
        mask = remaining_df['feature'].str.startswith(prefix + '_')
        if mask.sum() > 0:
            prefix_rows = remaining_df[mask]
            
            # Aggregate
            agg_importance = prefix_rows['mean_abs_shap'].sum()
            agg_std = np.sqrt((prefix_rows['std_shap'] ** 2).sum())  # Approximate
            
            aggregated_rows.append({
                'feature': f"{prefix} (aggregated)",
                'mean_abs_shap': agg_importance,
                'std_shap': agg_std
            })
            
            # Remove individual rows
            remaining_df = remaining_df[~mask]
    
    # Add aggregated rows
    if aggregated_rows:
        agg_df = pd.DataFrame(aggregated_rows)
        remaining_df = pd.concat([remaining_df, agg_df], ignore_index=True)
    
    # Re-sort
    remaining_df = remaining_df.sort_values('mean_abs_shap', ascending=False)
    
    return remaining_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Feature Engineering for Company Model")
    print("=" * 60)
    
    # Load data
    train_df, inst_features_df = load_data()
    
    # Extract features (FULL dataset for comprehensive analysis)
    print("\n=== Feature Extraction ===")
    features_df, targets = extract_features_batch(train_df, inst_features_df=inst_features_df, sample_size=None)
    
    # Save features
    features_df['target'] = targets
    features_df.to_csv(OUTPUT_DIR / 'feature_sample.csv', index=False)
    print(f"Saved features to {OUTPUT_DIR / 'feature_sample.csv'}")
    
    # Prepare for modeling
    X, y, feature_names = prepare_for_modeling(features_df, targets)
    
    # Train and analyze
    model, importance_df, shap_values, X_shap = train_and_analyze(X, y, feature_names)
    
    # Save importance
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    print(f"\nSaved importance to {OUTPUT_DIR / 'feature_importance.csv'}")
    
    # Print top features
    print("\n" + "=" * 60)
    print("TOP 30 FEATURES (SHAP Importance)")
    print("=" * 60)
    print(importance_df.head(30).to_string(index=False))
    
    # Feature groups analysis
    print("\n" + "=" * 60)
    print("FEATURE GROUP IMPORTANCE")
    print("=" * 60)
    
    # Define groups by prefix
    groups = {
        'Company Historical': ['hist_', 'n_hist_', 'behavioral_type', 'margin_trend'],
        'Institution': ['inst_', 'comp_inst_'],
        'Notice Competition': ['g_', 'l_', 'n_competitors', 'position_category', 'competition_intensity'],
        'Project': ['base_amt', 'amt_group', 'month', 'quarter', 'is_year']
    }
    
    group_importance = {}
    for group_name, prefixes in groups.items():
        total = 0
        for prefix in prefixes:
            mask = importance_df['feature'].str.contains(prefix, case=False, na=False)
            total += importance_df[mask]['mean_abs_shap'].sum()
        group_importance[group_name] = total
    
    for group_name, importance in sorted(group_importance.items(), key=lambda x: -x[1]):
        print(f"{group_name:25s}: {importance:.6f}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return {
        'model': model,
        'importance_df': importance_df,
        'feature_names': feature_names
    }


if __name__ == '__main__':
    results = main()
