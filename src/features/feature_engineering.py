"""
Feature Engineering Module

This module creates features for XGBoost models and clustering:
1. Company Behavioral Features - using fluctuation_calc and normal_bid_line_calc
2. Rank-Conditioned Statistics - company behavior at different ranks
3. Notice-Level Features - competitive context
4. Adaptive Window Features - by company tier (S/A/B/C)

Key insights from tech doc:
- fluctuation_calc = bid_rate - normal_bid_line_calc
  - Positive: Conservative (bid above threshold)
  - Negative: Aggressive (bid below threshold)
- Companies show different behaviors based on ranking position
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

# Company tier thresholds
TIER_THRESHOLDS = {
    'S': 500,   # Individual XGBoost
    'A': 100,   # Individual XGBoost (regularized)
    'B': 30,    # XGBoost with shared features
    'C': 0,     # Clustering → cluster models
}


def get_company_tier(n_bids: int) -> str:
    """Get company tier based on number of bids"""
    if n_bids >= TIER_THRESHOLDS['S']:
        return 'S'
    elif n_bids >= TIER_THRESHOLDS['A']:
        return 'A'
    elif n_bids >= TIER_THRESHOLDS['B']:
        return 'B'
    else:
        return 'C'


def get_adaptive_windows(n_bids: int) -> Dict[str, int]:
    """
    Get adaptive window sizes based on company data availability
    
    Returns dict with 'recent', 'medium', 'long' windows
    """
    tier = get_company_tier(n_bids)
    
    if tier == 'S':
        return {'recent': 50, 'medium': 150, 'long': 400}
    elif tier == 'A':
        return {'recent': 30, 'medium': 100, 'long': min(300, int(n_bids * 0.8))}
    elif tier == 'B':
        return {'recent': 15, 'medium': 40, 'long': min(100, int(n_bids * 0.8))}
    else:  # C-tier
        return {
            'recent': max(5, int(n_bids * 0.3)),
            'medium': max(10, int(n_bids * 0.6)),
            'long': max(15, int(n_bids * 0.9))
        }


def calculate_window_stats(series: pd.Series, window: int, prefix: str) -> Dict[str, float]:
    """
    Calculate statistics for a rolling window
    """
    if len(series) == 0:
        return {
            f'{prefix}_mean': np.nan,
            f'{prefix}_std': np.nan,
            f'{prefix}_min': np.nan,
            f'{prefix}_max': np.nan,
        }
    
    recent = series.tail(window)
    return {
        f'{prefix}_mean': recent.mean(),
        f'{prefix}_std': recent.std() if len(recent) > 1 else 0,
        f'{prefix}_min': recent.min(),
        f'{prefix}_max': recent.max(),
    }


def engineer_company_features(
    df: pd.DataFrame,
    company_code: str,
    up_to_date: pd.Timestamp = None
) -> Dict[str, float]:
    """
    Extract historical behavioral patterns for a company
    
    Key features using fluctuation_calc and normal_bid_line_calc:
    1. Overall behavior profile (mean, std of fluctuation)
    2. Rank-conditioned behavior (how they behave at different ranks)
    3. Trend indicators (recent vs long-term)
    4. Win rate and competition success
    
    Args:
        df: Historical data with announce_date, fluctuation_calc, etc.
        company_code: Company to profile
        up_to_date: Only use data before this date (to prevent leakage)
    
    Returns:
        Dict of features
    """
    # Filter to company and date
    company_df = df[df['company_code'] == company_code].copy()
    
    if up_to_date is not None:
        company_df = company_df[company_df['announce_date'] < up_to_date]
    
    # Sort by date
    company_df = company_df.sort_values('announce_date')
    
    n_bids = len(company_df)
    
    if n_bids == 0:
        return {'n_historical_bids': 0}
    
    features = {'n_historical_bids': n_bids}
    
    # Get adaptive windows
    windows = get_adaptive_windows(n_bids)
    
    # ==========================================
    # 1. FLUCTUATION-BASED FEATURES (fluctuation)
    # ==========================================
    # fluctuation_calc = bid_rate - normal_bid_line
    # Positive = conservative, Negative = aggressive
    
    fluct = company_df['fluctuation_calc'].dropna()
    
    # Overall stats
    features['fluct_overall_mean'] = fluct.mean()
    features['fluct_overall_std'] = fluct.std() if len(fluct) > 1 else 0
    
    # Window-based stats
    for window_name, window_size in windows.items():
        stats = calculate_window_stats(fluct, window_size, f'fluct_{window_name}')
        features.update(stats)
    
    # Trend: recent vs long-term
    if len(fluct) >= windows['long']:
        recent_mean = fluct.tail(windows['recent']).mean()
        long_mean = fluct.tail(windows['long']).mean()
        features['fluct_trend'] = recent_mean - long_mean  # Positive = becoming more conservative
    else:
        features['fluct_trend'] = 0
    
    # Behavioral classification
    mean_fluct = fluct.mean()
    if mean_fluct < -0.01:
        features['behavior_type'] = 0  # Aggressive
    elif mean_fluct < 0.01:
        features['behavior_type'] = 1  # Balanced
    else:
        features['behavior_type'] = 2  # Conservative
    
    # ==========================================
    # 2. RANK-CONDITIONED FEATURES
    # ==========================================
    # How does the company behave at different rankings?
    
    for rank_group, rank_range in [('top', (1, 3)), ('mid', (4, 7)), ('low', (8, 100))]:
        rank_mask = (company_df['ranking'] >= rank_range[0]) & (company_df['ranking'] <= rank_range[1])
        rank_fluct = company_df.loc[rank_mask, 'fluctuation_calc'].dropna()
        
        if len(rank_fluct) >= 3:
            features[f'fluct_{rank_group}_rank_mean'] = rank_fluct.mean()
            features[f'fluct_{rank_group}_rank_std'] = rank_fluct.std()
            features[f'count_{rank_group}_rank'] = len(rank_fluct)
        else:
            features[f'fluct_{rank_group}_rank_mean'] = np.nan
            features[f'fluct_{rank_group}_rank_std'] = np.nan
            features[f'count_{rank_group}_rank'] = len(rank_fluct)
    
    # Rank sensitivity: How much does behavior change with rank?
    if pd.notna(features.get('fluct_top_rank_mean')) and pd.notna(features.get('fluct_low_rank_mean')):
        features['rank_sensitivity'] = features['fluct_low_rank_mean'] - features['fluct_top_rank_mean']
    else:
        features['rank_sensitivity'] = np.nan
    
    # ==========================================
    # 3. WIN RATE FEATURES
    # ==========================================
    if 'is_winner' in company_df.columns:
        features['win_rate'] = company_df['is_winner'].mean()
        
        # Win rate by rank
        for rank_group, rank_range in [('top', (1, 3)), ('mid', (4, 7))]:
            rank_mask = (company_df['ranking'] >= rank_range[0]) & (company_df['ranking'] <= rank_range[1])
            rank_wins = company_df.loc[rank_mask, 'is_winner']
            if len(rank_wins) >= 5:
                features[f'win_rate_{rank_group}_rank'] = rank_wins.mean()
            else:
                features[f'win_rate_{rank_group}_rank'] = np.nan
    
    # ==========================================
    # 4. NORMAL BID LINE FEATURES (normal_bid_line)
    # ==========================================
    nbl = company_df['normal_bid_line_calc'].dropna()
    
    if len(nbl) > 0:
        features['nbl_overall_mean'] = nbl.mean()
        features['nbl_overall_std'] = nbl.std() if len(nbl) > 1 else 0
        
        # This represents the typical competitive position
        # Lower nbl = higher ranking typically
    
    # ==========================================
    # 5. BID RATE DIFFERENCE FEATURES (bid_rate_diff)
    # ==========================================
    brd = company_df['bid_rate_diff_calc'].dropna()
    
    if len(brd) > 0:
        features['bid_rate_diff_mean'] = brd.mean()
        features['bid_rate_diff_std'] = brd.std() if len(brd) > 1 else 0
    
    # ==========================================
    # 6. NORMALIZED RATE FEATURES
    # ==========================================
    if 'normalized_bid_rate' in company_df.columns:
        norm_rate = company_df['normalized_bid_rate'].dropna()
        
        features['norm_rate_mean'] = norm_rate.mean()
        features['norm_rate_std'] = norm_rate.std() if len(norm_rate) > 1 else 0
        
        # How often they bid near minimum (efficient bidding)
        near_min = (norm_rate < 1.02).mean()  # Within 2% of minimum
        features['near_minimum_rate'] = near_min
    
    return features


def engineer_notice_features(notice_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract features for a specific notice (competitive context)
    
    Args:
        notice_df: All records for a single notice
    
    Returns:
        Dict of notice-level features
    """
    features = {}
    
    # Competition intensity
    features['n_competitors'] = len(notice_df)
    
    # Ranking distribution
    if 'ranking' in notice_df.columns:
        features['max_ranking'] = notice_df['ranking'].max()
    
    # Tech score spread
    if 'tech_score' in notice_df.columns:
        tech_scores = notice_df['tech_score'].dropna()
        features['tech_score_range'] = tech_scores.max() - tech_scores.min()
        features['tech_score_mean'] = tech_scores.mean()
        features['tech_score_std'] = tech_scores.std() if len(tech_scores) > 1 else 0
    
    # Gap analysis
    if 'gap_to_1st' in notice_df.columns:
        gaps = notice_df['gap_to_1st'].dropna()
        features['gap_to_1st_mean'] = gaps.mean()
        features['gap_to_1st_max'] = gaps.max()
    
    # Min bid rate spread (competitive tightness)
    if 'min_bid_rate' in notice_df.columns:
        rates = notice_df['min_bid_rate'].dropna()
        features['min_bid_rate_spread'] = rates.max() - rates.min()
    
    # Normal bid line spread
    nbl_col = 'normal_bid_line_calc' if 'normal_bid_line_calc' in notice_df.columns else 'normal_bid_line'
    if nbl_col in notice_df.columns:
        nbl = notice_df[nbl_col].dropna()
        features['nbl_range'] = nbl.max() - nbl.min()
    
    # Project size category
    if 'base_amt' in notice_df.columns:
        base_amt = notice_df['base_amt'].iloc[0]
        if base_amt >= 1e9:
            features['project_size_cat'] = 2  # over_1B
        elif base_amt >= 5e8:
            features['project_size_cat'] = 1  # 500M_to_1B
        else:
            features['project_size_cat'] = 0  # under_500M
    
    return features


def engineer_record_features(
    record: pd.Series,
    company_features: Dict[str, float],
    notice_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Combine company and notice features with record-specific data
    
    Args:
        record: Single bid record
        company_features: Historical company features
        notice_features: Notice-level context features
    
    Returns:
        Complete feature dict for this record
    """
    features = {}
    
    # ==========================================
    # RECORD-SPECIFIC FEATURES
    # ==========================================
    features['ranking'] = record.get('ranking', np.nan)
    features['tech_score'] = record.get('tech_score', np.nan)
    features['gap_to_1st'] = record.get('gap_to_1st', np.nan)
    features['min_bid_rate'] = record.get('min_bid_rate', np.nan)
    
    nbl_col = 'normal_bid_line_calc' if 'normal_bid_line_calc' in record else 'normal_bid_line'
    features['normal_bid_line'] = record.get(nbl_col, np.nan)
    
    brd_col = 'bid_rate_diff_calc' if 'bid_rate_diff_calc' in record else 'bid_rate_diff'
    features['bid_rate_diff'] = record.get(brd_col, np.nan)
    
    # ==========================================
    # POSITION FEATURES
    # ==========================================
    # Position relative to competition
    features['is_top3'] = 1 if features['ranking'] <= 3 else 0
    features['is_mid'] = 1 if 4 <= features['ranking'] <= 7 else 0
    features['is_low'] = 1 if features['ranking'] > 7 else 0
    
    # ==========================================
    # ADD COMPANY FEATURES
    # ==========================================
    for key, value in company_features.items():
        features[f'company_{key}'] = value
    
    # ==========================================
    # ADD NOTICE FEATURES
    # ==========================================
    for key, value in notice_features.items():
        features[f'notice_{key}'] = value
    
    # ==========================================
    # INTERACTION FEATURES
    # ==========================================
    # How does current position compare to company's typical behavior?
    if pd.notna(features.get('company_fluct_overall_mean')):
        # Expected adjustment based on company behavior
        features['expected_fluct'] = features['company_fluct_overall_mean']
    
    # How competitive is this notice for this company?
    if pd.notna(features.get('bid_rate_diff')) and pd.notna(features.get('company_bid_rate_diff_mean')):
        features['bid_rate_diff_vs_typical'] = features['bid_rate_diff'] - features['company_bid_rate_diff_mean']
    
    return features


def prepare_features_for_training(
    df: pd.DataFrame,
    target_col: str = 'normalized_bid_rate'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare full feature matrix for training
    
    Args:
        df: Preprocessed dataframe with all columns
        target_col: Target variable column
    
    Returns:
        X (features), y (target)
    """
    logger.info(f"Preparing features for {len(df)} records...")
    
    feature_rows = []
    
    # Cache company features (expensive to compute)
    company_feature_cache = {}
    
    # Group by notice for notice features
    for notice_id, notice_df in df.groupby('notice_id'):
        notice_features = engineer_notice_features(notice_df)
        
        for idx, record in notice_df.iterrows():
            company_code = record['company_code']
            announce_date = record['announce_date']
            
            # Get or compute company features
            cache_key = (company_code, announce_date)
            if cache_key not in company_feature_cache:
                company_features = engineer_company_features(
                    df, company_code, up_to_date=announce_date
                )
                company_feature_cache[cache_key] = company_features
            else:
                company_features = company_feature_cache[cache_key]
            
            # Combine all features
            features = engineer_record_features(record, company_features, notice_features)
            features['record_id'] = record.get('record_id', idx)
            features['target'] = record[target_col]
            
            feature_rows.append(features)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_rows)
    
    # Separate X and y
    y = feature_df['target']
    X = feature_df.drop(columns=['target', 'record_id'], errors='ignore')
    
    logger.info(f"Created {len(X.columns)} features for {len(X)} records")
    logger.info(f"Feature columns: {list(X.columns)}")
    
    return X, y


def get_company_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign tier to each company based on bid count
    
    Returns DataFrame with company_code, n_bids, tier
    """
    company_counts = df.groupby('company_code').size().reset_index(name='n_bids')
    company_counts['tier'] = company_counts['n_bids'].apply(get_company_tier)
    
    tier_summary = company_counts.groupby('tier').size()
    logger.info(f"Company tier distribution: {tier_summary.to_dict()}")
    
    return company_counts
