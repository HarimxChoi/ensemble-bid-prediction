"""
Historical Profiling Module

Creates company behavioral profiles using HISTORICAL post-hoc data.

KEY INSIGHT:
- PROFILING: CAN use past bid_amt, margins, win_rates (from past projects)
- PREDICTION: CANNOT use current project's post-hoc data

This creates features like:
- hist_mean_margin_from_min_{recent,medium,long}
- hist_win_rate_{recent,medium,long}
- behavioral_type_{aggressive,balanced,conservative}
- margin_trend (strategic shift detection)
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def get_adaptive_windows(n_bids: int) -> Dict[str, int]:
    """
    Get adaptive window sizes based on company data availability
    """
    if n_bids >= 500:
        return {'recent': 50, 'medium': 150, 'long': 400}
    elif n_bids >= 150:
        return {'recent': 30, 'medium': 100, 'long': min(300, int(n_bids * 0.8))}
    elif n_bids >= 50:
        return {'recent': 15, 'medium': 40, 'long': min(100, int(n_bids * 0.8))}
    else:
        return {
            'recent': max(5, int(n_bids * 0.3)),
            'medium': max(10, int(n_bids * 0.6)),
            'long': max(15, int(n_bids * 0.9))
        }


def compute_historical_features(
    historical_df: pd.DataFrame,
    windows: Dict[str, int]
) -> Dict[str, float]:
    """
    Compute behavioral features from historical data
    
    Uses PAST bid_amt, margins, win_rates (allowed for profiling)
    """
    features = {}
    
    # Check required columns
    required = ['bid_rate', 'min_bid_rate', 'is_winner']
    if not all(col in historical_df.columns for col in required):
        return features
    
    # Calculate margins (ALLOWED - using PAST data)
    historical_df = historical_df.copy()
    historical_df['margin_from_min'] = historical_df['bid_rate'] - historical_df['min_bid_rate']
    
    if 'normal_bid_line_calc' in historical_df.columns:
        historical_df['margin_from_normal'] = historical_df['bid_rate'] - historical_df['normal_bid_line_calc']
    
    # Window-based features
    for window_name, window_size in windows.items():
        window_df = historical_df.tail(window_size)
        
        if len(window_df) == 0:
            continue
        
        # Margin statistics
        margin = window_df['margin_from_min']
        features[f'hist_mean_margin_from_min_{window_name}'] = margin.mean()
        features[f'hist_std_margin_from_min_{window_name}'] = margin.std() if len(margin) > 1 else 0
        features[f'hist_median_margin_from_min_{window_name}'] = margin.median()
        features[f'hist_q25_margin_{window_name}'] = margin.quantile(0.25)
        features[f'hist_q75_margin_{window_name}'] = margin.quantile(0.75)
        
        # Win rate
        features[f'hist_win_rate_{window_name}'] = window_df['is_winner'].mean()
        
        # Bid rate statistics
        features[f'hist_mean_bid_rate_{window_name}'] = window_df['bid_rate'].mean()
        features[f'hist_std_bid_rate_{window_name}'] = window_df['bid_rate'].std() if len(window_df) > 1 else 0
        
        # Normalized rate if available
        if 'normalized_bid_rate' in window_df.columns:
            norm = window_df['normalized_bid_rate']
            features[f'hist_mean_norm_rate_{window_name}'] = norm.mean()
            features[f'hist_std_norm_rate_{window_name}'] = norm.std() if len(norm) > 1 else 0
    
    return features


def compute_behavioral_classification(features: Dict[str, float]) -> Dict[str, int]:
    """
    Classify company behavioral type based on historical patterns
    """
    result = {}
    
    # Get long-term margin (most stable indicator)
    mean_margin = features.get('hist_mean_margin_from_min_long', 
                              features.get('hist_mean_margin_from_min_medium',
                              features.get('hist_mean_margin_from_min_recent', 0)))
    
    std_margin = features.get('hist_std_margin_from_min_long',
                             features.get('hist_std_margin_from_min_medium', 0))
    
    # Aggressiveness classification (based on margin)
    # Margin is in decimal form (e.g., 0.02 = 2%)
    if mean_margin < 0.015:  # < 1.5%
        result['behavioral_type_aggressive'] = 1
        result['behavioral_type_balanced'] = 0
        result['behavioral_type_conservative'] = 0
    elif mean_margin < 0.025:  # 1.5-2.5%
        result['behavioral_type_aggressive'] = 0
        result['behavioral_type_balanced'] = 1
        result['behavioral_type_conservative'] = 0
    else:  # > 2.5%
        result['behavioral_type_aggressive'] = 0
        result['behavioral_type_balanced'] = 0
        result['behavioral_type_conservative'] = 1
    
    # Consistency classification
    if std_margin < 0.005:  # < 0.5%
        result['consistency_high'] = 1
        result['consistency_medium'] = 0
        result['consistency_low'] = 0
    elif std_margin < 0.01:  # 0.5-1%
        result['consistency_high'] = 0
        result['consistency_medium'] = 1
        result['consistency_low'] = 0
    else:
        result['consistency_high'] = 0
        result['consistency_medium'] = 0
        result['consistency_low'] = 1
    
    return result


def compute_temporal_trend(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute temporal trend (strategic shift detection)
    """
    result = {}
    
    recent_margin = features.get('hist_mean_margin_from_min_recent')
    long_margin = features.get('hist_mean_margin_from_min_long')
    
    if recent_margin is not None and long_margin is not None:
        # Positive = becoming more conservative
        # Negative = becoming more aggressive
        result['margin_trend'] = recent_margin - long_margin
    else:
        result['margin_trend'] = 0
    
    return result


def compute_context_patterns(
    historical_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute context-dependent patterns (by ranking, project size)
    """
    features = {}
    
    historical_df = historical_df.copy()
    
    # Need margin column
    if 'bid_rate' in historical_df.columns and 'min_bid_rate' in historical_df.columns:
        historical_df['margin_from_min'] = historical_df['bid_rate'] - historical_df['min_bid_rate']
    else:
        return features
    
    # By ranking group
    for rank_group, (rank_min, rank_max) in [
        ('rank_1_2', (1, 2)),
        ('rank_3_5', (3, 5)),
        ('rank_6_plus', (6, 100))
    ]:
        if 'ranking' in historical_df.columns:
            rank_df = historical_df[
                (historical_df['ranking'] >= rank_min) & 
                (historical_df['ranking'] <= rank_max)
            ]
            
            if len(rank_df) >= 5:
                features[f'hist_mean_margin_{rank_group}'] = rank_df['margin_from_min'].mean()
                features[f'hist_win_rate_{rank_group}'] = rank_df['is_winner'].mean() if 'is_winner' in rank_df.columns else 0
    
    # By project size (if base_amt available)
    if 'base_amt' in historical_df.columns:
        for size_group, (size_min, size_max) in [
            ('small', (0, 5e8)),          # < 500M_KRW
            ('medium', (5e8, 1e9)),       # 500M_to_1B
            ('large', (1e9, float('inf'))) # > 1B_KRW
        ]:
            size_df = historical_df[
                (historical_df['base_amt'] >= size_min) & 
                (historical_df['base_amt'] < size_max)
            ]
            
            if len(size_df) >= 5:
                features[f'hist_mean_margin_size_{size_group}'] = size_df['margin_from_min'].mean()
    
    return features


def build_historical_profile(
    full_df: pd.DataFrame,
    company_code: str,
    current_date: pd.Timestamp
) -> Dict[str, float]:
    """
    Build complete historical profile for a company
    
    CRITICAL: Only uses data BEFORE current_date (no leakage)
    
    Args:
        full_df: Full dataset
        company_code: Company to profile
        current_date: Date of current bid (only use data before this)
    
    Returns:
        Dict of ~30-40 historical features
    """
    # Get historical data (BEFORE current date)
    historical_df = full_df[
        (full_df['company_code'] == company_code) &
        (full_df['announce_date'] < current_date)
    ].sort_values('announce_date')
    
    n_bids = len(historical_df)
    
    if n_bids == 0:
        return {'n_historical_bids': 0}
    
    features = {'n_historical_bids': n_bids}
    
    # Get adaptive windows
    windows = get_adaptive_windows(n_bids)
    
    # 1. Window-based historical features
    hist_features = compute_historical_features(historical_df, windows)
    features.update(hist_features)
    
    # 2. Behavioral classification
    behavioral = compute_behavioral_classification(hist_features)
    features.update(behavioral)
    
    # 3. Temporal trend
    trend = compute_temporal_trend(hist_features)
    features.update(trend)
    
    # 4. Context-dependent patterns
    context = compute_context_patterns(historical_df)
    features.update(context)
    
    return features


def build_all_profiles(
    df: pd.DataFrame,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Build historical profiles for all company-project pairs
    
    Args:
        df: Full dataset with announce_date
        show_progress: Show progress bar
    
    Returns:
        DataFrame with profile features for each row
    """
    from tqdm import tqdm
    
    logger.info(f"Building historical profiles for {len(df)} records...")
    
    profile_rows = []
    
    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Building profiles")
    
    for idx, row in iterator:
        company_code = row['company_code']
        current_date = row['announce_date']
        
        profile = build_historical_profile(df, company_code, current_date)
        profile['record_id'] = row.get('record_id', idx)
        
        profile_rows.append(profile)
    
    profile_df = pd.DataFrame(profile_rows)
    
    # Fill NaN with 0 for classification columns
    classification_cols = [
        'behavioral_type_aggressive', 'behavioral_type_balanced', 'behavioral_type_conservative',
        'consistency_high', 'consistency_medium', 'consistency_low'
    ]
    for col in classification_cols:
        if col in profile_df.columns:
            profile_df[col] = profile_df[col].fillna(0)
    
    logger.info(f"Created {len(profile_df.columns)} profile features")
    
    return profile_df
