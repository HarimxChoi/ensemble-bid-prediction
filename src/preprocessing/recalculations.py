"""
Recalculate derived columns from scratch

This module recalculates:
- normal_bid_line (normal_bid_line): Competitive threshold based on top 2 bidders
- bid_rate_diff (bid_rate_difference): Strategic disadvantage vs rank 1
- fluctuation (fluctuation): Actual bid deviation from normal_bid_line

Formulas (verified from gt.md):
- normal_bid_lineⱼ = lower_bid_rate × (R₁ + R₂) / 2 / Rⱼ
- bid_rate_diffⱼ = normal_bid_line₁ - normal_bid_lineⱼ  
- fluctuation = (bid_amt / base_amt × 100) - normal_bid_line

Where:
- R₁, R₂ = min_bid_rate of rank 1, 2 (as percentages)
- Rⱼ = min_bid_rate of company at rank j
- lower_bid_rate = typically 100 (for most PQ bids)
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# lower_bid_rate by project size (from tech doc Section 7.3)
# These are the floor rates for winning
LOWER_BID_RATE_BY_SIZE = {
    'over_1B': 79.995,     # ≥ 1 billion KRW
    '500M_to_1B': 85.495,    # 500M - 1B KRW
    'under_500M': 86.745,   # < 500M KRW
}

# Default lower_bid_rate (most PQ uses 100% as the reference)
DEFAULT_LOWER_BID_RATE = 100.0


def get_lower_bid_rate(base_amt: float) -> float:
    """
    Get lower_bid_rate based on project size
    
    Note: For most PQ calculations, lower_bid_rate = 100%
    The values in LOWER_BID_RATE_BY_SIZE are the minimum bid rates for perfect_score companies
    """
    # For normal_bid_line calculation, we use 100% (standard reference)
    return DEFAULT_LOWER_BID_RATE


def recalculate_normal_bid_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate normal_bid_line (normal_bid_line) for all records
    
    Formula: normal_bid_lineⱼ = L × (R₁ + R₂) / 2 / Rⱼ
    Where L = lower_bid_rate (100%)
    
    Note: min_bid_rate should be in PERCENTAGE form (e.g., 81.475) for this calculation
    If in decimal form (0.81475), convert first.
    """
    df = df.copy()
    
    # Determine if min_bid_rate is in decimal or percentage form
    mean_rate = df['min_bid_rate'].mean()
    if mean_rate < 1:
        # Decimal form - convert to percentage for calculation
        df['_min_bid_rate_pct'] = df['min_bid_rate'] * 100
        logger.info("Detected decimal min_bid_rate, converting to percentage for calculation")
    else:
        df['_min_bid_rate_pct'] = df['min_bid_rate']
    
    def calc_normal_bid_line(group):
        """Calculate normal_bid_line for a notice group"""
        # Sort by ranking to get rank 1 and 2
        group = group.sort_values('ranking')
        
        if len(group) < 2:
            # Need at least 2 bidders
            group['_normal_bid_line_calc'] = np.nan
            return group
        
        # Get min_bid_rate for rank 1 and 2
        rank1_rate = group.iloc[0]['_min_bid_rate_pct']
        rank2_rate = group.iloc[1]['_min_bid_rate_pct']
        
        # Average of top 2
        avg_top2 = (rank1_rate + rank2_rate) / 2
        
        # lower_bid_rate (use 100% as standard)
        lower_bid_rate = DEFAULT_LOWER_BID_RATE
        
        # Calculate normal_bid_line for each company
        # Formula: L × (R₁ + R₂) / 2 / Rⱼ
        group['_normal_bid_line_calc'] = (
            lower_bid_rate * avg_top2 / group['_min_bid_rate_pct']
        )
        
        return group
    
    # Apply to each notice
    df = df.groupby('notice_id', group_keys=False).apply(calc_normal_bid_line)
    
    # Convert to decimal if original was decimal
    if mean_rate < 1:
        df['normal_bid_line_calc'] = df['_normal_bid_line_calc'] / 100
    else:
        df['normal_bid_line_calc'] = df['_normal_bid_line_calc']
    
    # Clean up temp column
    df = df.drop(columns=['_min_bid_rate_pct', '_normal_bid_line_calc'])
    
    # Log comparison with original if exists
    if 'normal_bid_line' in df.columns:
        diff = (df['normal_bid_line_calc'] - df['normal_bid_line']).abs()
        logger.info(f"normal_bid_line recalculation: mean diff = {diff.mean():.6f}, max diff = {diff.max():.6f}")
    
    return df


def recalculate_bid_rate_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate bid_rate_diff (bid_rate_difference)
    
    Formula: bid_rate_diffⱼ = normal_bid_line₁ - normal_bid_lineⱼ
    
    Interpretation:
    - Rank 1 always has bid_rate_diff = 0
    - Higher rank = larger bid_rate_diff = more disadvantage
    """
    df = df.copy()
    
    # Use recalculated normal_bid_line if available
    nbl_col = 'normal_bid_line_calc' if 'normal_bid_line_calc' in df.columns else 'normal_bid_line'
    
    def calc_bid_rate_diff(group):
        """Calculate bid_rate_diff for a notice group"""
        group = group.sort_values('ranking')
        
        if len(group) < 1:
            group['bid_rate_diff_calc'] = np.nan
            return group
        
        # Get normal_bid_line for rank 1
        rank1_nbl = group.iloc[0][nbl_col]
        
        # bid_rate_diff = normal_bid_line₁ - normal_bid_lineⱼ
        group['bid_rate_diff_calc'] = rank1_nbl - group[nbl_col]
        
        return group
    
    df = df.groupby('notice_id', group_keys=False).apply(calc_bid_rate_diff)
    
    # Log comparison
    if 'bid_rate_diff' in df.columns:
        diff = (df['bid_rate_diff_calc'] - df['bid_rate_diff']).abs()
        logger.info(f"bid_rate_diff recalculation: mean diff = {diff.mean():.6f}, max diff = {diff.max():.6f}")
    
    return df


def recalculate_fluctuation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate fluctuation (fluctuation)
    
    Formula: fluctuation = bid_rate_vs_base - normal_bid_line
            = (bid_amt / base_amt × 100) - normal_bid_line
    
    In decimal terms (if both in decimal):
    fluctuation = bid_rate - normal_bid_line
    
    Interpretation:
    - Positive: Bid above threshold (conservative)
    - Negative: Bid below threshold (aggressive)
    """
    df = df.copy()
    
    # Use recalculated normal_bid_line if available
    nbl_col = 'normal_bid_line_calc' if 'normal_bid_line_calc' in df.columns else 'normal_bid_line'
    
    # Calculate bid_rate if not exists
    if 'bid_rate' not in df.columns:
        df['bid_rate'] = df['bid_amt'] / df['base_amt']
    
    # Detect if rates are in decimal or percentage
    mean_nbl = df[nbl_col].mean()
    is_decimal = mean_nbl < 1
    
    # fluctuation = bid_rate - normal_bid_line
    df['fluctuation_calc'] = df['bid_rate'] - df[nbl_col]
    
    # Log comparison
    if 'fluctuation_raw' in df.columns:
        # Original was in percentage, convert for comparison
        original_decimal = df['fluctuation_raw'] / 100 if df['fluctuation_raw'].mean() > 1 else df['fluctuation_raw']
        diff = (df['fluctuation_calc'] - original_decimal).abs()
        logger.info(f"fluctuation recalculation: mean diff = {diff.mean():.6f}, max diff = {diff.max():.6f}")
    
    return df


def validate_recalculations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate recalculated values
    
    Rules:
    1. If bid_rate_diff = 0 but rank ≠ 1 → Flag as error
    2. Rank 1 should have max normal_bid_line in notice
    """
    df = df.copy()
    
    nbl_col = 'normal_bid_line_calc' if 'normal_bid_line_calc' in df.columns else 'normal_bid_line'
    brd_col = 'bid_rate_diff_calc' if 'bid_rate_diff_calc' in df.columns else 'bid_rate_diff'
    
    # Rule 1: bid_rate_diff = 0 only for rank 1
    zero_diff_not_rank1 = (df[brd_col].abs() < 0.0001) & (df['ranking'] != 1)
    n_violations = zero_diff_not_rank1.sum()
    
    if n_violations > 0:
        logger.warning(f"Validation: {n_violations} records with bid_rate_diff=0 but rank≠1")
    
    # Rule 2: Check rank 1 has max normal_bid_line per notice (vectorized)
    max_nbl_per_notice = df.groupby('notice_id')[nbl_col].transform('max')
    rank1_mask = df['ranking'] == 1
    rank1_has_max = (df.loc[rank1_mask, nbl_col] - max_nbl_per_notice[rank1_mask]).abs() < 0.0001
    n_rank1_not_max = (~rank1_has_max).sum()
    
    if n_rank1_not_max > 0:
        logger.warning(f"Validation: {n_rank1_not_max} notices where rank 1 doesn't have max normal_bid_line")
    
    if n_violations == 0 and n_rank1_not_max == 0:
        logger.info("Validation passed: All recalculated values are consistent")
    
    return df


def apply_all_recalculations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all recalculations in order
    """
    logger.info("Starting recalculation of derived columns...")
    
    df = recalculate_normal_bid_line(df)
    df = recalculate_bid_rate_difference(df)
    df = recalculate_fluctuation(df)
    df = validate_recalculations(df)
    
    logger.info("Recalculation complete")
    
    return df
