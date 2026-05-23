"""
Validation functions for data preprocessing
"""
import pandas as pd
import numpy as np
import logging

from .config import MIN_BIDDERS_PER_NOTICE

logger = logging.getLogger(__name__)


def create_notice_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unique notice identifier from proj_name + announce_date + bid_date + choose_avg
    
    This handles cases where same project has multiple bidding rounds
    """
    df = df.copy()
    
    # Create composite key
    df['notice_key'] = (
        df['proj_name'].astype(str) + '_' +
        df['announce_date'].astype(str) + '_' +
        df['bid_date'].astype(str) + '_' +
        df['choose_avg'].astype(str)
    )
    
    logger.info(f"Created {df['notice_key'].nunique()} unique notice keys")
    
    return df


def validate_base_amt_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that base_amt is consistent within each notice
    If multiple values exist for same choose_avg, take mode
    """
    df = df.copy()
    
    def fix_base_amt(group):
        # For each unique choose_avg, check base_amt consistency
        for choose_avg_val in group['choose_avg'].unique():
            subgroup = group[group['choose_avg'] == choose_avg_val]
            
            if subgroup['base_amt'].nunique() > 1:
                # Multiple base_amt values - take mode
                mode_val = subgroup['base_amt'].mode()
                if len(mode_val) > 0:
                    mode_base_amt = mode_val.iloc[0]
                    logger.warning(
                        f"Notice {group.name}: Multiple base_amt for choose_avg={choose_avg_val}. "
                        f"Using mode: {mode_base_amt}"
                    )
                    group.loc[subgroup.index, 'base_amt'] = mode_base_amt
        
        return group
    
    df = df.groupby('notice_key', group_keys=False).apply(fix_base_amt)
    
    return df


def filter_invalid_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop notices where all base_amt=0 or all choose_avg=0
    """
    initial_count = len(df)
    initial_notices = df['notice_key'].nunique()
    
    # Mark notices with invalid amounts
    notice_validity = df.groupby('notice_key').apply(
        lambda g: (g['base_amt'] > 0).any() and (g['choose_avg'] > 0).any()
    )
    
    valid_notices = notice_validity[notice_validity].index
    df = df[df['notice_key'].isin(valid_notices)].copy()
    
    dropped_notices = initial_notices - df['notice_key'].nunique()
    dropped_records = initial_count - len(df)
    
    logger.info(f"Dropped {dropped_notices} notices ({dropped_records} records) with invalid amounts")
    
    return df


def filter_insufficient_bidders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop notices with fewer than MIN_BIDDERS_PER_NOTICE bidders
    """
    initial_count = len(df)
    initial_notices = df['notice_key'].nunique()
    
    # Count bidders per notice
    notice_counts = df.groupby('notice_key').size()
    valid_notices = notice_counts[notice_counts >= MIN_BIDDERS_PER_NOTICE].index
    
    df = df[df['notice_key'].isin(valid_notices)].copy()
    
    dropped_notices = initial_notices - len(valid_notices)
    dropped_records = initial_count - len(df)
    
    logger.info(f"Dropped {dropped_notices} notices ({dropped_records} records) with < {MIN_BIDDERS_PER_NOTICE} bidders")
    
    return df


def validate_minimum_constraint(df: pd.DataFrame, buffer_pct: float = 0.03) -> pd.DataFrame:
    """
    Validate bid_amt >= base_amt × min_bid_rate × (1 - buffer)
    
    The min_bid_rate is calculated assuming estimated_price/base_amt = 100%.
    In reality, yega_range varies ±2-3%, so we allow a buffer.
    
    Args:
        df: DataFrame with bid data
        buffer_pct: Allowable buffer below minimum (default 3% = 0.03)
                   This accounts for yega_range variation of 97-103%
    
    Drop records that violate this constraint even with buffer.
    """
    initial_count = len(df)
    
    # Calculate minimum allowed bid WITH buffer
    # min_bid_rate is decimal (0.8147), so min_allowed = base_amt × min_bid_rate × (1 - buffer)
    buffer_multiplier = 1.0 - buffer_pct
    df['_min_allowed_bid'] = df['base_amt'] * df['min_bid_rate'] * buffer_multiplier
    
    # Allow small tolerance (100 won) for rounding
    tolerance = 100
    df['_constraint_violated'] = df['bid_amt'] < (df['_min_allowed_bid'] - tolerance)
    
    violations = df['_constraint_violated'].sum()
    
    if violations > 0:
        logger.warning(f"Found {violations} records violating minimum bid constraint (with {buffer_pct*100:.0f}% buffer)")
        
        # Log some violators for debugging
        violators = df[df['_constraint_violated']][
            ['notice_key', 'company_name_orig', 'bid_amt', '_min_allowed_bid', 'min_bid_rate']
        ].head(10)
        logger.warning(f"Sample violators:\n{violators}")
    
    # Filter out violators
    df = df[~df['_constraint_violated']].copy()
    df = df.drop(columns=['_min_allowed_bid', '_constraint_violated'])
    
    logger.info(f"Dropped {violations} records violating minimum constraint (buffer={buffer_pct*100:.0f}%)")
    
    return df


def add_competitor_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add n_competitors column for each notice
    """
    df = df.copy()
    
    notice_counts = df.groupby('notice_key').size().rename('n_competitors')
    df = df.merge(notice_counts, on='notice_key', how='left')
    
    logger.info(f"Added n_competitors (mean: {df['n_competitors'].mean():.1f})")
    
    return df


def apply_all_validations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all validation steps
    """
    initial_count = len(df)
    logger.info(f"Starting validation with {initial_count} records")
    
    df = create_notice_key(df)
    df = validate_base_amt_consistency(df)
    df = filter_invalid_amounts(df)
    df = filter_insufficient_bidders(df)
    df = add_competitor_count(df)
    
    final_count = len(df)
    logger.info(f"Validation complete: {initial_count} → {final_count} records")
    
    return df
