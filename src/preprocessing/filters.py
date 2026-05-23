"""
Filtering functions for data preprocessing
"""
import pandas as pd
import numpy as np
import re
import logging

from .config import PLACEHOLDER_PATTERN, EXCLUDED_PQ_TYPES

logger = logging.getLogger(__name__)


def filter_invalid_pq_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid PQ types: negotiation, advanced_bid, general
    """
    initial_count = len(df)
    
    # Drop excluded pq_types
    mask = df['pq_type'].isin(EXCLUDED_PQ_TYPES)
    df = df[~mask].copy()
    
    dropped = initial_count - len(df)
    logger.info(f"Dropped {dropped} records with pq_type in {EXCLUDED_PQ_TYPES}")
    
    return df


def filter_placeholder_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out placeholder records like "total N companies participation"
    """
    initial_count = len(df)
    
    mask = df['company_name_orig'].str.contains(PLACEHOLDER_PATTERN, regex=True, na=False)
    df = df[~mask].copy()
    
    dropped = initial_count - len(df)
    logger.info(f"Dropped {dropped} placeholder records (placeholder participation rows)")
    
    return df


def filter_missing_pq_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out records without PQ date (general_competition)
    """
    initial_count = len(df)
    
    # Check for empty/null pq_date
    mask = df['pq_date'].isna() | (df['pq_date'] == '') | (df['pq_date'] == 0)
    df = df[~mask].copy()
    
    dropped = initial_count - len(df)
    logger.info(f"Dropped {dropped} records with missing pq_date (general_competition)")
    
    return df


def filter_zero_normal_bid_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out records where normal_bid_line = 0
    """
    initial_count = len(df)
    
    df = df[df['normal_bid_line_raw'] > 0].copy()
    
    dropped = initial_count - len(df)
    logger.info(f"Dropped {dropped} records with normal_bid_line=0")
    
    return df


def filter_zero_bid_amt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out records where bid_amt = 0
    """
    initial_count = len(df)
    
    df = df[df['bid_amt'] > 0].copy()
    
    dropped = initial_count - len(df)
    logger.info(f"Dropped {dropped} records with bid_amt=0")
    
    return df


def apply_all_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all initial filters in sequence
    """
    initial_count = len(df)
    logger.info(f"Starting filtering with {initial_count} records")
    
    df = filter_invalid_pq_types(df)
    df = filter_placeholder_records(df)
    df = filter_missing_pq_date(df)
    df = filter_zero_normal_bid_line(df)
    df = filter_zero_bid_amt(df)
    
    final_count = len(df)
    logger.info(f"Filtering complete: {initial_count} → {final_count} records ({initial_count - final_count} removed)")
    
    return df
