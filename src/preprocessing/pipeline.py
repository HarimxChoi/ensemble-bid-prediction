"""
Main preprocessing pipeline with recalculations

Usage:
    python src/preprocessing/pipeline.py
    
This pipeline:
1. Loads raw data with proper encoding (euc-kr)
2. Filters invalid records
3. Renames columns to standardized names
4. Recalculates normal_bid_line, bid_rate_diff, fluctuation from scratch
5. Applies minimum constraint with 3% buffer
6. Standardizes company names with fuzzy matching
7. Outputs train/val CSVs with utf-8-sig encoding (Excel compatible)
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.config import (
    RAW_DATA, PROCESSED_DIR, INTERIM_DIR, LOG_DIR,
    TRAIN_VAL_SPLIT_DATE, OUTPUT_COLUMNS
)
from src.preprocessing.filters import apply_all_filters
from src.preprocessing.validators import apply_all_validations, validate_minimum_constraint
from src.preprocessing.transformers import apply_all_transformations, rename_columns
from src.preprocessing.company_master import standardize_companies
from src.preprocessing.recalculations import apply_all_recalculations


def setup_logging():
    """Setup logging to file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"preprocessing_{timestamp}.log"
    
    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def load_data() -> pd.DataFrame:
    """Load raw data with proper encoding"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from {RAW_DATA}")
    
    # Try different encodings
    encodings = ['euc-kr', 'cp949', 'utf-8']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(RAW_DATA, encoding=encoding)
            logger.info(f"Successfully loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            logger.debug(f"Failed with encoding: {encoding}")
            continue
    
    if df is None:
        raise ValueError(f"Could not load {RAW_DATA} with any encoding")
    
    logger.info(f"Loaded {len(df)} records, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and reorder final output columns"""
    logger = logging.getLogger(__name__)
    
    # Define final column order with recalculated columns
    final_columns = [
        # IDs
        'record_id', 'notice_id',
        
        # Project context
        'announce_date', 'pq_date', 'bid_date',
        'project_type', 'area', 'proj_name',
        
        # Financial
        'base_amt', 'choose_avg',
        
        # Competitive position (scores - 0-100 scale)
        'ranking', 'tech_score', 'gap_to_1st',
        'n_competitors',
        
        # Company
        'company_code', 'company_name', 'company_first',
        
        # Rates (decimal, ×0.01)
        'min_bid_rate',            # 0.81535
        'normal_bid_line_calc',    # Recalculated normal_bid_line
        'bid_rate',                # 0.852
        'normalized_bid_rate',     # 1.025
        'norm_bid_margin',         # 0.025
        
        # Recalculated columns
        'bid_rate_diff_calc',      # Recalculated bid_rate_diff
        'fluctuation_calc',        # Recalculated fluctuation
        
        # Bid
        'bid_amt',
        
        # Winner info
        'is_winner',
        'winner_company_code',
        'winner_normalized_rate',
        'winner_norm_margin',
    ]
    
    # Select available columns
    available = [col for col in final_columns if col in df.columns]
    missing = [col for col in final_columns if col not in df.columns]
    
    if missing:
        logger.warning(f"Missing columns: {missing}")
    
    df = df[available].copy()
    
    logger.info(f"Selected {len(available)} output columns")
    
    return df


def split_train_val(df: pd.DataFrame, split_date: str = TRAIN_VAL_SPLIT_DATE) -> tuple:
    """Split data by announce_date"""
    logger = logging.getLogger(__name__)
    
    split_datetime = pd.to_datetime(split_date)
    
    train_df = df[df['announce_date'] <= split_datetime].copy()
    val_df = df[df['announce_date'] > split_datetime].copy()
    
    logger.info(f"Train/Val split at {split_date}:")
    logger.info(f"  Train: {len(train_df)} records, {train_df['notice_id'].nunique()} notices")
    logger.info(f"  Val: {len(val_df)} records, {val_df['notice_id'].nunique()} notices")
    
    if len(train_df) > 0:
        logger.info(f"  Train date range: {train_df['announce_date'].min()} to {train_df['announce_date'].max()}")
    if len(val_df) > 0:
        logger.info(f"  Val date range: {val_df['announce_date'].min()} to {val_df['announce_date'].max()}")
    
    return train_df, val_df


def save_data(train_df: pd.DataFrame, val_df: pd.DataFrame, master_df: pd.DataFrame):
    """Save processed data with utf-8-sig encoding (Excel compatible)"""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save files with utf-8-sig (Excel compatible)
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    master_path = PROCESSED_DIR / "company_master.csv"
    
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    master_df.to_csv(master_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Saved train.csv: {len(train_df)} records (utf-8-sig)")
    logger.info(f"Saved val.csv: {len(val_df)} records (utf-8-sig)")
    logger.info(f"Saved company_master.csv: {len(master_df)} entries (utf-8-sig)")
    
    # Generate summary
    summary_path = PROCESSED_DIR / "processing_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PREPROCESSING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Encoding: utf-8-sig (Excel compatible)\n\n")
        
        f.write("TRAIN SET:\n")
        f.write(f"  Records: {len(train_df)}\n")
        f.write(f"  Notices: {train_df['notice_id'].nunique()}\n")
        f.write(f"  Companies: {train_df['company_code'].nunique()}\n")
        if len(train_df) > 0:
            f.write(f"  Date range: {train_df['announce_date'].min()} to {train_df['announce_date'].max()}\n")
        f.write("\n")
        
        f.write("VALIDATION SET:\n")
        f.write(f"  Records: {len(val_df)}\n")
        f.write(f"  Notices: {val_df['notice_id'].nunique()}\n")
        f.write(f"  Companies: {val_df['company_code'].nunique()}\n")
        if len(val_df) > 0:
            f.write(f"  Date range: {val_df['announce_date'].min()} to {val_df['announce_date'].max()}\n")
        f.write("\n")
        
        f.write("COMPANY MASTER:\n")
        f.write(f"  Unique companies: {master_df['company_code'].nunique()}\n")
        f.write(f"  Total entries: {len(master_df)}\n")
        multi_groups = master_df[master_df['group_size'] > 1]['company_code'].nunique()
        f.write(f"  Groups with multiple names: {multi_groups}\n")
        f.write("\n")
        
        f.write("RECALCULATED COLUMNS:\n")
        if 'normal_bid_line_calc' in train_df.columns:
            f.write(f"  normal_bid_line_calc: mean={train_df['normal_bid_line_calc'].mean():.4f}\n")
        if 'bid_rate_diff_calc' in train_df.columns:
            f.write(f"  bid_rate_diff_calc: mean={train_df['bid_rate_diff_calc'].mean():.4f}\n")
        if 'fluctuation_calc' in train_df.columns:
            f.write(f"  fluctuation_calc: mean={train_df['fluctuation_calc'].mean():.4f}\n")
        f.write("\n")
        
        f.write("RATE STATISTICS (Train):\n")
        if len(train_df) > 0:
            f.write(f"  normalized_bid_rate: mean={train_df['normalized_bid_rate'].mean():.4f}, "
                   f"std={train_df['normalized_bid_rate'].std():.4f}\n")
            f.write(f"  norm_bid_margin: mean={train_df['norm_bid_margin'].mean():.4f}, "
                   f"std={train_df['norm_bid_margin'].std():.4f}\n")
    
    logger.info(f"Saved processing_summary.txt")


def save_interim(df: pd.DataFrame, step_name: str):
    """Save interim data for debugging"""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DIR / f"{step_name}.csv"
    df.to_csv(path, index=False, encoding='utf-8-sig')


def run_pipeline():
    """Execute full preprocessing pipeline"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE (with recalculations)")
    logger.info("=" * 60)
    
    # Step 0: Load data
    df = load_data()
    save_interim(df, "step0_raw")
    
    # Step 0.5: Rename columns to standardized names FIRST
    logger.info("\n--- STEP 0.5: Column Renaming ---")
    df = rename_columns(df)
    save_interim(df, "step0_5_renamed")
    
    # Step 1: Initial filtering
    logger.info("\n--- STEP 1: Initial Filtering ---")
    df = apply_all_filters(df)
    save_interim(df, "step1_filtered")
    
    # Step 2: Validation
    logger.info("\n--- STEP 2: Notice Validation ---")
    df = apply_all_validations(df)
    save_interim(df, "step2_validated")
    
    # Step 3-6: Transformations (IDs, rate conversion, winner extraction)
    logger.info("\n--- STEP 3-6: Transformations ---")
    df = apply_all_transformations(df)
    save_interim(df, "step3_6_transformed")
    
    # Step 6.5: Recalculate derived columns from scratch
    logger.info("\n--- STEP 6.5: Recalculate Derived Columns ---")
    df = apply_all_recalculations(df)
    save_interim(df, "step6_5_recalculated")
    
    # Step 7: Minimum constraint validation (after rates are calculated)
    logger.info("\n--- STEP 7: Minimum Constraint Validation ---")
    df = validate_minimum_constraint(df)
    save_interim(df, "step7_constraint_validated")
    
    # Step 8: Company standardization
    logger.info("\n--- STEP 8: Company Standardization ---")
    df, master_df = standardize_companies(df)
    save_interim(df, "step8_companies_standardized")
    
    # Step 9: Select output columns
    logger.info("\n--- STEP 9: Column Selection ---")
    df = select_output_columns(df)
    save_interim(df, "step9_columns_selected")
    
    # Step 10: Train/Val split
    logger.info("\n--- STEP 10: Train/Val Split ---")
    train_df, val_df = split_train_val(df)
    
    # Step 11: Save
    logger.info("\n--- STEP 11: Save Data (utf-8-sig) ---")
    save_data(train_df, val_df, master_df)
    
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"\nTrain: {len(train_df)} records, {train_df['notice_id'].nunique()} notices")
    print(f"Val: {len(val_df)} records, {val_df['notice_id'].nunique()} notices")
    print(f"Companies: {master_df['company_code'].nunique()} unique")
    print(f"\nOutput saved to: {PROCESSED_DIR}")
    print(f"Encoding: utf-8-sig (Excel compatible)")
    
    return train_df, val_df, master_df


def main():
    """Main entry point"""
    log_file = setup_logging()
    print(f"Logging to: {log_file}")
    
    try:
        train_df, val_df, master_df = run_pipeline()
        return train_df, val_df, master_df
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
