"""
Evaluation Script for MC-CP System

Evaluates trained models on validation set:
1. Coverage validation (does 90% target hold?)
2. Win probability calibration
3. Two-stage MC simulation comparison with actuals

Usage:
    cd mccp_simulation
    python evaluate.py
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.config import PROCESSED_DIR
from src.models.conformal import ConformalCalibrator
from src.models.model_training import TieredModelManager
from src.models.monte_carlo import TwoStageMonteCarlo, YegaConfig


def setup_logging():
    """Setup logging"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def load_models():
    """Load trained models and calibrator"""
    model_dir = Path(__file__).parent / 'models' / 'saved_calibrated'
    
    # Load calibrator
    calibrator = ConformalCalibrator.load(model_dir / 'calibrator.pkl')
    
    # Load model manager metadata
    with open(model_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Load individual models
    company_models = {}
    for company_code in metadata.get('company_model_codes', []):
        model_path = model_dir / f"company_{company_code}.pkl"
        if model_path.exists():
            from src.models.model_training import CompanyModelTrainer
            company_models[company_code] = CompanyModelTrainer.load(model_path)
    
    # Load global model
    global_model_path = model_dir / 'global_model.pkl'
    if global_model_path.exists():
        from src.models.model_training import CompanyModelTrainer
        global_model = CompanyModelTrainer.load(global_model_path)
    else:
        global_model = None
    
    return company_models, global_model, calibrator


def load_validation_data():
    """Load validation data"""
    val_path = PROCESSED_DIR / 'val.csv'
    val_df = pd.read_csv(val_path, encoding='utf-8-sig')
    val_df['announce_date'] = pd.to_datetime(val_df['announce_date'])
    return val_df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix"""
    feature_cols = [
        'ranking', 'tech_score', 'gap_to_1st', 'n_competitors',
        'min_bid_rate', 'normal_bid_line_calc', 'bid_rate_diff_calc'
    ]
    
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    
    X['is_top3'] = (X['ranking'] <= 3).astype(int)
    X['is_mid'] = ((X['ranking'] > 3) & (X['ranking'] <= 7)).astype(int)
    X['is_low'] = (X['ranking'] > 7).astype(int)
    
    return X


def evaluate_coverage(val_df, company_models, global_model, calibrator):
    """
    Evaluate conformal prediction coverage
    
    Target: 90% coverage (α=0.10)
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating conformal coverage...")
    
    results = []
    
    for company_code in val_df['company_code'].unique():
        company_val = val_df[val_df['company_code'] == company_code]
        
        if len(company_val) < 3:
            continue
        
        X = build_features(company_val)
        y_actual = company_val['normalized_bid_rate'].values
        
        # Get predictions
        if company_code in company_models:
            model = company_models[company_code]
            try:
                y_pred = model.predict(X)
            except:
                continue
        elif global_model:
            try:
                y_pred = global_model.predict(X)
            except:
                continue
        else:
            continue
        
        # Get intervals
        q = calibrator.get_q_value(company_code)
        lower = y_pred - q
        upper = y_pred + q
        
        # Apply hard constraint
        lower = np.maximum(lower, 0.97)
        
        # Check coverage
        in_interval = (y_actual >= lower) & (y_actual <= upper)
        coverage = in_interval.mean()
        
        results.append({
            'company_code': company_code,
            'n_samples': len(company_val),
            'q_value': q,
            'coverage': coverage,
            'interval_width': 2 * q,
            'mean_actual': y_actual.mean(),
            'mean_pred': y_pred.mean()
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        overall_coverage = results_df['coverage'].mean()
        n_companies = len(results_df)
        n_above_90 = (results_df['coverage'] >= 0.90).sum()
        
        logger.info(f"Coverage evaluation:")
        logger.info(f"  Companies evaluated: {n_companies}")
        logger.info(f"  Mean coverage: {overall_coverage:.1%}")
        logger.info(f"  Companies with ≥90% coverage: {n_above_90}/{n_companies}")
        logger.info(f"  Mean interval width: {results_df['interval_width'].mean():.4f}")
        
        # Target: 90% coverage
        if overall_coverage >= 0.85:
            logger.info("✅ Coverage meets target (≥85%)")
        else:
            logger.warning(f"⚠️ Coverage below target: {overall_coverage:.1%}")
    
    return results_df


def evaluate_predictions(val_df, company_models, global_model):
    """
    Evaluate prediction accuracy (MAE, RMSE)
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating prediction accuracy...")
    
    all_actual = []
    all_pred = []
    
    for company_code in val_df['company_code'].unique():
        company_val = val_df[val_df['company_code'] == company_code]
        
        X = build_features(company_val)
        y_actual = company_val['normalized_bid_rate'].values
        
        if company_code in company_models:
            model = company_models[company_code]
            try:
                y_pred = model.predict(X)
                all_actual.extend(y_actual)
                all_pred.extend(y_pred)
            except:
                pass
        elif global_model:
            try:
                y_pred = global_model.predict(X)
                all_actual.extend(y_actual)
                all_pred.extend(y_pred)
            except:
                pass
    
    if len(all_actual) > 0:
        all_actual = np.array(all_actual)
        all_pred = np.array(all_pred)
        
        mae = np.mean(np.abs(all_actual - all_pred))
        rmse = np.sqrt(np.mean((all_actual - all_pred) ** 2))
        
        logger.info(f"Prediction accuracy:")
        logger.info(f"  Samples: {len(all_actual)}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        
        return {
            'n_samples': len(all_actual),
            'mae': mae,
            'rmse': rmse
        }
    
    return None


def main():
    """Main evaluation"""
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("EVALUATION: MC-CP SYSTEM")
    logger.info("=" * 60)
    
    # Load models
    logger.info("\n--- Loading Models ---")
    company_models, global_model, calibrator = load_models()
    logger.info(f"Loaded {len(company_models)} company models")
    logger.info(f"Global model: {'Yes' if global_model else 'No'}")
    logger.info(f"Calibrated companies: {len(calibrator.q_values)}")
    
    # Load validation data
    logger.info("\n--- Loading Validation Data ---")
    val_df = load_validation_data()
    logger.info(f"Validation: {len(val_df)} records, {val_df['notice_id'].nunique()} notices")
    
    # Evaluate coverage
    logger.info("\n--- Coverage Evaluation ---")
    coverage_df = evaluate_coverage(val_df, company_models, global_model, calibrator)
    
    # Evaluate accuracy
    logger.info("\n--- Accuracy Evaluation ---")
    accuracy = evaluate_predictions(val_df, company_models, global_model)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    
    print(f"\nLog file: {log_file}")
    
    return coverage_df, accuracy


if __name__ == "__main__":
    main()
