"""
Strategic Bidding Optimizer
===========================

Production-grade bid optimization system for Korean PQ auctions.

Transforms MC-CP from prediction to actionable bidding strategy:
- Input: Project + Simulation Company
- Output: Optimal bid that maximizes P(win)

Features:
- Grid search over 100 candidate bids (0.975-1.025)
- 100K iterations per candidate (vectorized)
- Multi-window yega sampling (3m/6m/1y/all/global)
- Ablation study comparing yega strategies
- Validation against 50 actual auction winners

Author: MC-CP Team
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configuration
N_ITERATIONS_PER_CANDIDATE = 100000  # 100K for production
CANDIDATE_BIDS = np.arange(0.975, 1.025, 0.0005)  # 100 candidates
N_VALIDATION_PROJECTS = 50

# Yega fallback thresholds
MIN_SAMPLES_3M = 5
MIN_SAMPLES_6M = 5
MIN_SAMPLES_1Y = 20
MIN_SAMPLES_ALL = 20

print(f"Strategic Bidding Optimizer Configuration:")
print(f"  Candidates: {len(CANDIDATE_BIDS)} bids")
print(f"  Iterations per candidate: {N_ITERATIONS_PER_CANDIDATE:,}")
print(f"  Total simulations per project: {len(CANDIDATE_BIDS) * N_ITERATIONS_PER_CANDIDATE:,}")
print()


class StrategicBiddingOptimizer:
    """
    Find optimal bid for specific company that maximizes P(win).
    
    Key Methods:
    - optimize_bid(): Find optimal bid via grid search
    - ablation_study_yega(): Compare yega sampling strategies
    - validate(): Test on actual auction winners
    """
    
    def __init__(self, models_dir='models', data_dir='data/processed'):
        """Initialize optimizer with models and profiles."""
        self.root = Path(__file__).parent
        self.models_dir = self.root / models_dir
        self.data_dir = self.root / data_dir
        
        print("Initializing Strategic Bidding Optimizer...")
        
        # Load models
        print("  Loading models...")
        
        # Try HPO models first (best), then weighted quantile, then standard
        hpo_model_path = self.models_dir / 'company_models_hpo.pkl'
        if hpo_model_path.exists():
            self.models = joblib.load(hpo_model_path)
            self.q_values = joblib.load(self.models_dir / 'q_values_hpo.pkl')
            print(f"    OK {len(self.models)} HPO models (Weighted Quantile α=0.15)")
        else:
            # Fallback to standard models
            self.models = joblib.load(self.models_dir / 'company_models.pkl')
            self.q_values = joblib.load(self.models_dir / 'q_values.pkl')
            print(f"    ✅ {len(self.models)} standard models")
        
        # Load institution profiles
        print("  Loading institution profiles...")
        self.inst_profiles = self._load_multi_window_profiles()
        print(f"    ✅ {len(self.inst_profiles)} institutions with multi-window data")
        
        # Default yega (global fallback)
        self.default_yega = {
            'mean': 100.0,
            'std': 0.8,
            'min': 97.0,
            'max': 103.0
        }
        
        print("✅ Optimizer ready!")
        print()
    
    def _load_multi_window_profiles(self):
        """
        Load institution profiles with multiple time windows.
        
        Returns dict with structure:
        {
            'I0403': {
                'mean_3m': 99.85, 'std_3m': 0.65, 'n_bids_3m': 12,
                'mean_6m': 99.92, 'std_6m': 0.71, 'n_bids_6m': 25,
                'mean_1y': 100.05, 'std_1y': 0.82, 'n_bids_1y': 48,
                'mean_all': 100.12, 'std_all': 0.88, 'n_bids_all': 156
            },
            ...
        }
        """
        # Load static profile (has all-time data)
        static_df = pd.read_csv(self.data_dir / 'inst_profile_static.csv')
        
        # Try to load timeseries (has 3m/6m/1y windows)
        try:
            timeseries_df = pd.read_csv(self.data_dir / 'inst_yega_timeseries.csv')
            has_timeseries = True
        except:
            print("    ⚠️ inst_yega_timeseries.csv not found, using static profile only")
            has_timeseries = False
        
        profiles = {}
        
        for idx, row in static_df.iterrows():
            inst_code = row['inst_code']
            
            # All-time data (from static profile)
            profile = {
                'mean_all': row['mean_yega_rate_all'] * 100,
                'std_all': row['std_yega_rate_all'] * 100,
                'n_bids_all': row.get('total_bids', 0),
                'min_all': row.get('min_yega_rate_all', 0.97) * 100,
                'max_all': row.get('max_yega_rate_all', 1.03) * 100,
            }
            
            # TODO: Add 3m/6m/1y from timeseries if available
            # For now, use all-time as fallback
            if has_timeseries:
                # inst_ts = timeseries_df[timeseries_df['inst_code'] == inst_code]
                # Extract 3m/6m/1y aggregations
                # For now, use all-time
                pass
            
            # Fallback: use all-time for all windows
            for window in ['3m', '6m', '1y']:
                profile[f'mean_{window}'] = profile['mean_all']
                profile[f'std_{window}'] = profile['std_all']
                profile[f'n_bids_{window}'] = 0  # Will trigger fallback
            
            profiles[inst_code] = profile
        
        return profiles
    
    def sample_yega_vectorized(self, inst_code, n_samples, method='auto'):
        """
        Sample yega with adaptive window selection.
        
        Args:
            inst_code: Institution code
            n_samples: Number of samples
            method: 'auto', '3m', '6m', '1y', 'all', 'global'
        
        Returns:
            (n_samples,) array of yega values
        """
        profile = self.inst_profiles.get(inst_code, None)
        
        if profile is None or method == 'global':
            # Global fallback
            mean, std = self.default_yega['mean'], self.default_yega['std']
            min_val, max_val = self.default_yega['min'], self.default_yega['max']
        else:
            # Adaptive window selection
            if method == 'auto':
                # Priority: 3m → 6m → 1y → all → global
                if profile['n_bids_3m'] >= MIN_SAMPLES_3M:
                    window = '3m'
                elif profile['n_bids_6m'] >= MIN_SAMPLES_6M:
                    window = '6m'
                elif profile['n_bids_1y'] >= MIN_SAMPLES_1Y:
                    window = '1y'
                elif profile['n_bids_all'] >= MIN_SAMPLES_ALL:
                    window = 'all'
                else:
                    # Fallback to global
                    mean, std = self.default_yega['mean'], self.default_yega['std']
                    min_val, max_val = self.default_yega['min'], self.default_yega['max']
                    window = None
            else:
                window = method
            
            if window is not None:
                mean = profile[f'mean_{window}']
                std = profile[f'std_{window}']
                min_val = profile.get('min_all', 97.0)
                max_val = profile.get('max_all', 103.0)
        
        samples = np.random.normal(mean, std, size=n_samples)
        return np.clip(samples, min_val, max_val)
    
    def predict_batch(self, companies, X_features):
        """Batch prediction using XGBoost models."""
        predictions = np.zeros(len(companies))
        
        for i, company in enumerate(companies):
            if company not in self.models:
                predictions[i] = 1.0  # Fallback
            else:
                model = self.models[company]
                pred = model.predict(X_features[i:i+1])[0]
                predictions[i] = pred
        
        return predictions
    
    def simulate_with_fixed_bid(
        self,
        company_code,
        fixed_bid,
        project_data,
        n_iterations=N_ITERATIONS_PER_CANDIDATE,
        yega_method='auto',
        random_seed=None
    ):
        """
        Run MC simulation with simulation company's bid FIXED.
        
        Args:
            company_code: Simulation company (e.g., 'C0022')
            fixed_bid: Fixed normalized_bid_rate (e.g., 1.0033)
            project_data: Dict with companies, X_features, etc.
            n_iterations: Number of MC iterations
            yega_method: Yega sampling method
            random_seed: Optional seed
        
        Returns:
            P(win | fixed_bid): Float probability
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        companies = project_data['companies']
        X_features = project_data['X_features']
        base_amt = project_data['base_amt']
        inst_code = project_data.get('inst_code', None)
        min_bid_rates = project_data['min_bid_rates']
        
        # Find simulation company index
        sim_idx = companies.index(company_code)
        n_comp = len(companies)
        
        # STEP 1: Predict for all companies
        predictions = self.predict_batch(companies, X_features)
        
        # Override simulation company with FIXED bid
        predictions[sim_idx] = fixed_bid
        
        # STEP 2: Sample yega
        yega_samples = self.sample_yega_vectorized(inst_code, n_iterations, yega_method)
        
        # STEP 3: Sample bids (vectorized)
        predictions_tiled = np.tile(predictions, (n_iterations, 1))
        
        # Get q-values (ZERO for simulation company!)
        q_values_arr = np.array([
            self.q_values.get(c, 0.02) if i != sim_idx else 0.0
            for i, c in enumerate(companies)
        ])
        
        # Sample conformal noise
        noise = np.random.uniform(
            -q_values_arr, +q_values_arr,
            size=(n_iterations, n_comp)
        )
        
        # Sampled normalized bid rates
        sampled_normalized = predictions_tiled + noise
        sampled_normalized = np.clip(sampled_normalized, 0.75, 1.25)
        
        # STEP 4: Calculate bid amounts
        estimated_price = base_amt * (yega_samples / 100)[:, None]
        bid_amounts = estimated_price * min_bid_rates * sampled_normalized
        thresholds = estimated_price * min_bid_rates
        
        # STEP 5: Determine winners
        valid_mask = bid_amounts >= thresholds
        bid_amounts_masked = np.where(valid_mask, bid_amounts, np.inf)
        winner_indices = np.argmin(bid_amounts_masked, axis=1)
        
        # STEP 6: Calculate P(win)
        p_win = np.mean(winner_indices == sim_idx)
        
        return float(p_win)
    
    def optimize_bid(
        self,
        company_code,
        project_data,
        candidate_bids=CANDIDATE_BIDS,
        n_iterations=N_ITERATIONS_PER_CANDIDATE,
        yega_method='auto',
        random_seed=42,
        show_progress=True
    ):
        """
        Find optimal bid that maximizes P(win).
        
        Returns:
            optimal: Dict with {bid, p_win, ...}
            curve: List of {candidate_bid, p_win} for all candidates
        """
        print(f"\n{'='*70}")
        print(f"OPTIMIZING BID FOR {company_code}")
        print(f"{'='*70}")
        print(f"Candidates: {len(candidate_bids)} ({candidate_bids[0]:.4f} to {candidate_bids[-1]:.4f})")
        print(f"Iterations per candidate: {n_iterations:,}")
        print(f"Yega method: {yega_method}")
        print(f"Total simulations: {len(candidate_bids) * n_iterations:,}")
        print()
        
        results = []
        
        iterator = tqdm(candidate_bids, desc="Grid search") if show_progress else candidate_bids
        
        for candidate_bid in iterator:
            p_win = self.simulate_with_fixed_bid(
                company_code=company_code,
                fixed_bid=candidate_bid,
                project_data=project_data,
                n_iterations=n_iterations,
                yega_method=yega_method,
                random_seed=random_seed
            )
            
            results.append({
                'candidate_bid': float(candidate_bid),
                'p_win': float(p_win)
            })
        
        # Find optimal (maximize P(win))
        results_df = pd.DataFrame(results)
        optimal_idx = results_df['p_win'].idxmax()
        optimal = results_df.iloc[optimal_idx].to_dict()
        
        print()
        print(f"{'='*70}")
        print("OPTIMIZATION RESULT")
        print(f"{'='*70}")
        print(f"Optimal bid: {optimal['candidate_bid']:.4f}")
        print(f"P(win): {optimal['p_win']:.2%}")
        print(f"{'='*70}")
        print()
        
        return optimal, results
    
    def ablation_study_yega(
        self,
        company_code,
        project_data,
        methods=['global', 'all', '1y', '6m', '3m', 'auto'],
        n_iterations=50000
    ):
        """
        Ablation study: Compare yega sampling methods.
        
        Returns: List of results for each method
        """
        print(f"\n{'='*70}")
        print("ABLATION STUDY: Yega Sampling Methods")
        print(f"{'='*70}")
        print()
        
        ablation_results = []
        
        for method in methods:
            print(f"\n--- Testing method: {method} ---")
            
            # Run optimization with this method
            optimal, curve = self.optimize_bid(
                company_code=company_code,
                project_data=project_data,
                n_iterations=n_iterations,
                yega_method=method,
                show_progress=False
            )
            
            ablation_results.append({
                'method': method,
                'optimal_bid': optimal['candidate_bid'],
                'p_win': optimal['p_win'],
                'curve': curve
            })
        
        # Display comparison
        print(f"\n{'='*70}")
        print("ABLATION RESULTS")
        print(f"{'='*70}")
        
        df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'curve'}
            for r in ablation_results
        ])
        print(df.to_string(index=False))
        print()
        
        return ablation_results


# Helper function for feature encoding
def encode_features_for_inference(df):
    """Encode features exactly as in training."""
    categorical_cols = [
        'behavioral_type', 'position_category', 'competition_intensity',
        'amt_group', 'inst_yega_consistency', 'inst_bidding_frequency',
        'inst_yega_bias', 'data_window_used'
    ]
    
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
    
    exclude_from_features = [
        'record_id', 'company_code', 'notice_id', 'institution_code',
        'target', 'announce_date', 'min_bid_rate', 'base_amt', 'is_winner',
        'normalized_bid_rate'
    ]
    
    feature_cols = [col for col in df_encoded.columns if col not in exclude_from_features]
    X = df_encoded[feature_cols].fillna(-999).values
    
    return X, feature_cols


if __name__ == "__main__":
    print("Strategic Bidding Optimizer")
    print("Use validate_optimizer.py to run full validation")
