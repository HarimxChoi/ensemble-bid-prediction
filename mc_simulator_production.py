"""
Production-Grade MC-CP Simulator
=================================

Features:
1. ✅ Uses actual XGBoost model predictions (not random!)
2. ✅ Institution-specific yega sampling (from inst_profile_static.csv)
3. ✅ Vectorized NumPy operations (10-100x faster)
4. ✅ Proper feature extraction from train.csv structure
5. ⏳ ONNX Runtime support (to be added later)

Architecture:
- Load models + q-values + institution profiles once
- For each project: extract features → predict → simulate (vectorized)
- No DataFrame loops, pure NumPy broadcasting
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import json

class MCCPSimulator:
    """Production MC-CP Simulator with vectorization."""
    
    def __init__(self, models_dir='models', data_dir='data/processed'):
        """
        Initialize simulator.
        
        Loads:
        - XGBoost models (114 companies)
        - Q-values (conformal uncertainty)
        - Institution yega profiles
        """
        self.root = Path(__file__).parent
        self.models_dir = self.root / models_dir
        self.data_dir = self.root / data_dir
        
        print("Initializing MC-CP Simulator...")
        
        # Load models
        print("  Loading models...")
        self.models = joblib.load(self.models_dir / 'company_models.pkl')
        print(f"    ✅ {len(self.models)} company models")
        
        # Load q-values
        self.q_values = joblib.load(self.models_dir / 'q_values.pkl')
        print(f"    ✅ {len(self.q_values)} q-values")
        
        # Load institution yega profiles
        print("  Loading institution profiles...")
        self.inst_profiles = self._load_institution_profiles()
        print(f"    ✅ {len(self.inst_profiles)} institutions")
        
        # Default yega profile (if institution not found)
        self.default_yega = {
            'mean': 100.0,
            'std': 1.67,
            'min': 97.0,
            'max': 103.0
        }
        
        print("✅ Simulator ready!")
    
    def _load_institution_profiles(self):
        """Load institution yega profiles from inst_profile_static.csv."""
        df = pd.read_csv(self.data_dir / 'inst_profile_static.csv')
        
        profiles = {}
        for idx, row in df.iterrows():
            inst_code = row['inst_code']
            profiles[inst_code] = {
                'mean': row['mean_yega_rate_all'] * 100,  # Convert to percentage
                'std': row['std_yega_rate_all'] * 100,
                'min': row['min_yega_rate_all'] * 100,
                'max': row['max_yega_rate_all'] * 100
            }
        
        return profiles
    
    def sample_yega_vectorized(self, inst_code, n_samples):
        """
        Sample yega from institution-specific distribution.
        
        Args:
            inst_code: Institution code
            n_samples: Number of samples (MC iterations)
        
        Returns:
            yega_samples: (n_samples,) array in percentage (e.g., 100.5)
        """
        profile = self.inst_profiles.get(inst_code, self.default_yega)
        
        samples = np.random.normal(
            loc=profile['mean'],
            scale=profile['std'],
            size=n_samples
        )
        
        return np.clip(samples, profile['min'], profile['max'])
    
    def predict_batch(self, companies, X_features):
        """
        Batch predict using XGBoost models.
        
        Args:
            companies: List of company codes
            X_features: (n_companies, n_features) array
        
        Returns:
            predictions: (n_companies,) array of normalized_bid_rate predictions
        """
        predictions = np.zeros(len(companies))
        
        for i, company in enumerate(companies):
            if company not in self.models:
                # Use mean prediction if model not available
                predictions[i] = 1.0
            else:
                model = self.models[company]
                pred = model.predict(X_features[i:i+1])[0]
                predictions[i] = pred
        
        return predictions
    
    def simulate_vectorized(self, project_data, n_iterations=50000, random_seed=None):
        """
        Vectorized MC simulation (NO LOOPS over iterations!).
        
        Args:
            project_data: Dict with:
                - companies: List of company codes
                - X_features: (n_comp, n_feat) feature array
                - base_amt: Project base amount
                - inst_code: Institution code
               - min_bid_rates: (n_comp,) array of min_bid_rates
            n_iterations: Number of MC iterations
            random_seed: Optional random seed
        
        Returns:
            results: Dict with win_probabilities and diagnostics
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        companies = project_data['companies']
        X_features = project_data['X_features']
        base_amt = project_data['base_amt']
        inst_code = project_data.get('inst_code', None)
        min_bid_rates = project_data['min_bid_rates']
        
        n_comp = len(companies)
        
        # STEP 1: Predict once for all companies using actual models!
        predictions = self.predict_batch(companies, X_features)  # (n_comp,)
        
        # STEP 2: Sample yega (institution-specific)
        yega_samples = self.sample_yega_vectorized(inst_code, n_iterations)  # (n_iter,)
        
        # STEP 3: Sample all bids (vectorized, uses model predictions!)
        # Tile predictions across iterations
        predictions_tiled = np.tile(predictions, (n_iterations, 1))  # (n_iter, n_comp)
        
        # Get q-values for all companies
        q_values_arr = np.array([
            self.q_values.get(c, 0.02) for c in companies
        ])  # (n_comp,)
        
        # Sample conformal noise
        noise = np.random.uniform(
            -q_values_arr, +q_values_arr,
            size=(n_iterations, n_comp)
        )  # (n_iter, n_comp)
        
        # Sampled normalized bid rates (centered at model prediction!)
        sampled_normalized = predictions_tiled + noise  # (n_iter, n_comp)
        sampled_normalized = np.clip(sampled_normalized, 0.75, 1.25)
        
        # STEP 4: Calculate bid amounts (broadcasting magic!)
        estimated_price = base_amt * (yega_samples / 100)[:, None]  # (n_iter, 1)
        bid_amounts = estimated_price * min_bid_rates * sampled_normalized  # (n_iter, n_comp)
        thresholds = estimated_price * min_bid_rates  # (n_iter, n_comp)
        
        # STEP 5: Determine winners (vectorized, no loops!)
        valid_mask = bid_amounts >= thresholds  # (n_iter, n_comp)
        bid_amounts_masked = np.where(valid_mask, bid_amounts, np.inf)
        winner_indices = np.argmin(bid_amounts_masked, axis=1)  # (n_iter,)
        
        # STEP 6: Count wins
        win_counts = np.bincount(winner_indices, minlength=n_comp)
        win_probabilities = win_counts / n_iterations
        
        # Package results
        results = {
            'win_probabilities': {
                company: float(win_probabilities[i])
                for i, company in enumerate(companies)
            },
            'n_iterations': n_iterations,
            'n_companies': n_comp,
            'model_predictions': {
                company: float(predictions[i])
                for i, company in enumerate(companies)
            },
            'yega_stats': {
                'mean': float(yega_samples.mean()),
                'std': float(yega_samples.std()),
                'min': float(yega_samples.min()),
                'max': float(yega_samples.max())
            }
        }
        
        return results


def encode_features_for_inference(df):
    """
    Encode features EXACTLY as in training (train_company_models.py).
    
    CRITICAL: Must match encode_full_dataset() to get 106 features!
    """
    categorical_cols = [
        'behavioral_type', 'position_category', 'competition_intensity',
        'amt_group', 'inst_yega_consistency', 'inst_bidding_frequency',
        'inst_yega_bias', 'data_window_used'
    ]
    
    df_encoded = df.copy()
    
    # One-hot encode categorical columns
    for col in categorical_cols:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
    
    # Exclude metadata columns
    exclude_from_features = [
        'record_id', 'company_code', 'notice_id', 'institution_code',
        'target', 'announce_date', 'min_bid_rate', 'base_amt', 'is_winner',
        'normalized_bid_rate'
    ]
    
    feature_cols = [col for col in df_encoded.columns if col not in exclude_from_features]
    
    # Fill NaN with -999 (as in training)
    X = df_encoded[feature_cols].fillna(-999).values
    
    return X, feature_cols


def validate_simulator():
    """Run validation on test projects."""
    print("=" * 70)
    print("MC-CP SIMULATOR VALIDATION")
    print("=" * 70)
    print()
    
    # Initialize simulator
    simulator = MCCPSimulator()
    print()
    
    # Load feature data
    print("Loading validation data...")
    root = Path(__file__).parent
    features_df = pd.read_csv(root / 'analysis_results' / 'feature_sample.csv')
    train_df = pd.read_csv(root / 'data' / 'processed' / 'train_clean.csv')
    
    # Merge
    df = features_df.merge(
        train_df[['record_id', 'company_code', 'notice_id', 'base_amt', 'min_bid_rate',
                  'institution_code', 'is_winner', 'normalized_bid_rate']],
        on='record_id',
        how='inner'
    )
    
    print(f"✅ Loaded {len(df):,} records")
    
    # CRITICAL: Encode features EXACTLY as in training!
    print("Encoding features...")
    X_all, feature_cols = encode_features_for_inference(df)
    print(f"✅ Features: {len(feature_cols)} (X shape: {X_all.shape})")
    print()
    
    # Test on 10 projects
    test_projects = df['notice_id'].unique()[:10]
    
    results = []
    
    for notice_id in tqdm(test_projects, desc="Simulating"):
        project_df = df[df['notice_id'] == notice_id]
        project_indices = df[df['notice_id'] == notice_id].index
        
        # Get actual winner
        actual_winner_rows = project_df[project_df['is_winner'] == True]
        if len(actual_winner_rows) == 0:
            continue
        
        actual_winner = actual_winner_rows.iloc[0]['company_code']
        
        # Extract project data with ENCODED features
        companies = project_df['company_code'].tolist()
        X_features = X_all[project_indices]  # Use encoded features!
        
        project_data = {
            'companies': companies,
            'X_features': X_features,
            'base_amt': project_df.iloc[0]['base_amt'],
            'inst_code': project_df.iloc[0].get('institution_code', None),
            'min_bid_rates': project_df['min_bid_rate'].values
        }
        
        # Run simulation!
        sim_results = simulator.simulate_vectorized(
            project_data,
            n_iterations=50000,
            random_seed=42
        )
        
        # Get predicted winner
        win_probs = sim_results['win_probabilities']
        predicted_winner = max(win_probs, key=win_probs.get)
        
        results.append({
            'notice_id': str(notice_id),
            'actual_winner': actual_winner,
            'predicted_winner': predicted_winner,
            'correct': (predicted_winner == actual_winner),
            'p_win_actual': win_probs.get(actual_winner, 0.0),
            'p_win_predicted': win_probs[predicted_winner],
            'n_companies': sim_results['n_companies']
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    accuracy = np.mean([r['correct'] for r in results])
    mean_p_win_actual = np.mean([r['p_win_actual'] for r in results])
    
    print(f"Projects: {len(results)}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Mean P(win) for actual winners: {mean_p_win_actual:.1%}")
    print()
    
    # Show details
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"{status} {r['notice_id']}: "
              f"pred={r['predicted_winner']} ({r['p_win_predicted']:.1%}), "
              f"actual={r['actual_winner']} ({r['p_win_actual']:.1%})")
    
    # Save
    output_path = root / 'analysis_results' / 'mc_simulator_validation.json'
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'n_projects': len(results),
                'accuracy': float(accuracy),
                'mean_p_win_actual': float(mean_p_win_actual)
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Saved: {output_path}")


if __name__ == "__main__":
    validate_simulator()
