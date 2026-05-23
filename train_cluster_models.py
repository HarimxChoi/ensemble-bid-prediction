"""
Cluster-Level Model Training for Small Companies
=================================================

For companies with 10-29 bids (too few for per-company models):
  1. Cluster by behavioral features
  2. Find optimal K using silhouette score
  3. Train one XGBoost model per cluster
  4. Compute conformal q-values per cluster

Output:
  - models/cluster_models.pkl: Trained models per cluster
  - models/cluster_q_values.pkl: Conformal q-values per cluster
  - models/company_to_cluster.pkl: Mapping company_code -> cluster_id
  - models/cluster_training_stats.csv: Training statistics
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ==============================================================================
# CONFIGURATION
# ==============================================================================

MIN_BIDS_FOR_CLUSTERING = 10   # Minimum bids to include in clustering
MAX_BIDS_FOR_CLUSTERING = 29   # Companies with more go to per-company models
MIN_CALIBRATION_SAMPLES = 10   # Per cluster (lower since we pool)
K_RANGE = range(3, 12)         # Test K from 3 to 11

VALIDATION_RATIO = 0.2
CALIBRATION_RATIO = 0.2
CONFORMAL_ALPHA = 0.1

XGB_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 150,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'gpu_id': 0
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_project_root():
    return Path(__file__).parent


def load_data():
    """Load features and company metadata."""
    root = get_project_root()
    features_path = root / 'analysis_results' / 'feature_sample.csv'
    train_path = root / 'data' / 'processed' / 'train.csv'
    
    print("Loading data...")
    features_df = pd.read_csv(features_path)
    train_df = pd.read_csv(train_path, usecols=['record_id', 'company_code', 'announce_date'])
    
    df = features_df.merge(train_df, on='record_id', how='left')
    print(f"  Total records: {len(df):,}")
    
    return df


def get_clustering_features(df):
    """Extract company-level behavioral features for clustering."""
    # Aggregate per company (use last available features)
    company_agg = df.groupby('company_code').agg({
        'n_hist_bids': 'last',
        'hist_mean_margin_30': 'last',
        'hist_std_margin_30': 'last',
        'hist_mean_margin_100': 'last',
        'hist_std_margin_100': 'last',
        'hist_mean_margin_300': 'last',
        'hist_std_margin_300': 'last',
        'hist_win_rate_30': 'last',
        'hist_win_rate_100': 'last',
        'hist_win_rate_300': 'last',
        'target': ['mean', 'std']  # Historical bid behavior
    }).reset_index()
    
    # Flatten column names
    company_agg.columns = [
        'company_code', 'n_hist_bids',
        'mean_margin_30', 'std_margin_30',
        'mean_margin_100', 'std_margin_100',
        'mean_margin_300', 'std_margin_300',
        'win_rate_30', 'win_rate_100', 'win_rate_300',
        'mean_bid', 'std_bid'
    ]
    
    # Add bid count
    bid_counts = df['company_code'].value_counts().reset_index()
    bid_counts.columns = ['company_code', 'bid_count']
    company_agg = company_agg.merge(bid_counts, on='company_code')
    
    return company_agg


def find_optimal_k(X_scaled, k_range):
    """Find optimal K using silhouette score."""
    print(f"\nFinding optimal K in range {list(k_range)}...")
    
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append((k, score))
        print(f"  K={k}: silhouette={score:.4f}")
    
    # Find best K
    best_k, best_score = max(scores, key=lambda x: x[1])
    print(f"\n✅ Optimal K={best_k} (silhouette={best_score:.4f})")
    
    return best_k, scores


def compute_conformal_qvalue(model, X_cal, y_cal, alpha=0.1):
    """Compute conformal prediction q-value."""
    y_pred_cal = model.predict(X_cal)
    scores = np.abs(y_cal - y_pred_cal)
    
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    
    q_value = np.quantile(scores, q_level)
    actual_coverage = np.mean(scores <= q_value)
    
    return {
        'q_value': float(q_value),
        'actual_coverage': float(actual_coverage),
        'n_calibration': n
    }


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    print("=" * 60)
    print("Cluster-Level Model Training for Small Companies")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    root = get_project_root()
    
    # 1. Load data
    df = load_data()
    
    # 2. Get company-level features for clustering
    company_features = get_clustering_features(df)
    
    # 3. Filter to companies with 10-29 bids
    cluster_companies = company_features[
        (company_features['bid_count'] >= MIN_BIDS_FOR_CLUSTERING) &
        (company_features['bid_count'] <= MAX_BIDS_FOR_CLUSTERING)
    ].copy()
    
    print(f"\nCompanies for clustering: {len(cluster_companies)}")
    print(f"Total bids in these companies: {cluster_companies['bid_count'].sum():,}")
    
    if len(cluster_companies) < 10:
        print("⚠️ Too few companies for clustering")
        return
    
    # 4. Prepare clustering features
    feature_cols = [
        'mean_margin_30', 'std_margin_30',
        'mean_margin_100', 'std_margin_100',
        'mean_margin_300', 'std_margin_300',
        'win_rate_30', 'win_rate_100', 'win_rate_300',
        'mean_bid', 'std_bid'
    ]
    
    X_cluster = cluster_companies[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # 5. Find optimal K
    optimal_k, k_scores = find_optimal_k(X_scaled, K_RANGE)
    
    # 6. Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_companies['cluster_id'] = kmeans.fit_predict(X_scaled)
    
    # Show cluster distribution
    print("\nCluster distribution:")
    cluster_dist = cluster_companies.groupby('cluster_id').agg({
        'company_code': 'count',
        'bid_count': 'sum'
    }).rename(columns={'company_code': 'n_companies'})
    print(cluster_dist)
    
    # 7. Create company_code -> cluster_id mapping
    company_to_cluster = dict(zip(
        cluster_companies['company_code'],
        cluster_companies['cluster_id']
    ))
    
    # 8. Encode full dataset for training
    print("\nEncoding features...")
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
    
    exclude_cols = ['record_id', 'company_code', 'target', 'announce_date']
    feature_names = [c for c in df_encoded.columns if c not in exclude_cols]
    
    X_all = df_encoded[feature_names].fillna(-999).values
    y_all = df['target'].values
    company_codes_all = df['company_code'].values
    announce_dates_all = df['announce_date'].values
    
    print(f"  Feature matrix: {X_all.shape}")
    
    # 9. Train per-cluster models
    print(f"\n{'='*60}")
    print("TRAINING CLUSTER MODELS")
    print(f"{'='*60}\n")
    
    cluster_models = {}
    cluster_q_values = {}
    training_stats = []
    
    for cluster_id in range(optimal_k):
        # Get companies in this cluster
        cluster_company_codes = cluster_companies[
            cluster_companies['cluster_id'] == cluster_id
        ]['company_code'].tolist()
        
        # Get all bids from these companies
        mask = np.isin(company_codes_all, cluster_company_codes)
        X_cluster_data = X_all[mask]
        y_cluster_data = y_all[mask]
        dates_cluster = announce_dates_all[mask]
        
        n_samples = len(y_cluster_data)
        print(f"Cluster {cluster_id}: {len(cluster_company_codes)} companies, {n_samples} bids")
        
        if n_samples < 30:
            print(f"  ⚠️ Skipping - too few samples")
            continue
        
        # Sort by date
        date_order = np.argsort(dates_cluster)
        X_cluster_data = X_cluster_data[date_order]
        y_cluster_data = y_cluster_data[date_order]
        
        # 3-way split
        n_cal = max(int(n_samples * CALIBRATION_RATIO), 5)
        n_val = max(int(n_samples * VALIDATION_RATIO), 5)
        n_train = n_samples - n_cal - n_val
        
        X_train = X_cluster_data[:n_train]
        y_train = y_cluster_data[:n_train]
        X_val = X_cluster_data[n_train:n_train+n_val]
        y_val = y_cluster_data[n_train:n_train+n_val]
        X_cal = X_cluster_data[-n_cal:]
        y_cal = y_cluster_data[-n_cal:]
        
        # Train
        try:
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Conformal calibration
            conformal_result = compute_conformal_qvalue(model, X_cal, y_cal, CONFORMAL_ALPHA)
            
            cluster_models[cluster_id] = model
            cluster_q_values[cluster_id] = conformal_result['q_value']
            
            # Metrics
            val_mae = np.mean(np.abs(y_val - model.predict(X_val)))
            
            stats = {
                'cluster_id': cluster_id,
                'n_companies': len(cluster_company_codes),
                'n_total': n_samples,
                'n_train': n_train,
                'n_validation': n_val,
                'n_calibration': n_cal,
                'val_mae': val_mae,
                'q_value': conformal_result['q_value'],
                'actual_coverage': conformal_result['actual_coverage']
            }
            training_stats.append(stats)
            
            print(f"  ✅ val_mae={val_mae:.4f}, q={conformal_result['q_value']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:50]}")
    
    # 10. Save results
    models_dir = root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(cluster_models, models_dir / 'cluster_models.pkl')
    joblib.dump(cluster_q_values, models_dir / 'cluster_q_values.pkl')
    joblib.dump(company_to_cluster, models_dir / 'company_to_cluster.pkl')
    joblib.dump(scaler, models_dir / 'cluster_scaler.pkl')
    joblib.dump(kmeans, models_dir / 'kmeans_model.pkl')
    
    with open(models_dir / 'cluster_q_values.json', 'w') as f:
        json.dump({str(k): v for k, v in cluster_q_values.items()}, f, indent=2)
    
    pd.DataFrame(training_stats).to_csv(models_dir / 'cluster_training_stats.csv', index=False)
    
    # Save K selection results
    k_results = pd.DataFrame(k_scores, columns=['k', 'silhouette_score'])
    k_results.to_csv(models_dir / 'k_selection_results.csv', index=False)
    
    print(f"\n{'='*60}")
    print("CLUSTER TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Clusters trained: {len(cluster_models)}")
    print(f"Companies mapped: {len(company_to_cluster)}")
    if cluster_q_values:
        print(f"Average cluster q-value: {np.mean(list(cluster_q_values.values())):.4f}")
    print(f"{'='*60}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
