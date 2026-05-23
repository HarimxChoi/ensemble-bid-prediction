# ensemble-bid-prediction

R2CCP-based system for Korean PQ (qualification review) bidding. Distribution prediction + Monte Carlo optimization.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What it does

Predicts an optimal `normalized_bid_rate` for Korean public procurement PQ auctions.
Treats the winning bid as a distribution problem, not a point prediction.

## Why

PQ bidding is structurally unequal. Technical score determines each company's `min_bid_rate`, so higher-scoring companies have wider bidding freedom. Predicting "the winning bid" as a single number ignores both the structural asymmetry and the multi-modal shape of competitor bid distributions.

## Approach

### R2CCP custom impl (`r2ccp_2.py`)

The pip R2CCP package uses APS cumulative-mass intervals that collapse bimodal distributions. This implementation switches to **per-bin threshold** for distribution-shape preservation.

| | pip R2CCP | this impl |
|---|---|---|
| Interval | APS cumulative | Per-bin threshold |
| Bimodal preserved | no | yes |
| entropy_weight | fixed | configurable (default 3.0) |
| loss_weight | 1.0 | winner 5x |
| Adaptive range | min/max | percentile 0.5/99.5 |

### 8 context models (Q x BRD)

Q-group: ranking quantile (Q1 head-of-pack vs Q2 tail). BRD-group: bid-rate-difference quartiles.

```
              BRD_1  BRD_2  BRD_3  BRD_4
       Q1     m1     m2     m3     m4
       Q2     m5     m6     m7     m8
```

Per-context conformal alpha (`group_alphas`) tuned for 90-95% coverage by context.

### Monte Carlo simulation (500K iterations)

For each candidate bid rate `r` in grid `[0.975, 1.025]` step `0.0005`:

1. Sample `yega` from `Normal(institution_mean, institution_std)`
2. Sample competitor rates via tiered model (dedicated / behavioral / global)
3. Tally wins

Law of large numbers gives `P(win | r)`. `tol_validity` strategy picks the highest-validity rate within `tolerance=0.02` of max `P(win)`.

## Results (aggregate)

- Per-context lift: argmax / tol_validity strategies (+1.2 to +6.0 percentage points by context)
- Coverage diagnostics: mean 84.2% within 90% CI on out-of-time bids
- Mean win rate uplift on backtest: roughly +2.0 percentage points (per-company range varies)

## Repository layout

```
ensemble-bid-prediction/
├── r2ccp_2.py                       # core R2CCP custom impl (per-bin threshold)
├── mc_cp_simulation.py              # MC simulation engine
├── mc_cp_optimization.py            # Grid-search optimizer
├── mc_cp_optimization_fast.py       # Reduced-iter variant for testing
├── mc_validation_full.py            # Full validation (50K iter)
├── mc_simulator_production.py       # Production-grade simulator
├── feature_engineering.py           # Position / gap / notice / project / institution features
├── evaluate.py                      # Evaluation suite
├── backtest_simulation.py           # Backtest
├── backtest_simulation_tolerance.py # Backtest with tolerance strategy
├── strategic_bidding_optimizer.py   # Strategy comparison
├── copula_test.py / rank_copula_test.py  # Independence test for competitor sampling
├── train_cluster_models.py          # Per-cluster R2CCP models
├── train_company_models.py          # Per-company R2CCP models
├── train_strategic_mimicry.py       # Strategic mimicry training
├── train_weighted_quantile.py       # Weighted quantile regression
├── validate_inference.py / validate_optimizer.py / validation_backtest*.py
├── verify_normalized_formula.py     # Sanity check on normalization
├── analyze_expr_params.py           # Hyperparameter analysis
├── test_*.py                        # Unit tests
└── src/
    ├── data_pipeline/               # Metric calculators (min_bid_rate, normal_bid_line, ...)
    ├── features/                    # Feature engineering pipeline
    ├── models/conformal.py          # R2CCP wrapper
    └── preprocessing/               # Filters, recalculations, validators
```

## Quick Start

```bash
pip install -r requirements.txt

# Train per-context models (requires features + processed data, not included)
python train_cluster_models.py
python train_company_models.py

# Run validation with full MC simulation
python mc_validation_full.py

# Backtest with tolerance strategy
python backtest_simulation_tolerance.py
```

## Repository note

This repo provides the methodology, R2CCP custom impl, MC simulation engine, and per-context training pipeline. Training data, fitted model artifacts, and company-specific backtest tables are intentionally not included.

## References

- Guha et al. 2024, *R2CCP: Regression-as-Classification with Conformal Coverage Prediction*: https://arxiv.org/abs/2404.08168

## License

MIT
