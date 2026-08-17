# R2CCP 입찰 예측

[English](./README.md) | 한국어

8개 경쟁환경의 다봉형 PQ 투찰분포를 모델링하고 50만 회 Monte Carlo simulation으로 낙찰확률을 계산한 의사결정 시스템입니다.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Why

PQ 입찰은 기술점수와 가격점수를 함께 평가합니다. 기술점수가 낮아질수록 가격점수로 만회해야 하므로 유효한 투찰구간이 분리되고, 중심 가격대의 위와 아래에 입찰 불가능 구간이 생깁니다. 이 때문에 실제 투찰행태가 하나의 봉우리가 아닌 다봉분포로 나타나며, 하나의 점예측이나 연속된 예측구간으로는 이를 표현하기 어려웠습니다.

## How

경쟁사의 가격점수와 투찰행태를 특징으로 사용해 경쟁환경을 8개 context로 분리하고, context별 R2CCP 분포를 학습했습니다. 공개 R2CCP 구현체가 두 봉우리와 그 사이의 빈 구간을 하나의 interval로 합치는 문제를 발견해 entropy regularization과 per-bin conformal threshold를 적용했습니다. 이후 후보별 50만 회 Monte Carlo simulation으로 투찰 범위별 예상 낙찰확률을 계산했습니다.

## Result

- 69,934건으로 8개 context model을 학습했습니다.
- 13,984건의 시간순 validation에서 α=0.10 기준 **coverage 90.73%**를 기록했습니다.
- 실제 운영에서 내부 PQ 낙찰 KPI를 **35%** 개선했습니다.

## Method details

### Interval collapse 개선 (`r2ccp_2.py`)

공개 R2CCP 구현체는 APS cumulative-mass interval을 사용해 다봉분포의 서로 떨어진 봉우리와 그 사이 빈 구간까지 하나의 interval로 합칩니다. 이 프로젝트에서는 **per-bin threshold**로 전환해 분리된 유효구간을 보존했습니다.

![pip R2CCP vs r2ccp_2 (bimodal 분포)](./r2ccp-comparison.png)

*기존 구현은 두 봉우리 사이의 빈 구간까지 포함하지만, `r2ccp_2.py`는 두 봉우리를 분리된 interval로 유지합니다.*

| | pip R2CCP | this impl |
|---|---|---|
| Interval | APS cumulative | Per-bin threshold |
| Bimodal preserved | no | yes |
| entropy_weight | fixed | configurable (default 3.0) |
| loss_weight | 1.0 | winner 5x |
| Adaptive range | min/max | percentile 0.5/99.5 |

### 8 context models (Q x BRD)

Q-group: ranking quantile (Q1 head-of-pack vs Q2 tail). BRD-group: bid-rate-difference quartile.

```
              BRD_1  BRD_2  BRD_3  BRD_4
       Q1     m1     m2     m3     m4
       Q2     m5     m6     m7     m8
```

Context별 conformal alpha (`group_alphas`)를 90-95% coverage로 튜닝.

### Monte Carlo simulation (500K iterations)

`[0.975, 1.025]` 구간에 `0.0005` 간격의 후보 입찰률 `r`을 만들고, 각 후보에 대해 다음 과정을 수행합니다.

1. `yega`를 `Normal(institution_mean, institution_std)`에서 샘플
2. 경쟁사 입찰률을 tiered model (dedicated / behavioral / global)로 샘플
3. 승수 카운트

반복 결과로 `P(win | r)`을 추정합니다. `tol_validity` 전략은 최대 `P(win)`과의 차이가 `tolerance=0.02` 이내인 후보 중 validity가 가장 높은 입찰률을 선택합니다.

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

이 저장소에는 R2CCP 개선 구현, Monte Carlo simulation engine과 context별 학습 파이프라인이 포함되어 있습니다. 학습 데이터, 학습된 model artifact와 회사별 backtest 표는 공개하지 않았습니다.

## References

- Guha et al. 2024, *R2CCP: Regression-as-Classification with Conformal Coverage Prediction*: https://arxiv.org/abs/2404.08168

## License

MIT
