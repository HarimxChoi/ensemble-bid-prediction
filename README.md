# Ensemble Bid Prediction System

![Status](https://img.shields.io/badge/Status-Production-success)
![Win Rate](https://img.shields.io/badge/Win_Rate-7.5%25-brightgreen)
![Improvement](https://img.shields.io/badge/Improvement-3x_Baseline-blue)
![Tech](https://img.shields.io/badge/Tech-NGBoost_+_XGBoost-orange)

Hybrid machine learning system for government procurement bidding optimization, 
achieving **3x improvement in win rate** (2.5% → 7.5%) in highly competitive markets.

---

## Problem Context

### Market Characteristics

**Ultra-Competitive Environment:**
- Average participants per bid: **~100 companies**
- Market baseline win rate: **~1%** (1 winner out of 100)
- Company baseline: **2.5%** (2.5x market average)
- Target: Maximize win rate in binary outcome (win or nothing)

**Auction Mechanism:**
- Sealed-bid format with randomized floor price
- Floor rate range: Pre-defined by agency (e.g., 70~90% of base price)
- Actual floor: Randomly determined within range at bidding close
- Winner: Lowest bid **above** the floor price
- **Outcome:** Binary (Win = Contract, Lose = Zero)

**Challenge:**
- Information asymmetry: Cannot know competitors' bids
- Randomness: Floor price unknown until reveal
- High stakes: Wrong prediction = guaranteed loss

---

## Solution Approach

### Two-Stage Ensemble Framework

**Stage 1: Uncertainty Quantification (NGBoost)**

Model agency-specific bidding behavior patterns and quantify systematic bias.

**Why NGBoost?**
- Captures probabilistic distribution of floor price behavior
- Quantifies prediction uncertainty
- **Agency bias detection:**
  - **Downward bias:** Agency consistently sets floor below expected range
  - **Upward bias:** Agency consistently sets floor above expected range
  - **Neutral:** No statistically significant bias

NGBoost tests statistical significance of bias patterns and quantifies them.

**Output:** Bias scores + Uncertainty scores for each agency

---

**Stage 2: Bid Rate Prediction (XGBoost)**

Predict optimal **bid rate** (bid price / base price) using bias/uncertainty as features.

**Why XGBoost?**
- Handles complex feature interactions
- Robust to noise and outliers
- Production-proven reliability

**Key Features:**
- **NGBoost bias scores** (Stage 1 output)
- **NGBoost uncertainty scores** (Stage 1 output)
- Additional proprietary features derived from historical data

Feature engineering methodology contains competitive intelligence 
and is not publicly disclosed.

**Output:** Recommended bid rate (not absolute price)

**Objective:**
- **NOT profit maximization**
- **Pure win/loss problem:** Bid correctly = Win contract, Bid incorrectly = Zero
- Predict bid rate that maximizes P(win) in binary outcome market

---

### System Architecture
```
Historical Bid Data (10 years, 1M records)
    ↓
[Stage 1: NGBoost]
    ├─ Agency bias detection (statistical testing)
    ├─ Uncertainty quantification
    └─ Confidence scoring
    ↓
Bias Scores + Uncertainty Scores + Other Features
    ↓
[Stage 2: XGBoost]
    ├─ Bid rate prediction (bid/base price)
    ├─ Binary outcome optimization (win or nothing)
    └─ Bid recommendation
    ↓
Final Bid Rate → Bid Price
```

---

## Performance Results

### Win Rate Improvement

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Company Win Rate** | 2.5% | **8%** | **3.2x** |
| **vs. Market Avg** | 2.5x | 8.0x | - |
| **Error Reduction** | - | - | ~68.75% |

**Training Data:**
- Historical records: **1M bids** (10 years)
- Training samples: 800K
- Validation samples: 100K
- Test samples: 100K

**Production Validation:**
- Test period: 6 months
- Sample size: **75 bids** (actual production deployment)
- Win count: 7 wins (vs. expected 1-2 at baseline)

**Context:**
- Market: ~100 participants per bid (1% baseline)
- Company: Started at 2.5% (already 2.5x market)
- **Result: 8.0% win rate = 8.0x market average**

### Business Impact

**Quantitative:**
- Win rate: 2.5% → 8.0% (+220%)
- Opportunities captured: 3.2x increase
- Production deployment: Handling 10+ bids/month

**Qualitative:**
- Systematic decision-making replacing intuition
- Reduced manual analysis time
- Consistent performance across varying market conditions
- **Binary outcome handled:** Win entire contract or nothing

---

## Technical Implementation

### Stage 1: NGBoost for Bias Detection

**Model Configuration:**
- Base learner: Decision trees
- Distribution: Normal (Gaussian)
- Training: Agency-stratified sampling

**Agency Bias Analysis:**

Different agencies exhibit systematic biases in floor price selection:

| Bias Type | Behavior | NGBoost Detection |
|-----------|----------|-------------------|
| **Downward** | Floor consistently below expected | Mean μ < range midpoint (statistically significant) |
| **Upward** | Floor consistently above expected | Mean μ > range midpoint (statistically significant) |
| **Neutral** | No systematic pattern | No significant deviation from expected |

NGBoost performs statistical testing to confirm bias significance, 
then quantifies both bias direction and magnitude as features.

**Feature Categories:**
- Agency historical floor patterns
- Bid characteristics
- Market conditions

**Output:**
- Bias direction (↓ / → / ↑)
- Bias magnitude (strength of pattern)
- Uncertainty (prediction confidence)

**Stage 2: Bid Rate Prediction (XGBoost)**

Predict optimal **bid rate** (bid price / base price) using bias/uncertainty as features.

**Why XGBoost?**
- Handles complex feature interactions
- Robust to noise and outliers
- Production-proven reliability

**Key Features:**
- **NGBoost bias scores** (Stage 1 output)
- **NGBoost uncertainty scores** (Stage 1 output)
- Additional proprietary features derived from historical data

Feature engineering methodology contains competitive intelligence 
and is not publicly disclosed.

**Output:** Recommended bid rate (not absolute price)

**Objective:**
- **NOT profit maximization**
- **Pure win/loss problem:** Bid correctly = Win contract, Bid incorrectly = Zero
- Predict bid rate that maximizes P(win) in binary outcome market

---

## Methodology Highlights

### 1. Agency Bias Quantification

**Statistical Testing:**
For each agency, NGBoost tests:
```
H0: Floor price ~ Uniform(range_min, range_max)
H1: Floor price exhibits systematic bias
```

If H1 confirmed (p < 0.05):
- Measure bias direction (↓ or ↑)
- Quantify bias strength (effect size)
- Estimate prediction uncertainty

**Example:**
```
Agency A: Downward bias detected
  - Mean floor: 87.8% (range: 87.745% - 88.113%)
  - Bias strength: -0.13 percentage points
  - Confidence: High (low uncertainty)

Agency B: Neutral
  - Mean floor: 87.93% (range: 87.745% - 88.113%)
  - Bias strength: N/A
  - Confidence: Moderate
```

### 2. Binary Outcome Optimization

Unlike continuous optimization problems, this is purely **binary:**
- Bid above floor + lowest → **Win entire contract**
- Bid below floor OR not lowest → **Zero**

This changes the optimization strategy:
- Focus: Minimize P(bid < floor)
- Secondary: Minimize bid among valid bids
- NOT concerned with: Profit margins, expected values

### 3. Temporal Validation

Training/validation split respects chronological order:
- Training: Years 1-8 (800K bids)
- Validation: Year 9 (100K bids)
- Test: Year 10 (100K bids)
- Production: Real deployment (100 bids over 6 months)

Prevents data leakage and ensures realistic performance estimates.

---

## System Characteristics

**Strengths:**
- Probabilistic approach handles inherent randomness
- Agency bias detection improves accuracy
- Two-stage design separates concerns cleanly
- Production-proven over 6+ months
- Binary outcome handling (win or nothing)

**Limitations:**
- Cannot predict exact floor price (random by design)
- Performance depends on historical data quality
- Occasional losses unavoidable due to market randomness
- Binary market: No partial wins

---

## Code Availability

**What's included:**
- ✅ High-level methodology
- ✅ Performance metrics
- ✅ System architecture

**What's not included:**
- ❌ Feature engineering implementation
- ❌ Model training code
- ❌ Agency bias detection algorithms
- ❌ Optimization strategies

**Reason:** Proprietary bidding strategies developed for commercial deployment.

---

## Technology Stack

**Machine Learning:**
- NGBoost: Bias detection + Uncertainty quantification
- XGBoost: Bid rate prediction
- scikit-learn: Preprocessing & evaluation

**Production:**
- Python 3.10+
- pandas, numpy
- Production API integration

**Infrastructure:**
- CPU-only deployment
- Real-time prediction (<1s latency)

**Data Scale:**
- Training: 1M records (10 years)
- Production: 75 bids (6 months validation)

---

## Project Context

This system was developed for a civil engineering firm operating in 
Korean government procurement markets. The solution has been in 
production for 6+ months, consistently delivering 7.5% win rates 
in a binary outcome (win or nothing) market.

**Skills demonstrated:**
- Ensemble learning
- Statistical bias detection
- Uncertainty quantification
- Binary outcome optimization
- Production ML deployment
- Domain expertise application

---

## Contact

**Author:** Harim Choi  
**Email:** 2.harim.choi@gmail.com  
**GitHub:** [@HarimxChoi](https://github.com/HarimxChoi)

For collaboration or detailed methodology inquiries, please reach out via email.

---

## License

This project documentation is available under MIT License. 
Implementation code is proprietary.
