The two models complement each other:
- **NGBoost:** Captures what we don't know (bias patterns)
- **XGBoost:** Optimizes based on detected patterns

---

## Training and Validation

### Data Split Strategy

**Temporal validation (prevents data leakage):**

| Phase | Period | Records | Purpose |
|-------|--------|---------|---------|
| Training | Years 1-8 | 800K | Model training |
| Validation | Year 9 | 100K | Hyperparameter tuning |
| Test | Year 10 | 100K | Performance evaluation |
| Production | 6 months | 75 bids | Real deployment |

**Critical:** Chronological order maintained to ensure realistic estimates.

### Evaluation Metrics

**Primary metric:** Win rate (business objective)
- Baseline: 2.5%
- Target: Maximize
- Achieved: 8.0%

**Secondary metrics:**
- Bias detection accuracy
- Prediction MAE (model quality)
- Consistency across agencies

**Tertiary consideration:**
- Economic viability (implicit in binary outcome)

---

## Production Considerations

### Real-World Deployment

**Operational characteristics:**
- Real-time prediction: <1 second per bid
- Batch processing: 10+ bids monthly
- Monitoring: Weekly win rate tracking

**Maintenance:**
- Quarterly retraining with new data
- Bias pattern updates
- Performance degradation alerts

### Key Success Factors

1. **Agency bias quantification**
   - 91% accuracy in bias classification
   - Strong correlation between detected bias and actual behavior

2. **Binary outcome handling**
   - Optimization strategy fundamentally different from continuous problems
   - Focus on win probability, not profit margins

3. **Ensemble synergy**
   - Neither model alone achieves target performance
   - Bias detection + Optimization = 3.2x baseline

---

## Limitations and Future Work

### Current Limitations

**Fundamental constraints:**
- Cannot predict exact floor price (random by design)
- Performance depends on historical data quality
- New agencies require cold-start handling

**Market constraints:**
- Binary outcome: No partial wins possible
- Information asymmetry remains
- Competition also improving

### Potential Improvements

**Technical enhancements:**
- Multi-objective optimization (risk vs. return)
- Transfer learning for new agencies
- Adaptive strategies based on market dynamics

**Business enhancements:**
- Portfolio optimization (which bids to pursue)
- Risk management (exposure limits)
- Market condition adaptation

---

## Conclusion

This methodology demonstrates the value of:

1. **Statistical bias detection** (NGBoost)
   - Quantifies agency-specific patterns
   - Provides confidence levels

2. **Binary outcome optimization** (XGBoost)
   - Different from profit maximization
   - Win-or-nothing strategy

3. **Ensemble synergy** (Two-stage)
   - Separates bias detection from optimization
   - Complementary strengths

**Result:** 3.2x improvement in win rate (2.5% → 8%)

This represents a practical application of probabilistic ML 
to strategic decision-making in binary outcome markets.
