# QuantForge Machine Learning Research Review (Sprint 2.5)

This document provides a rigorous academic and institutional review of the machine learning algorithms, design choices, trade-offs, and risk factors associated with the newly implemented QuantForge Model Training Platform.

---

## 1. Algorithm Selection: Tabular GBDTs

QuantForge initially supports **LightGBM**, **XGBoost**, and **CatBoost**. These gradient-boosted decision tree (GBDT) frameworks represent the state-of-the-art for tabular market data.

### Why Decision Trees Over Deep Learning?

- **Feature Representability**: Financial time-series feature spaces consist of heterogeneous technical indicators, price ratios, and volume characteristics. Trees natively handle multi-scale signals, whereas neural networks (e.g., MLPs, LSTMs) require extensive normalization and are highly sensitive to covariate shifts.
- **Robustness to Noise**: Market data exhibits low signal-to-noise ratios (SNR). GBDTs handle uninformative features through automatic feature selection (splitting on informative nodes and ignoring noise).
- **Efficiency and Scale**: Iterative model exploration requires light footprint training. GBDT models train in seconds/minutes, unlike deep learning architectures that require heavy GPU infrastructure.

### Algorithm-Specific Strengths

1. **LightGBM**: Optimized for training speed and low memory usage. Employs _Gradient-based One-Side Sampling (GOSS)_ and _Exclusive Feature Bundling (EFB)_ to speed up split evaluation. Extremely fast for hyperparameter sweeps.
2. **XGBoost**: Employs regularized objective functions ($L_1$ and $L_2$) to prevent overfitting. Its cache-aware and out-of-core computing implementations render it exceptionally robust for high-dimensional feature spaces.
3. **CatBoost**: Excels at handling categorical attributes natively (using symmetric trees) and implements a unique _ordered boosting_ technique to combat target leakage during training.

---

## 2. Risk & Leakage Analysis

In financial machine learning, data leakage is the single most common cause of backtest outperformance that fails catastrophically in live trading.

### Data Leakage Sources & Prevention

- **Feature Computations**: Technical indicators (RSI, EMA, Bollinger Bands) must be computed in a causal manner. Our upstream `IndicatorEngine` and `FeatureEngine` use strict causal lookbacks, ensuring that for candle $t$, indicators rely _only_ on candles up to and including $t$.
- **Target Labeling**:
  - For forward-looking returns (e.g., $t + k$), the label for candles $t$ from $t-k$ to $t$ cannot be computed without lookahead into the future.
  - **Prevention**: Validation and test splits are partitioned sequentially by open time (`df.iloc` index splits) without random shuffling. This ensures train data always precedes validation data, and validation data always precedes test data.
- **Overlap Leakage**: When training binary labels spanning multiple bars (e.g., horizons $> 1$), targets overlap. If train and validation sets are adjacent, information leaks.
  - **Mitigation**: Future implementations of `TimeSeriesSplit` will incorporate purging (dropping labels that span across the train/validation boundary) to guarantee absolute isolation.

### Class Imbalance Analysis

- Financial classification targets (e.g., price increase $> 1.0\%$) are naturally sparse.
- **Risk**: A classifier trained on rare events will learn to predict the majority class (e.g., always predict 0), achieving high accuracy but zero economic utility.
- **Mitigation**:
  - `DatasetValidator` enforces a minority class threshold (default $5\%$). Training fails early if class ratios drop below this margin.
  - Trainers support library-specific balance controls (e.g., `scale_pos_weight` in XGBoost, `is_unbalance` in LightGBM, `class_weights` in CatBoost) via hyperparameters passed through `TrainingConfig`.

### Overfitting & Generalization Risks

- **Overfitting Indicators**: A substantial gap between train performance (e.g., F1 $= 0.95$) and val/test performance (e.g., F1 $= 0.51$) points to overfitting.
- **Mitigation**:
  - Regularization hyperparameters ($L_1/L_2$ penalties, `max_depth`, `min_child_weight`) are tuned systematically.
  - Candidate status validation threshold prevents weak or overfitted models from progressing to shadow execution pipelines.

---

## 3. Institutional Readiness Assessment

### Current Status

**Approved for Staging/A/B Validation**

The platform achieves a robust software foundation:

- **Independent Prediction System**: No downstream systems couple with specific ML packages; predictions route through `PredictionModel`.
- **Reproducibility**: Identical random seeds produce identical training results across LightGBM, XGBoost, and CatBoost.
- **Quality Assurance**: Complete validation of data quality prior to executing fit loops prevents trash-in-trash-out failures.

### Recommended Next Steps for Production Deployment

1. **Purged Cross-Validation**: Move from single split train-val-test to K-fold TimeSeriesSplit with embargo and purging.
2. **Feature Importance Monitoring**: Continuously track feature attribution drift in live pipelines to flag covariate shifts.
3. **Hyperparameter Optimization (HPO)**: Integrate parameter grids via `ParameterOptimizer` to systematically locate optimal regularizers.
4. **Lifecycle Audits**: Utilize the newly built `ModelRegistry` state transition events and manifests to compile full historic lineage for institutional compliance audits.
