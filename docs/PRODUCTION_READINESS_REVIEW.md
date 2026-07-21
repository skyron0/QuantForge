# QuantForge Production Readiness Review (Sprint 2.8.1)

This document evaluates the operational readiness, audit compliance, security safeguards, and contingency strategies of the QuantForge Model Training and Registry Platform.

---

## 1. Lifecycle State Promotion Gates

To ensure only rigorously backtested and validated models are executed in staging and production environment settings, we enforce a strict promotion sequencing policy.

### Implemented State Machine Checklist

Every model starts in the `TRAINING` state and must proceed strictly in order:

$$\text{TRAINING} \longrightarrow \text{CANDIDATE} \longrightarrow \text{VALIDATED} \longrightarrow \text{SHADOW} \longrightarrow \text{PRODUCTION}$$

At any point, a model can be transitioned out of the active loop to `DEPRECATED` or `ARCHIVED`.

### Gate Criteria & Verification Rules

| Transition                | Enforced Validation Gate                                                                                                                          | Operational Action                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **TRAINING → CANDIDATE**  | Validation thresholds met (e.g. F1 $\ge 0.55$) in training pipeline.                                                                              | Auto-generated candidate status.                         |
| **CANDIDATE → VALIDATED** | Walk-forward performance and leakage-safe purged cross-validation (with embargo) satisfy validation thresholds. Human auditor signature required. | Manual check entry logged.                               |
| **VALIDATED → SHADOW**    | Passed isolation testing. Run in sandboxed paper trading environment.                                                                             | Ingestion by live market consumers (no order execution). |
| **SHADOW → PRODUCTION**   | Minimal tracker drawdown and stable predictive accuracy over target time window.                                                                  | Enabled order execution permission.                      |

---

## 2. Platform Audit Trails

Operational visibility and model lineage are captured chronologically inside model manifests and experiment tracking databases.

- **Immutable Serialization**: The registry records all parameters (dataset ID, feature columns, git commit, hyperparameters, metrics) inside `data/registry/model_registry.json`.
- **Transition History logs**: Every state modification appends a `TransitionEvent` timeline block tracking:
  - `from_status`
  - `to_status`
  - `timestamp`
  - `notes` (reasons for promotion/deprecation)
- **Experiment Tracking**: Transitions log directly into the centralized Git-linked experiment registry database.

---

## 3. Version Pinning & Safety

- **Decoupled Architecture**: Downstream executors (paper/live engines) import only the `PredictionModel` interface, isolating them from raw framework pickling dependencies (XGBoost, LightGBM, CatBoost).
- **Model Version Pinning**: Execution components instantiate model objects using explicitly pinned version strings (UUIDs) fetched from the registry rather than picking the "latest" dynamically.

---

## 4. Rollback & Contingency Plan

### Risk: Model Degradation in Production

If a model's live PnL or predictive ROC-AUC falls below trailing limits:

1. **Trigger Alert**: Dashboard signals threshold warning.
2. **Registry Demote**: Operator executes `ModelRegistry.deprecate(model_version, notes="Live PnL degradation")`. This transition is recorded in the audit trail.
3. **Rollback**: Downstream orchestrators dynamically query the registry for the last validated stable version using `ModelRegistry.find_best(metric_name="f1", task_type="classification", status="SHADOW")` or load a fixed fallback version.
4. **Archival**: Pinned degradation version is moved to `ARCHIVED` state to prevent any future consumption.

---

## 5. Model Validation Readiness (Sprint 2.7.1)

### Readiness for Shadow Mode

With the completion of the `ValidationPipeline` and the `PurgedTimeSeriesSplit` integration, models are formally prepared to transition from `VALIDATED` to `SHADOW` execution with absolute assurance of leakage-free validation:

- **Purged Cross-Validation with Embargo**: Generalizability validation is evaluated on the dataset using a chronological purged cross-validation scheme. It blocks target spillover leakage by purging data points in a window equal to the label horizon before validation ranges, and shields serial-correlation bias with an embargo window after validation ranges.
- **Sandbox Isomorphic Paths**: Validation is executed by monkeypatching the production `MarketConsumer` to utilize a `ValidationPaperExecutor` that inherits from the core `PaperExecutor`. This ensures that backtesting simulations follow the exact path-dependency and execution loops of live trading.
- **Friction Readiness**: Models must beat all 6 baseline benchmarks and exceed 70% retention under stressful commissions and slippage before they can transition out of Candidate status.

### Status of Prior Architectural Risks

1. **Calibration Interfaces**: **Fully Implemented**. Platt Scaling and Isotonic Regression modules are integrated with ECE, Brier, and Log Loss validation.
2. **Active Drift Detection**: **Fully Implemented**. `ActiveDriftObserver` computes PSI and KS distribution deviations against training baselines in real-time.

### Institutional Compliance Level

- **Level**: **Grade A (Leakage-Aware Financial Audit Lineage)**.
- **Lineage Integrity**: The registry fully logs all intermediate candidate metrics, temporal cross-validation parameters (folds count, purge horizon, embargo size, pruned sample counts), transition timelines, and audit comments. All stages are fully reproducible and auditable.

---

## 6. Real-Time Inference Platform Checklist (Sprint 2.8 & 2.8.1)

### Production Gate Review

#### 1. Can runtime inference execute without any trading authority?

**Yes.** The `InferenceEngine` is completely isolated from trading libraries. It possesses no imports or interfaces for order submission, position management, or risk checks. Its sole purpose is generating predictions and probabilities from input features.

#### 2. Can an invalid feature schema reach the model?

**No.** All feature matrices are processed by a dedicated `Schema Validator` which validates:

- Schema dimension match.
- Missing/unexpected features.
- Value representation (rejection of NaNs and $\pm\text{Inf}$).
- Alignment of features to the canonical manifest feature order (eliminating dictionary key-order assumptions).
  On any incompatibility, the system immediately fails closed and raises a structured `SchemaValidationError`.

#### 3. Can a model version silently change?

**No.** Version routing is entirely explicit. Each `InferenceRequest` must pin the model version by its exact registry UUID. The engine will not execute predictions using generic "latest" strings or automatically resolve missing paths.

#### 4. Can inference failure accidentally become a trade signal?

**No.** The engine uses strict distinction between success and failure paths. `predict_one()` returns a payload only upon complete validation and prediction success. Any failure raises an `InferenceError` subtype. The engine never emits default or fallback predictions (such as default BUY or SELL outputs).

#### 5. How are corrupted/unverified artifacts handled?

**Fail-Closed.** Deserialization is wrapped in SHA-256 validation. Any checksum mismatch or missing checksum under STRICT policies raises `ArtifactIntegrityError`.

#### 6. Shadow Mode Gate Checklist

All requirements for live paper trading under Shadow Mode have been successfully finalized:

- **Active Drift Checks**: Completed. Rolling distribution checks are executed via the `ActiveDriftObserver`.
- **Calibrated Scores**: Completed. Calibrators are dynamically loaded and applied to prediction outputs.
- **Artifact Checksums**: Completed. `ArtifactIntegrityVerifier` enforces SHA-256 verification on every weight load.
