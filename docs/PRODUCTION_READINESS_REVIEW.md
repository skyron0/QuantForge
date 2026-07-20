# QuantForge Production Readiness Review (Sprint 2.6)

This document evaluates the operational readiness, audit compliance, security safeguards, and contingency strategies of the QuantForge Model Training and Registry Platform.

---

## 1. Lifecycle State Promotion Gates

To ensure only rigorously backtested and validated models are executed in staging and production environment settings, we enforce a strict promotion sequencing policy.

### Implemented State Machine Checklist

Every model starts in the `TRAINING` state and must proceed strictly in order:

$$\text{TRAINING} \longrightarrow \text{CANDIDATE} \longrightarrow \text{VALIDATED} \longrightarrow \text{SHADOW} \longrightarrow \text{PRODUCTION}$$

At any point, a model can be transitioned out of the active loop to `DEPRECATED` or `ARCHIVED`.

### Gate Criteria & Verification Rules

| Transition                | Enforced Validation Gate                                                            | Operational Action                                       |
| ------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **TRAINING → CANDIDATE**  | Validation thresholds met (e.g. F1 $\ge 0.55$) in training pipeline.                | Auto-generated candidate status.                         |
| **CANDIDATE → VALIDATED** | Walk-forward simulation matches training metrics. Human auditor signature required. | Manual check entry logged.                               |
| **VALIDATED → SHADOW**    | Passed isolation testing. Run in sandboxed paper trading environment.               | Ingestion by live market consumers (no order execution). |
| **SHADOW → PRODUCTION**   | Minimal tracker drawdown and stable predictive accuracy over target time window.    | Enabled order execution permission.                      |

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
