from typing import Optional
from backend.training.validation import ValidationResult


def generate_validation_report(
    model_version: str,
    dataset_version: str,
    experiment_id: str,
    result: ValidationResult,
    reviewer_notes: Optional[str] = None,
) -> str:
    """
    Generates a structured markdown report presenting model performance metrics,
    benchmark comparison tables, stability indexes, and promotion recommendations.
    """
    # Build benchmark table
    bench_rows = []
    for bench_name, metrics in result.benchmark_comparison.items():
        beaten_str = "**PASS (YES)**" if metrics["beaten"] else "FAIL (NO)"
        bench_rows.append(
            f"| {bench_name} | {metrics['model_profit']:.2f} | {metrics['benchmark_profit']:.2f} | {metrics['difference']:.2f} | {beaten_str} |"
        )
    bench_table = "\n".join(bench_rows)

    report = f"""# QuantForge Institutional Model Validation Report

## 1. Metadata Details
- **Model Version**: `{model_version}`
- **Dataset Version**: `{dataset_version}`
- **Experiment ID**: `{experiment_id}`
- **Validation Decsion**: **{result.validation_decision}**
- **Promotion Recommendation**: **{result.promotion_recommendation.value}**
- **Calibration Status**: `{result.calibration_status}`

---

## 2. Walk Forward Validation Summary
- **Average Return (Net Profit)**: {result.walk_forward_metrics['average_return']:.2f}
- **Average Drawdown**: {result.walk_forward_metrics['average_drawdown'] * 100.0:.2f}%
- **Average F1 Score**: {result.walk_forward_metrics['average_f1']:.4f}
- **Sharpe Ratio Proxy**: {result.walk_forward_metrics['average_sharpe']:.4f}

---

## 3. Benchmark Strategies Comparison
| Strategy | Model Net Profit | Benchmark Net Profit | Profit Difference | Outperformed |
|---|---|---|---|---|
{bench_table}

---

## 4. Stability Stress Analysis
- **Stability Score**: {result.stability_score * 100.0:.2f}%
- **Performance Details**:
  - *Baseline Scenario*: PnL = {result.stability_results['baseline']['net_profit']:.2f} (trades: {result.stability_results['baseline']['total_trades']})
  - *Medium Stress Scenario (0.1% fee/slip)*: PnL = {result.stability_results['medium_stress']['net_profit']:.2f}
  - *High Stress Scenario (0.3% fee/slip)*: PnL = {result.stability_results['high_stress']['net_profit']:.2f}

---

## 5. Purged Time-Series Cross Validation Summary
- **Splitter Type**: `{result.splitter_info.get('splitter_type', 'N/A')}`
- **Number of Folds**: {result.splitter_info.get('n_splits', 0)}
- **Purge Size (Horizon)**: {result.splitter_info.get('purge_size', 0)}
- **Embargo Size**: {result.splitter_info.get('embargo_size', 0)}
- **Total Samples Purged**: {result.splitter_info.get('total_samples_purged', 0)}
- **Total Samples Embargoed**: {result.splitter_info.get('total_samples_embargoed', 0)}
- **Average Out-Of-Sample CV Score**: {result.cv_metrics.get('average_cv_score', 0.0):.4f}

---

## 6. Reviewer Comments & Sign-off
- **Notes**: {reviewer_notes or "None provided."}
- **Validation Timestamp**: {datetime.datetime.now(datetime.timezone.utc).isoformat()}
"""
    return report


import datetime  # Import here for timezone use above
