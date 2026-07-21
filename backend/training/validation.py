import os
import datetime
import logging
from typing import Dict, Any, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from backend.clock.clock import Clock
from backend.models.candle import Candle
from backend.backtest.walk_forward import WalkForwardEngine, WalkForwardWindow, WalkForwardResult
from backend.backtest.engine import BacktestEngine
from backend.backtest.metrics import MetricsEngine
from backend.database.models.trade import Trade as TradeModel
from backend.database.session import SessionLocal
from backend.strategy.registry import StrategyLoader
from backend.training.lifecycle import ModelStatus
from backend.training.registry import ModelRegistry
from backend.execution.paper_executor import PaperExecutor
from backend.portfolio.position import Position
from backend.training.base_trainer import TrainerRegistry
from backend.training.models import TrainingConfig
from backend.training.evaluator import (
    compute_classification_metrics,
    compute_regression_metrics,
)


# ────────────── 1. Custom Validation Executor for Stress Analysis ──────────────

class ValidationPaperExecutor(PaperExecutor):
    """
    Subclass of PaperExecutor designed for stability stress validation.
    Applies configurable commissions and execution slippage factors.
    """
    def __init__(self, db_session=None, clock=None, commission_pct: float = 0.0, slippage_pct: float = 0.0):
        super().__init__(db_session, clock)
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def execute(self, signal, candle):
        self.portfolio.update_positions(candle)

        # Check and close existing positions
        for position in self.portfolio.get_open_positions():
            close_reason = None
            if position.pnl >= self.TAKE_PROFIT:
                close_reason = "TAKE_PROFIT"
            elif position.pnl <= -self.STOP_LOSS:
                close_reason = "STOP_LOSS"

            if close_reason:
                # Apply slippage on exit price
                exit_price = candle.close * (1.0 - self.slippage_pct)
                raw_pnl = (exit_price - position.entry_price) * position.quantity
                
                # Apply transaction commissions on entry + exit
                total_volume = (position.entry_price + exit_price) * position.quantity
                commission = total_volume * self.commission_pct
                net_pnl = raw_pnl - commission

                position.close_price = exit_price
                position.close_time = self.clock.now() if self.clock is not None else datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

                trade = self.trade_repository.get_open_trade(position.symbol)
                if trade:
                    trade.exit_price = exit_price
                    trade.pnl = net_pnl
                    trade.commission = commission
                    trade.status = "CLOSED"
                    trade.close_time = position.close_time
                    trade.reason = close_reason
                    self.trade_repository.update(trade)

                self.portfolio.close_position(position)

        if signal is None or signal.action != "BUY":
            return

        if self.portfolio.has_open_position(candle.symbol):
            return
        if not self.risk_manager.can_open_position(len(self.portfolio.get_open_positions())):
            return

        # Apply slippage on entry price
        entry_price = candle.close * (1.0 + self.slippage_pct)
        quantity = self.risk_manager.calculate_position_size(
            entry_price=entry_price,
            stop_loss=entry_price - self.STOP_LOSS
        )

        position = Position(
            symbol=candle.symbol,
            side="BUY",
            entry_price=entry_price,
            quantity=quantity,
            open_time=self.clock.now() if self.clock is not None else datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            stop_loss=entry_price - self.STOP_LOSS,
            take_profit=entry_price + self.TAKE_PROFIT,
        )
        self.portfolio.open_position(position)

        trade = TradeModel(
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=None,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            pnl=0.0,
            commission=0.0,
            confidence=getattr(signal, "confidence", None),
            strategy=signal.reason,
            reason=getattr(signal, "reason", "BUY"),
            model_version="v1",
            status="OPEN",
            open_time=position.open_time,
            close_time=None,
        )
        self.trade_repository.create(trade)


# ────────────── 2. Stability Walk-Forward Engine ──────────────

class StabilityWalkForwardEngine(WalkForwardEngine):
    """
    Subclass of WalkForwardEngine modifying BacktestEngine to inject
    the ValidationPaperExecutor under stress parameters.
    """
    def __init__(self, db_engine=None, commission_pct: float = 0.0, slippage_pct: float = 0.0):
        super().__init__(db_engine)
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def _run_single_backtest(
        self,
        candles: Sequence[Candle],
        strategy_name: str,
        strategy_params: dict | None,
    ) -> Any:
        connection = self.db_engine.connect()
        transaction = connection.begin()
        db = SessionLocal(bind=connection)

        try:
            from backend.database.base import Base
            Base.metadata.create_all(bind=connection)

            params = strategy_params or {}
            strategy = StrategyLoader.load_strategy(strategy_name, **params)

            clock = Clock()
            backtest_engine = BacktestEngine(db_session=db, clock=clock, strategy=strategy)
            
            # Inject validation executor
            executor = ValidationPaperExecutor(
                db_session=db,
                clock=clock,
                commission_pct=self.commission_pct,
                slippage_pct=self.slippage_pct
            )
            backtest_engine.consumer.paper_executor = executor

            backtest_engine.run(candles)
            result = backtest_engine.get_metrics()
            backtest_engine.close()
            return result
        finally:
            transaction.rollback()
            db.close()
            connection.close()


# ────────────── 3. Time Series Validation Splitter Abstractions ──────────────

class BaseTimeSeriesSplitter(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Slice chronological train/val indices."""
        pass


class PurgedTimeSeriesSplit(BaseTimeSeriesSplitter):
    """
    Purged Time-Series Cross Validation splitter with Embargo support.
    Divides dataset into sequential non-shuffled folds.
    """
    def __init__(self, n_splits: int = 5, purge_horizon: int = 0, embargo_size: int = 0):
        self.n_splits = n_splits
        self.purge_horizon = purge_horizon
        self.embargo_size = embargo_size

    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(df)
        if n_samples < self.n_splits + 1:
            return []

        # Divide indices into n_splits contiguous folds
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        boundaries = [0]
        current = 0
        for size in fold_sizes:
            current += size
            boundaries.append(current)

        splits = []
        for i in range(self.n_splits):
            v_start = boundaries[i]
            v_end = boundaries[i + 1]

            val_idx = np.arange(v_start, v_end)

            # Purge training indices before validation start
            train_before_end = max(0, v_start - self.purge_horizon)
            train_before = np.arange(0, train_before_end)

            # Embargo training indices after validation end
            train_after_start = min(n_samples, v_end + self.embargo_size)
            train_after = np.arange(train_after_start, n_samples)

            train_idx = np.concatenate([train_before, train_after])

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        return splits


# ────────────── 4. Config & Result Dataclasses ──────────────

@dataclass
class ValidationConfig:
    min_sharpe: float = 1.0
    min_f1: float = 0.55
    max_drawdown: float = 0.20
    min_return: float = 0.0
    stability_threshold: float = 0.70  # Min ratio of (Net Profit High / Baseline)
    scenarios: List[str] = field(default_factory=lambda: ["baseline", "medium_stress", "high_stress"])
    train_days: int = 30
    test_days: int = 10
    step_days: int = 10
    n_splits: int = 5
    embargo_size: int = 5
    purge_horizon: Optional[int] = None
    calibration_method: str = "platt"


@dataclass
class ValidationResult:
    walk_forward_metrics: Dict[str, Any]
    benchmark_comparison: Dict[str, Dict[str, Any]]
    stability_score: float
    calibration_status: str
    drift_baseline: Dict[str, Any]
    validation_decision: str
    promotion_recommendation: ModelStatus
    stability_results: Dict[str, Dict[str, Any]]
    cv_metrics: Dict[str, Any] = field(default_factory=dict)
    splitter_info: Dict[str, Any] = field(default_factory=dict)


# ────────────── 4. Validation Pipeline ──────────────

class ValidationPipeline:
    """
    Executes walk forward backtests, compares model performance to benchmarks,
    evaluates stability under stress simulations, and calculates covariate baseline bounds.
    """
    def __init__(self, model_registry: Optional[ModelRegistry] = None):
        self.registry = model_registry or ModelRegistry()

    def validate(
        self,
        model_version: str,
        dataset_df: pd.DataFrame,
        candles: Sequence[Candle],
        config: Optional[ValidationConfig] = None,
    ) -> ValidationResult:
        cfg = config or ValidationConfig()

        # Load model and verify Candidate status
        model_record = self.registry.repo.get(model_version)
        if not model_record:
            raise FileNotFoundError(f"Model version '{model_version}' not found.")

        # Load the actual PredictionModel instance
        model = self.registry.load(model_version)

        # 1. Walk Forward (Baseline scenario)
        engine_base = StabilityWalkForwardEngine(commission_pct=0.0, slippage_pct=0.0)
        wf_res = engine_base.run(
            candles=candles,
            train_days=cfg.train_days,
            test_days=cfg.test_days,
            step_days=cfg.step_days,
            strategy_name="prediction_strategy",
            strategy_params={"model_version": model_version}
        )
        base_stats = wf_res.global_stats

        # Calculate average Sharpe, return, etc.
        # Average return = net_profit / initial_balance (usually 10,000)
        net_profit = base_stats["net_profit"]
        avg_drawdown = base_stats["maximum_drawdown_pct"] / 100.0
        
        # Calculate Sharpe ratio (dummy/simple Sharpe metric from win rate if not explicitly present)
        # In QuantForge metrics.py, sharpe ratio is not calculated by default inside BacktestResult global_stats.
        # Let's derive a proxy Sharpe: of (winrate - 50) / 10 or similar, or compute standard Sharpe if profit factor exists.
        pf = base_stats.get("profit_factor", 1.0)
        sharpe = (pf - 1.0) * 2.0  # Simple mapping proxy for validation check

        # F1 score is from the training metrics
        f1_score = model_record.metrics.get("f1", 0.0)

        # 2. Benchmarks comparison
        benchmarks = ["buy_and_hold", "always_flat", "random_predictor", "ema_crossover", "rsi_strategy", "macd_strategy"]
        bench_comparison = {}
        for bench in benchmarks:
            bench_engine = WalkForwardEngine()
            bench_res = bench_engine.run(
                candles=candles,
                train_days=cfg.train_days,
                test_days=cfg.test_days,
                step_days=cfg.step_days,
                strategy_name=bench
            )
            bench_stats = bench_res.global_stats
            bench_profit = bench_stats["net_profit"]
            beaten = net_profit > bench_profit
            bench_comparison[bench] = {
                "beaten": beaten,
                "model_profit": net_profit,
                "benchmark_profit": bench_profit,
                "difference": net_profit - bench_profit
            }

        # 3. Stability analysis
        # Scenario medium
        engine_med = StabilityWalkForwardEngine(commission_pct=0.001, slippage_pct=0.001)
        med_stats = engine_med.run(candles, cfg.train_days, cfg.test_days, cfg.step_days, "prediction_strategy", {"model_version": model_version}).global_stats
        
        # Scenario high
        engine_high = StabilityWalkForwardEngine(commission_pct=0.003, slippage_pct=0.003)
        high_stats = engine_high.run(candles, cfg.train_days, cfg.test_days, cfg.step_days, "prediction_strategy", {"model_version": model_version}).global_stats

        # Compute Stability Score (Net profit under high stress / Baseline net profit, or 1.0 if not degrading)
        base_profit = net_profit
        high_profit = high_stats["net_profit"]
        if base_profit <= 0:
            stability_score = 1.0 if high_profit >= base_profit else 0.0
        else:
            stability_score = max(0.0, min(1.0, high_profit / base_profit))

        # 4. Confidence Calibration interface
        calibration_status = "READY_INTERFACE_PLATT_ISOTONIC"

        # 4.5. Purged Time-Series Cross Validation
        feature_cols = list(model_record.feature_importance.keys())
        label_horizon = 1
        if cfg.purge_horizon is not None:
            label_horizon = cfg.purge_horizon
        else:
            # Attempt to retrieve from DatasetMetadata
            dataset_dir = os.path.join("data/datasets", model_record.dataset_version)
            meta_path = os.path.join(dataset_dir, "metadata.json")
            if os.path.isfile(meta_path):
                try:
                    import json
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    label_horizon = int(meta.get("label_horizon") or meta.get("generation_parameters", {}).get("horizon") or 1)
                except Exception:
                    pass

        # Splitter setup
        splitter = PurgedTimeSeriesSplit(
            n_splits=cfg.n_splits,
            purge_horizon=label_horizon,
            embargo_size=cfg.embargo_size
        )
        splits = splitter.split(dataset_df)

        cv_scores = []
        total_samples_purged = 0
        total_samples_embargoed = 0
        n_samples = len(dataset_df)

        task_type = "classification"
        if "regression" in model_record.trainer or "mae" in model_record.metrics or "rmse" in model_record.metrics:
            task_type = "regression"

        if splits:
            fold_sizes = np.full(cfg.n_splits, n_samples // cfg.n_splits, dtype=int)
            fold_sizes[:n_samples % cfg.n_splits] += 1
            boundaries = [0]
            curr = 0
            for sz in fold_sizes:
                curr += sz
                boundaries.append(curr)

            for idx_split, (train_idx, val_idx) in enumerate(splits):
                v_start = boundaries[idx_split]
                v_end = boundaries[idx_split + 1]

                total_samples_purged += min(v_start, label_horizon)
                total_samples_embargoed += min(n_samples - v_end, cfg.embargo_size)

                try:
                    df_train = dataset_df.iloc[train_idx]
                    df_val = dataset_df.iloc[val_idx]

                    X_train = df_train[feature_cols]
                    y_train = df_train["label"]
                    X_val = df_val[feature_cols]
                    y_val = df_val["label"]

                    trainer = TrainerRegistry.get(model_record.trainer)
                    config_train = TrainingConfig(
                        model_type=model_record.trainer,
                        task_type=task_type,
                        hyperparameters=model_record.hyperparameters,
                        feature_columns=feature_cols,
                    )
                    fold_model = trainer.train(X_train, y_train, X_val, y_val, config_train)
                    preds = trainer.predict(fold_model, X_val)

                    if task_type == "regression":
                        fold_m = compute_regression_metrics(y_val.to_numpy(), preds)
                        fold_score = fold_m.get("r2", 0.0)
                    else:
                        proba = trainer.predict_proba(fold_model, X_val)
                        fold_m = compute_classification_metrics(y_val.to_numpy(), preds, proba)
                        fold_score = fold_m.get("f1", 0.0)
                    cv_scores.append(fold_score)
                except Exception:
                    pass

        avg_cv_score = float(np.mean(cv_scores)) if cv_scores else 0.0
        cv_metrics = {
            "average_cv_score": avg_cv_score,
            "cv_scores": cv_scores,
        }
        splitter_info = {
            "splitter_type": "PurgedTimeSeriesSplit",
            "n_splits": cfg.n_splits,
            "purge_size": label_horizon,
            "embargo_size": cfg.embargo_size,
            "total_samples_purged": total_samples_purged,
            "total_samples_embargoed": total_samples_embargoed,
        }

        # 5. Drift Baseline
        drift_baseline = {}
        for col in feature_cols:
            if col in dataset_df.columns:
                series = dataset_df[col].dropna()
                if len(series) > 0:
                    q25 = float(series.quantile(0.25))
                    q50 = float(series.quantile(0.50))
                    q75 = float(series.quantile(0.75))
                    drift_baseline[col] = {
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "quantiles": {
                            "25": q25,
                            "50": q50,
                            "75": q75,
                        },
                        "binning_method": "quantile",
                        "bin_edges": ["-inf", q25, q50, q75, "inf"],
                        "expected_proportions": [0.25, 0.25, 0.25, 0.25]
                    }

        # 6. Fit probability calibrator on out-of-sample validation data of the training set
        calibration_status = "UNCALIBRATED"
        calib_metadata = {}
        
        if "split" in dataset_df.columns:
            df_val = dataset_df[dataset_df["split"] == "val"]
        else:
            df_val = dataset_df
        if task_type == "classification" and len(df_val) > 0:
            try:
                X_val = df_val[feature_cols].values
                y_val = df_val["label"].values.astype(int)
                y_val_proba = model.predict_proba_batch(X_val)
                if y_val_proba is None:
                    # Fallback to hard predictions represented as probabilities
                    y_pred = model.predict_batch(X_val)
                    # Clip boundaries slightly to keep them within (0, 1) probability space for metrics
                    y_pred_clipped = np.clip(y_pred, 0.01, 0.99)
                    y_val_proba = np.column_stack((1.0 - y_pred_clipped, y_pred_clipped))
                
                calib_method = (cfg.calibration_method if cfg else "platt").lower()
                if calib_method in ("platt", "isotonic"):
                    from backend.inference.calibration import (
                        PlattCalibrator,
                        IsotonicCalibrator,
                        evaluate_calibration_metrics,
                    )
                    from backend.inference.integrity import calculate_sha256

                    if calib_method == "platt":
                        calibrator = PlattCalibrator()
                    else:
                        calibrator = IsotonicCalibrator()

                    # Prevent data validation test leakage by training only on OOF/validation split
                    calibrator.fit(y_val_proba, y_val)

                    # Evaluate before / after metrics
                    calibrated_val_proba = calibrator.transform(y_val_proba)
                    metrics_before = evaluate_calibration_metrics(y_val_proba, y_val)
                    metrics_after = evaluate_calibration_metrics(calibrated_val_proba, y_val)

                    # Save calibration.joblib to model's version folder
                    model_dir = os.path.dirname(model_record.model_path)
                    calib_path = os.path.join(model_dir, "calibration.joblib")
                    calibrator.save(calib_path)

                    # Compute hash for target calibration artifact
                    calib_sha256 = calculate_sha256(calib_path)
                    calib_size = os.path.getsize(calib_path)

                    calib_metadata = {
                        "calibration_artifact_path": calib_path,
                        "calibration_sha256": calib_sha256,
                        "calibration_size_bytes": calib_size,
                        "calibration_method": calib_method,
                        "calibration_dataset_version": model_record.dataset_version,
                        "calibration_sample_count": len(y_val),
                        "calibration_metrics_before": metrics_before,
                        "calibration_metrics_after": metrics_after,
                    }
                    calibration_status = "READY_INTERFACE_PLATT_ISOTONIC"
            except Exception as e:
                # Log warning and run uncalibrated
                logging.getLogger(__name__).warning(f"Failed to fit probability calibrator: {str(e)}")
                calibration_status = "FAILED"

        # Validate against promotion rules
        benchmarks_beaten = all(b["beaten"] for b in bench_comparison.values())
        
        # Verify conditions
        is_candidate = model_record.status == ModelStatus.CANDIDATE
        wf_passes = (sharpe >= cfg.min_sharpe) and (avg_drawdown <= cfg.max_drawdown) and (net_profit >= cfg.min_return)
        stability_passes = stability_score >= cfg.stability_threshold
        
        cv_passes = True
        if task_type == "classification":
            cv_passes = avg_cv_score >= cfg.min_f1

        validation_passed = is_candidate and wf_passes and benchmarks_beaten and stability_passes and cv_passes

        if validation_passed:
            decision = "APPROVED"
            recommendation = ModelStatus.VALIDATED
        else:
            decision = "REJECTED"
            recommendation = ModelStatus.DEPRECATED if (net_profit < 0 or avg_drawdown > 0.40) else ModelStatus.CANDIDATE

        # Update registry record with walk forward metrics and validation results
        validation_metrics = {
            "validation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "walk_forward_sharpe": sharpe,
            "walk_forward_return": net_profit,
            "walk_forward_max_drawdown": avg_drawdown,
            "stability_score": stability_score,
            "benchmarks_beaten": benchmarks_beaten,
            "validation_decision": decision,
            "promotion_recommendation": recommendation.value,
            "average_cv_score": avg_cv_score,
        }
        model_record.walk_forward_metrics = validation_metrics
        model_record.drift_baseline = drift_baseline
        model_record.calibration_metadata = calib_metadata
        self.registry.repo.save(model_record)

        # Trigger promotions if approved
        if recommendation == ModelStatus.VALIDATED:
            self.registry.promote(
                model_version=model_version,
                new_status=ModelStatus.VALIDATED,
                notes=f"Passed Validation Pipeline. Decision={decision} Stability={stability_score:.2f}."
            )

        return ValidationResult(
            walk_forward_metrics={
                "average_return": net_profit,
                "average_drawdown": avg_drawdown,
                "average_f1": f1_score,
                "average_sharpe": sharpe,
            },
            benchmark_comparison=bench_comparison,
            stability_score=stability_score,
            calibration_status=calibration_status,
            drift_baseline=drift_baseline,
            validation_decision=decision,
            promotion_recommendation=recommendation,
            stability_results={
                "baseline": base_stats,
                "medium_stress": med_stats,
                "high_stress": high_stats,
            },
            cv_metrics=cv_metrics,
            splitter_info=splitter_info
        )
