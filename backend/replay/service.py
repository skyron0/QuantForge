"""
Historical Replay Service orchestrator.
Manages the loop of running simulation steps, advancing ReplayClock, 
ingesting data, and calling coordinator.
"""
import logging
import uuid
import random
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from decimal import Decimal

from backend.replay.exceptions import ReplayValidationError, ReplayInvariantError
from backend.replay.models import (
    ReplayStatus,
    ReplayDatasetMetadata,
    ReplaySessionConfig,
    ReplayProgress,
    ReplaySessionResult,
)
from backend.replay.policy import ReplayPolicy
from backend.replay.clock import ReplayClock
from backend.replay.loader import CSVHistoricalCandleLoader
from backend.replay.scheduler import ReplayScheduler
from backend.replay.telemetry import ReplayTelemetrySink, ConsoleReplayTelemetrySink
from backend.replay.metrics import calculate_replay_metrics
from backend.replay.dataset import generate_dataset_hash

# Container and Coordinator
from backend.application.container import QuantForgeContainer
from backend.application.coordinator import IntegratedPaperTradingCoordinator
from backend.application.models import CycleTriggerType, IntegratedRuntimeStatus
from backend.market_data.adapters.replay import ReplayMarketDataProvider
from backend.market_data.models import MarketDataType, TickerSnapshot, OrderBookSnapshot, Candle, MarketDataSnapshot
from backend.market_data.bridge import MarketDataSnapshotUpdated

# Policy Imports (to establish standard policies if not custom-provided)
from backend.application.policy import IntegratedRuntimePolicy
from backend.market_data.policy import MarketDataPolicy
from backend.feature_runtime.policy import FeatureRuntimePolicy
from backend.feature_runtime.schema import FeatureSchema
from backend.decision.policy import FusionPolicy
from backend.risk.policy import RiskPolicy
from backend.positioning.policy import PositionSizingPolicy
from backend.execution_authorization.policy import ExecutionPolicy
from backend.execution_adapter.policy import PaperExecutionPolicy
from backend.portfolio.policy import PortfolioPolicy
from backend.position_lifecycle.policy import PositionLifecyclePolicy
from backend.orchestration.policy import TradingCyclePolicy
from backend.persistence.policy import PersistencePolicy

class HistoricalReplayService:
    """
    Main orchestrator governing historical simulation execution.
    Handles sequence step iterations, dependency configuration, 
    and validates execution constraints.
    """
    def __init__(
        self,
        policy: Optional[ReplayPolicy] = None,
        telemetry: Optional[ReplayTelemetrySink] = None
    ) -> None:
        self.policy = policy or ReplayPolicy()
        self.telemetry = telemetry or ConsoleReplayTelemetrySink()
        self.logger = logging.getLogger("QuantForge.ReplayService")

    def run_replay_session(
        self,
        config: ReplaySessionConfig,
        predict_fn: Optional[Any] = None,
        custom_policies: Optional[Dict[str, Any]] = None
    ) -> ReplaySessionResult:
        """
        Executes a historical simulation session run.
        """
        # 1. Validate Service Policy
        self.policy.validate()

        # 2. Basic config validation
        if not config.enabled_symbols:
            raise ReplayValidationError("enabled_symbols list cannot be empty in config")

        # 3. Load historical candles dataset
        symbol = config.enabled_symbols[0]
        loader = CSVHistoricalCandleLoader(config.dataset_path, symbol, config.timeframe)
        raw_records = loader.load()

        if not raw_records:
            raise ReplayValidationError(f"Dataset at {config.dataset_path} loaded zero records")

        # 4. Check dataset size against policy limit
        if len(raw_records) > self.policy.max_dataset_rows:
            raise ReplayValidationError(
                f"Dataset rows ({len(raw_records)}) exceeds maximum allowed limit ({self.policy.max_dataset_rows})"
            )

        # 5. Dataset metadata and hash fingerprint
        dataset_hash = generate_dataset_hash(raw_records)
        metadata = ReplayDatasetMetadata(
            dataset_hash=dataset_hash,
            row_count=len(raw_records),
            symbols=[symbol.upper().replace("/", "")],
            start_time=raw_records[0]["timestamp"],
            end_time=raw_records[-1]["timestamp"]
        )

        # 6. Initialize Replay Clock and Seeded UUID Generator
        clock = ReplayClock(initial_time=metadata.start_time)
        
        # Seeded Random Generator for reproducible UUID creation
        rng = random.Random(config.seed)
        def seeded_uuid_generator() -> str:
            val_bytes = bytearray(rng.getrandbits(8) for _ in range(16))
            val_bytes[6] = (val_bytes[6] & 0x0f) | 0x40  # Set version to 4
            val_bytes[8] = (val_bytes[8] & 0x3f) | 0x80  # Set variant to RFC 4122
            return str(uuid.UUID(bytes=bytes(val_bytes)))

        # 7. Normalize raw records for the ReplayMarketDataProvider queue
        # Ensure timeframe, open_time, close_time and keys for normalization are present
        replay_queue = []
        for i, rec in enumerate(raw_records):
            pld = dict(rec)
            # Guarantee close_time is present to prevent KeyError during normalization
            if "close_time" not in pld:
                pld["close_time"] = pld.get("open_time") or pld.get("timestamp")
            
            replay_queue.append({
                "symbol": pld.get("symbol", symbol),
                "data_type": MarketDataType.CANDLE.value,
                "payload": pld
            })

        # 8. Setup Policies
        pols = custom_policies or {}
        # Ensure fallback defaults matching standard configurations if omitted
        if "runtime" not in pols:
            pols["runtime"] = IntegratedRuntimePolicy(
                policy_version="1.0",
                paper_only=True,
                enabled_symbols=config.enabled_symbols,
                enabled_timeframes=[config.timeframe]
            )
        if "mkt" not in pols:
            pols["mkt"] = MarketDataPolicy(
                allowed_symbols=set(config.enabled_symbols)
            )
        if "feat" not in pols:
            pols["feat"] = FeatureRuntimePolicy(
                policy_version="1.0",
                minimum_history=2,
                staleness_limit_seconds=999999999.0,
                default_timeframe=config.timeframe
            )
        else:
            from dataclasses import replace
            pols["feat"] = replace(pols["feat"], staleness_limit_seconds=999999999.0)
        if "schema" not in pols:
            pols["schema"] = FeatureSchema(
                schema_id="replay_schema",
                schema_version="1.0",
                feature_names=["f1", "f2"]
            )
        if "fusion" not in pols:
            pols["fusion"] = FusionPolicy(
                policy_version="1.0",
                ml_weight=1.0,
                intelligence_weight=0.0,
                minimum_ml_confidence=0.0,
                minimum_fusion_confidence=0.0,
                minimum_agreement_score=-1.0,
                allow_ml_only=True,
                reject_on_critical_drift=False,
                reject_on_intelligence_risk_flags=[],
                proposal_ttl_seconds=86400.0
            )
        if "risk" not in pols:
            pols["risk"] = RiskPolicy(
                policy_version="1.0.0",
                minimum_proposal_confidence=0.0,
                maximum_proposal_age_seconds=86400.0,
                maximum_daily_loss_fraction=1.0,
                maximum_drawdown_fraction=1.0,
                maximum_portfolio_exposure_fraction=1.0,
                maximum_symbol_exposure_fraction=1.0,
                maximum_leverage=100.0,
                maximum_open_positions=100,
                maximum_symbol_open_positions=100,
                maximum_consecutive_losses=100,
                reject_on_critical_volatility=False,
                reject_on_critical_liquidity=False,
                reject_on_critical_drift=False,
                base_risk_fraction=1.0,
                maximum_risk_fraction=1.0,
                minimum_risk_fraction=0.0,
                volatility_adjustments={"HIGH": 1.0, "NORMAL": 1.0, "LOW": 1.0}
            )
        if "sizing" not in pols:
            pols["sizing"] = PositionSizingPolicy(
                policy_version="1.0.0",
                minimum_position_notional=1.0,
                maximum_position_notional=config.initial_capital * 10.0,
                minimum_quantity=0.0001,
                maximum_quantity=10000.0,
                maximum_leverage=100.0,
                maximum_margin_fraction=1.0,
                maximum_symbol_exposure_fraction=1.0,
                maximum_portfolio_exposure_fraction=1.0,
                rounding_mode="DOWN",
                reject_if_below_min_quantity=True,
                reject_if_above_max_quantity=True,
                reject_if_stop_distance_invalid=False,
                reject_if_market_data_stale=False,
                market_data_max_age_seconds=86400.0,
                authorization_max_age_seconds=86400.0
            )
        from backend.execution_authorization.models import OrderType
        if "exec" not in pols:
            from backend.execution_authorization.models import ExecutionEnvironment
            pols["exec"] = ExecutionPolicy(
                policy_version="1.0",
                allowed_environments=[ExecutionEnvironment.PAPER],
                maximum_market_data_age_seconds=86400.0,
                order_intent_ttl_seconds=86400.0,
                minimum_quantity=0.0001,
                maximum_quantity=10000.0,
                require_stop_loss=False,
                require_take_profit=False,
                allowed_order_types=[OrderType.MARKET, OrderType.LIMIT],
                allow_live_execution_intents=False,
                require_execution_enabled=True,
                reject_when_kill_switch_active=False,
                require_symbol_enabled=True,
                maximum_clock_skew_seconds=86400.0
            )
        if "paper_exec" not in pols:
            pols["paper_exec"] = PaperExecutionPolicy(
                policy_version="1.0",
                maximum_market_data_age_seconds=86400.0,
                maximum_future_clock_skew_seconds=86400.0,
                fee_rate=0.0,
                slippage_rate=0.0,
                allow_partial_fills=False,
                minimum_fill_quantity=0.0001,
                reject_if_insufficient_liquidity=False,
                intent_max_age_seconds=86400.0,
                execution_result_ttl_seconds=86400.0
            )
        if "portfolio" not in pols:
            pols["portfolio"] = PortfolioPolicy(
                policy_version="1.0",
                supported_instrument_types=["linear_perpetual", "spot"],
                allow_position_reversal=True,
                maximum_open_positions=100,
                maximum_symbol_positions=100,
                maximum_gross_exposure_fraction=Decimal("100.0"),
                maximum_net_exposure_fraction=Decimal("100.0"),
                maximum_leverage=Decimal("100.0"),
                market_price_max_age_seconds=86400.0,
                maximum_future_clock_skew_seconds=86400.0,
                accounting_tolerance=Decimal("0.001")
            )
        if "lifecycle" not in pols:
            pols["lifecycle"] = PositionLifecyclePolicy(
                policy_version="1.0.0",
                allow_stop_loss=True,
                require_stop_loss=False,
                allow_take_profit=True,
                require_take_profit=False,
                allow_trailing_stop=True,
                minimum_stop_distance_fraction=Decimal("0.0001"),
                maximum_stop_distance_fraction=Decimal("1.0"),
                minimum_take_profit_distance_fraction=Decimal("0.0001"),
                trailing_distance_mode="ABSOLUTE",
                minimum_trailing_distance=Decimal("0.0001"),
                maximum_trailing_distance=Decimal("100000.0"),
                allow_breakeven=True,
                breakeven_activation_fraction=Decimal("0.5"),
                breakeven_offset_fraction=Decimal("0.0")
            )
        if "cycle" not in pols:
            pols["cycle"] = TradingCyclePolicy(
                policy_version="1.0"
            )
        if "persist" not in pols:
            pols["persist"] = PersistencePolicy(
                persistence_enabled=False
            )

        # 9. Build and Wire Container and Components
        # Construct provider first as required by container
        provider = ReplayMarketDataProvider()
        
        session_id = seeded_uuid_generator()
        container = QuantForgeContainer(
            runtime_policy=pols["runtime"],
            market_data_policy=pols["mkt"],
            feature_policy=pols["feat"],
            feature_schema=pols["schema"],
            predict_fn=predict_fn or (lambda x: 0.0),
            fusion_policy=pols["fusion"],
            risk_policy=pols["risk"],
            sizing_policy=pols["sizing"],
            execution_policy=pols["exec"],
            paper_exec_policy=pols["paper_exec"],
            portfolio_policy=pols["portfolio"],
            lifecycle_policy=pols["lifecycle"],
            cycle_policy=pols["cycle"],
            persistence_policy=pols["persist"],
            market_data_provider=provider,
            session_id=session_id,
            clock=clock,
            uuid_generator=seeded_uuid_generator
        )

        # Link provider and market service
        provider.service = container.market_service
        
        # Subscribe provider to configured symbols to ensure they are ingested
        for sym in config.enabled_symbols:
            provider.subscribe(sym)

        # Overwrite default portfolio engine initial balance and reset to apply
        container.portfolio_engine.initial_balance = Decimal(str(config.initial_capital))
        container.portfolio_engine.clear()

        # Load normalized candle payloads to provider
        provider.load_data(replay_queue)

        # Initialize coordinator
        coordinator = IntegratedPaperTradingCoordinator(container)
        coordinator.start_session()

        # 10. Core Simulation Loop Event Stepper
        portfolio_snapshots = []
        cycle_count = 0
        error_msg = None
        status = ReplayStatus.RUNNING

        def on_scheduler_step(record: Dict[str, Any]) -> None:
            nonlocal cycle_count
            now_iso = clock.now().isoformat().replace("+00:00", "Z")
            
            # Step the ReplayMarketDataProvider to ingest the data
            norm_symbol = symbol.upper().replace("/", "")
            
            # Step provider to normalize, validate, sequencer, and cache
            provider.step()

            # Construct snapshots (Ticker and Book) for marking and exit validation
            price_val = Decimal(str(record["close"]))
            ticker_snap = TickerSnapshot(
                symbol=norm_symbol,
                bid=price_val - Decimal("0.01"),
                ask=price_val + Decimal("0.01"),
                last=price_val,
                bid_quantity=Decimal("100"),
                ask_quantity=Decimal("100"),
                volume_24h=Decimal(str(record.get("volume", 0))),
                timestamp=now_iso,
                source="replay",
                received_at=now_iso
            )
            book_snap = OrderBookSnapshot(
                symbol=norm_symbol,
                bids=[],
                asks=[],
                sequence=cycle_count + 1,
                timestamp=now_iso,
                source="replay",
                received_at=now_iso
            )
            
            # Update store to make build_snapshot run smoothly
            container.market_store.update_ticker(ticker_snap)
            container.market_store.update_order_book(book_snap)

            # Broadcast MarketDataSnapshotUpdated event so downstream feature buffers receive it
            snapshot_from_store = container.market_snapshot_builder.build_snapshot(norm_symbol, now_iso)
            
            # Set the exact loaded candle in the snapshot list to circumvent timestamp limits
            # Check container.market_service cache to retrieve the Normalized Candle envelope
            candles_list = container.market_store.get_candles(norm_symbol, config.timeframe)
            latest_candle = candles_list[-1] if candles_list else None
            if latest_candle:
                snapshot_from_store = MarketDataSnapshot(
                    symbol=norm_symbol,
                    timestamp=now_iso,
                    ticker=ticker_snap,
                    latest_trade=None,
                    candles={config.timeframe: [latest_candle]},
                    order_book=book_snap,
                    source_health="CONNECTED",
                    data_age=0.0,
                    sequence_state={},
                    metadata={}
                )

            event = MarketDataSnapshotUpdated(
                event_id=seeded_uuid_generator(),
                event_type="MarketDataSnapshotUpdated",
                timestamp=now_iso,
                runtime_id="replay-runtime",
                session_id=session_id,
                cycle_id=None,
                snapshot=snapshot_from_store
            )
            container.event_bus.publish(event)

            # Trigger continuous trading loop cycle
            cycle_results = coordinator.step(
                trigger_type=CycleTriggerType.MARKET_UPDATE,
                manual_timestamp=now_iso
            )
            
            # Keep history of snapshots for metrics calculations
            portfolio_state = container.portfolio_engine.get_state()
            portfolio_snapshots.append(container.portfolio_engine.create_snapshot())

            cycle_count += 1
            self.telemetry.record_step(
                step_index=cycle_count,
                total_steps=len(raw_records),
                details={"symbol": symbol, "close": float(record["close"])}
            )

        scheduler = ReplayScheduler(
            clock=clock,
            dataset=raw_records,
            on_step=on_scheduler_step
        )

        try:
            # Execute simulation up to max cycles config or data exhaust
            limit = min(config.max_cycles, scheduler.total_steps)
            for _ in range(limit):
                if not scheduler.step():
                    break
            
            status = ReplayStatus.COMPLETED
        except Exception as e:
            self.logger.exception("Simulation execution failed due to an uncaught exception")
            error_msg = str(e)
            status = ReplayStatus.FAILED

        # 11. Finalize and report results
        coord_summary = coordinator.stop_session("historical replay complete")
        
        final_portfolio_state = container.portfolio_engine.get_state()
        metrics = calculate_replay_metrics(
            initial_equity=config.initial_capital,
            final_state=final_portfolio_state,
            history=portfolio_snapshots
        )

        # Build repeatable determinism seed hash
        run_determinism_hash = dataset_hash  # baseline seed

        progress = ReplayProgress(
            total_steps=len(raw_records),
            processed_steps=cycle_count,
            current_time=clock.now().isoformat().replace("+00:00", "Z"),
            percent_complete=float(cycle_count) / len(raw_records) * 100.0 if raw_records else 0.0
        )

        result = ReplaySessionResult(
            session_id=session_id,
            status=status,
            config=config,
            dataset_metadata=metadata,
            progress=progress,
            initial_equity=config.initial_capital,
            final_equity=metrics["final_equity"],
            realized_pnl=metrics["realized_pnl"],
            unrealized_pnl=metrics["unrealized_pnl"],
            fees=metrics["total_fees"],
            determinism_hash=run_determinism_hash,
            error_message=error_msg,
            portfolio_history=portfolio_snapshots,
            metadata={
                "min_equity": metrics["min_equity"],
                "max_equity": metrics["max_equity"],
                "gross_return": metrics["gross_return"],
                "max_drawdown": metrics["max_drawdown"]
            }
        )

        return result
