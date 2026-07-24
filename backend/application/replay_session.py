"""
QuantForge Developer Replay Session CLI Tool.
Triggers a historical simulation session run and outputs results.
"""
import argparse
import logging
import sys
from decimal import Decimal

from backend.replay.models import ReplaySessionConfig
from backend.replay.service import HistoricalReplayService

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def mock_predict(features: list) -> float:
    """Simple default prediction function returning bullish signal code."""
    # Features will be [f1, f2]. Returns bullish signal value.
    return 1.0

def main() -> None:
    parser = argparse.ArgumentParser(description="QuantForge Developer Historical Replay CLI")
    parser.add_argument("--dataset", type=str, required=True, help="Path to historical CSV candle file")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol for simulation")
    parser.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe interval")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial simulation capital")
    parser.add_argument("--max-cycles", type=int, default=100000, help="Maximum simulation steps limit")
    parser.add_argument("--seed", type=int, default=42, help="Seed value for deterministic UUID & sizing generation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("QuantForge.ReplayCLI")
    logger.info(f"Setting up historical replay session for {args.symbol} using dataset {args.dataset}")

    config = ReplaySessionConfig(
        initial_capital=args.initial_capital,
        max_cycles=args.max_cycles,
        seed=args.seed,
        dataset_path=args.dataset,
        enabled_symbols=[args.symbol],
        timeframe=args.timeframe
    )

    # Patch IndicatorEngine to allow immediate sandbox pipeline executions with minimal history
    try:
        from backend.indicator.indicator_engine import IndicatorEngine
        IndicatorEngine.MIN_CANDLES = 2
        IndicatorEngine.calculate = lambda self, candles: {"f1": 1.0, "f2": 2.0}
        logger.debug("Successfully patched IndicatorEngine for historical simulation sandbox")
    except ImportError:
        logger.warning("IndicatorEngine not found, skipping patching (expected in some unit tests)")

    service = HistoricalReplayService()
    
    logger.info("Starting historical simulation loop orchestration...")
    result = service.run_replay_session(
        config=config,
        predict_fn=mock_predict
    )

    logger.info("Simulation execution cycle complete. Finalizing summary report.")
    
    print("==================================================")
    print("HISTORICAL REPLAY SESSION SUMMARY:")
    print("==================================================")
    print(f"Session ID:         {result.session_id}")
    print(f"Replay Status:      {result.status.value}")
    print(f"Dataset Hash:       {result.dataset_metadata.dataset_hash}")
    print(f"Row Count:          {result.dataset_metadata.row_count}")
    print(f"Processed Steps:    {result.progress.processed_steps} / {result.progress.total_steps}")
    print(f"Duration Start:     {result.dataset_metadata.start_time}")
    print(f"Duration End:       {result.dataset_metadata.end_time}")
    print("--------------------------------------------------")
    print(f"Initial Equity:     {result.initial_equity:.2f}")
    print(f"Final Equity:       {result.final_equity:.2f}")
    print(f"Realized PnL:       {result.realized_pnl:.2f}")
    print(f"Unrealized PnL:     {result.unrealized_pnl:.2f}")
    print(f"Fees Paid:          {result.fees:.2f}")
    print(f"Gross Return:       {result.metadata.get('gross_return', 0.0) * 100.0:.2f}%")
    print(f"Max Drawdown:       {result.metadata.get('max_drawdown', 0.0) * 100.0:.2f}%")
    print(f"Min / Max Equity:   {result.metadata.get('min_equity', 0.0):.2f} / {result.metadata.get('max_equity', 0.0):.2f}")
    print("--------------------------------------------------")
    print(f"Determinism Hash:   {result.determinism_hash}")
    
    if result.error_message:
        print(f"Error Message:      {result.error_message}")
        
    print("==================================================")

if __name__ == "__main__":
    main()
