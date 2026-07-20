import sys
from sqlalchemy.orm import Session
from backend.database.session import engine
from backend.clock.clock import Clock
from backend.backtest.engine import BacktestEngine
from backend.database.base import Base
from backend.models.candle import Candle
from backend.database.models.trade import Trade
from backend.database.models.feature_snapshot import FeatureSnapshot
from configs.logging import app_logger


import argparse
from datetime import datetime, timezone
from backend.repositories.candle_repository import CandleRepository


def main():
    parser = argparse.ArgumentParser(description="QuantForge Backtest Runner")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to backtest (default: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--start", type=str, default="2026-07-01", help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end", type=str, default=None, help="End date (default: today)")

    args = parser.parse_args()

    # Parse dates
    try:
        start_dt = datetime.fromisoformat(args.start)
    except ValueError:
        print(f"Error: Invalid start date format '{args.start}'. Expected ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).", file=sys.stderr)
        sys.exit(1)

    if args.end:
        try:
            end_dt = datetime.fromisoformat(args.end)
        except ValueError:
            print(f"Error: Invalid end date format '{args.end}'. Expected ISO format.", file=sys.stderr)
            sys.exit(1)
    else:
        end_dt = datetime.now()

    # Ensure all tables are created in the target database
    Base.metadata.create_all(bind=engine)

    connection = engine.connect()
    # Start the external connection transaction
    transaction = connection.begin()
    db = Session(bind=connection)

    try:
        # Load candles from DB using CandleRepository
        candle_repo = CandleRepository(db)
        candles = candle_repo.get_between(
            symbol=args.symbol,
            start=start_dt,
            end=end_dt,
            timeframe=args.timeframe,
        )

        if not candles:
            print(
                f"Error: No historical candles found for symbol '{args.symbol}', "
                f"timeframe '{args.timeframe}' between {args.start} and {args.end or 'now'}. "
                "Please verify that data has been ingested.",
                file=sys.stderr,
            )
            sys.exit(1)

        if len(candles) < 200:
            print(
                f"Error: Insufficient historical candles for symbol '{args.symbol}', "
                f"timeframe '{args.timeframe}' between {args.start} and {args.end or 'now'}. "
                f"Found {len(candles)} candles, but at least 200 are required for simulation.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Loaded {len(candles)} candles for simulation.")

        clock = Clock()
        backtest_engine = BacktestEngine(db_session=db, clock=clock)

        print("Starting backtest execution...")
        backtest_engine.run(candles)

        # Pull and print results
        result = backtest_engine.get_metrics()
        print("\n=== BACKTEST PERFORMANCE REPORT ===")
        print(f"Total Trades:           {result.total_trades}")
        print(f"Winning Trades:         {result.winning_trades}")
        print(f"Losing Trades:          {result.losing_trades}")
        print(f"Win Rate:               {result.win_rate:.2f}%")
        print(f"Net Profit:             {result.net_profit:.2f}")
        print(f"Gross Profit:           {result.gross_profit:.2f}")
        print(f"Gross Loss:             {result.gross_loss:.2f}")
        print(f"Average Profit:         {result.average_profit:.2f}")
        print(f"Average Win:            {result.average_win:.2f}")
        print(f"Average Loss:           {result.average_loss:.2f}")
        print(f"Profit Factor:          {result.profit_factor:.2f}")
        print(f"Largest Win:            {result.largest_win:.2f}")
        print(f"Largest Loss:           {result.largest_loss:.2f}")
        print(f"Avg Trade Duration:     {result.average_trade_duration / 60.0:.2f} mins")
        print(f"Max Drawdown Pct:       {result.maximum_drawdown_pct:.2f}%")
        print(f"Max Drawdown Abs:       {result.maximum_drawdown_abs:.2f}")
        print(f"Equity Curve Points:    {len(result.equity_curve)}")
        print("===================================\n")

        backtest_engine.close()

    except SystemExit:
        # Exit propagates up cleanly
        raise
    except Exception as e:
        app_logger.exception("Backtest run failed")
        print(f"Backtest error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        print(
            "Rolling back transaction. No persistent data has been written to the database."
        )
        transaction.rollback()
        db.close()
        connection.close()


if __name__ == "__main__":
    main()
