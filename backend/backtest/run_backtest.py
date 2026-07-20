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


def main():
    # Ensure all tables are created in the target database
    Base.metadata.create_all(bind=engine)

    connection = engine.connect()
    # Start the external connection transaction
    transaction = connection.begin()
    db = Session(bind=connection)

    try:
        # Load candles from DB
        candles = (
            db.query(Candle)
            .filter(Candle.symbol == "BTCUSDT")
            .order_by(Candle.open_time.asc())
            .all()
        )

        if len(candles) < 200:
            print(
                "Database has insufficient candles for 'BTCUSDT'. "
                "Generating 300 synthetic candles inside the rollbackable transaction..."
            )
            from datetime import datetime, timedelta, timezone
            import random
            import math

            start_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=15)
            price = 45000.0

            for i in range(300):
                # Generate a cyclic price path to trigger both buy and sell decisions
                price = (
                    45000.0
                    + 5000.0 * math.sin(i / 15.0)
                    + random.uniform(-100, 100)
                )
                c = Candle(
                    symbol="BTCUSDT",
                    timeframe="1h",
                    open=price - random.uniform(5, 50),
                    high=price + random.uniform(20, 100),
                    low=price - random.uniform(20, 100),
                    close=price,
                    volume=random.uniform(5, 25),
                    open_time=start_time + timedelta(hours=i),
                )
                db.add(c)
            # Flush changes to the connection.
            db.commit()

            candles = (
                db.query(Candle)
                .filter(Candle.symbol == "BTCUSDT")
                .order_by(Candle.open_time.asc())
                .all()
            )

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

    except Exception as e:
        app_logger.exception("Backtest run failed")
        print(f"Backtest error: {e}", file=sys.stderr)
    finally:
        print(
            "Rolling back transaction. No persistent data has been written to the database."
        )
        transaction.rollback()
        db.close()
        connection.close()


if __name__ == "__main__":
    main()
