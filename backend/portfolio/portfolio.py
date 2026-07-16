from backend.portfolio.position import Position


class Portfolio:

    START_BALANCE = 10000.0

    def __init__(self):

        self.balance = self.START_BALANCE
        self.cash = self.START_BALANCE

        self.positions = []

        self.total_profit = 0.0

    def open_position(self, position: Position):

        self.positions.append(position)

    def close_position(self, position: Position):

        if position not in self.positions:
            return

        position.is_open = False

        self.positions.remove(position)

        self.total_profit += position.pnl

    def get_open_positions(self):

        return [

            p

            for p in self.positions

            if p.is_open

        ]

    def has_open_position(self, symbol):

        return any(

            p.symbol == symbol and p.is_open

            for p in self.positions

        )

    def update_positions(self, candle):

        for position in self.get_open_positions():

            if position.symbol != candle.symbol:
                continue

            if position.side == "BUY":

                position.pnl = (

                    candle.close - position.entry_price

                ) * position.quantity

    def equity(self):

        return self.balance + self.total_profit