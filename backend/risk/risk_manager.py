class RiskManager:

    def __init__(
        self,
        balance: float = 10000.0,
        risk_percent: float = 1.0,
        max_open_positions: int = 3,
        max_position_size: float = 5.0,
    ):
        self.balance = balance
        self.risk_percent = risk_percent
        self.max_open_positions = max_open_positions
        self.max_position_size = max_position_size

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
    ) -> float:

        stop_distance = abs(entry_price - stop_loss)

        if stop_distance <= 0:
            return 0.0

        risk_amount = self.balance * (self.risk_percent / 100)

        quantity = risk_amount / stop_distance

        return round(min(quantity, self.max_position_size), 6)

    def can_open_position(
        self,
        open_positions: int,
    ) -> bool:

        return open_positions < self.max_open_positions