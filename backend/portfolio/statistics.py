class Statistics:

    @staticmethod
    def win_rate(history):

        if history.total_trades() == 0:
            return 0

        return (

            history.wins()

            / history.total_trades()

        ) * 100

    @staticmethod
    def average_profit(history):

        if history.total_trades() == 0:
            return 0

        return (

            history.total_profit()

            / history.total_trades()

        )