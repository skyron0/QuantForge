from backend.decision.models import Decision


class SignalValidator:

    def __init__(self):

        self.last_action = None

    def validate(
        self,
        decision: Decision,
    ):

        if decision is None:
            return None

        if decision.action == "HOLD":
            return None

        if decision.action == self.last_action:
            return None

        self.last_action = decision.action

        return decision