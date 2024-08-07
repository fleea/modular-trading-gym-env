from src.environments.base_environment.base_environment import BaseEnvironment


def NonZeroBuyReward(self: BaseEnvironment):
    if len(self.equity) < 2:
        return 0

    equity_change = self.equity[-1] - self.equity[-2]

    if equity_change != 0:
        return equity_change

    current_step_data = self.get_step_data(0)
    previous_step_data = self.get_step_data(-1)

    delta_bid_price = 1000 * (
        current_step_data["bid_price"] - previous_step_data["bid_price"]
    )

    return -delta_bid_price


__all__ = ["NonZeroBuyReward"]
