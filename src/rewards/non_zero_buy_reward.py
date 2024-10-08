from src.environments.base_environment.base_environment import BaseEnvironment


def NonZeroBuyReward(self: BaseEnvironment):
    if len(self.equity) < 2:
        return 0

    equity_change = self.equity[-1] - self.equity[-2]

    if equity_change != 0:
        return equity_change

    current_step_data = self.get_step_data(0)
    previous_step_data = self.get_step_data(-1)

    if current_step_data is None or previous_step_data is None:
        return 0

    # 1 for stock
    # 1000 for forex
    delta_bid_price = 1 * (
        current_step_data["bid_price"] - previous_step_data["bid_price"]
    )

    return -delta_bid_price


__all__ = ["NonZeroBuyReward"]
