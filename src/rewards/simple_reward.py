from src.environments.base_environment.base_environment import BaseEnvironment


def SimpleReward(self: BaseEnvironment):
    equity_change = self.equity[-1] - self.equity[-2] if len(self.equity) >= 2 else 0
    # step_reward = 0.001 if equity_change > 0 else -0.001 if equity_change < 0 else 0
    # position_closed_reward = equity_change if not self.orders and self.closed_orders else 0
    return equity_change


__all__ = ["SimpleReward"]
