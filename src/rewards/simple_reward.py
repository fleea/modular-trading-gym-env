from src.environments.base_environment.base_environment import BaseEnvironment


def SimpleReward(self: BaseEnvironment):
    return self.equity[-1] - self.equity[-2] if len(self.equity) >= 2 else 0


__all__ = ["SimpleReward"]
