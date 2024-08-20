from src.observations.base_observation import BaseObservation
from src.interfaces.order_interface import OrderAction
from typing import List, Protocol, Any
from gymnasium import spaces
import numpy as np
import pandas as pd


# Define a protocol for environments that can use TrendObservation
class TrendEnvironment(Protocol):
    current_step: int
    tick_data: pd.DataFrame
    orders: List[Any]

    def get_current_price(self, order_action: Any) -> float:
        pass


class TrendObservation(BaseObservation[TrendEnvironment]):
    def __init__(self, trend_offsets: List[int]):
        self.trend_offsets = trend_offsets

    def get_space(self) -> spaces.Space:
        low = np.array([0] + ([-np.inf] * len(self.trend_offsets)), dtype=np.float32)
        high = np.array([1] + [np.inf] * len(self.trend_offsets), dtype=np.float32)
        shape = (1 + len(self.trend_offsets),)
        # Boolean for order, then the trend offset calculation
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def get_min_periods(self) -> int:
        return (
            max(self.trend_offsets) + 1
        )  # +1 to ensure we have data for the current step

    def get_observation(self, env: TrendEnvironment) -> np.ndarray:
        trends = self._calculate_trend(env)
        return np.array([bool(env.orders)] + trends, dtype=np.float32)

    def _calculate_trend(self, env: TrendEnvironment) -> List[float]:
        current_price = env.get_current_price(OrderAction.CLOSE)
        trends = [
            current_price - ((env.tick_data.loc[env.current_step - offset].bid_price + env.tick_data.loc[
                env.current_step - offset].ask_price) / 2)
            if env.current_step - offset >= 0 else 0.0
            for offset in self.trend_offsets
        ]
        return trends
