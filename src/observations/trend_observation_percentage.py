# Extending trend_observation
from .trend_observation import TrendObservation, TrendEnvironment
from src.interfaces.order_interface import OrderAction
from typing import List
from gymnasium import spaces
import numpy as np


class TrendObservationPercentage(TrendObservation):
    def get_space(self) -> spaces.Space:
        # Adjust the space to accommodate percentage changes
        # Typical percentage changes in forex might be within ±10%
        low = np.array([0] + ([-10.0] * len(self.trend_offsets)), dtype=np.float32)
        high = np.array([1] + [10.0] * len(self.trend_offsets), dtype=np.float32)
        shape = (1 + len(self.trend_offsets),)
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def _calculate_trend(self, env: TrendEnvironment) -> List[float]:
        current_price = env.get_current_price(OrderAction.CLOSE)
        trends = []
        for offset in self.trend_offsets:
            if env.current_index - offset >= env.start_index:
                previous_time_index = env.current_index - offset
                previous_data = env.data.loc[previous_time_index]
                past_price = (
                    previous_data.bid_price
                )  # If OrderAction.CLOSE right now, what is the price we get?
                percentage_change = ((current_price - past_price) / past_price) * 100
                trends.append(percentage_change)
            else:
                trends.append(0.0)
        return trends
