# Extending trend_observation
from .trend_observation import TrendObservation, TrendEnvironment
from src.interfaces.order_interface import OrderAction
from typing import List
from gymnasium import spaces
import numpy as np


class TrendObservationPercentageArrayStock(TrendObservation):
    def get_space(self) -> spaces.Space:
        # 1st element is the boolean of order
        # subsequent elements are the percentage changes of close to close, close to high, close to low of the current data compared to the data - n
        low = np.array([0] + [0] * (len(self.trend_offsets) * 3), dtype=np.float32)
        high = np.array([1] + [1] * (len(self.trend_offsets) * 3), dtype=np.float32)
        shape = (1 + len(self.trend_offsets) * 3,)
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def _calculate_trend(self, env: TrendEnvironment) -> List[float]:
        current_data = env.get_current_data()
        trends = []
        for offset in self.trend_offsets:
            if env.current_index - offset >= env.start_index:
                previous_time_index = env.current_index - offset
                previous_data = env.data.loc[previous_time_index]
                close_to_close = (
                    current_data.close - previous_data.close
                ) / previous_data.close
                close_to_high = (
                    current_data.close - previous_data.high
                ) / previous_data.high
                close_to_low = (
                    current_data.close - previous_data.low
                ) / previous_data.low
                trends.extend([close_to_close, close_to_high, close_to_low])
            else:
                trends.extend([0, 0, 0])
        return trends
