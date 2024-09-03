# Extending trend_observation
from src.observations.trend_observation import TrendObservation, TrendEnvironment
from typing import List
from gymnasium import spaces
import numpy as np
from collections import deque
from itertools import chain


class TrendObservationRMS(TrendObservation):
    def __init__(self, trend_offsets: List[int], history_size: int = 30, rms_multiplier: float = 1.0):
        super().__init__(trend_offsets)
        self.history_size = history_size
        self.trend_history = deque(maxlen=history_size)
        self.rms_multiplier = rms_multiplier

    def get_space(self) -> spaces.Space:
        # RMS should be between 1 and -1
        # Add 10 to the high and low values to accommodate for the change
        low = np.array([0] + ([-10.0] * len(self.trend_offsets)), dtype=np.float32)
        high = np.array([1] + [10.0] * len(self.trend_offsets), dtype=np.float32)
        shape = (1 + len(self.trend_offsets),)
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def _calculate_trend(self, env: TrendEnvironment) -> List[float]:
        current_price = env.tick_data.loc[env.current_step].bid_price
        trends = np.array([
            (current_price - env.tick_data.loc[env.current_step - offset].bid_price)
            if env.current_step - offset >= 0 else 0.0
            for offset in self.trend_offsets
        ])
        self.trend_history.append(trends)

        multiplier = get_multiplier(self.trend_history) * self.rms_multiplier
        return trends * multiplier


def get_multiplier(row: list):
    row = list(chain.from_iterable(row))
    rms_from_change = np.sqrt(np.mean(np.square(row)))
    return 1/rms_from_change if rms_from_change != 0 else 1


if __name__ == "__main__":
    d = deque([[-0.1, 0.2, 0.3]])
    print(get_multiplier(d))