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
        # LOGGING
        # print(f"Calculating trend. Current index: {env.current_index}")
        # print(f"Tick data shape: {env.tick_data.shape}")
        # print(f"Tick data index: {env.tick_data.index}")
        try:
            current_tick = env.tick_data.loc[env.current_index]
            current_price = current_tick.bid_price
            trends = np.array([
                (current_price - env.tick_data.loc[env.current_index - offset].bid_price)
                if env.current_index - offset >= env.start_index else 0.0
                for offset in self.trend_offsets
            ], dtype=np.float32)

            self.trend_history.append(trends)

            multiplier = get_multiplier(self.trend_history) * self.rms_multiplier
            return trends * multiplier
        except Exception as e:
            print(f"Error in _calculate_trend: {type(e).__name__}: {e}")
            print(f"current_index: {env.current_index}")
            print(f"tick data: {env.tick_data}")
            print(f"trend_offsets: {self.trend_offsets}")
            print(f"trend_history: {list(self.trend_history)}")
            print(f"rms_multiplier: {self.rms_multiplier}")
            raise

def get_multiplier(row: list):
    row = list(chain.from_iterable(row))
    rms_from_change = np.sqrt(np.mean(np.square(row)))
    return 1/rms_from_change if rms_from_change != 0 else 1


if __name__ == "__main__":
    d = deque([[-0.1, 0.2, 0.3]])
    print(get_multiplier(d))