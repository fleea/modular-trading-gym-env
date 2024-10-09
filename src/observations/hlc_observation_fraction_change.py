from src.observations.base_observation import BaseObservation
from gymnasium import spaces
import numpy as np
from src.observations.hlc_observation import HLCEnvironment

from src.preprocessing.hlc import (
    change,
    d_high,
    d_low,
    d_close,
    w_high,
    w_low,
    w_close,
    m_high,
    m_low,
    m_close,
)


class HLCObservationFractionChange(BaseObservation[HLCEnvironment]):
    def __init__(self):
        self.column_names = [
            change + d_high,
            change + d_low,
            change + d_close,
            change + w_high,
            change + w_low,
            change + w_close,
            change + m_high,
            change + m_low,
            change + m_close,
        ]

    def get_space(self) -> spaces.Space:
        low = np.array([-np.inf] * len(self.column_names), dtype=np.float32)
        high = np.array([np.inf] * len(self.column_names), dtype=np.float32)
        shape = (len(self.column_names),)
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def get_observation(self, env: HLCEnvironment) -> np.ndarray:
        current_data = env.get_current_data()
        observation = current_data[self.column_names]
        return np.array(observation, dtype=np.float32)

    def get_start_index(self) -> int:
        return 31
        # Find the first index where m_high is not NaN
        # first_valid_index = self.data[m_high].first_valid_index()
        # if first_valid_index is None:
        #     return 0
        # return self.data.index.get_loc(first_valid_index)
