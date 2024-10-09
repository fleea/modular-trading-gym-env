from src.observations.base_observation import BaseObservation
from typing import List, Protocol, Any
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.preprocessing.hlc import (
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


class HLCEnvironment(Protocol):
    current_index: int
    data: pd.DataFrame
    orders: List[Any]

    def get_current_data(self) -> pd.Series:
        pass


class HLCObservation(BaseObservation[HLCEnvironment]):
    def __init__(self):
        self.column_names = [
            d_high,
            d_low,
            d_close,
            w_high,
            w_low,
            w_close,
            m_high,
            m_low,
            m_close,
        ]

    def get_space(self) -> spaces.Space:
        low = np.array([-np.inf] * len(self.column_names), dtype=np.float32)
        high = np.array([np.inf] * len(self.column_names), dtype=np.float32)
        shape = (len(self.column_names),)
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def get_observation(self, env: HLCEnvironment) -> np.ndarray:
        current_data = env.get_current_data()
        close = current_data["close"]
        observation = [close - current_data[col] for col in self.column_names]
        return np.array(observation, dtype=np.float32)

    def get_start_index(self, data: pd.DataFrame) -> int:
        if len(data) == 0:
            return 0
        # Find the first index where m_high is not NaN
        first_valid_index = data[m_high].first_valid_index()
        if first_valid_index is None:
            return 0
        return data.index.get_loc(first_valid_index)
