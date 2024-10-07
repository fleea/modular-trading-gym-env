from src.observations.base_observation import BaseObservation
from typing import Protocol
from gymnasium import spaces
import numpy as np
import pandas as pd


class PriceEnvironment(Protocol):
    current_index: int
    
    def get_current_data(self) -> pd.Series:
        pass
    
    def get_data(self) -> pd.DataFrame:
        pass


class PriceObservation(BaseObservation[PriceEnvironment]):
    def __init__(self, column_name: str = "close"):
        # Define the observation space for a single normalized price
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.min_price = None
        self.max_price = None
        self.column_name = column_name

    def get_space(self) -> spaces.Space:
        low = np.array([0], dtype=np.float32)
        high = np.array([1], dtype=np.float32)
        shape = (1,)
        # Boolean for order, then the trend offset calculation
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def get_price_range(self, env: PriceEnvironment) -> tuple[float, float]:
        if self.min_price is None or self.max_price is None:
            historical_data = env.get_data()
            self.min_price = historical_data[self.column_name].min()
            self.max_price = historical_data[self.column_name].max()
        return self.min_price, self.max_price

    def get_observation(self, env: PriceEnvironment) -> np.ndarray:
        try:
            data = env.get_current_data()
            min_price, max_price = self.get_price_range(env)

            # Normalize the price
            normalized_price = (data[self.column_name] - min_price) / (
                max_price - min_price
            )

            return np.array([normalized_price], dtype=np.float32)
        except:
            print(f"Current index: {env.current_index}")
            print(f"Column name: {self.column_name}")
            print(f"Min max price: {self.get_price_range(env)}")
            data = env.get_data()
            print(f"Data: {data}")
            print(f"Current data: {env.get_current_data()}")
            print(f"Data shape: {data.shape}")
            print(f"Data head:\n{data.head()}")
            print(f"Data tail:\n{data.tail()}")
