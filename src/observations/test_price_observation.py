import pytest
import numpy as np
import pandas as pd
from src.observations.price_observation import PriceObservation, PriceEnvironment


class MockPriceEnvironment:
    def __init__(self, current_price, historical_prices):
        self.current_price = current_price
        self.historical_prices = historical_prices

    def get_current_data(self):
        return pd.Series({"bid_price": self.current_price})

    def __getitem__(self, key):
        if key == "data":
            return pd.DataFrame({"bid_price": self.historical_prices})


@pytest.fixture
def price_observation():
    return PriceObservation(column_name="bid_price")


def test_observation_space(price_observation):
    assert price_observation.observation_space.shape == (1,)
    assert price_observation.observation_space.low[0] == 0
    assert price_observation.observation_space.high[0] == 1


def test_get_price_range(price_observation):
    env = MockPriceEnvironment(100, [50, 75, 100, 125, 150])
    min_price, max_price = price_observation.get_price_range(env)
    assert min_price == 50
    assert max_price == 150


def test_get_observation_normalization(price_observation):
    env = MockPriceEnvironment(100, [50, 75, 100, 125, 150])
    observation = price_observation.get_observation(env)
    expected_normalized_price = (100 - 50) / (150 - 50)
    np.testing.assert_almost_equal(observation, [expected_normalized_price])


def test_get_observation_edge_cases(price_observation):
    # Test minimum price
    env_min = MockPriceEnvironment(50, [50, 75, 100, 125, 150])
    observation_min = price_observation.get_observation(env_min)
    np.testing.assert_almost_equal(observation_min, [0])

    # Test maximum price
    env_max = MockPriceEnvironment(150, [50, 75, 100, 125, 150])
    observation_max = price_observation.get_observation(env_max)
    np.testing.assert_almost_equal(observation_max, [1])


def test_get_observation_caching(price_observation):
    env1 = MockPriceEnvironment(100, [50, 75, 100, 125, 150])
    env2 = MockPriceEnvironment(125, [0, 25, 50, 75, 100])

    # First call should set min_price and max_price
    price_observation.get_observation(env1)
    assert price_observation.min_price == 50
    assert price_observation.max_price == 150

    # Second call should use cached values, even with different historical data
    observation = price_observation.get_observation(env2)
    expected_normalized_price = (125 - 50) / (150 - 50)
    np.testing.assert_almost_equal(observation, [expected_normalized_price])


def test_custom_column_name():
    custom_observation = PriceObservation(column_name="bid_price")
    env = MockPriceEnvironment(100, [50, 75, 100, 125, 150])
    observation = custom_observation.get_observation(env)
    expected_normalized_price = (100 - 50) / (150 - 50)
    np.testing.assert_almost_equal(observation, [expected_normalized_price])
