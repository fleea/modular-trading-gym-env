import pytest
import numpy as np
import pandas as pd
from pytest import approx
from unittest.mock import Mock
from src.observations.trend_observation import TrendEnvironment
from src.observations.trend_observation_std import TrendObservationStd


@pytest.fixture
def trend_observation_std():
    return TrendObservationStd([1, 3, 5], history_size=5)


@pytest.fixture
def mock_env():
    env = Mock(spec=TrendEnvironment)
    env.current_step = 5  # the 10th step
    env.get_current_price.return_value = 0.6
    env.tick_data = pd.DataFrame({
        'bid_price': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'ask_price': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    })
    env.orders = []
    return env


def test_get_space(trend_observation_std):
    space = trend_observation_std.get_space()
    assert space.shape == (4,)  # 1 for order + 3 trend offsets
    np.testing.assert_array_equal(space.low, np.array([0, -10.0, -10.0, -10.0]))
    np.testing.assert_array_equal(space.high, np.array([1, 10.0, 10.0, 10.0]))


def test_get_min_periods(trend_observation_std):
    assert trend_observation_std.get_min_periods() == 5  # max offset


def test_calculate_trend(trend_observation_std, mock_env):
    for _ in range(3):
        trends = trend_observation_std._calculate_trend(mock_env)
        mock_env.current_step+=1
        print(trends)
        print("=====")


    assert len(trends) == 3
    # Note: The exact values might differ due to normalization
    assert all(isinstance(trend, float) for trend in trends)
    assert all(-10.0 <= trend <= 10.0 for trend in trends)


def test_get_observation(trend_observation_std, mock_env):
    # Call _calculate_trend multiple times to fill the trend_history
    for _ in range(5):
        trend_observation_std._calculate_trend(mock_env)

    observation = trend_observation_std.get_observation(mock_env)

    assert len(observation) == 4
    assert observation[0] == 0  # No orders
    assert all(-10.0 <= obs <= 10.0 for obs in observation[1:])


def test_get_observation_with_orders(trend_observation_std, mock_env):
    mock_env.orders = [Mock()]  # Non-empty orders list
    observation = trend_observation_std.get_observation(mock_env)
    assert observation[0] == 1  # Orders exist


def test_calculate_trend_insufficient_data(trend_observation_std):
    mock_env = Mock(spec=TrendEnvironment)
    mock_env.current_step = 4  # Not enough data for all offsets
    mock_env.get_current_price.return_value = 1.1
    mock_env.tick_data = pd.DataFrame({
        'bid_price': [0.9, 0.95, 1.0, 1.05, 1.1],
        'ask_price': [0.91, 0.96, 1.01, 1.06, 1.11]
    })

    trends = trend_observation_std._calculate_trend(mock_env)

    assert len(trends) == 3
    assert trends[1] == 0.0  # Not enough data for offset 5
    assert trends[2] == 0.0  # Not enough data for offset 9


def test_trend_history(trend_observation_std, mock_env):
    for _ in range(10):
        trend_observation_std._calculate_trend(mock_env)

    assert len(trend_observation_std.trend_history) == 5  # history_size is 5


def test_normalization(trend_observation_std, mock_env):
    # Call _calculate_trend multiple times with different prices
    prices = [1.0, 1.1, 1.2, 1.3, 1.4]
    for price in prices:
        mock_env.get_current_price.return_value = price
        trends = trend_observation_std._calculate_trend(mock_env)
        print(trends)
        assert all(-10.0 <= trend <= 10.0 for trend in trends)


if __name__ == "__main__":
    pytest.main()