import pytest
import numpy as np
import pandas as pd
from pytest import approx
from unittest.mock import Mock
from .trend_observation_percentage import (
    TrendObservationPercentage,
)


@pytest.fixture
def trend_observation():
    return TrendObservationPercentage([1, 5, 9])


@pytest.fixture
def mock_env():
    env = Mock()
    env.current_index = 9  # the 10th step
    env.get_current_price.return_value = 1.35
    env.data = pd.DataFrame(
        {
            "bid_price": [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35],
            "ask_price": [0.91, 0.96, 1.01, 1.06, 1.11, 1.16, 1.21, 1.26, 1.31, 1.36],
        }
    )
    env.orders = []
    return env


def test_get_space(trend_observation):
    space = trend_observation.get_space()
    assert space.shape == (4,)  # 1 for order + 3 trend offsets
    np.testing.assert_array_equal(space.low, np.array([0, -10, -10, -10]))
    np.testing.assert_array_equal(space.high, np.array([1, 10, 10, 10]))


def test_get_start_index(trend_observation):
    assert trend_observation.get_start_index() == 9


def test_calculate_trend(trend_observation, mock_env):
    trends = trend_observation._calculate_trend(mock_env)
    print(trends, trend_observation.trend_offsets)

    assert len(trends) == 3
    assert trends[0] == approx(3.8461538461538494, rel=1e-5)
    assert trends[1] == approx(
        22.727272727272727, rel=1e-5
    )  # ((1.35 - 1.1) / 1.1) * 100
    assert trends[2] == approx(50, rel=1e-5)  # ((1.35 - 0.9) / 0.9) * 100


def test_get_observation(trend_observation, mock_env):
    observation = trend_observation.get_observation(mock_env)

    assert len(observation) == 4
    assert observation[0] == 0  # No orders
    assert observation[1] == approx(3.8461538461538494, rel=1e-5)
    assert observation[2] == approx(22.727272727272727, rel=1e-5)
    assert observation[3] == approx(50, rel=1e-5)


def test_get_observation_with_orders(trend_observation, mock_env):
    mock_env.orders = [Mock()]  # Non-empty orders list
    observation = trend_observation.get_observation(mock_env)
    assert observation[0] == 1  # Orders exist


def test_calculate_trend_insufficient_data(trend_observation):
    mock_env = Mock()
    mock_env.current_index = 4  # Not enough data for all offsets
    mock_env.start_index = 0
    mock_env.get_current_price.return_value = 1.1
    mock_env.data = pd.DataFrame(
        {
            "bid_price": [0.9, 0.95, 1.0, 1.05, 1.1],
            "ask_price": [0.91, 0.96, 1.01, 1.06, 1.11],
        }
    )

    trends = trend_observation._calculate_trend(mock_env)

    assert len(trends) == 3
    assert trends[0] == approx(4.761904761904765, rel=1e-5)
    assert trends[1] == 0.0  # Not enough data for offset 5
    assert trends[2] == 0.0  # Not enough data for offset 10


if __name__ == "__main__":
    pytest()
