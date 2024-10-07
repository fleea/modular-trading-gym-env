# pytest src/observations/test_trend_observation.py

import pytest
import numpy as np
import pandas as pd
from pytest import approx
from unittest.mock import Mock
from src.interfaces.order_interface import OrderAction
from src.observations.trend_observation import TrendObservation, TrendEnvironment


@pytest.fixture
def trend_observation():
    return TrendObservation([1, 5, 9])


@pytest.fixture
def mock_env():
    env = Mock(spec=TrendEnvironment)
    env.current_index = 9  # the 10th step
    env.get_current_price.return_value = 1.35
    env.start_index = 0
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
    np.testing.assert_array_equal(space.low, np.array([0, -np.inf, -np.inf, -np.inf]))
    np.testing.assert_array_equal(space.high, np.array([1, np.inf, np.inf, np.inf]))


def test_get_start_index(trend_observation):
    assert trend_observation.get_start_index() == 9  # max offset


def test_calculate_trend(trend_observation, mock_env):
    trends = trend_observation._calculate_trend(mock_env)

    assert len(trends) == 3
    assert trends[0] == approx(0.05, rel=1e-5)  # 1.35 - 1.3
    assert trends[1] == approx(0.25, rel=1e-5)  # 1.35 - 1.1
    assert trends[2] == approx(0.45, rel=1e-5)  # 1.35 - 0.9


def test_get_observation(trend_observation, mock_env):
    observation = trend_observation.get_observation(mock_env)

    assert len(observation) == 4
    assert observation[0] == 0  # No orders
    assert observation[1] == approx(0.05, rel=1e-5)
    assert observation[2] == approx(0.25, rel=1e-5)
    assert observation[3] == approx(0.45, rel=1e-5)


def test_get_observation_with_orders(trend_observation, mock_env):
    mock_env.orders = [Mock()]  # Non-empty orders list
    observation = trend_observation.get_observation(mock_env)
    assert observation[0] == 1  # Orders exist


def test_calculate_trend_insufficient_data(trend_observation):
    mock_env = Mock(spec=TrendEnvironment)
    mock_env.start_index = 0
    mock_env.current_index = 4  # Not enough data for all offsets
    mock_env.get_current_price.return_value = 1.1
    mock_env.data = pd.DataFrame(
        {
            "bid_price": [0.9, 0.95, 1.0, 1.05, 1.1],
            "ask_price": [0.91, 0.96, 1.01, 1.06, 1.11],
        }
    )

    trends = trend_observation._calculate_trend(mock_env)

    assert len(trends) == 3
    assert trends[0] == approx(0.05, rel=1e-5)  # 1.1 - 1.05
    assert trends[1] == 0.0  # Not enough data for offset 5
    assert trends[2] == 0.0  # Not enough data for offset 9


if __name__ == "__main__":
    pytest.main()
