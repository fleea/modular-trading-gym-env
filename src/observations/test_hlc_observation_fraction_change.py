# pytest -v src/observations/test_hlc_observation_fraction_change.py

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from src.observations.hlc_observation import HLCEnvironment
from src.observations.hlc_observation_fraction_change import HLCObservationFractionChange


@pytest.fixture
def hlc_observation_fraction_change():
    return HLCObservationFractionChange()

@pytest.fixture
def mock_env():
    env = Mock(spec=HLCEnvironment)
    df = pd.read_csv("src/observations/test_hlc_observation_fraction_change.csv")
    df.set_index("time", inplace=True)
    env.start_index = 0
    env.current_index = 5  # the 10th step
    env.get_current_price.return_value = 0.6
    env.data = df
    env.orders = []
    return env


def test_get_space(hlc_observation_fraction_change):
    space = hlc_observation_fraction_change.get_space()
    assert space.shape == (9,)  # 3 * 3 day, week, month trends
    np.testing.assert_array_equal(space.low, np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]))
    np.testing.assert_array_equal(space.high, np.array([ np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))


def test_get_start_index(hlc_observation_fraction_change):
    assert hlc_observation_fraction_change.get_start_index() == 4  # max offset


# def test_calculate_trend(hlc_observation_fraction_change, mock_env):
#     for _ in range(3):
#         trends = hlc_observation_fraction_change._calculate_trend(mock_env)
#         mock_env.current_index += 1

#     print(f"trends: {trends}")
#     assert len(trends) == 3
#     trends_list = trends.tolist()
#     assert all(isinstance(trend, float) for trend in trends_list)
#     assert all(-10.0 <= trend <= 10.0 for trend in trends_list)


# def test_get_observation(hlc_observation_fraction_change, mock_env):
#     # Call _calculate_trend multiple times to fill the trend_history
#     for _ in range(5):
#         hlc_observation_fraction_change._calculate_trend(mock_env)

#     observation = hlc_observation_fraction_change.get_observation(mock_env)

#     assert len(observation) == 4
#     assert observation[0] == 0  # No orders
#     assert all(-10.0 <= obs <= 10.0 for obs in observation[1:])


# def test_get_observation_with_orders(hlc_observation_fraction_change, mock_env):
#     mock_env.orders = [Mock()]  # Non-empty orders list
#     observation = hlc_observation_fraction_change.get_observation(mock_env)
#     assert observation[0] == 1  # Orders exist


# def test_calculate_trend_insufficient_data(hlc_observation_fraction_change):
#     mock_env = Mock(spec=TrendEnvironment)
#     mock_env.current_index = 4  # Not enough data for all offsets
#     mock_env.start_index = 0
#     mock_env.get_current_price.return_value = 1.1
#     mock_env.data = pd.DataFrame(
#         {
#             "bid_price": [0.9, 0.95, 1.0, 1.05, 1.1],
#             "ask_price": [0.91, 0.96, 1.01, 1.06, 1.11],
#         }
#     )

#     trends = hlc_observation_fraction_change._calculate_trend(mock_env)

#     assert len(trends) == 3
#     assert trends[2] == 0.0  # Not enough data for offset 5


# def test_trend_history(hlc_observation_fraction_change, mock_env):
#     for _ in range(10):
#         hlc_observation_fraction_change._calculate_trend(mock_env)

#     assert len(hlc_observation_fraction_change.trend_history) == 5  # history_size is 5


# def test_normalization(hlc_observation_fraction_change, mock_env):
#     # Call _calculate_trend multiple times with different prices
#     prices = [1.0, 1.1, 1.2, 1.3, 1.4]
#     for price in prices:
#         mock_env.get_current_price.return_value = price
#         trends = hlc_observation_fraction_change._calculate_trend(mock_env)
#         assert all(-10.0 <= trend <= 10.0 for trend in trends)


# def test_rms_calculation(hlc_observation_fraction_change, mock_env):
#     from collections import deque

#     # Set up a known sequence of trends
#     hlc_observation_fraction_change.trend_history = deque(
#         [
#             np.array([1.0, 2.0, 3.0]),
#             np.array([2.0, 3.0, 4.0]),
#             np.array([3.0, 4.0, 5.0]),
#         ]
#     )

#     # Test the full calculation
#     mock_env.current_index = 5
#     mock_env.data.loc[5].bid_price = 1.0
#     mock_env.data.loc[4].bid_price = 0.9
#     mock_env.data.loc[2].bid_price = 0.8
#     mock_env.data.loc[0].bid_price = 0.7

#     trends = hlc_observation_fraction_change._calculate_trend(mock_env)

#     # Calculate raw trends
#     current_price = mock_env.data.loc[5].bid_price
#     raw_trends = np.array(
#         [
#             current_price - mock_env.data.loc[4].bid_price,
#             current_price - mock_env.data.loc[2].bid_price,
#             current_price - mock_env.data.loc[0].bid_price,
#         ]
#     )

#     # Apply RMS normalization
#     actual_multiplier = get_multiplier(hlc_observation_fraction_change.trend_history)
#     expected_trends = (
#         raw_trends * actual_multiplier * hlc_observation_fraction_change.rms_multiplier
#     )
#     # print(f"trends: {trends}, expected_trends: {expected_trends}")

#     np.testing.assert_array_almost_equal(trends, expected_trends)


if __name__ == "__main__":
    pytest.main()
