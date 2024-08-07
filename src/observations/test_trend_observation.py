import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
from gymnasium import spaces
from src.interfaces.order_interface import OrderAction
from src.observations.trend_observation import TrendObservation, TrendEnvironment


class TestTrendObservation(unittest.TestCase):

    def setUp(self):
        self.trend_offsets = [1, 5, 10]
        self.observation = TrendObservation(self.trend_offsets)

        # Create a mock environment
        self.mock_env = Mock(spec=TrendEnvironment)
        self.mock_env.current_step = 20
        self.mock_env.orders = []

        # Create sample tick data
        self.mock_env.tick_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
                "bid_price": np.random.uniform(90, 110, 100),
                "ask_price": np.random.uniform(90, 110, 100),
            }
        )

        self.mock_env.get_current_price.return_value = 100.0

    def test_init(self):
        self.assertEqual(self.observation.trend_offsets, self.trend_offsets)

    def test_get_space(self):
        space = self.observation.get_space()
        self.assertIsInstance(space, spaces.Box)
        self.assertEqual(space.shape, (1 + len(self.trend_offsets),))
        np.testing.assert_array_equal(
            space.low, np.array([0] + [-np.inf] * len(self.trend_offsets))
        )
        np.testing.assert_array_equal(
            space.high, np.array([1] + [np.inf] * len(self.trend_offsets))
        )

    def test_get_min_periods(self):
        self.assertEqual(
            self.observation.get_min_periods(), max(self.trend_offsets) + 1
        )

    def test_get_observation_no_orders(self):
        self.mock_env.orders = []
        obs = self.observation.get_observation(self.mock_env)
        self.assertEqual(obs[0], 0)  # No orders
        self.assertEqual(len(obs), 1 + len(self.trend_offsets))

    def test_get_observation_with_orders(self):
        self.mock_env.orders = [Mock()]  # Add a mock order
        obs = self.observation.get_observation(self.mock_env)
        self.assertEqual(obs[0], 1)  # Has orders
        self.assertEqual(len(obs), 1 + len(self.trend_offsets))

    def test_calculate_trend(self):
        trends = self.observation._calculate_trend(self.mock_env)
        self.assertEqual(len(trends), len(self.trend_offsets))
        for trend in trends:
            self.assertIsInstance(trend, float)

    def test_calculate_trend_insufficient_history(self):
        self.mock_env.current_step = 5  # Set current step to a low value
        trends = self.observation._calculate_trend(self.mock_env)
        self.assertEqual(len(trends), len(self.trend_offsets))
        self.assertEqual(
            trends[-1], 0.0
        )  # The last trend should be 0.0 due to insufficient history

    def test_observation_values(self):
        # Set specific values in the mock environment
        self.mock_env.current_step = 20
        self.mock_env.get_current_price.return_value = 105.0
        self.mock_env.tick_data.loc[19, "bid_price"] = 100.0
        self.mock_env.tick_data.loc[19, "ask_price"] = 101.0
        self.mock_env.tick_data.loc[15, "bid_price"] = 95.0
        self.mock_env.tick_data.loc[15, "ask_price"] = 96.0
        self.mock_env.tick_data.loc[10, "bid_price"] = 90.0
        self.mock_env.tick_data.loc[10, "ask_price"] = 91.0

        obs = self.observation.get_observation(self.mock_env)

        # Check if the trends are calculated correctly
        self.assertAlmostEqual(obs[1], 105.0 - 100.5, places=6)  # Current - 1 step ago
        self.assertAlmostEqual(obs[2], 105.0 - 95.5, places=6)  # Current - 5 steps ago
        self.assertAlmostEqual(obs[3], 105.0 - 90.5, places=6)  # Current - 10 steps ago


if __name__ == "__main__":
    unittest.main()
