import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from gymnasium import spaces
from src.interfaces.order_interface import OrderType, OrderAction
from src.observations.base_observation import BaseObservation
from src.environments.single_buy_discrete.single_buy_discrete_env import (
    SingleBuyDiscreteEnv,
)


class TestSingleBuyDiscreteEnv(unittest.TestCase):
    def setUp(self):
        # Create a mock observation
        self.mock_observation = Mock(spec=BaseObservation)
        self.mock_observation.get_min_periods.return_value = 0
        self.mock_observation.get_space.return_value = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,)
        )
        self.mock_observation.get_observation.return_value = np.array([0.0])

        # Create sample tick data
        self.tick_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
                "bid_price": np.random.uniform(90, 110, 100),
                "ask_price": np.random.uniform(90, 110, 100),
            }
        )

        # Mock reward function
        self.mock_reward_func = Mock(return_value=0.0)

        # Create the environment
        self.env = SingleBuyDiscreteEnv(
            initial_balance=10000,
            tick_data=self.tick_data,
            observation=self.mock_observation,
            reward_func=self.mock_reward_func,
            lot=0.01,
        )

    def test_initialization(self):
        self.assertEqual(self.env.initial_balance, 10000)
        self.assertEqual(len(self.env.tick_data), 100)
        self.assertEqual(self.env.lot, 0.01)
        self.assertEqual(self.env.action_space.n, 2)
        self.assertIsInstance(self.env.observation_space, spaces.Box)

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(len(self.env.orders), 0)
        self.assertEqual(len(self.env.closed_orders), 0)
        np.testing.assert_array_equal(obs, np.array([0.0]))

    def test_step_no_position(self):
        self.env.reset()
        obs, reward, done, truncated, info = self.env.step(0)  # No position
        self.assertEqual(len(self.env.orders), 0)
        self.assertEqual(self.env.current_step, 1)
        np.testing.assert_array_equal(obs, np.array([0.0]))

    def test_step_open_position(self):
        self.env.reset()
        obs, reward, done, truncated, info = self.env.step(1)  # Open position
        self.assertEqual(len(self.env.orders), 1)
        self.assertEqual(self.env.current_step, 1)
        np.testing.assert_array_equal(obs, np.array([0.0]))

    def test_step_close_position(self):
        self.env.reset()
        self.env.step(1)  # Open position
        obs, reward, done, truncated, info = self.env.step(0)  # Close position
        self.assertEqual(len(self.env.orders), 0)
        self.assertEqual(len(self.env.closed_orders), 1)
        self.assertEqual(self.env.current_step, 2)
        np.testing.assert_array_equal(obs, np.array([0.0]))

    def test_get_current_price(self):
        self.env.reset()
        open_price = self.env.get_current_price(OrderAction.OPEN)
        close_price = self.env.get_current_price(OrderAction.CLOSE)
        self.assertEqual(open_price, self.tick_data.loc[0, "ask_price"])
        self.assertEqual(close_price, self.tick_data.loc[0, "bid_price"])

    @patch(
        "src.environments.single_buy_discrete.single_buy_discrete_env.SingleBuyDiscreteEnv._update_account_state"
    )
    def test_account_update_called(self, mock_update):
        self.env.reset()
        self.env.step(1)  # Open position
        mock_update.assert_called_once()

    def test_done_condition(self):
        self.env.reset()
        for _ in range(98):  # 99 steps (starting from 0)
            obs, reward, done, truncated, info = self.env.step(0)
            self.assertFalse(done)

        obs, reward, done, truncated, info = self.env.step(0)  # 100th step
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
