# pytest src/environments/buy_environment/test_buy_environment.py

import unittest
from unittest.mock import Mock, patch
import numpy as np
from gymnasium import spaces
from src.environments.buy_environment.buy_environment import (
    BuyEnvironment,
)
from src.interfaces.order_interface import OrderType, OrderAction, OrderObjectType
from src.observations.base_observation import BaseObservation
from src.utils.tick_data import simulate_prices


class TestBuyEnvironment(unittest.TestCase):
    def setUp(self):
        self.initial_balance = 10000
        down = {
            "step_amount": 20,
            "spread_func": lambda step: 0.00011,  # Increasing spread
            "delta_func": lambda step: -0.001,  # Constant downward trend
        }
        up = {
            "step_amount": 20,
            "spread_func": lambda step: 0.00011,  # Increasing spread
            "delta_func": lambda step: 0.001,  # Constant downward trend
        }
        self.data, _ = simulate_prices(1.015055, [down, up, down, up])
        self.mock_observation = Mock(spec=BaseObservation)
        self.mock_observation.get_start_index.return_value = 1
        self.mock_observation.get_space.return_value = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,)
        )
        self.mock_observation.get_observation.return_value = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )

        self.reward_func = Mock(return_value=0.0)

        self.env = BuyEnvironment(
            initial_balance=self.initial_balance,
            data=self.data,
            observation=self.mock_observation,
            reward_func=self.reward_func,
            max_orders=3,
        )

    def test_initialization(self):
        self.assertEqual(self.env.initial_balance, self.initial_balance)
        self.assertEqual(len(self.env.data), 80)  # 4 * 20 steps
        self.assertEqual(self.env.max_orders, 3)
        self.assertIsInstance(self.env.action_space, spaces.Box)
        self.assertEqual(self.env.action_space.shape, (1,))

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(self.env.current_index, 1)  # start_padding
        np.testing.assert_array_equal(obs, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertIn("balance", info)
        self.assertIn("equity", info)

    @patch.object(BuyEnvironment, "_open_new_orders")
    @patch.object(BuyEnvironment, "_close_excess_orders")
    def test_step(self, mock_close, mock_open):
        self.env.reset()

        # Test opening orders
        obs, reward, done, truncated, info = self.env.step(
            np.array([0.7], dtype=np.float32)
        )  # Should open 2 orders
        mock_open.assert_called_once_with(2)
        mock_close.assert_not_called()

        mock_open.reset_mock()
        mock_close.reset_mock()

        # Test closing orders
        self.env.orders = [
            Mock(spec=OrderObjectType, open_price=1.0, volume=0.01),
            Mock(spec=OrderObjectType, open_price=1.0, volume=0.01),
            Mock(spec=OrderObjectType, open_price=1.0, volume=0.01),
        ]

        # Mock the get_current_price method to return a specific value
        with patch.object(self.env, "get_current_price", return_value=1.001):
            obs, reward, done, truncated, info = self.env.step(
                np.array([0.3], dtype=np.float32)
            )  # Should close 2 orders

        mock_close.assert_called_once_with(2)
        mock_open.assert_not_called()

    def test_max_orders(self):
        self.env.reset()
        self.env.max_orders = 1
        up = {
            "step_amount": 20,
            "spread_func": lambda step: 0.00011,  # Increasing spread
            "delta_func": lambda step: 0.001,  # Constant downward trend
        }
        self.env.data, _ = simulate_prices(1.015055, [up])
        self.env.step(np.array([0.7], dtype=np.float32))
        self.env.step(np.array([0.7], dtype=np.float32))
        self.env.step(np.array([0.7], dtype=np.float32))
        self.env.current_index = 14
        self.env.step(np.array([0.1], dtype=np.float32))
        self.assertEqual(len(self.env.orders), 0)
        self.assertEqual(len(self.env.closed_orders), 1)

    def test_open_new_orders(self):
        self.env.reset()
        self.env._open_new_orders(2)
        self.assertEqual(len(self.env.orders), 2)
        for order in self.env.orders:
            self.assertEqual(order.type, OrderType.BUY)
            self.assertEqual(order.volume, self.env.lot)

    def test_close_excess_orders(self):
        self.env.reset()
        # Create some mock orders
        self.env.orders = [
            OrderObjectType(OrderType.BUY, 0.01, 1.0000, "2023-01-01 00:00:00", "1"),
            OrderObjectType(OrderType.BUY, 0.01, 1.0001, "2023-01-01 00:01:00", "2"),
            OrderObjectType(OrderType.BUY, 0.01, 1.0002, "2023-01-01 00:02:00", "3"),
        ]
        self.env._close_excess_orders(2)
        self.assertEqual(len(self.env.orders), 1)
        self.assertEqual(len(self.env.closed_orders), 2)

    def test_select_orders_to_close_lifo(self):
        self.env.reset()
        self.env.orders = [
            OrderObjectType(OrderType.BUY, 0.01, 1.0000, "2023-01-01 00:00:00", "1"),
            OrderObjectType(OrderType.BUY, 0.01, 1.0001, "2023-01-01 00:01:00", "2"),
            OrderObjectType(OrderType.BUY, 0.01, 1.0002, "2023-01-01 00:02:00", "3"),
        ]
        orders_to_close = self.env._select_orders_to_close(2)
        self.assertEqual(len(orders_to_close), 2)
        self.assertEqual(orders_to_close[0].open_price, 1.0001)
        self.assertEqual(orders_to_close[1].open_price, 1.0002)

    def test_calculate_profit(self):
        self.env.reset()
        order = OrderObjectType(OrderType.BUY, 0.01, 1.0000, "2023-01-01 00:00:00", "1")
        with patch.object(self.env, "get_current_price", return_value=1.0005):
            profit = self.env._calculate_profit(order)
            expected_profit = (1.0005 - 1.0000) * 0.01
            self.assertAlmostEqual(profit, expected_profit, places=10)

    def test_get_current_price(self):
        self.env.reset()
        self.env.current_index = 15  # Arbitrary step
        open_price = self.env.get_current_price(OrderAction.OPEN)
        close_price = self.env.get_current_price(OrderAction.CLOSE)
        self.assertEqual(open_price, self.data.loc[15, "ask_price"])
        self.assertEqual(close_price, self.data.loc[15, "bid_price"])

    def test_different_market_conditions(self):
        self.env.reset()

        # Test during upward trend (first 20 steps)
        for _ in range(5):
            obs, reward, done, truncated, info = self.env.step(
                np.array([0.7], dtype=np.float32)
            )  # Open 2 orders
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(info, dict)

        # Move to downward trend (next 20 steps)
        self.env.current_index = 25
        for _ in range(5):
            obs, reward, done, truncated, info = self.env.step(
                np.array([0.3], dtype=np.float32)
            )  # Close orders
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(info, dict)

        # Move to next upward trend
        self.env.current_index = 45

        for _ in range(5):
            obs, reward, done, truncated, info = self.env.step(
                np.array([0.7], dtype=np.float32)
            )  # Open 2 orders
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
