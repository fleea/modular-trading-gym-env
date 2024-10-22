# pytest -s src/environments/buy_environment_stock_full_equity/test_buy_environment_stock_full_equity.py
# python3.12 src/environments/buy_environment_stock_full_equity/test_buy_environment_stock_full_equity.py

import unittest
import numpy as np
from src.environments.buy_environment_stock_full_equity.buy_environment_stock_full_equity import BuyEnvironmentStockFullEquity
from src.preprocessing.test_hlc import get_hlc_data
from unittest.mock import Mock
from src.observations.base_observation import BaseObservation
from gymnasium import spaces
from src.preprocessing.hlc import augment_with_hlc
from src.interfaces.order_interface import OrderObjectType, OrderType

class TestBuyEnvironmentStockFullEquity(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.data = augment_with_hlc(get_hlc_data())
        self.data = self.data.reset_index()
        self.mock_observation = Mock(spec=BaseObservation)
        self.mock_observation.get_start_index.return_value = 1
        self.mock_observation.get_space.return_value = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,)
        )
        self.mock_observation.get_observation.return_value = np.array(
            [0.8661785930870153, 0.9563227115649926, 0.8634259549708271, 1.6021444636122812,
             1.6256345371293421, 1.6020351583411125, 1.5075567228888183, 1.5666989036012804,
             1.5075567228888183]
        )
        self.reward_func = Mock(return_value=0.0)
        self.env = BuyEnvironmentStockFullEquity(
            initial_balance=122 * 6 - 1, # Max order should be 5
            data=self.data,
            observation=self.mock_observation,
            reward_func=self.reward_func,
        )
        self.env.reset()

    def test_action_zero_no_orders(self):
        """
        Test that no orders are opened or closed when action is 0 and there are no existing orders.
        """
        # Ensure there are no existing orders
        self.env.orders = []
        self.env.closed_orders = []
        action = np.array([0], dtype=np.float32)
        initial_orders = len(self.env.orders)
        self.env.step(action)
        final_orders = len(self.env.orders)
        self.assertEqual(final_orders, initial_orders, "Number of orders should remain the same for action 0 when there are no existing orders.")

    def test_action_point_two_open_two_orders(self):
        """
        Test that 2 orders are opened when action is 0.2 and max_order is 10.
        """
        # Ensure there are no existing orders
        self.env.orders = []
        self.env.closed_orders = []
        action = np.array([0.2], dtype=np.float32)
        self.env.step(action)
        final_orders = len(self.env.orders)
        expected_orders = 1
        self.assertEqual(final_orders, expected_orders, f"Number of orders should be {expected_orders} for action 0.2 with max_order 5.")

    def test_action_point_seven_open_two_orders(self):
        """
        Test that 2 orders are opened when action is 0.7, current_orders is 5, and max_order is 10.
        """
        # Set existing orders to 5
        mock_order = OrderObjectType(
            volume=1, type=OrderType.BUY, open_price=122, timestamp="01-01-2024"
        )
        self.env.orders = [mock_order for _ in range(3)]
        self.env.closed_orders = []
        action = np.array([0.5], dtype=np.float32)
        self.env.step(action)
        final_orders = len(self.env.orders)
        expected_orders = 2
        self.assertEqual(final_orders, expected_orders, f"Number of orders should be {expected_orders} for action 0.5 with current_orders 3 and max_order 5.")

if __name__ == "__main__":
    unittest.main()
