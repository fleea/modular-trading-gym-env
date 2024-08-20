import unittest
from decimal import Decimal
import pandas as pd
import numpy as np
from gymnasium import spaces
from src.interfaces.order_interface import OrderObjectType, OrderType, OrderAction
from src.environments.base_environment.base_environment import BaseEnvironment


class TestEnvironment(BaseEnvironment):
    def __init__(self, initial_balance: int, data: pd.DataFrame, max_step: int):
        super().__init__(initial_balance, data, max_step)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        # Implement a simple step function for testing
        self.current_step += 1
        return self._get_observation(), 0, self._is_done(), False, self._get_info()

    def _get_observation(self):
        return np.array([self.current_step], dtype=np.float32)

    def get_current_price(
        self, order_action: OrderAction, order_type: OrderType = None
    ) -> float:
        # Return a dummy price for testing
        return 100.0


class TestBaseEnvironment(unittest.TestCase):

    def setUp(self):
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
                "price": np.random.uniform(90, 110, 100),
            }
        )
        self.env = TestEnvironment(initial_balance=10000, data=data, max_step=100)

    def test_initial_state(self):
        self.assertEqual(self.env.initial_balance, 10000)
        self.assertEqual(self.env.balance, [10000])
        self.assertEqual(self.env.equity, [10000])
        self.assertEqual(self.env.current_step, 0)

    def test_open_order(self):
        self.env._open_order(
            OrderType.BUY, volume=1.0, price=100.0, timestamp="2023-01-01 00:00:00"
        )
        self.assertEqual(len(self.env.orders), 1)
        self.assertEqual(self.env.orders[0].volume, 1.0)
        self.assertEqual(self.env.orders[0].open_price, 100.0)

    def test_close_order(self):
        order = OrderObjectType(
            volume=1.0,
            type=OrderType.BUY,
            open_price=100.0,
            timestamp="2023-01-01 00:00:00",
        )
        self.env.orders.append(order)
        self.env._close_order(order, close_price=105.0)
        self.assertEqual(len(self.env.orders), 0)
        self.assertEqual(len(self.env.closed_orders), 1)
        self.assertEqual(self.env.closed_orders[0].close_price, 105.0)

    def test_update_account_state_no_orders(self):
        self.env._update_account_state()
        self.assertEqual(self.env.balance[-1], 10000)
        self.assertEqual(self.env.equity[-1], 10000)

    def test_update_account_state_with_open_order(self):
        self.env._open_order(
            OrderType.BUY, volume=1.0, price=100.0, timestamp="2023-01-01 00:00:00"
        )
        self.env._update_account_state()
        self.assertEqual(self.env.balance[-1], 10000)
        self.assertAlmostEqual(self.env.equity[-1], 10000, places=2)

    def test_update_account_state_with_closed_order_profit(self):
        order = OrderObjectType(
            volume=1.0,
            type=OrderType.BUY,
            open_price=100.0,
            timestamp="2023-01-01 00:00:00",
        )
        self.env.closed_orders.append(order)
        order.close_price = 105.0
        self.env._update_account_state()
        self.assertAlmostEqual(self.env.balance[-1], 10005, places=2)
        self.assertAlmostEqual(self.env.equity[-1], 10005, places=2)

    def test_update_account_state_with_closed_order_loss(self):
        order = OrderObjectType(
            volume=1.0,
            type=OrderType.BUY,
            open_price=100.0,
            timestamp="2023-01-01 00:00:00",
        )
        self.env.closed_orders.append(order)
        order.close_price = 95.0
        self.env._update_account_state()
        self.assertAlmostEqual(self.env.balance[-1], 9995, places=2)
        self.assertAlmostEqual(self.env.equity[-1], 9995, places=2)

    def test_update_account_state_with_multiple_orders(self):
        # Open order
        self.env._open_order(
            OrderType.BUY, volume=1.0, price=100.0, timestamp="2023-01-01 00:00:00"
        )

        # Closed order with profit
        closed_order1 = OrderObjectType(
            volume=1.0,
            type=OrderType.BUY,
            open_price=100.0,
            timestamp="2023-01-01 01:00:00",
        )
        closed_order1.close_price = 105.0
        self.env.closed_orders.append(closed_order1)

        # Closed order with loss
        closed_order2 = OrderObjectType(
            volume=1.0,
            type=OrderType.BUY,
            open_price=100.0,
            timestamp="2023-01-01 02:00:00",
        )
        closed_order2.close_price = 97.0
        self.env.closed_orders.append(closed_order2)

        self.env._update_account_state()

        expected_balance = 10000 + 5 - 3  # Initial balance + profit - loss
        expected_equity = (
            expected_balance + (100 - 100) * 1.0
        )  # Balance + unrealized P/L of open order

        self.assertAlmostEqual(self.env.balance[-1], expected_balance, places=2)
        self.assertAlmostEqual(self.env.equity[-1], expected_equity, places=2)

    def test_is_done(self):
        self.assertFalse(self.env._is_done())
        self.env.current_step = 100
        self.assertTrue(self.env._is_done())
        self.env.current_step = 0
        self.env.balance = [0]
        self.assertTrue(self.env._is_done())


if __name__ == "__main__":
    unittest.main()
