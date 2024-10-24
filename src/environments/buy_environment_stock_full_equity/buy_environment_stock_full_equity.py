from src.environments.buy_environment.buy_environment import BuyEnvironment
from src.interfaces.order_interface import OrderAction
import numpy as np
from typing import Tuple, Any, SupportsFloat
from gymnasium import spaces
import pandas as pd
# There is no leverage in this environment
# We always buy at max amount possible in balance available

class BuyEnvironmentStockFullEquity(BuyEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_current_price(self, data: pd.DataFrame) -> float:
        current = self.get_current_data()
        return current.close

    def step(
        self, action: np.ndarray
    ) -> Tuple[spaces, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment based on the given action.

        Args:
            action (np.ndarray): The action to take, a float between 0 and 1.

        Returns:
            Tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
            Observation, reward, terminated, truncated, and info dictionary.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        tick_data = self.get_current_data()
        max_order = self.equity[-1] // tick_data['close']
        action_value = action[0]
        target_orders = 0 if max_order is 0 else min(int(action_value * max_order), max_order)
        current_orders = len(self.orders)

        # Calculate the change in number of orders
        order_change = target_orders - current_orders

        if order_change > 0:
            # Open new orders
            orders_to_open = min(order_change, self.max_orders - current_orders)
            if orders_to_open > 0:
                self._open_new_orders(orders_to_open)
        elif order_change < 0:
            # Close excess orders
            orders_to_close = abs(order_change)
            self._close_excess_orders(orders_to_close)

        self._update_account_state()

        done = self.current_index >= self.max_index
        rwd = self.reward_function(self)
        self.rewards.append(rwd)
        obs = self._get_observation()
        info = self._get_info()
        self.current_index += 1

        return (obs, rwd, done, False, info)