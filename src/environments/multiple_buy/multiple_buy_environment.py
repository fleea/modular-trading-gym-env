from typing import List, Callable, Self, Any, SupportsFloat, Tuple, Optional
import numpy as np
import pandas as pd
from gymnasium import spaces
from src.environments.base_environment.base_environment import BaseEnvironment
from src.interfaces.order_interface import OrderObjectType, OrderType, OrderAction
from src.observations.base_observation import BaseObservation
from src.enums.closing_strategy_enum import OrderClosingStrategy


class MultipleBuyEnvironment(BaseEnvironment):
    """
    A trading environment supporting multiple buy orders with a gear-based action space.

    This environment allows for multiple concurrent orders, with the number of active orders
    determined by a continuous action value. It supports various order closing strategies.

    Attributes:
        data (pd.DataFrame): DataFrame of tick data for the trading session.
        current_index (int): The current index in the environment.
        max_orders (int): Maximum number of concurrent orders allowed.
        closing_strategy (OrderClosingStrategy): Strategy for closing orders when reducing positions.
    """

    def __init__(
        self,
        initial_balance: int,
        data: pd.DataFrame,
        observation: BaseObservation["MultipleBuyEnvironment"],
        reward_func: Callable[[Self, ...], float],
        lot: float = 0.01 * 100_000,
        max_orders: int = 3,
        closing_strategy: OrderClosingStrategy = OrderClosingStrategy.LIFO,
        start_index: int = 0,
    ):
        """
        Initialize the multiple buy gear trading environment.

        Args:
            initial_balance (int): The initial account balance.
            data (pd.DataFrame): DataFrame of data for the trading session.
            observation (BaseObservation[MultipleBuyGearEnvironment]): Observation class with get_space and get_observation methods.
            reward_func (Callable[[Self, float], float]): A function to calculate rewards.
            lot (float): The lot size for each order.
            max_orders (int): Maximum number of concurrent orders allowed.
            closing_strategy (OrderClosingStrategy): Strategy for closing orders when reducing positions.
        """
        start_padding = observation.get_start_padding()
        if len(data) <= start_padding:
            raise ValueError(
                f"Not enough data. Need at least {start_padding} periods, but got {len(data)}"
            )

        super().__init__(
            initial_balance, data=data, start_index=start_index + start_padding, max_index=len(data) + start_index - 1, 
        )

        self.observation = observation
        self.reward_function = reward_func
        self.lot = lot
        self.max_orders = max_orders
        self.closing_strategy = closing_strategy
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = observation.get_space()

    def reset(self, seed =None):
        super().reset(seed=seed)
        observation = self._get_observation()
        # assert self.observation_space.contains(observation), "Observation not in space"
        return observation, self._get_info()

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

        action_value = action[0]
        target_orders = min(int(action_value * (self.max_orders + 1)), self.max_orders)
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

    def _open_new_orders(self, num_orders: int):
        for _ in range(num_orders):
            open_price = self.get_current_price(OrderAction.OPEN)
            self._open_order(
                OrderType.BUY, self.lot, open_price, self.get_current_data().timestamp
            )

    def _close_excess_orders(self, num_orders: int):
        """Close excess orders based on the closing strategy."""
        orders_to_close = self._select_orders_to_close(num_orders)
        close_price = self.get_current_price(OrderAction.CLOSE)
        for order in orders_to_close:
            self._close_order(order, close_price)

    def _select_orders_to_close(self, num_orders: int) -> List[OrderObjectType]:
        """Select orders to close based on the closing strategy."""
        if self.closing_strategy == OrderClosingStrategy.LIFO:
            return self.orders[-num_orders:]
        elif self.closing_strategy == OrderClosingStrategy.FIFO:
            return self.orders[:num_orders]
        elif self.closing_strategy == OrderClosingStrategy.MOST_PROFITABLE:
            sorted_orders = sorted(
                self.orders, key=lambda x: self._calculate_profit(x), reverse=True
            )
            return sorted_orders[:num_orders]
        elif self.closing_strategy == OrderClosingStrategy.LEAST_PROFITABLE:
            sorted_orders = sorted(self.orders, key=lambda x: self._calculate_profit(x))
            return sorted_orders[:num_orders]
        else:
            raise NotImplementedError(
                f"Closing strategy {self.closing_strategy} not implemented"
            )

    def _calculate_profit(self, order: OrderObjectType) -> float:
        """Calculate the current profit of an order."""
        current_price = self.get_current_price(OrderAction.CLOSE)
        return (current_price - order.open_price) * order.volume

    def _get_observation(self: Self) -> spaces:
        return self.observation.get_observation(self)

    def get_current_price(
        self, order_action: OrderAction, order_type: Optional[OrderType] = None
    ) -> float:
        current = self.get_current_data()
        return (
            current.ask_price if order_action == OrderAction.OPEN else current.bid_price
        )

