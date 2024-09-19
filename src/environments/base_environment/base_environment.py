from abc import abstractmethod
from dataclasses import field
from typing import List, Optional, Any, TypeVar, Tuple, Self

import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from numpy.typing import NDArray
from decimal import Decimal, getcontext

from src.interfaces.order_interface import OrderObjectType, OrderType, OrderAction

# Spaces like Box or Discrete, should be defined by child environment
ObservationType = TypeVar("ObservationType", bound=spaces)
ActionType = TypeVar("ActionType", bound=spaces)
DataType = TypeVar("DataType", bound=pd.DataFrame)


# Observation type: gymnasium spaces
# Di base_environment environment, with default
# - get_observation, is_done, is_truncated, get_info, get_reward,
# - all calculation for equity and balances, create new order, close order
# Child environment
# - get_observation, get_info, action space, step, observation space
# Input
# - reward_function
class BaseEnvironment(Env[NDArray, dict[str, Any]]):
    initial_balance: int = 0
    balance: List[float] = []
    equity: List[float] = []
    closed_orders: List[OrderObjectType] = field(default_factory=list)
    orders: List[OrderObjectType] = field(default_factory=list)
    current_index: int = 0
    max_index: int = (
        1000  # Default value, based on tick data, can be overridden in child classes
    )
    data: DataType = field(
        default_factory=pd.DataFrame
    )  # Tick Data or trading data per minute

    def __init__(
        self: Self, initial_balance: int, data: pd.DataFrame, start_index: int = 0, max_index: int = None
    ) -> None:
        self.initial_balance = initial_balance
        self.balance = [initial_balance]
        self.equity = [initial_balance]
        self.rewards = [0]
        self.closed_orders = []
        self.orders = []
        self.data = data
        self.start_index = start_index
        self.max_index = max_index if max_index is not None else len(data)
        self.current_index = start_index

        # Define a default observation space
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )

        # Define a default action space (can be overridden in child classes)
        self.action_space = spaces.Discrete(2)  # 0: No action, 1: Take action

    # Reset should return: Observation space, info
    def reset(
        self: Self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[ObservationType, dict[str, Any]]:
        super().reset(seed=seed)
        self.balance = [self.initial_balance]
        self.equity = [self.initial_balance]
        self.rewards = [0]
        self.closed_orders = []
        self.orders = []
        self.current_index = self.start_index
        # NEED TO RETURN tuple[[NDArray], dict[str, Any]]
        # WARN: The obs returned by the `reset()` method is not within the observation space.
        return self._get_observation(), self._get_info()

    @abstractmethod
    def step(
        self: Self, action: ActionType
    ) -> Tuple[ObservationType, float, bool, bool, dict[str, Any]]:
        # This method should be implemented in child classes
        # Step should return: Observation space, reward, done, truncated, info
        # On buy, get tick_data, open position
        # On close, get tick_data, close position
        pass

    @abstractmethod
    def _get_observation(self: Self) -> ObservationType:
        pass

    def _is_done(self: Self) -> bool:
        return self.current_index - self.start_index >= self.max_index or self.balance[-1] <= 0

    def _is_truncated(self: Self) -> bool:
        return False  # Default implementation, can be overridden in child classes

    def _get_info(self: Self) -> dict[str, Any]:
        return {
            "balance_history": self.balance,
            "equity_history": self.equity,
            "balance": self.balance[-1],
            "equity": self.equity[-1],
            "num_orders": len(self.orders),
            "closed_orders": self.closed_orders,
            "orders": self.orders,
            "num_closed_orders": len(self.closed_orders),
            "current_index": self.current_index,
            "max_index": self.max_index,
            "rewards": self.rewards,
            "reward": self.rewards[-1],  # Add the reward to the info
            "final_equity": (
                self.equity[-1] if self.current_index >= self.max_index else None
            ),
        }

    def _open_order(
        self: Self, order_type: OrderType, volume: float, price: float, timestamp: str
    ) -> None:
        """
        Open a new order.

        Args:
            order_type (str): The type of order (e.g., 'buy', 'sell').
            volume (float): The volume of the order.
            price (float): The price at which to open the order.
            timestamp (str): The timestamp of the order.
        """
        order = OrderObjectType(
            volume=volume, type=order_type, open_price=price, timestamp=timestamp
        )
        self.orders.append(order)

    def _close_order(self, order: OrderObjectType, close_price: float) -> None:
        """
        Close an existing order and calculate its profit.

        Args:
            order (OrderType): The order to close.
            close_price (float): The price at which to close the order.
        """
        if order in self.orders:
            self.orders.remove(order)
            order.close_price = close_price

            # Calculate profit
            if order.type == OrderType.BUY:
                order.profit = (close_price - order.open_price) * order.volume
            elif order.type == OrderType.SELL:
                order.profit = (order.open_price - close_price) * order.volume
            else:
                raise ValueError(f"Unknown order type: {order.type}")

            self.closed_orders.append(order)

    def _update_account_state(self) -> None:
        """
        Update the account state (balance and equity) for the current tick.
        This method uses Decimal for high-precision calculations.
        """
        # Set decimal precision (adjust as needed)
        getcontext().prec = 10

        # Convert initial balance to Decimal
        initial_balance = Decimal(str(self.initial_balance))

        # Calculate realized PnL from closed orders
        realized_pnl = sum(
            (Decimal(str(order.close_price)) - Decimal(str(order.open_price)))
            * Decimal(str(order.volume))
            for order in self.closed_orders
            if order.close_price is not None
        )

        # Update balance (affected by realized PnL)
        new_balance = initial_balance + realized_pnl
        self.balance.append(float(new_balance))  # Convert back to float for storage

        # Calculate unrealized PnL from open orders
        current_price = Decimal(str(self.get_current_price(OrderAction.CLOSE)))
        unrealized_pnl = sum(
            (current_price - Decimal(str(order.open_price)))
            * Decimal(str(order.volume))
            for order in self.orders
        )

        # Update equity (balance plus unrealized PnL)
        new_equity = new_balance + unrealized_pnl
        self.equity.append(float(new_equity))  # Convert back to float for storage

    @abstractmethod
    def get_current_price(
        self, order_action: OrderAction, order_type: Optional[OrderType] = None
    ) -> float:
        """
        Get the current price for a given order. This method should be implemented
        in child classes based on the specific market data structure.
        NOT a private class because observation class should be able to access this

        Args:
            order_action (OrderAction): The action to be performed.
            order_type (OrderType): The type of order (e.g., 'buy', 'sell'). This is optional

        Returns:
            float: The current price for the order.
        """
        pass

    def get_step_data(self, offset):
        index = self.current_index + offset
        try:
            return self.data.iloc[index]
        except IndexError:
            return None
        
    def get_data(self) -> pd.DataFrame:
        return self.data

    def get_current_data(self) -> pd.Series:
        return self.data.loc[self.current_index]


__all__ = ["BaseEnvironment"]
