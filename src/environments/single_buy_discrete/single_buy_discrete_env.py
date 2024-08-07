from typing import List, Callable, Self, Any, SupportsFloat, Tuple, Optional
import pandas as pd
from src.environments.base_environment.base_environment import BaseEnvironment
from src.interfaces.data_interface import TickData
from src.interfaces.order_interface import OrderObjectType, OrderType, OrderAction
from src.observations.base_observation import BaseObservation
from gymnasium import spaces


def _calculate_profit(exit_price: float, order: OrderObjectType) -> float:
    """
    Calculate the profit or loss for the given position.

    Args:
        exit_price (float): The price at which the position is being closed.
        order (OrderObjectType): The order being closed.

    Returns:
        float: The calculated profit or loss.
    """
    return (exit_price - order.open_price) * order.volume


class SingleBuyDiscreteEnv(BaseEnvironment):
    """
    A simple trading environment for a single buy position based on gymnasium and BaseEnvironment.

    This environment simulates a basic trading scenario with two discrete actions:
    Position (1) and NoPosition (0). It uses tick data and trends for observations.

    Attributes:
        tick_data (List[TickData]): List of tick data for the trading session.
        current_step (int): The current step in the environment.
    """

    def __init__(
        self,
        initial_balance: int,
        tick_data: pd.DataFrame,
        observation: BaseObservation["SingleBuyDiscreteEnv"],
        reward_func: Callable[[Self, ...], float],
        lot: float = 0.01 * 100_000,
    ) -> None:
        """
        Initialize the single buy trading environment.

        Args:
            initial_balance (int): The initial account balance.
            tick_data (List[TickData]): List of tick data for the trading session.
            observation (BaseObservation[SingleBuyDiscreteEnv]): Observation class that has get_space and get_observation methods
            reward_func (Callable[[Self, float], float]): A function to calculate rewards. Used in steps, not needed in base_environment environment.
        """
        min_periods = observation.get_min_periods()
        if len(tick_data) <= min_periods:
            raise ValueError(
                f"Not enough data. Need at least {min_periods} periods, but got {len(tick_data)}"
            )

        super().__init__(
            initial_balance, data=tick_data, max_step=len(tick_data) - min_periods
        )
        self.observation = observation
        self.tick_data = tick_data
        self.reward_function = reward_func
        self.lot = lot
        self.action_space = spaces.Discrete(2)  # 0: NoPosition, 1: Position
        self.observation_space = observation.get_space()
        self.current_step = min_periods  # Start at the minimum required step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.observation.get_min_periods()
        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[spaces, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment based on the given action.

        Args:
            action (int): The action to take (0: NoPosition, 1: Position).

        Returns:
            Tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
            Observation, reward, terminated, truncated, and info dictionary.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if action == 0 and self.orders:  # NoPosition
            close_price = self.get_current_price(OrderAction.CLOSE)
            self._close_order(self.orders[0], close_price)
        elif action == 1 and len(self.orders) == 0:  # Position
            tick_data = self._get_tick_data()
            open_price = self.get_current_price(OrderAction.OPEN)
            self._open_order(
                order_type=OrderType.BUY,
                volume=self.lot,
                price=open_price,
                timestamp=tick_data.timestamp,
            )

        self._update_account_state()

        self.current_step += 1
        done = self.current_step >= len(self.tick_data) - 1
        rwd = self.reward_function(self)
        self.rewards.append(rwd)

        return (self._get_observation(), rwd, done, False, self._get_info())

    def _get_observation(self: Self) -> spaces:
        return self.observation.get_observation(self)

    def get_current_price(
        self, order_action: OrderAction, order_type: Optional[OrderType] = None
    ) -> float:
        current = self._get_tick_data()
        return (
            current.ask_price if order_action == OrderAction.OPEN else current.bid_price
        )

    def _get_tick_data(self: Self) -> TickData:
        return self.tick_data.loc[self.current_step]
