from datetime import datetime
from enum import Enum
from typing import List, Callable, TypedDict, NamedTuple, Tuple
import pandas as pd


class Direction(Enum):
    UP = 1
    DOWN = -1


class PriceCommand(TypedDict):
    step_amount: int
    spread_func: Callable[[int], float]
    delta_func: Callable[[int], float]


class PriceData(NamedTuple):
    bid_price: float
    ask_price: float
    timestamp: datetime


def calculate_max_profit(df: pd.DataFrame) -> float:
    if "bid_price" not in df.columns or "ask_price" not in df.columns:
        raise ValueError("DataFrame must contain 'bid_price' and 'ask_price' columns")

    current_max, max_profit = 0, 0
    for _, price in df.iloc[::-1].iterrows():
        current_max = max(current_max, price["bid_price"])
        potential_profit = current_max - price["ask_price"]
        max_profit = max(max_profit, potential_profit)
    return max(max_profit, 0)  # min should be 0 (not buying anything)


def simulate_prices(initial_price: float, commands: List[PriceCommand]) -> pd.DataFrame:
    """
    Simulates price movements based on initial price and a series of commands.

    Parameters:
    initial_price (float): The initial mid price.
    commands (List[PriceCommand]): List of command objects, each containing:
        - step_amount (int): Number of steps for this command.
        - spread_func (Callable[[int], float]): Function to calculate spread, takes step index.
        - delta_func (Callable[[int], float]): Function to calculate price change, takes step index.

    Returns:
    pd.DataFrame: DataFrame containing 'mid_price', 'bid_price', and 'ask_price' for each step.
    """
    data: List[PriceData] = []
    segment_data: List[PriceData] = []
    segment_profits: List[float] = []
    current_mid_price = initial_price
    total_steps = 0

    for command in commands:
        step_amount = command["step_amount"]
        spread_func = command["spread_func"]
        delta_func = command["delta_func"]

        segment_data.clear()

        for step in range(step_amount):
            timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(
                minutes=total_steps + step
            )
            spread = spread_func(total_steps + step)
            bid_price = current_mid_price - spread / 2
            ask_price = current_mid_price + spread / 2

            price_data = PriceData(
                bid_price=bid_price, ask_price=ask_price, timestamp=timestamp
            )
            data.append(price_data)
            segment_data.append(price_data)

            delta = delta_func(total_steps + step)
            current_mid_price += delta

        segment_profit = calculate_max_profit(pd.DataFrame(segment_data))
        segment_profits.append(segment_profit)
        print(f"Max profit for segment {len(segment_profits)}: {segment_profit}")

        total_steps += step_amount

    total_max_profit = sum(segment_profits)
    print(f"Total max profit: {total_max_profit}")

    return pd.DataFrame(data)


def get_data():
    # WRAPPER FOR COMMANDS FUNCTION
    # Feel free to change or copy and paste to agent

    # replacing
    # up = get_directional_price_dataframe(Direction.UP, step=15, spread=11, points=100)
    # down = get_directional_price_dataframe(Direction.DOWN, step=15, spread=11, points=100, start_price=1.0150)
    # tick_data = merge_price_dataframes(down, up, down, up)

    initial_price: float = 1.015055
    down = {
        "step_amount": 15,
        "spread_func": lambda step: 0.00011,  # Increasing spread
        "delta_func": lambda step: -0.001,  # Constant downward trend
    }
    up = {
        "step_amount": 15,
        "spread_func": lambda step: 0.00011,  # Increasing spread
        "delta_func": lambda step: 0.001,  # Constant downward trend
    }
    commands: List[PriceCommand] = [down, up, down, up]
    return simulate_prices(initial_price, commands)


if __name__ == "__main__":
    print(get_data())
