from datetime import datetime
from enum import Enum
from typing import List, Callable, TypedDict, NamedTuple
import pandas as pd
import random


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
            # print(f"Step {total_steps + step}: {price_data}")

            delta = delta_func(total_steps + step)
            current_mid_price += delta

        segment_profit = calculate_max_profit(pd.DataFrame(segment_data))
        segment_profits.append(segment_profit)
        # print(f"Max profit for segment {len(segment_profits)}: {segment_profit}")

        total_steps += step_amount

    total_max_profit = sum(segment_profits)
    # print(f"Total max profit: {total_max_profit}")

    return pd.DataFrame(data), total_max_profit


def get_data(size=1000, initial_price: float = 1.015055, trend_probability: float = 0.5):
    total_points = size

    # Calculate average segment length based on total_points
    if total_points <= 100:
        avg_segment_length = 5
    elif total_points <= 500:
        avg_segment_length = 8
    elif total_points <= 1000:
        avg_segment_length = 12
    else:
        avg_segment_length = 15

    commands: List[PriceCommand] = []
    total_steps = 0
    cumulative_change = 0

    while total_steps < total_points:
        segment_length = max(1, int(random.gauss(avg_segment_length, avg_segment_length / 4)))
        segment_length = min(segment_length, total_points - total_steps)
        # print(f"segment_length: {segment_length}")


        spread_func = lambda step: max(0.00001, random.gauss(0.00011, 0.00001)) # Around 0.00011
        # print(f"base_spread: {spread_func}")


        direction = Direction.UP if random.random() < trend_probability else Direction.DOWN

        # print(f"direction: {direction}")
        # Generate random delta function
        base_delta = random.uniform(0.0005, 0.002) * direction.value
        # print(f"base_delta: {base_delta}")

        command = PriceCommand(
            step_amount=segment_length,
            spread_func=spread_func,
            delta_func=create_delta_func(direction)
        )

        # print (command)
        commands.append(command)
        total_steps += segment_length

        # Update cumulative change
        cumulative_change += base_delta * segment_length

    return simulate_prices(initial_price, commands)


def create_delta_func(direction: Direction, base_delta=0.001, std_dev=0.0001, spike_probability=0.05,
                      spike_multiplier=3):
    def delta_func(step: int) -> float:
        base = base_delta * direction.value
        delta = random.gauss(base, std_dev)

        # Occasional spikes
        # if random.random() < spike_probability:
        #     delta *= random.uniform(1, spike_multiplier) * (1 if random.random() < 0.5 else -1)

        return delta

    return delta_func


def plot_tick_data(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['bid_price'], label='Bid Price')
    plt.plot(df['timestamp'], df['ask_price'], label='Ask Price')
    plt.title('Generated Price Data with Downward Trend and Upside Movements')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # print(get_data())
    df, max_profit = get_data()
    # df.to_csv('output.csv', index=False)
    # print(df)
    # print(f"Generated {len(df)} data points")
    # print(f"Initial price: {df['bid_price'].iloc[0]:.6f}")
    # print(f"Max profit: {max_profit:.6f}")
    # print(f"Final price: {df['bid_price'].iloc[-1]:.6f}")
    # print(f"Overall change: {(df['bid_price'].iloc[-1] - df['bid_price'].iloc[0]):.6f}")
    # print(f"Average spread: {(df['ask_price'] - df['bid_price']).mean():.6f}")
    # plot_tick_data(df)

    # print(df.head())
    # print(df['bid_price'].std())
    # print(augment_col_difference(df, ['bid_price', 'ask_price']))
    # Plotting code to visualize the data

