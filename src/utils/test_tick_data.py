import pytest
from datetime import datetime
import pandas as pd
from typing import List

from src.utils.tick_data import (
    Direction,
    PriceCommand,
    PriceData,
    calculate_max_profit,
    simulate_prices,
    create_delta_func,
)


def test_direction_enum():
    assert Direction.UP.value == 1
    assert Direction.DOWN.value == -1


def test_price_data_namedtuple():
    price_data = PriceData(bid_price=1.0, ask_price=1.1, timestamp=datetime.now())
    assert isinstance(price_data, PriceData)
    assert price_data.bid_price == 1.0
    assert price_data.ask_price == 1.1
    assert isinstance(price_data.timestamp, datetime)


def test_calculate_max_profit():
    df = pd.DataFrame(
        {"bid_price": [1.0, 1.1, 1.2, 1.1, 1.3], "ask_price": [1.1, 1.2, 1.3, 1.2, 1.4]}
    )
    assert calculate_max_profit(df) == pytest.approx(
        0.2, rel=1e-6
    )  # Max profit should be 1.3 - 1.1 = 0.2

    # Test with no profit scenario
    df_no_profit = pd.DataFrame(
        {"bid_price": [1.3, 1.2, 1.1, 1.0], "ask_price": [1.4, 1.3, 1.2, 1.1]}
    )
    assert calculate_max_profit(df_no_profit) == 0

    # Test with missing columns
    df_missing_columns = pd.DataFrame({"mid_price": [1.0, 1.1, 1.2]})
    with pytest.raises(ValueError):
        calculate_max_profit(df_missing_columns)


def test_simulate_prices():
    initial_price = 1.0
    commands: List[PriceCommand] = [
        {
            "step_amount": 3,
            "spread_func": lambda step: 0.1,
            "delta_func": lambda step: 0.01,
        },
        {
            "step_amount": 2,
            "spread_func": lambda step: 0.2,
            "delta_func": lambda step: -0.02,
        },
    ]

    result = simulate_prices(initial_price, commands)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5  # Total steps: 3 + 2
    assert "bid_price" in result.columns
    assert "ask_price" in result.columns
    assert "timestamp" in result.columns


def test_create_delta_func():
    up_delta = create_delta_func(Direction.UP, base_delta=0.001, std_dev=0)
    down_delta = create_delta_func(Direction.DOWN, base_delta=0.001, std_dev=0)

    assert up_delta(0) == pytest.approx(0.001, rel=1e-9)
    assert down_delta(0) == pytest.approx(-0.001, rel=1e-9)


if __name__ == "__main__":
    pytest.main()
