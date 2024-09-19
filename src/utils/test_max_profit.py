import pandas as pd
from src.utils.max_profit import calculate_max_profit
import pytest

def test_basic_functionality():
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='min'),
        'bid_price': [1.00, 1.01, 1.02, 1.03, 1.04],
        'ask_price': [1.01, 1.02, 1.03, 1.04, 1.05]
    }
    df = pd.DataFrame(data)
    result = calculate_max_profit(df)
    assert result == 0.03  # Buy at 1.01 (first ask), sell at 1.04 (last bid)

def test_no_profit():
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='min'),
        'bid_price': [1.00, 0.99, 0.98, 0.97, 0.96],
        'ask_price': [1.01, 1.00, 0.99, 0.98, 0.97]
    }
    df = pd.DataFrame(data)
    result = calculate_max_profit(df)
    assert result == 0.0  # No profit can be made

def test_single_transaction():
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=2, freq='min'),
        'bid_price': [1.00, 1.04],
        'ask_price': [1.01, 1.05]
    }
    df = pd.DataFrame(data)
    result = calculate_max_profit(df)
    assert result == 0.03  # Buy at 1.01 (first ask), sell at 1.04 (last bid)

def test_multiple_transactions():
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=6, freq='min'),
        'bid_price': [1.00, 1.04, 1.02, 1.03, 1.05, 1.06],
        'ask_price': [1.01, 1.05, 1.03, 1.04, 1.06, 1.07]
    }
    df = pd.DataFrame(data)
    result = calculate_max_profit(df)
    assert result == 0.06  # Buy at 1.01, sell at 1.04, buy at 1.03, sell at 1.06

def test_edge_case_empty_dataframe():
    df = pd.DataFrame(columns=['timestamp', 'bid_price', 'ask_price'])
    result = calculate_max_profit(df)
    assert result == 0.0  # No data to calculate profit

def test_edge_case_single_row():
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=1, freq='min'),
        'bid_price': [1.00],
        'ask_price': [1.01]
    }
    df = pd.DataFrame(data)
    result = calculate_max_profit(df)
    assert result == 0.0  # No profit can be made with a single data point

if __name__ == "__main__":
    pytest.main()