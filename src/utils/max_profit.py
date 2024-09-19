import pandas as pd


def calculate_max_profit(df: pd.DataFrame) -> float:
    if "bid_price" not in df.columns or "ask_price" not in df.columns:
        raise ValueError("DataFrame must contain 'bid_price' and 'ask_price' columns")

    total_profit = 0
    n = len(df)
    buy_price = float('inf')

    for i in range(n):
        current_ask = df.at[i, 'ask_price']
        current_bid = df.at[i, 'bid_price']

        # If we haven't bought yet or if we can buy at a lower price
        if current_ask < buy_price:
            buy_price = current_ask

        # If we can sell for a profit
        elif current_bid > buy_price:
            # Check if there's a lower ask price in the future
            future_min_ask = min(df['ask_price'][i+1:], default=float('inf'))
            
            # If current bid is higher than future minimum ask, sell now
            if current_bid > future_min_ask:
                total_profit += current_bid - buy_price
                buy_price = float('inf')  # Reset buy_price for next purchase

    return total_profit