from dataclasses import dataclass


@dataclass
class TickData:
    ask_price: float  # purchase price, on open position
    bid_price: float  # sell asset to market, on close position
    timestamp: str

@dataclass
class StockData:
    high: float
    low: float
    open: float
    close: float
    volume: float
    timestamp: str

__all__ = ["TickData", "StockData"]
