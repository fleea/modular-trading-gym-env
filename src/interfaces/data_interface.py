from dataclasses import dataclass


@dataclass
class TickData:
    ask_price: float  # purchase price, on open position
    bid_price: float  # sell asset to market, on close position
    timestamp: str


__all__ = ["TickData"]
