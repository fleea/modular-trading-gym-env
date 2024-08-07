from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OrderType(Enum):
    BUY = 1
    SELL = 2


class OrderAction(Enum):
    OPEN = 1
    CLOSE = 2


@dataclass
class OrderObjectType:
    type: OrderType
    volume: float
    open_price: float
    timestamp: str
    close_price: Optional[float] = None
    profit: Optional[float] = None


__all__ = ["OrderObjectType", "OrderType", "OrderAction"]
