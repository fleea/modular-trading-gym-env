from enum import Enum, auto


class OrderClosingStrategy(Enum):
    FIFO = auto()
    LIFO = auto()
    MOST_PROFITABLE = auto()
    LEAST_PROFITABLE = auto()
    LARGEST_POSITION = auto()
    SPECIFIC_ASSET = auto()
    TIME_BASED = auto()
    STOP_LOSS_TAKE_PROFIT = auto()
    RISK_REWARD_RATIO = auto()
    RANDOM = auto()

    def describe(self):
        descriptions = {
            OrderClosingStrategy.FIFO: "Close the oldest open order first.",
            OrderClosingStrategy.LIFO: "Close the most recently opened order first.",
            OrderClosingStrategy.MOST_PROFITABLE: "Close the order with the highest current profit.",
            OrderClosingStrategy.LEAST_PROFITABLE: "Close the order with the lowest profit or highest loss.",
            OrderClosingStrategy.LARGEST_POSITION: "Close the order with the largest position size.",
            OrderClosingStrategy.SPECIFIC_ASSET: "Close orders for a specific asset based on market conditions.",
            OrderClosingStrategy.TIME_BASED: "Close orders that have been open for a certain duration.",
            OrderClosingStrategy.STOP_LOSS_TAKE_PROFIT: "Close orders when stop-loss or take-profit levels are hit.",
            OrderClosingStrategy.RISK_REWARD_RATIO: "Close orders based on a predefined risk-reward ratio.",
            OrderClosingStrategy.RANDOM: "Randomly select an order to close.",
        }
        return descriptions[self]
