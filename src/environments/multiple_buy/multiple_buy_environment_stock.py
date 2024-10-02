from src.environments.multiple_buy.multiple_buy_environment import MultipleBuyEnvironment
from src.environments.base_environment.base_environment import BaseEnvironment
from src.interfaces.data_interface import StockData
from src.interfaces.order_interface import OrderAction

class MultipleBuyEnvironmentStock(MultipleBuyEnvironment):
    def __init__(self, spread: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spread = spread

    def get_current_price(self, order_action: OrderAction) -> float:
        current = self.get_current_data()
        if order_action == OrderAction.OPEN:
            return current.close * (1 + self.spread / 100)
        else:
            return current.close * (1 - self.spread / 100)