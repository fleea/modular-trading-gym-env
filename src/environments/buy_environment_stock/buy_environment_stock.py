from src.environments.buy_environment.buy_environment import BuyEnvironment
from src.interfaces.order_interface import OrderAction

class BuyEnvironmentStock(BuyEnvironment):
    def __init__(self, spread: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spread = spread
    
    def get_current_price(self, order_action: OrderAction) -> float:
        current = self.get_current_data()
        if order_action == OrderAction.OPEN:
            return current.close * (1 + self.spread / 100)
        else:
            return current.close * (1 - self.spread / 100)
