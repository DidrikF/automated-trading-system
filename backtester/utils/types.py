from abc import ABC, abstractmethod

class Strategy(ABC):
    def generate_signals(self):
        pass

    # @staticmethod
    def generate_orders_from_signals(self):
        pass

    def get_order_id(self):
        pass

    def get_signal_id(self):
        pass

    def handle_event(self):
        pass



class CommissionModel(ABC):
    def __init__(self):
        pass
    def calculate(self, amount: int, price: float):
        pass



class SlippageModel(ABC):
    def __init__(self):
        pass

    def calculate(self):
        pass


class BrokerBase(ABC):
    def __init__(self):
        pass

    def process_orders(self, portfolio, orders_event):
        pass

    def manage_active_trades(self, portfolio):
        pass

class PortfolioBase(ABC):
    def __init__(self):
        pass
    def generate_signals(self):
        pass
    def generate_orders_from_signals(self):
        pass
    def handle_trades_event(self):
        pass
    def handle_cancelled_orders_event(self):
        pass
    def handle_position_liquidations_event(self):
        pass
    def handle_margin_account_update(self):
        pass
    def increase_margin_account(self):
        pass
    def charge(self):
        pass
    def charge_commission(self):
        pass
    def charge_margin_interest(self):
        pass


    
