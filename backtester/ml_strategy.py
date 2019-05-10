
#______________________ STRATEGY ___________________________



"""
    NOTE: The strategy must take into account margin requirement when makeing short orders. If not enough on the balance, the order will fail.
    
    event.date.weekday() == self.rebalance_weekday -> to control when signals are generated...

    # NOTE: Maybe if orders are cancelled, the portfolio would like to make another trade instead.

"""
from utils.types import Strategy


class MLStrategy(Strategy):
    def __init__(self):
        self.rebalance_weekday = 0
        pass

    def generate_signals(self):
        """
        
        Only generate signals on days where we rebalance. By controlling emition of signals, one controls 
        invocation of other strategy methods.
        """
        pass

    def generate_orders_from_signals(self): 
        pass

    def get_order_id(self):
        pass

    def get_signal_id(self):
        pass
    
    def handle_event(self):
        pass

