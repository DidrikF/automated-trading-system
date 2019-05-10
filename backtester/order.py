import numpy as np

class Order(): # what is best, have long arg list or a class hierarchy
    """
    Parameters:	
    asset (Asset) – The asset that this order is for.
    amount (int) – The amount of shares to order. If amount is positive, this is the number of shares to buy or cover. If amount is negative, this is the number of shares to sell or short.
    limit_price (float, optional) – The limit price for the order.
    stop_price (float, optional) – The stop price for the order.
    style (ExecutionStyle, optional) – The execution style for the order.
    Returns:	
    order_id – The unique identifier for this order, or None if no order was placed.
    """
    def __init__(self, order_id, ticker, amount, date, signal, stop_loss=None, take_profit=None, timeout=None):
        self.id = order_id
        self.ticker = ticker
        self.amount = amount
        self.direction = np.sign(amount)
        self.date = date
        self.signal = signal
        self.stop_loss = stop_loss # This needs to be a price that I can compare the open/close to
        self.take_profit = take_profit # This needs to be a price that I can compare the open/close to
        self.timeout = timeout
        self.type = "MARKET ORDER"

    def __str__(self):
        string_representation = "Order id: {}, ticker: {}, date: {}, direction: {}".format(self.id, self. ticker, self.date, self.direction)
        return string_representation

    @classmethod
    def order_value(cls, ticker, value):
        pass

    @classmethod
    def order_percent(cls, ticker, value):
        pass

    def get_limit_price(self):
        pass

    def get_stop_price(self):
        pass

    # Order target methods, probably not needed


class CancelledOrder():
    def __init__(self, order: Order, error):
        self.order_id = order.id
        self.ticker = order.ticker
        self.date = order.date

        self.order = order

        self.error = error



# Don't know if this is necessary
class MarketOrder(Order):
  def __init__(self):
    pass

class LimitStopOrder(Order):
  def __init__(self):
    pass
