import numpy as np

class Order():
    def __init__(self, order_id, ticker, amount, date, signal, stop_loss=None, take_profit=None, timeout=None):
        self.id = order_id
        self.ticker = ticker
        self.amount = amount
        self.direction = np.sign(amount)
        self.date = date
        self.signal = signal
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.timeout = timeout
        self.type = "MARKET ORDER"

    def __str__(self):
        string_representation = "Order id: {}, ticker: {}, date: {}, direction: {}".format(self.id, self. ticker, self.date, self.direction)
        return string_representation



class CancelledOrder():
    def __init__(self, order: Order, error):
        self.order_id = order.id
        self.ticker = order.ticker
        self.date = order.date

        self.order = order

        self.error = error
