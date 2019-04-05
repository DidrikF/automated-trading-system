import math

from helpers import sign
from strategy import Strategy, Signal
from event import Event
from errors import MarketDataNotAvailableError, BalanceTooLowError

class Portfolio():
    def __init__(self, market_data, balance: float, strategy: Strategy):
        self.balance = balance
        self.strategy = strategy
        self.market_data = market_data # Shared with backtester and broker


        self.order_history = []
        self.portfolio = {} # ticker -> position is the wording used in the project
        self.order_id_seed = 0
        self.signals = []
        self.blotter = []

        self.max_position_size = None

        """
        self.portfolio = {
            "AAPL": { NO
                "fills": [Fill, Fill],
                "position": 12400, # net number of stocks (int), sign gives direction
            },
            "MSFT": {
                "fills": [Fill, Fill],
                "position": 12400, # net number of stocks (int), sign gives direction
            }
            ...
        }
        """

    def generate_signals(self, feature_data_event):
        return self.strategy.generate_signals(feature_data_event)


    def generate_orders_from_signals(self, signals_event):
        """
        Need some serious work...
        """
        signals = signals_event.data
        orders = []
        for signal in signals:
            ticker = signal.ticker
            amount = sign(signal.direction) * signal.certainty * self.max_position_size
            orders.append(Order(order_id=self.get_order_id(), ticker=ticker, amount=amount, signal=signal))

        return Event(event_type="ORDERS", data=orders)


    def simple_order(self, ticker, amount):
        order = Order(order_id=self.get_order_id(), ticker=ticker, amount=amount, signal=Signal.from_nothing()) 
        return order


    def handle_fill_event(self, fills_event):
        """
        Fills tell the portfolio what price the orders was filled at. (one fill per order?)
        The commission, slippage and whatever other information associated with the processing of an order
        is also captured Fill objects. 
        Fill objects are important for the portfolio to track its active positions.
        Knowing exactly how its position changed at at what tiem
        The fill objects are important for tracking portfolio performance, value, etc.
        """
        # I NEED TO READ MORE ABOUT THIS

        fills = fills_event.data
        for fill in fills:
            if fill.ticker not in self.portfolio:
                self.portfolio[fill.ticker] = {
                    "fills": [fill],
                    "amount": fill.amount,
                }
            else:
                self.portfolio[fill.ticker]["fills"].append(fill)
                self.portfolio[fill.ticker]["amount"] += fill.amount


        self.blotter.extend(fills)


    def get_value(self):
        """
        Calculates are returns portfolio value at the end of the current day.
        """
        value = 0
        for ticker, position in self.portfolio.items():
            try:
                daily_data = self.market_data.current_for_ticker(ticker)
            except MarketDataNotAvailableError:
                return None
            price = daily_data["close"]

            value += price * position["amount"]

        return value

    def calculate_daily_return(self):
        """
        Somewhat complicated, becuase I need to take into account the fill price, not the open for both sells and buys.
        """
    
    def calculate_return_over_period(self):
        """
        Complicated.
        Get back to this later
        """

    def charge(self, amount):
        """
        Amount must be a positive value to deduct from the portfolios balance.
        """
        if self.balance >= amount:
            self.balance -= amount
        else:
            raise BalanceTooLowError("Cannot charge portfolio because balance is {} and wanted to charge {}".format(self.balance, amount))


    def charge_commission(self, commission):
        """
        Charge the portfolio the commission.
        To avoid senarios where the portfolios balance is too low to even exit its current positions.
        """
        self.balance -= commission

    def get_order_id(self): # now we can only have orders from one portfolio, maybe introduce a portfolio id
        self.order_id_seed += 1
        order_id = self.order_id_seed
        return order_id




    def set_max_position_size(self, max_size):
        """
        Set a limit on the number of shares and/or dollar value held for the given sid. Limits are treated 
        as absolute values and are enforced at the time that the algo attempts to place an order for sid. 
        This means that it’s possible to end up with more than the max number of shares due to splits/dividends, 
        and more than the max notional due to price improvement.
        If an algorithm attempts to place an order that would result in increasing the absolute value of shares/dollar 
        value exceeding one of these limits, raise a TradingControlException.
        """
        self.max_position_size = max_size



    def set_order_validators(self):
        """
        Set list of validators that must be passed for an order to be valid.
        """
        pass

    def set_do_not_order_list(self):
        pass

    def set_long_only(self):
        pass

    def set_max_leverage(self): # Dont think I will have leverage
        pass

    def set_max_order_count(self):
        pass

    def set_max_order_size(self):
        """
        Set a limit on the number of shares and/or dollar value of any single order placed for sid. 
        Limits are treated as absolute values and are enforced at the time that the algo attempts to place an order for sid.
        If an algorithm attempts to place an order that would result in exceeding one of these limits, 
        raise a TradingControlException.
        """






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
    def __init__(self, order_id, ticker, amount, signal, stop_loss=None, take_profit=None, time_out=None):
        self.id = order_id
        self.ticker = ticker
        self.amount = amount
        self.direction = sign(amount)
        self.signal = signal
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.time_out = time_out
        self.type = "MARKET ORDER"

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





# Don't know if this is necessary
class MarketOrder(Order):
  def __init__(self):
    pass

class LimitStopOrder(Order):
  def __init__(self):
    pass


