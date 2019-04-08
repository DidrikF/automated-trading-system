import math
import pandas as pd
import copy

from helpers import sign
from strategy import Strategy, Signal
from event import Event
from errors import MarketDataNotAvailableError, BalanceTooLowError

"""
For now only long positions are possible.
"""

class Portfolio():
    def __init__(self, market_data, balance: float, strategy: Strategy):
        self.balance = balance
        self.strategy = strategy
        self.market_data = market_data # Shared with backtester and broker


        self.order_history = [] # All orders ever created (Not all gets sent to the broker necessarily (but most will I envision, why not all?))
        self.portfolio_history = [] # To be able to play through how the portfolio developed over time.
        self.cancelled_orders = [] # Cancelled orders

        self.commissions_charged = pd.DataFrame(index=self.market_data.date_index_to_iterate) # Not implemented
        self.commissions_charged["amount"] = 0
        self.slippage_suffered = pd.DataFrame(index=self.market_data.date_index_to_iterate)
        self.slippage_suffered["amount"] = 0
        # Can the above just be merged with the portfolio_value dataframe?
        self.portfolio_value = pd.DataFrame(index=self.market_data.date_index_to_iterate) # This is the basis for portfolio return calculations
        
        self.portfolio_value["balance"] = pd.NaT
        self.portfolio_value["market_value"] = pd.NaT


        # Active positions: (rename?)
        self.portfolio = {} # ticker -> position is the wording used in the project
            # In order to support long and short positions this might have to be extended some, but not sure about this.
        
        self.signals = [] # All signals ever produced
        self.blotter = [] # Fills

        self.order_id_seed = 0
        
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
        signals_event =  self.strategy.generate_signals(feature_data_event)

        self.signals.extend(signals_event.data)

        return signals_event

    def generate_orders_from_signals(self, signals_event):
        """
        This is complicated.
        I need to see the signals in relation to the expectation the portfolio has for its current possitions
        Then based on the restrictions and new signals need to decide what to buy/sell (if any) and whether 
        to liquidate some portion of its position. 
        Here the portfolio also needs to calculate stop-loss, take-profit and set timeout of the orders.

        Orders must respect various limitations... These must be taken into account when generating orders. validate_orders is not really needed
        if orders are made with the restrictions in mind.
        """
        signals = signals_event.data
        orders = []
        for signal in signals:
            ticker = signal.ticker
            amount = sign(signal.direction) * signal.certainty * self.max_position_size
            orders.append(Order(order_id=self.get_order_id(), ticker=ticker, amount=amount, date=self.market_data.cur_date, signal=signal))

        
        self.order_history.extend(orders)

        return Event(event_type="ORDERS", data=orders)


    def simple_order(self, ticker, amount):
        """ Return order for $amount number of shares of $ticker's stock. (Does not create orders event, in which case it should be added to order_history) """
        order = Order(order_id=self.get_order_id(), ticker=ticker, amount=amount, date=self.market_data.cur_date, signal=Signal.from_nothing()) 
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
            self.commissions_charged.at[fill.date, "amount"] += fill.commission
            self.slippage_suffered.at[fill.date, "amount"] += self.calculate_slippage(fill)

            # Update portfolio (active positions)
            if fill.ticker not in self.portfolio:
                self.portfolio[fill.ticker] = {
                    # "fills": [fill], tracking the fill here is redundant, I have the blotter...
                    # "close": close, # Usefull for calculating portfolio asset allocation
                    "amount": fill.amount,
                }
            else:
                # self.portfolio[fill.ticker]["fills"].append(fill)
                # self.portfolio[fill.ticker]["close"] = close
                self.portfolio[fill.ticker]["amount"] += fill.amount


        self.blotter.extend(fills)


    def handle_cancelled_orders_event(self, event):
        cancelled_orders = event.data
        for cancelled_order in cancelled_orders:
            # Do whatever with the cancelled order. Don't know what...
            pass

        self.cancelled_orders.extend(cancelled_orders)

    def handle_position_liquidations_event(self, event):
        """
        A position liquidation will conclude a trade and the balance must be updated and effects on portfolio
        return must be dealt with (this may be as simple as capturing the market value of the portfolio at the close)
        """
        pass

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

    def validate_orders(self):
        """ This is better to have here, the broker does not need to know about the portfolios restrictions"""
        pass

    """The below methods are used to capture the state of the portfolio at the end of the backtest"""

    
    def calculate_daily_return(self):
        """
        MAYBE NOT NEEDED
        Somewhat complicated, becuase I need to take into account the fill price, not the open for both sells and buys.

        THIS IS WHERE ALL THE COMPLEXITY "SHOULD" LIE

        Portfolio return is a weighted return of the individual assets in the portfolio's return.
        With this in mind it becomes easy, because you only need to check what amount of the asset
        you purchased today (fill price)

        note that you may also increase a current position!

        weighting : (number of stocks * price of stocks) / total portfolio value (what time to get the price? start of day? end of day?)
        return for each stock bought today = (todays close / fill price) - 1 * 
        return for each stock not bought today = (yesterdays close / todays close) - 1 

        sum(weight * return for stocks)

        """
    
    def calculate_return_over_period(self, start_date, end_date):
        """
        Complicated.
        Get back to this later

        Once daily returns are calculated precicely, is simply becomes a matter of multiplying
        the daily returns over the period together (1+return!)
        """
        if start_date < self.market_data.start or (end_date > self.market_data.cur_date):
            raise ValueError("Attempted to calculate return over period where the start date was lower\
                 than backtest's start date or the end date was higher than the current date.")

        start_row = self.portfolio_value[start_date]
        end_row = self.portfolio_value[end_date]

        start_val = start_row["balance"] + start_row["market_value"]
        end_val = end_row["balance"] + end_row["market_value"]

        return (end_val / start_val) - 1


    def calculate_market_value(self):
        """
        Calculate market value of portfolio (only support long positions atm)
        """
        market_value = 0

        for ticker in self.portfolio:
            position = self.portfolio[ticker]
            try:
                ticker_close = self.market_data.current_for_ticker(ticker)["close"]
            except:
                ticker_close = self.market_data.last_for_ticker(ticker)["close"]
            
            ticker_amount = position["amount"] # Implies direction of position
            
            market_value += ticker_close * ticker_amount

        return market_value


    def calculate_total_commission(self):
        """ Return total commission charged from fill objects """
        total_commission = 0
        for data, row in self.commissions_charged.iterrows():
            total_commission += row["amount"]

        return total_commission

    def calculate_total_slippage(self):
        """ Return total slippage suffered from fill objects. """
        total_slippage = 0
        for date, row in self.slippage_suffered.iterrows():
            total_commission += row["amount"]

        return total_slippage

    def calculate_slippage(self, fill):
        # fill.slippage: price difference between open and fill (can be negative and positive)
        # fill.amount: number of stocks bought or sold

        # positive slippage when buying is a cost, negative slippage when buying is profit
        # dollar slippage per stock * number of stocks bought/sold (+/-) = slippage cost/reward (desired sign)
        # -10 * -10 = 100 (-)
        # -10 * 10 = -100 (+)
        # 10 * 10 = 100 (-)
        # 10 * -10 = (+)
        # Conclusion: Sign must be inverted when multiplying slippage and amount
        dollar_slippage = (fill.slippage * fill.amount) * -1
        return dollar_slippage

    def capture_portfolio_state(self):
        """
        Capture the state of the portfolio and append it to portfolio_history
        """
        positions = copy.deepcopy(self.portfolio)

        portfolio = {}

        for ticker in positions:
            try:
               close = self.market_data.current_for_ticker(ticker)["close"]
            except:
                close = math.nan

            positions[ticker]["close"] = close
            positions[ticker]["value"] = close * positions[ticker]["amount"] # OBS: This does not support short positions!

        positions["Cash"] = {
            "value": self.balance,
            "amount": math.nan,
            "close": math.nan,
        }

        portfolio["positions"] = positions
        portfolio["cur_date"] = self.market_data.cur_date
        portfolio["cur_balance"] = self.balance # Would be nice to see this on the portfolio allocation chart as well
        cur_market_value = self.calculate_market_value()
        portfolio["cur_market_value"] = cur_market_value

        # print(portfolio) # this object is correct...

        self.portfolio_history.append(portfolio)

        # This is used in return calculations and graphing portfolio value and portfolio return. 
        # Not really related to the portfolio_history list
        self.portfolio_value.loc[self.market_data.cur_date, "balance"] = self.balance
        self.portfolio_value.loc[self.market_data.cur_date, "market_value"] = cur_market_value # This does not support short positions.



    def order_history_to_df(self):
        df = pd.DataFrame(columns=["order_id", "date", "ticker", "amount", "direction", "stop_loss", "take_profit", "time_out", "type", "signal_id"]) # SET COLUMNS, so sorting and set index does not fail
        for index, order in enumerate(self.order_history):
            df.at[index, "order_id"] = order.id
            df.at[index, "date"] = order.date
            df.at[index, "ticker"] = order.ticker
            df.at[index, "amount"] = order.amount
            df.at[index, "direction"] = order.direction
            df.at[index, "stop_loss"] = order.stop_loss
            df.at[index, "take_profit"] = order.take_profit
            df.at[index, "time_out"] = order.time_out
            df.at[index, "type"] = order.type
            df.at[index, "signal_id"] = order.signal.signal_id

        df.set_index("order_id")
        df = df.sort_index()

        return df


    def portfolio_history_to_df(self):
        """
        What i need to record is how positoisn and allocations to diffrent positons changed over time...
        "2012-10-04"   "ticker" "amount"  "close" "value"
                        "AAPL":  12          40      480
                        "MSFT":  12          40      480
                        # "cur_balalnce":             5000 # get this from portfolio_value
                        # "cur_market_value":         6500 # get this from portfolio_value
        "2012-10-05"
                        "STAT":
                        "AAPL":
        """
        df = pd.DataFrame(columns=["date", "ticker", "amount", "close", "value"]) # index=self.market_data.date_index_to_iterate
        # df["date"] = self.market_data.date_index_to_iterate
        # df = df.set_index(["date", "ticker"])
        for portfolio in self.portfolio_history:
            port_df = pd.DataFrame()
            date = portfolio["cur_date"]
            # df.at[date, "cur_balance"] = portfolio["cur_balance"] # But I have this in portfolio_value
            # df.at[date, "cur_market_value"] = portfolio["cur_market_value"] # But I have this in portfolio_value
            #  print(portfolio["positions"])
            # port_df.at[date, "cur_balance"] = portfolio["cur_balance"]
            for index, ticker in enumerate(portfolio["positions"]):
                position = portfolio["positions"][ticker]
                port_df.at[index, "date"] = date
                port_df.at[index, "ticker"] = ticker
                port_df.at[index, "amount"] = position["amount"]
                port_df.at[index, "close"] = position["close"]
                port_df.at[index, "value"] = position["value"]

            df = df.append(port_df)

        df = df.set_index(["date", "ticker"])
        df = df.sort_index()

        return df

    def cancelled_orders_to_df(self):
        df = pd.DataFrame(columns=["order_id", "date", "ticker", "amount", "error"])
        for index, c_ord in enumerate(self.cancelled_orders):
            df.at[index, "order_id"] = c_ord.id
            df.at[index, "date"] = c_ord.date
            df.at[index, "ticker"] = c_ord.ticker
            df.at[index, "amount"] = c_ord.order.amount
            df.at[index, "error"] = str(c_ord.error)
            # Might want to extend this, don't know atm

        df = df.set_index("order_id")
        df = df.sort_index()

        return df

    def signals_to_df(self):
        df = pd.DataFrame(columns=["signal_id", "ticker", "direction", "certainty", "ewastd", "upper_barrier", "lower_barrier", "vertical_barrier", "feature_data_index", "feature_data_date"])
        for index, signal in enumerate(self.signals):
            df.at[index, "signal_id"] = signal.signal_id
            df.at[index, "ticker"] = signal.ticker
            df.at[index, "direction"] = signal.direction
            df.at[index, "certainty"] = signal.certainty
            df.at[index, "ewastd"] = signal.ewastd
            df.at[index, "upper_barrier"] = signal.barriers[0]
            df.at[index, "lower_barrier"] = signal.barriers[1]
            df.at[index, "vertical_barrier"] = signal.barriers[2]
            df.at[index, "feature_data_index"] = signal.feature_data_index
            df.at[index, "feature_data_date"] = signal.feature_data_date

        df = df.set_index("signal_id")
        df = df.sort_index()

        return df


    def blotter_to_df(self):
        df = pd.DataFrame(columns=["order_id", "ticker", "direction", "amount", "date", "price", "commission", "slippage"])

        for index, fill in enumerate(self.blotter):
            df.at[index, "order_id"] = fill.order_id
            df.at[index, "ticker"] = fill.ticker
            df.at[index, "direction"] = fill.direction
            df.at[index, "amount"] = fill.amount
            df.at[index, "date"] = fill.date
            df.at[index, "price"] = fill.price
            df.at[index, "commission"] = fill.commission
            df.at[index, "slippage"] = fill.slippage
            # May want more of the order properties here, but not sure
        
        df = df.set_index("order_id")
        df = df.sort_index() # This may not be the order things are executed? (or will it be? I decide the order the orders are processed, so I guess, the order should be filled in order.)

        return df



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
    def __init__(self, order_id, ticker, amount, date, signal, stop_loss=None, take_profit=None, time_out=None):
        self.id = order_id
        self.ticker = ticker
        self.amount = amount
        self.direction = sign(amount)
        self.date = date
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


