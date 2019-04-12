"""
Maybe just have an abstract base class here, setting the interface that the automated trading system
must adhere to.
"""
from abc import ABC, abstractmethod
import random
from event import Event
from helpers import sign
from portfolio import Order
from errors import MarketDataNotAvailableError
from logger import Logger
from dateutil.relativedelta import *


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


class RandomLongShortStrategy():
    def __init__(self, desc, tickers, amount):
        self.description = desc
        self.amount = amount
        self.tickers = tickers
        self.signal_id_seed = 0

        # Restrictions
        self.max_position_size = None

    def generate_signals(self, feature_data_event: Event):
        signals = []
        possible_directions = [-1, 1]

        for _ in range(self.amount):
            ticker = random.choice(self.tickers)
            direction = random.choice(possible_directions)
            certainty = random.uniform(0, 1)
            signals.append(Signal(signal_id=self.get_signal_id(), ticker=ticker, direction=direction, certainty=certainty, ewmstd=0.15, ptSl=[0.8, -0.8]))

        return Event(event_type="SIGNALS", data=signals, date=feature_data_event.date)


    def generate_orders_from_signals(self, portfolio, signals_event: Event):
        orders = []
        signals = signals_event.data

        for signal in signals:
            ticker = signal.ticker
            amount = sign(signal.direction) * signal.certainty * self.max_position_size
            cur_date = portfolio.market_data.cur_date
            try:
                cur_price = portfolio.market_data.current_for_ticker(ticker)
            except MarketDataNotAvailableError as e:
                Logger.logr.warning("Failed to generate order from signal because market data was not available for ticker {} in {}".format(ticker, cur_date))
                continue
                
            take_profit = cur_price + signal.eqmstd * signal.ptSl[0]
            stop_loss = cur_price - signal.ewmstd * signal.ptSl[1] # may not use the ptSl settings behind the signal
            time_out = cur_date + relativedelta(months=1)
            # order_id, ticker, amount, date, signal, stop_loss=None, take_profit=None, time_out=None
            orders.append(Order(order_id=self.get_order_id(), ticker=ticker, amount=amount, date=self.market_data.cur_date, signal=signal, \
                stop_loss=stop_loss , take_profit=take_profit, time_out=time_out))


        return orders


    # @staticmethod
    def generate_orders_from_signals_real(self, portfolio, signals_event: Event):
        orders = []
        signals = signals_event.data

        cur_portfolio = portfolio.portfolio

        desired_portfolio = self.get_desired_portfolio(cur_portfolio, signals)

        orders = self.generate_orders(cur_portfolio, desired_portfolio)

        return Event(event_type="ORDERS", data=orders, date=signals_event.date) # Orders are 

    def get_desired_portfolio(self, cur_portfolio, signals):
        """
        I need a prioritized list of "moves", then I execute as many as I can under the given restrictions
        """
        # Is this where enforce portfolio restrictiond and make sure I have the balance, margin required?


        desired_portfolio = {}

        return desired_portfolio

    
    def generate_orders(self, cur_portfolio, desired_portfolio):
        """
        Returns ordered list of market orders to send to the broker, that will to the greatest extent move the portfolio 
        to the desired state given the restrictions.
        """
        orders = []

        # Liquidate first to get capital for new orders

        # Then the rest of the orders, don't think the order matters

        # Its very possible to not generate any orders, this should probably happen most of the time.

        return orders


    def set_order_restrictions(self, max_size, max_positions, min_positions, max_orders_per_day, max_orders_per_month, max_hold_period):
        """
        Set a limit on the number of shares and/or dollar value held for the given sid. Limits are treated 
        as absolute values and are enforced at the time that the algo attempts to place an order for sid. 
        This means that it’s possible to end up with more than the max number of shares due to splits/dividends, 
        and more than the max notional due to price improvement.
        If an algorithm attempts to place an order that would result in increasing the absolute value of shares/dollar 
        value exceeding one of these limits, raise a "TradingControlException".
        """
        
        self.max_position_size_percentage = max_size # Percentage of total portfolio value
        
        self.max_positions = max_positions
        self.min_positions = min_positions

        self.max_orders_per_day = max_orders_per_day
        self.max_orders_per_month = max_orders_per_month

        self.max_hold_period = max_hold_period
        
        # max position size together with max orders per time will limit turnaround

    def get_order_id(self):


    def get_signal_id(self):
        self.signal_id_seed += 1
        return self.signal_id_seed



class BuyAppleStrategy():

    def __init__(self, desc):
        self.description = desc
        self.signal_id_seed = 0

    def generate_signals(self, feature_data_event: Event):
        signals = []
        
        # make predictions based on the features in $feature_data_event
        signals.append(Signal(signal_id=self.get_signal_id(), ticker="AAPL", direction=1, certainty=1, ewmstd=None, ptSl=None))
        # signals.append(Signal(signal_id=self.get_signal_id(), ticker="FCX", direction=1, certainty=0.5, , ewmstd=None, ptSl=None))

        return Event(event_type="SIGNALS", data=signals, date=feature_data_event.date)

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

        
        # self.order_history.extend(orders) # What does this do here?

        return Event(event_type="ORDERS", data=orders)


    def get_order_id(self):


    def get_signal_id(self):
        self.signal_id_seed += 1
        return self.signal_id_seed
    
    def get_order_id(self): # now we can only have orders from one portfolio, maybe introduce a portfolio id
        self.order_id_seed += 1
        order_id = self.order_id_seed
        return order_id



class Signal():
    def __init__(self, signal_id, ticker, direction, certainty, ewmstd, ptSl):
        self.signal_id = signal_id
        self.ticker = ticker
        self.direction = direction
        self.certainty = certainty

        self.ewmstd = ewmstd # The variablity measure behind the barriers, may also be relevant
        self.ptSl = ptSl
        # I dont really have barriers for new predictions, but I have the ewmstd and ptSl which was used to generate barriers for training.
        # self.barriers = (None, None, None) # Relevant for calculations of stop-loss, take-profit and also relevent for rebalancing

        self.feature_data_index = None # Need to take this as arguments, makes it easier to track the data behind signals
        self.feature_data_date = None # Need to take this as arguments, makes it easier to track the data behind signals


    @classmethod
    def from_nothing(cls):
        return cls("NONE", "NONE", "NONE", "NONE", "NONE")