from event import Event
from sklearn.base import BaseEstimator
from dateutil.relativedelta import *
from numpy import sign
import numpy as np
import sys
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd

from utils.logger import Logger
from utils.types import Strategy
from utils.utils import Signal
from order import Order
from data_handler import MLFeaturesDataHandler



class RandomStrategy(Strategy):
    def __init__(self, 
        rebalance_weekdays: list, 
        side_classifier: BaseEstimator, 
        certainty_classifier: BaseEstimator,
        ptSl: list,
        feature_handler: MLFeaturesDataHandler,
        features: list,
        initial_margin_requirement: float,
        logger: Logger,
        time_of_day: str="open",
        accepted_signal_age=relativedelta(days=7)
    
    ):
        
        self.rebalance_weekdays = rebalance_weekdays
        self.side_classifier = side_classifier
        self.certainty_classifier = certainty_classifier
        self.ptSl = ptSl
        self.feature_handler = feature_handler
        self.features = features
        self.initial_margin_requirement = initial_margin_requirement
        self.time_of_day = time_of_day
        self.accepted_signal_age = accepted_signal_age 

        self.feature_handler.feature_data["signal_id"] = np.array(range(len(self.feature_handler.feature_data.index)))

        self.order_id_seed = 0

        self.signals_that_have_been_acted_upon = set()

        self.logger = logger

    def set_order_restrictions(self, 
        max_position_size: float, 
        max_positions: int, 
        minimum_balance: float,
        max_percent_to_invest_each_period: float,
        max_orders_per_period: int,
        min_order_size_limit: float,
        num_short_positions: int,
        volume_limit: float=0.1,
    ):
        
        self.max_position_size_percentage = max_position_size # Percentage of total portfolio value
        self.max_positions = max_positions # Order generation will converge towards this number of positions, but not exceed it.
        self.minimum_balance = minimum_balance
        self.max_percent_to_invest_each_period = max_percent_to_invest_each_period
        self.max_orders_per_period = max_orders_per_period
        self.min_order_size_limit = min_order_size_limit
        self.num_short_positions = num_short_positions
        self.volume_limit = volume_limit


    def compute_predictions(self):
        pass


    def generate_signals(self, cur_date, market_data):

        if cur_date.weekday() in self.rebalance_weekdays: # NOTE: Max two weekdays
            signals = []
            feature_data = self.feature_handler.get_range(cur_date-self.accepted_signal_age, cur_date-relativedelta(days=1))
            random_features = self._get_random_features(feature_data, 15) # NOTE: CONTINUE

            for date, features in random_features.iterrows():
                can_trade_res = market_data.can_trade(features["ticker"])
                if (isinstance(can_trade_res, str)) or (can_trade_res != True): continue

                signal = Signal(
                    signal_id=features["signal_id"],
                    ticker=features["ticker"], 
                    direction=1, 
                    certainty=0.51, 
                    ewmstd=features["ewmstd_2y_monthly"], 
                    timeout=features["timeout"],
                    features_date=date
                )
                signals.append(signal)

            return Event(event_type="SIGNALS", data=signals, date=cur_date)
        else:
            return None

    def _get_random_features(self, features, amount):
        amount = min(amount, len(features))
        if amount == 0:
            return features
    

        random_sample = features.sample(n=amount)
        return random_sample

    def generate_orders_from_signals(self, portfolio, signals_event):
        """
        Generate random orders, but subject to the same restrictions as ml_strategy.
        """
        date = signals_event.date
        used_signal_ids = portfolio.broker.blotter.used_signal_ids
        signals = []
        for signal in signals_event.data:
            if signal.signal_id not in used_signal_ids:
                signals.append(signal)

        stocks_with_position_in = set([trade.ticker for trade in portfolio.broker.blotter.active_trades])

        portfolio_value = portfolio.calculate_portfolio_value()
        available_funds = min(
            portfolio_value*self.max_percent_to_invest_each_period,
            portfolio.balance - self.minimum_balance
        )
        # short_funds = portfolio_value * self.percentage_short - portfolio.get_dollar_amount_short()
        # remaining_funds = available_funds - short_funds

        cur_num_trades =  portfolio.get_num_trades()
        if cur_num_trades < (self.max_positions * 0.5):
            num_orders = min(self.max_orders_per_period, self.max_positions - cur_num_trades)
            num_short_orders = 0
        else:
            num_orders = min(self.max_orders_per_period, self.max_positions - cur_num_trades)
            num_short_orders = max(self.num_short_positions - portfolio.get_num_short_trades(), 0)

        top_signals = self._get_top_signals(signals, num_short_orders, num_orders, stocks_with_position_in)
        allocations = self._get_allocation_between_signals(top_signals) 

        orders = []

        for allocation, signal in zip(allocations, top_signals):
            try:
                prev_ticker_data = portfolio.broker.market_data.prev_for_ticker(signal.ticker) # TODO: implement
            except:
                continue # Very bad error handling...
            ticker_data = portfolio.broker.market_data.current_for_ticker(signal.ticker) # we generate order at the open
            ticker_price = ticker_data[self.time_of_day]
            # direction = int(Decimal(signal.direction).quantize(0, ROUND_HALF_UP))

            if signal.direction == 1:
                dollar_amount = min( 
                    portfolio_value * self.max_position_size_percentage,
                    available_funds * allocation
                )
            elif signal.direction == -1: # Short trades require more money as 150% or order size must be put in a margin account
                dollar_amount = min( 
                    (portfolio_value * self.max_position_size_percentage) / (1 + self.initial_margin_requirement),
                    (available_funds * allocation) / (1 + self.initial_margin_requirement)
                )

            number_of_stocks = int(dollar_amount / ticker_price)

            prev_days_volume = prev_ticker_data["volume"]
            number_of_stocks = min(int(prev_days_volume*self.volume_limit), int(dollar_amount / ticker_price)) # TODO: this may free up funds to use for other positions...

            if number_of_stocks == int(prev_days_volume*self.volume_limit):
                self.logger.logr.warning("ML_STRATEGY: Was restricted to order only {} percent of last days volume (vol: {}, price: {}, total: {}) \
                for ticker {} at date {} given signal {}.".format(self.volume_limit*100, prev_days_volume, ticker_price, prev_days_volume*ticker_price, signal.ticker, date, signal.signal_id))

            order_dollar_amount = number_of_stocks * ticker_price
            if order_dollar_amount < self.min_order_size_limit:
                self.logger.logr.warning("ML_STRATEGY: Order generation was aborted because order size was too low for ticker {} on date {} given signal {}.".format(signal.ticker, date, signal.signal_id))
                continue

            order = Order(
                order_id=self.get_order_id(),
                ticker=signal.ticker,
                amount=number_of_stocks * signal.direction,
                date=date,
                signal=signal,
                take_profit=signal.ewmstd * self.ptSl[0],
                stop_loss=signal.ewmstd * self.ptSl[1], # Can be set to stop out early
                timeout=date+relativedelta(months=1), # signal.timeout
            )
            orders.append(order)

            # spent += order_dollar_amount
        if len(orders) == 0:
            return None
        else:
            return Event(event_type="ORDERS", data=orders, date=signals_event.date)
        
        # return orders

    def _get_top_signals(self, signals: list, num_short_orders: int, num_orders: int, stocks_with_position_in: set):
        signals.sort(key=lambda signal: signal.certainty, reverse=True)
    
        top_signals = []
        tickers_in_top_signals = set() # set([top_signal.ticker for top_signal in top_signals])
        num_short_signals_added = 0

        # Add short signals
        for signal in signals:
            if len(top_signals) < num_orders:
                if (signal.direction == -1) and \
                (num_short_signals_added < num_short_orders) and \
                (signal.ticker not in tickers_in_top_signals) and \
                (signal.ticker not in stocks_with_position_in):
                    top_signals.append(signal)
                    tickers_in_top_signals.add(signal.ticker)
                    num_short_signals_added += 1
        
        # Add rest of signals
        for signal in signals:
            if len(top_signals) < num_orders:
                if (signal.ticker not in tickers_in_top_signals) and (signal.ticker not in stocks_with_position_in):
                    top_signals.append(signal)
            
            else:
                break

        return top_signals


    def _get_allocation_between_signals(self, signals: list):
        """
        Returns a list of the percentage of the available balance to allocate to each trade/signal.
        The available funds are allocated according to a weighted average calculation on certainty above 0.5
        """
        certainties = np.array([signal.certainty - 0.5 for signal in signals])
        
        negs = sum(n <= 0 for n in certainties)
        if negs > 0:
            raise ValueError("A signals certainty was less than or equal to 0.5!")

        allocations = [certainty / certainties.sum() for certainty in certainties]

        return allocations


    def handle_event(self, portfolio, event: Event):
        if event.type == "TRADES":
            pass
        elif event.type == "CANCELLED_ORDERS":
            pass
    

    def get_order_id(self):
        self.order_id_seed += 1
        return self.order_id_seed


