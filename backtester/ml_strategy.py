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

"""
    NOTE: The strategy must take into account margin requirement when makeing short orders. If not enough on the balance, the order will fail.
    
    event.date.weekday() == self.rebalance_weekday -> to control when signals are generated...

    # NOTE: Maybe if orders are cancelled, the portfolio would like to make another trade instead.


    # Turnaround vs costs 

    # make sure to have sufficient balance to exploit big opportunities


"""


class MLStrategy(Strategy):
    def __init__(self, 
        rebalance_weekday: int, 
        side_classifier: BaseEstimator, 
        certainty_classifier: BaseEstimator,
        ptSl: list,
        feature_handler: MLFeaturesDataHandler,
        features: list,
        initial_margin_requirement: float,
        log_path: str,
        time_of_day: str="open",
        accepted_signal_age=relativedelta(days=7)
    
    ):
        
        self.rebalance_weekday = 0
        self.side_classifier = side_classifier
        self.certainty_classifier = certainty_classifier
        self.ptSl = ptSl
        self.feature_handler = feature_handler
        self.features = features
        self.initial_margin_requirement = initial_margin_requirement
        self.time_of_day = time_of_day
        self.accepted_signal_age = accepted_signal_age 

        self.signal_id_seed = 0
        self.order_id_seed = 0

        self.logger = Logger("ML_STRATEGY", log_path + "/ml_strategy.log")

    def set_order_restrictions(self, 
        max_position_size: float, 
        max_positions: int, 
        minimum_balance: float,
        max_percent_to_invest_each_period: float,
        max_orders_per_period: int,
        min_order_size_limit: float,
        percentage_short: float,
        # max_orders_per_day: int, 
        # max_orders_per_month: int, 
        # max_hold_period: int # NOTE: set by the timeout...
    ):
        """
        Set a limit on the number of shares and/or dollar value held for the given sid. Limits are treated 
        as absolute values and are enforced at the time that the algo attempts to place an order for sid. 
        This means that it’s possible to end up with more than the max number of shares due to splits/dividends, 
        and more than the max notional due to price improvement.
        If an algorithm attempts to place an order that would result in increasing the absolute value of shares/dollar 
        value exceeding one of these limits, raise a "TradingControlException".
        """
        
        self.max_position_size_percentage = max_position_size # Percentage of total portfolio value
        self.max_positions = max_positions # Order generation will converge towards this number of positions, but not exceed it.
        self.minimum_balance = minimum_balance
        self.max_percent_to_invest_each_period = max_percent_to_invest_each_period
        self.max_orders_per_period = max_orders_per_period
        self.min_order_size_limit = min_order_size_limit
        self.percentage_short = percentage_short
        # self.max_orders_per_month = max_orders_per_month
        # self.max_hold_period = max_hold_period
    
        # max position size together with max orders per time will limit turnaround

    def compute_predictions(self):
        self.feature_handler.feature_data["side_prediction"] = self.side_classifier.predict(self.feature_handler.feature_data[self.features])
        # print(pd.Series(self.side_classifier.predict(self.feature_handler.feature_data[self.features])).value_counts())
        # sys.exit()
        self.feature_handler.feature_data["certainty_prediction"] = self.certainty_classifier.predict_proba(self.feature_handler.feature_data[self.features])[:,1]

        """
        # print(features[self.features])
        # side_prediction = self.side_classifier.predict_proba(np.array(features[self.features]).reshape(1, -1))
        side_prediction = self.side_classifier.predict(np.array(features[self.features]).reshape(1, -1))[-1] # (-1, 1) labels
        # print("Side_prediciton: ", side_prediction)
        certainty_prediction = self.certainty_classifier.predict_proba(np.array(features[self.features]).reshape(1, -1))[-1][1] # (0, 1) labels 
        # certainty_prediction2 = self.certainty_classifier.predict(np.array(features[self.features]).reshape(1, -1))
        # print("certainty_prediction", certainty_prediction)
        """

    def compute_ml_model_statistics(self):

        # Calculate precision, recall and accuracy for both models over the backtested period
        # test_x_pred = side_classifier.predict(certainty_test_x)
        # accuracy = accuracy_score(test_y, test_x_pred)
        # precision = precision_score(test_y, test_x_pred)
        # recall = recall_score(test_y, test_x_pred)
        pass

    def generate_signals(self, cur_date):
        """
        Only generate signals on days where we rebalance. By controlling emition of signals, one controls 
        invocation of other strategy methods.
        """
        if cur_date.weekday() == self.rebalance_weekday:
            feature_data = self.feature_handler.get_range(cur_date-self.accepted_signal_age, cur_date-relativedelta(days=1))
            # print("Feature data received: ", [features["ticker"] + " "+ str(date) for date, features in feature_data.iterrows()])
            
            signals = []

            # NOTE: Need to make faster
            # predict ahead of time
            # order feature_data
            # generate only signals for the top n-number of signlas (twice the number of allowed orders f eks)
            feature_data = feature_data.sort_values(by=["certainty_prediction"], ascending=False)
            
            # feature_data_best = feature_data[:self.max_orders_per_period*2]
            feature_data_short  = feature_data.loc[feature_data.direction == -1]
            feature_data_short = feature_data_short.iloc[:self.max_orders_per_period]
            feature_data_long  = feature_data.loc[feature_data.direction == 1]  
            feature_data_long = feature_data_long.iloc[:self.max_orders_per_period]

            signals = {
                "best": [], # NOTE: Not implemented
                "short": [],
                "long": [],
            }
            for date, features in feature_data_short.iterrows():
                signal = Signal(
                    signal_id=self.get_signal_id(), 
                    ticker=features["ticker"], 
                    direction=features["side_prediction"], 
                    certainty=features["certainty_prediction"], 
                    ewmstd=features["ewmstd_2y_monthly"], 
                    timeout=features["timeout"],
                    features_date=date
                )
                signals["short"].append(signal)

            for date, features in feature_data_long.iterrows():
                signal = Signal(
                    signal_id=self.get_signal_id(), 
                    ticker=features["ticker"], 
                    direction=features["side_prediction"], 
                    certainty=features["certainty_prediction"], 
                    ewmstd=features["ewmstd_2y_monthly"], 
                    timeout=features["timeout"],
                    features_date=date
                )
                signals["long"].append(signal)


            return Event(event_type="SIGNALS", data=signals, date=cur_date)
        else:
            return None


    def generate_orders_from_signals(self, portfolio, signals_event: Event): 
        """
        All signals are relevent here, because they adhere to both the rebalance date and age restriction.
        
        Description of algorithm:
        At each reabalance try to make trades that moves the portfolio as close as possible to the desired portfolio.
        The desired portfolio is based on some simple diversification principles without any advanced techniques:

        X Position in 30 Stocks at the same time is the goal
        X Can hold long and short positions concurrently, but cannot go long and short the same time
        No more than 5% of the total value of the portfolio in a single stock
        You can invest in a maximum of 10 stocks each week
        You can only invest 33% of total portfolio value each week
        Must maintain the minimum balance

        # How to deal with short trades and the extra requirement for the margin account...
        """
        
        signals = signals_event.data

        long_signals = signals["long"]
        short_signals = signals["short"]

        if self.percentage_short != 0 or self.percentage_short is not None:
            orders = self._generate_long_short_orders(portfolio, short_signals, long_signals, signals_event.date)
        else:
            signals = short_signals.copy()
            signals.extend(long_signals)
            signals.sort(key=lambda signal: signal.certainty, reverse=True)
            best_signals = signals[:self.max_orders_per_period]
            orders = self._generate_best_orders(portfolio, best_signals, signals_event.date)
    
        if len(orders) == 0:
            return None
        else:
            return Event(event_type="ORDERS", data=orders, date=signals_event.date)


    def _generate_best_orders(self, portfolio, signals: list, date: pd.datetime):
        """
        Generate the best orders irrespective of allocation to long or short positions.
        # TODO: make it aware of what it has ordered when generating new orders.
        # TODO: make it go short 25% of the time... but make this optional...
        """
        stocks_with_position_in = set([trade.ticker for trade in portfolio.broker.blotter.active_trades])
        number_of_orders_to_generate = min(self.max_orders_per_period, self.max_positions - len(stocks_with_position_in)) # Will converge the number of positions towards self.max_positions
        
        top_signals = self._get_top_signals(signals, number_of_orders_to_generate, list(stocks_with_position_in)) # Only one signal per ticker
        allocations = self._get_allocation_between_signals(top_signals)

        portfolio_value = portfolio.calculate_portfolio_value()
        # Available funds for going both long and short, but short "costs" more
        available_funds = min(
            portfolio_value*self.max_percent_to_invest_each_period, # Max 33% of port value can be invested each week
            portfolio.balance - self.minimum_balance
        )        
        orders = []
        
        # spent = 0

        for allocation, signal in zip(allocations, top_signals):
            prev_ticker_data = portfolio.broker.market_data.prev_for_ticker(signal.ticker) # TODO: implement
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
            number_of_stocks = min(prev_days_volume, number_of_stocks) # TODO: this may free up funds to use for other positions...

            order_dollar_amount = number_of_stocks * ticker_price
            if order_dollar_amount < self.min_order_size_limit:
                continue

            order = Order(
                order_id=self.get_order_id(),
                ticker=signal.ticker,
                amount=number_of_stocks * signal.direction,
                date=date,
                signal=signal,
                take_profit=signal.ewmstd * self.ptSl[0],
                stop_loss=signal.ewmstd * self.ptSl[1], # Can be set to stop out early
                timeout=signal.timeout,
            )
            orders.append(order)

            # spent += order_dollar_amount
        return orders
        
    def _generate_long_short_orders(self, portfolio, short_signals, long_signals, date):
        """
        Generate orders such that a minimum percentage of the available capital is allocated to short trades.
        The method does not take into consideration how large the current short position is.
        # TODO: make it aware of what it has ordered when generating new orders.
        # TODO: make it go short 25% of the time... but make this optional...
        # NOTE: you don't have to consider cases where multiple trades happen in the same stock, because of the timeout on trades.
        """
        stocks_with_position_in = set([trade.ticker for trade in portfolio.broker.blotter.active_trades])
        number_of_orders_to_generate = min(self.max_orders_per_period, self.max_positions - len(stocks_with_position_in)) # Will converge the number of positions towards self.max_positions
        
        # Number of shorts and longs
        number_of_short = int(number_of_orders_to_generate*self.percentage_short)
        number_of_long = number_of_orders_to_generate - number_of_short

        # Get top short and long signals
        top_short_signals = self._get_top_signals(short_signals, number_of_short)
        top_long_signals = self._get_top_signals(short_signals, number_of_long)

        # Get allocations for long and short signals
        short_allocations = self._get_allocation_between_signals(top_short_signals)
        long_allocations = self._get_allocation_between_signals(top_long_signals)

        allocations = short_allocations.copy()
        allocations.extend(long_allocations)
        top_signals = short_signals.copy()
        top_signals.extend(long_signals)

        portfolio_value = portfolio.calculate_portfolio_value()
        # Available funds for going both long and short, but short "costs" more
        available_funds = min(
            portfolio_value*self.max_percent_to_invest_each_period, # Max 33% of port value can be invested each week
            portfolio.balance - self.minimum_balance
        )
        available_funds_for_short = available_funds * self.percentage_short
        available_funds_for_long = available_funds - available_funds_for_short

        orders = []
        # Generate orders for long and short signals
        for allocation, signal in zip(allocations, top_signals):
            ticker_data = portfolio.broker.market_data.current_for_ticker(signal.ticker) # we generate order at the open
            ticker_price = ticker_data[self.time_of_day]
            # direction = int(Decimal(signal.direction).quantize(0, ROUND_HALF_UP))

            if signal.direction == 1:
                dollar_amount = min( 
                    portfolio_value * self.max_position_size_percentage,
                    available_funds_for_long * allocation
                )
            elif signal.direction == -1: # Short trades require more money as 150% or order size must be put in a margin account
                dollar_amount = min( 
                    (portfolio_value * self.max_position_size_percentage) / (1 + self.initial_margin_requirement),
                    (available_funds_for_short * allocation) / (1 + self.initial_margin_requirement)
                )

            prev_ticker_data = portfolio.broker.market_data.prev_for_ticker(signal.ticker) # TODO: implement
            prev_days_volume = prev_ticker_data["volume"]
            number_of_stocks = min(int(prev_days_volume*self.volume_limit), int(dollar_amount / ticker_price)) # TODO: this may free up funds to use for other positions...
            
            if number_of_stocks == int(prev_days_volume*self.volume_limit):
                self.logger.logr.warning("Was restricted to order only {} percent of last days volume for ticker {} at date {} given signal {}.".format(self.volume_limit*100, signal.ticker, date, signal.signal_id))

            order_dollar_amount = number_of_stocks * ticker_price
            if order_dollar_amount < self.min_order_size_limit:
                self.logger.logr.warning("Order generation was aborted because order size was too low for ticker {} on date {} given signal {}.".format(signal.ticker, date, signal.signal_id))
                continue

            order = Order(
                order_id=self.get_order_id(),
                ticker=signal.ticker,
                amount=number_of_stocks * signal.direction,
                date=date,
                signal=signal,
                take_profit=signal.ewmstd * self.ptSl[0],
                stop_loss=signal.ewmstd * self.ptSl[1], # Can be set to stop out early
                timeout=signal.timeout,
            )
            orders.append(order)

        # NOTE: We may not have used all available capital if some restriction was met, see if this is a problem

        return orders

    def _get_top_signals(self, signals: list, amount: int, current_positions_tickers: list):
        """
        returns a list of the top signals for the &amount number of stocks that are not in &current_positions_tickers.
        Priority is based purely on the certainty prediction.
        
        NOTE:
        Only return signals for stocks that have no current position!
        This limitation makes order generation easier, because I don't have to take into account
        the current position in a stock when calculating how to size the order.
        This limitation also has no real consequence on the systems ability extract usefull signals
        from the ML models, because signals are only updated once a month (ish) and the timeout
        of trades is also 1 month.
        """
        signals.sort(key=lambda signal: signal.certainty, reverse=True)
        top_signals = []
        for signal in signals:
            tickers_in_top_signals = set([top_signal.ticker for top_signal in top_signals])

            if len(tickers_in_top_signals) < amount:
                if (signal.ticker not in tickers_in_top_signals) and (signal.ticker not in current_positions_tickers):
                    top_signals.append(signal)
            elif len(tickers_in_top_signals) >= amount:
                break

        return top_signals

    def _get_allocation_between_signals(self, signals: list):
        """
        Returns a list of the percentage of the available balance to allocate to each trade/signal.
        The available funds are allocated according to a weighted average calculation on certainty above 0.5
        """
        certainties = np.array([signal.certainty - 0.5 for signal in signals])

        allocations = [certainty / certainties.sum() for certainty in certainties]

        return allocations

    def free_up_money(self):
        """
        Free up money by closing trades.
        Needed in case the margin account must be increased.
        """
        pass


    def validate_orders(self, orders):
        # maybe just build this into the order generation algorithm
        pass


    def handle_event(self, portfolio, event: Event):
        if event.type == "TRADES":
            pass
        elif event.type == "CANCELLED_ORDERS":
            pass
    
    
    def get_signal_id(self):
        self.signal_id_seed += 1
        return self.signal_id_seed

    def get_order_id(self):
        self.order_id_seed += 1
        return self.order_id_seed




class MockSideClassifier(BaseEstimator):
    def __init__(self):
        pass

    def predict_proba(self, features):
        return [[0.55, 0.45]]
    

class MockCertaintyClassifier(BaseEstimator):
    def __init__(self):
        pass

    def predict_proba(self, features):
        return [[0.55, 0.45]]


"""


def _get_desired_portfolio(self, cur_portfolio, signals):

    # I need a prioritized list of "moves", then I execute as many as I can under the given restrictions

    # Is this where enforce portfolio restrictiond and make sure I have the balance, margin required?
    desired_portfolio = {}
    return desired_portfolio


def _generate_orders(self, cur_portfolio, desired_portfolio):

    # Returns ordered list of market orders to send to the broker, that will to the greatest extent move the portfolio 
    # to the desired state given the restrictions.

    orders = []

    # Liquidate first to get capital for new orders
    # Then the rest of the orders, don't think the order matters
    # Its very possible to not generate any orders, this should probably happen most of the time.
    return orders


"""