import pandas as pd
import math

from strategy import Strategy
from portfolio import Portfolio, Order
from utils import CommissionModel, SlippageModel
import logging
import time

from event import EventQueue, Event
from data_handler import DataHandler, DailyBarsDataHander, MLFeaturesDataHandler
from broker import Broker
from utils import CommissionModel, SlippageModel, report_progress
from portfolio import Portfolio
from errors import MarketDataNotAvailableError, BalanceTooLowError
from visualization.visualization import plot_data
from logger import Logger


# NOT SURE ABOUT THIS, WHERE TO PUT METHODS ETC
class Backtester():
    def __init__(self, market_data_handler: DataHandler, feature_data_handler: DataHandler, start, end, output_path, \
        initialize_hook=None, handle_data_hook=None, analyze_hook=None):
        """
        data_handler: DataHandler for price data.
        feature_data: DataHandler for feature data informing the strategy.
        start: start date of backtest.
        end: end date of backtest.
        output_path: path where perf object is stored.
        initialize_hook: Called before the backtest starts. Can be used to setup any needed state
        handle_data_hook: Called every tick of the backtest. Can be used to calculate and track data to add to perf data frame.
        analyze_hook: Called at the end of the backtest. Can be used to output performance statistics.
        """

        self.market_data = market_data_handler
        self.feature_data = feature_data_handler

        # self.market_data = {} # Add data as it becomes available, to avoid look-ahead bias, do it another way

        self.start = start
        self.end = end
        self.output_path = output_path

        # How to best make these available in the hooks?
        start_end_index = self.market_data.date_index_to_iterate
        self.perf = pd.DataFrame(index=start_end_index) # This is where performance metrics will be stored
        self.time_context = pd.DataFrame(index=start_end_index) # To store time data between calls to handle_data
        self.context = {} # To store anything between subsequent calls to handle_data

        self.initialize = initialize_hook
        self.handle_data = handle_data_hook
        self.analyze = analyze_hook

        self._event_queue = EventQueue()
        # Need to be set via setter methods
        self.portfolio = None
        # self._commission_model = None
        # self._slippage_model = None
        # self._strategy = None
        self.broker = None

        # self._benchmark = None # Not implemented, not sure...
        # ..


    def set_portfolio(self, portfolio_cls, **kwargs):
        """
        Set the portfolio instance to use during the backtest. The portfolio object is responsible for
        taking in signals from the strategy and use these recommendations in the context of the portfolio's
        active positions and constraints to generate orders.

        Being charged or receive money (when selling or receiving dividends) from the broker?
        - Can I assume dividend reinvestment plans are available for all stocks? Even if it only is a service provided by
        the broker?

        Taxes?
        """
        if not issubclass(portfolio_cls, Portfolio):
            raise TypeError("Must be subclass of Portfolio")

        self.portfolio = portfolio_cls(market_data=self.market_data, **kwargs)
        


    def set_broker(self, broker_cls, **kwargs):
        """
        Set broker instance that will execute orders. Orders come from a portfolio object and contain information like: ticker, 
        amount, side, stop-loss and take-profit. 
        The broker fills only orders with amount under 10% of the previous day's volume. If the order exceeds this limit, the order will be broken up.
        This is to ensure that slippage does not become a major factor.

        When filling an order the broker calculates commission and slippage (costs) and charges the portfolio accordingly.
        Orders are filled at the opening price (plus slippage).
        Multiple consecutive orders are filled so fast that any random fluctuation the market may have exibited over the few seconds (or even milliseconds)
        needed to complete the trades are averaged out to an insignificant effect and ignored in the backtest.
        """
        if not issubclass(broker_cls, Broker):
            raise TypeError("Must be subclass of Broker")

        self.broker = broker_cls(self.market_data, **kwargs)
   
    
    def set_benchmark(self):
        """
        Set the benchmark asset/strategy.
        I could formulate a benchmark I am interested in tracking as a strategy object...
        """
        pass

    def run(self):
        # Check configuration

        """
        The whole point of doing it in an event driven way is to make it easier to reason about
        """

        if self.initialize is not None:
            self.initialize(self) # self.context, self.time_context, self.perf

        time0 = time.time()
        while True: # This loop generates new "ticks" until the backtest is completed.
            try:
                market_data_event = next(self.market_data.tick) # THIS MUST ONLY BE CALLED HERE!
            except Exception as e: # What error does the generator give?
                print(e)
                break
            else:
                self._event_queue.add(market_data_event)
            

            # This is executed until all events for the tick has been processed
            while True: 
                try:
                    event = self._event_queue.get()
                except: # queue is empty
                    break
                else:
                    if event is not None:
                        if event.type == 'DAILY_MARKET_DATA':
                            
                            # need this to add the feature data to the queue, so it can be "processed" (used to generate predictions)
                            feature_data_event = self.feature_data.current(event.date) # the function need access to the event queue
                            
                            self._event_queue.add(feature_data_event) # The next iteration of the loop, the event queue is not empty.


                        elif event.type == "FEATURE_DATA":
                            # Contain signals for multiple tickers
                            signals_event = self.portfolio.generate_signals(event) 
                            # Returns many signals 
                            # # I make the signals available here if I want to do something with them in multiple places
                            self._event_queue.add(signals_event) # Process in bunches

                        elif event.type == 'SIGNALS': 
                            # Maybe not have this type of event, just create signals and make order no FEATURE_DATA events
                            # It comes down to if I want to do more with it than just 
                            orders_event = self.portfolio.generate_orders_from_signals(event) # Process in bunches
                            self._event_queue.add(orders_event)

                        elif event.type == 'ORDERS':
                            # Order events contain only one event, one order is executed at a time! Or not?
                            # I need the order to be executed in sequence sometimes (when selling to make fund available for a buy for example)
                            
                            # I think it is best to charge the portfolio etc here, and update the portfolio regarding the details of the fill, though fill events.
                            fills_event = self.broker.process_orders(self.portfolio, event)
                            self._event_queue.add(fills_event)


                        elif event.type == 'FILLS':
                            # Here the portfolios state with regards to active positions and return calculation can be handeled
                            self.portfolio.handle_fill_event(event) # Don't know what this will do yet. Dont know what it will return

            if self.handle_data is not None:
                # Called at the end of every trading day
                self.handle_data(self) # self.context, self.time_context, self.perf

            report_progress(self.market_data.cur_date, self.start, self.end, time0, "Backtest")

        if self.analyze is not None:
            self.analyze(self) # self.context, self.time_context, self.perf

        return self.perf

    def get_info(self): 
        """Get initial setting of the backtest."""
        print("Start: ", self.start)
        print("End: ", self.end)
        print("etc.")
    





def Performance():
    def __init__(self, index):
        self.data = pd.DataFrame(index=index)

    def record_price(self, ticker):
        # ETC...
        pass
        