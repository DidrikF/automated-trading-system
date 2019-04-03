import pandas as pd
import math

from strategy import some_strat
# import CommissionModel
# import SlippageModel
from event import EventQueue, Event
from data_handler import DataHandler, DailyBarsDataHander, MLFeaturesDataHandler
from broker import Broker
from utils import CommissionModel, SlippageModel

# NOT SURE ABOUT THIS, WHERE TO PUT METHODS ETC
class Backtester():
  def __init__(self, data_handler: DataHandler, feature_handler: DataHandler, start, end, output_path, \
      initialize_hook=None, handle_data_hook=None, analyze_hook=None):

    self.data_handler = data_handler
    self.feature_handler = feature_handler
    self.start = start
    self.end = end
    self.output_path = output_path

    # How to best make these available in the hooks?
    self.perf = pd.DataFrame(index=data_handler.get_index()) # This is where performance metrics will be stored
    self.time_context = pd.DataFrame(index=data_handler.get_index()) # To store time data between calls to handle_data
    self.context = {} # To store anything between subsequent calls to handle_data

    self.initialize = initialize_hook
    self.handle_data = handle_data_hook
    self.analyze = analyze_hook

    self._event_queue = EventQueue()
    # Need to be set via setter methods
    self._portfolio = None
    self._slippage_model = None
    self._commission_model = None
    self._strategy = None
    self._broker = None
    # ..

    def set_strategy(self):
        """Set the strategy to be used to generate trading signals"""

    def set_portfolio(self):
        """
        Set the portfolio instance to use during the backtest. The portfolio object is responsible for
        taking in signals from the strategy and use these recommendations in the context of the portfolio's
        active positions and constraints to generate orders.

        Being charged or receive money (when selling or receiving dividends) from the broker?
        - Can I assume dividend reinvestment plans are available for all stocks? Even if it only is a service provided by
        the broker?

        Taxes?
        """

    def set_benchmark(self):
        """
        Set the benchmark asset/strategy.
        I could formulate a benchmark I am interested in tracking as a strategy object...
        """

    def set_commission(self, commission_model: CommissionModel):
        """
        Set commission model to use.
        The commission model is responsible for modeling the costs associated with executing orders. 
        """


    def set_slippage(self, slippage_model: SlippageModel):
        """
        Set slippage model to use. The slippage model is responsible for modeling the effect your order has on the stock price.
        Generally the stock price will move against you when submitting an order to the market.
        """

    def set_broker(self, broker: Broker):
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

    def run(self):
        # Check configuration

        """
        The whole point of doing it in an event driven way is to make it easier to reason about
        """

        if self.initialize is not None:
            self.initialize()

        while True: # This loop generates new "ticks" until the backtest is completed.
            if self.data_handler.continue_backtest() == True:
                market_data_event = self.data_handler.next_tick()
                self.event_queue.add(market_data_event)
            else:
                break

            while True: # This is executed until all events for the tick has been processed
                try:
                    event = self.event_queue.get(False)
                except: # queue is empty
                    break
                else:
                    if event is not None:
                        if event.type == 'DAILY_MARKET_DATA':
                            if self.handle_data is not None:
                                self.handle_data()

                            # need this to add the feature data to the queue, so it can be "processed" (used to generate predictions)
                            # Contain features for multiple tickers
                            feature_data_event = self.feature_handler.get_events(event) # the function need access to the event queue
                            
                            self.event_queue.add(feature_data_event) # The next iteration of the loop, the event queue is not empty.


                        elif event.type == "FEATURE_DATA":
                            # Contain signals for multiple tickers
                            signal_event = self.strategy.calculate_signals(event) # Returns many signals
                            
                            self.event_queue.add(signal_event)

                        elif event.type == 'SIGNAL':
                            order_events = self.portfolio.generate_orders(signal_events)
                            self.event_queue.add(order_events)

                        elif event.type == 'ORDER':
                            # Order events contain only one event, one order is executed at a time! Or not?
                            # I need the order to be executed in sequence sometimes (when selling to make fund available for a buy for example)
                            fill_events, order_events = self.broker.execute_order(event)
                            self.event_queue.add(fill_events)

                        elif event.type == 'FILL':
                            self.portfolio.handle_fill_event(event) # Don't know what this will do yet.

        if self.analyze is not None:
            self.analyze()


    def get_info(self): 
        """Get initial setting of the backtest."""
        print("Start: ", self.start)
        print("End: ", self.end)
        print("etc.")
    


        