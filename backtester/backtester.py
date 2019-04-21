import pandas as pd
import math
import datetime

# from strategy import Strategy
from portfolio import Portfolio, Order, Strategy
from utils import CommissionModel, SlippageModel
import logging
import time
import pickle

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


        start_end_index = self.market_data.date_index_to_iterate

        # Since I am the only user, I can store both system default metrics and "user specified" metrics here
        self.perf = pd.DataFrame(index=start_end_index) # This is where performance metrics will be stored
        self.stats = {}
        # OBS: Not used atm, might just drop these and store all in perf. And whatever that might had gone into context, is handled by the portfolio and strategy
        # self.time_context = pd.DataFrame(index=start_end_index) # To store time data between calls to handle_data
        # self.context = {} # To store anything between subsequent calls to handle_data

        self.initialize = initialize_hook
        self.handle_data = handle_data_hook
        self.analyze = analyze_hook

        self._event_queue = EventQueue()
        # Need to be set via setter methods
        self.portfolio = None
        self.broker = None
        

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

        if not isinstance(self.broker, Broker):
            raise ValueError("broker must set to an instance of Broker, before instatiating the portfolio. ")


        self.portfolio = portfolio_cls(market_data=self.market_data, broker=self.broker, **kwargs)
        

    
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
                    # maybe I should account for returned events may be None, and should not be added to the event_queue
                    if event.type == 'DAILY_MARKET_DATA':
                        
                        # need this to add the feature data to the queue, so it can be "processed" (used to generate predictions)
                        feature_data_event = self.feature_data.current(event.date) # the function need access to the event queue
                        
                        if feature_data_event is not None:
                            self._event_queue.add(feature_data_event) # The next iteration of the loop, the event queue is not empty.

                    elif event.type == "FEATURE_DATA":
                        # Contain signals for multiple tickers
                        signals_event = self.portfolio.generate_signals(event) # Might get nothing
                        # Returns many signals 
                        # # I make the signals available here if I want to do something with them in multiple places
                        if signals_event is not None:
                            self._event_queue.add(signals_event)

                    elif event.type == 'SIGNALS': 
                        # Maybe not have this type of event, just create signals and make order no FEATURE_DATA events
                        # It comes down to if I want to do more with it than just 
                        orders_event = self.portfolio.generate_orders_from_signals(self.portfolio, event)
                        
                        if orders_event is not None:
                            self.portfolio.order_history.extend(orders_event.data)
                            self._event_queue.add(orders_event)

                    elif event.type == 'ORDERS':

                        fills_event, cancelled_orders_event = self.broker.process_orders(self.portfolio, event) # Might get no fills or cancelled orders
                        
                        if fills_event is not None:
                            self._event_queue.add(fills_event)
                        
                        if cancelled_orders_event is not None:
                            self._event_queue.add(cancelled_orders_event)

                    elif event.type == 'FILLS':
                        # Here the portfolios state with regards to active positions and return calculation can be handeled
                        self.portfolio.handle_fill_event(event) # Don't know what this will do yet. Dont know what it will return

                    elif event.type == 'CANCELLED_ORDERS':

                        self.portfolio.handle_cancelled_orders_event(event)


                    # Possible that these events are handled outside this loop
                    elif event.type == 'POSITION_LIQUIDATIONS':
                        # This happens at the end of the trading day. Funds that become available as the result of a position being
                        # liquidated is first available to use the next day.
                        self.portfolio.handle_position_liquidations_event(event)
                    
                    elif event.type == "BANKRUPTCIES":
                        self.broker.handle_bankruptcies(event)

                    elif event.type == "CORPORATE_ACTIONS": # DIVIDENDS
                        # Don't know if I want both to handle this event...
                        self.broker.handle_corporate_actions(event) # dividend is one type of corporate action, if this is the only event I end up covering are rename this stuff
                        self.portfolio.handle_corporate_actions(event)


            # Here the day is over, before ending the day and starting a new one we want to update the margin account according to latest close prices
            # and update the balance and positions if any was liquidated throughout the day.
            position_liquidations_event, margin_account_update_event = self.broker.manage_active_positions(self.portfolio) #  This generates margin account update

            if position_liquidations_event is not None:
                self.portfolio.handle_position_liquidations_event(position_liquidations_event)

            if margin_account_update_event is not None:
                self.portfolio.handle_margin_account_update(margin_account_update_event) # Now the mony from liquidated positions are available to update the margin account


            if self.handle_data is not None:
                self.handle_data(self) # self.context, self.time_context, self.perf
            

            # Notice that this is called after margin_account has been updated and liquidation events have updated the balance and the portfolio state
            self.capture_daily_state()
            
            report_progress(self.market_data.cur_date, self.start, self.end, time0, "Backtest")
            

        self.calculate_statistics()


        if self.analyze is not None:
            self.analyze(self) # self.context, self.time_context, self.perf



        return self.perf

    def get_info(self): 
        """Get initial setting of the backtest."""
        print("Start: ", self.start)
        print("End: ", self.end)
        print("etc.")
    

    def capture_daily_state(self):
        """
        Calculate various backtest statistics. These calculations are split into their 
        own function, but the work is sentralized here.
        """
        
        self.portfolio.capture_state()
        self.broker.capture_state()
        


    def calculate_statistics(self):
        self.stats["total_commission"] = None
        self.stats["total_slippage"] = None
        self.stats["sharpe_ratio"] = None # https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
        

        self.stats["total_charged_for_stock_purchases"] = self.portfolio.total_charged_for_stock_purchases
        self.stats["total_commission"] = self.portfolio.total_commission
        self.stats["total_margin_interest"] = self.portfolio.total_margin_interest

        


        # self.perf.at[self.market_data.cur_date, "max_trades_per_month"] = None
        # self.perf.at[self.market_data.cur_date, "min_trades_per_month"] = None      

    def save_state_to_disk_and_return(self):
        """
        This will save a snapshot of all relevant state of the backtest and save it 
        to disk as a pickle. This way the backtest can be replayed and all results
        are available for later inspection.
        
        The output can be used to generate a dashboard for the backtest.
        
        What to save:

        From backtester:
        - perf
        - stats
        
        From Portfolio:
        - order history
        - Portfolio history (Need to record all changes...) # Need to visualize long, short and net position
        - Active positions history (might not show this in the dashboard)
        - portfolio blotter  -> contains order -> contains signal
        - portfolio signlas -> all signals generated -> with reference to feature data, so it can be retreved
        - portfolio order history -> all orders generated and send to the broker -> can be used with the blotter to see what orders was not filled

        From broker:
        - cancelled_orders
        - Active positions history

        """


        backtest_state = {
            "timestamp": time.time(),
            "settings": {
                "start": self.start,
                "end": self.end,
                "output_path": self.output_path
            },
            "perf": self.perf,
            "stats": self.stats,
            "portfolio": {
                "commissions_charged": self.portfolio.commissions_charged,
                "slippage_suffered": self.portfolio.slippage_suffered,
                "order_history": self.portfolio.order_history_to_df(),
                "portfolio_history": self.portfolio.portfolio_history_to_df(),
                "cancelled_orders": self.portfolio.cancelled_orders_to_df(),
                "active_positions_history": self.portfolio.active_positions_history_to_df(),
                "signals": self.portfolio.signals_to_df(),
                "blotter": self.portfolio.blotter_to_df(),
                "portfolio_value": self.portfolio.portfolio_value,
            },
            "broker": {
                "cancelled_orders": self.broker.cancelled_orders_to_df(), # completely redundant as far as I understand
                "active_positions_history": self.broker.active_positions_history_to_df(), # this is the fills that are active at any one time, the fills combines makes up the portfolio, this is somewhat different from portfolio.portfolio
                "blotter": self.broker.blotter_to_df(), # Also redundant I guess, but also nice to know that is correct
            }
        }
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        pickle_path = self.output_path + "/backtest_state_" + timestamp + ".pickle"

        pickle_out = open(pickle_path,"wb")
        pickle.dump(backtest_state, pickle_out)
        pickle_out.close()

        return backtest_state

        



def Performance():
    def __init__(self, index):
        self.data = pd.DataFrame(index=index)

    def record_price(self, ticker):
        # ETC...
        pass
        