import pandas as pd
import math
import datetime
from typing import Callable
import time
import pickle

from portfolio import Portfolio
from utils.types import CommissionModel, SlippageModel, Strategy
from utils.errors import MarketDataNotAvailableError, BalanceTooLowError
from utils.logger import Logger
from utils.utils import report_progress
from order import Order
from event import EventQueue, Event
from data_handler import DataHandler
from broker import Broker
from portfolio import Portfolio
from metrics import calculate_frequency_of_bets, calculate_average_holding_period, calculate_annualized_turnover



class Backtester(object):
    def __init__(
        self, 
        market_data_handler: DataHandler, 
        start: pd.datetime, 
        end: pd.datetime,
        log_path: str,
        output_path: str,
        initialize_hook: Callable=None, 
        handle_data_hook: Callable=None, 
        analyze_hook: Callable=None,
        print_state: bool=False,
    ):
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
        self.start = start
        self.end = end
        self.log_path = log_path
        self.output_path = output_path

        start_end_index = self.market_data.date_index_to_iterate

        self.perf = pd.DataFrame(index=start_end_index) # This is where performance metrics will be stored
        self.stats = {}

        self.initialize = initialize_hook
        self.handle_data = handle_data_hook
        self.analyze = analyze_hook
        self.print = print_state

        self._event_queue = EventQueue()

        # Need to be set via setter methods
        self.portfolio = None
        self.broker = None
        self.logger = Logger("BACKTESTER", log_path + "/backtester.log")

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
        """
        if not issubclass(portfolio_cls, Portfolio):
            raise TypeError("Must be subclass of Portfolio")

        if not isinstance(self.broker, Broker):
            raise ValueError("broker must set to an instance of Broker, before instatiating the portfolio. ")

        self.portfolio = portfolio_cls(market_data=self.market_data, broker=self.broker, **kwargs)
        


    def run(self):

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

                        if (event.is_business_day == True):
                            if self.print: print("Generating signals")
                            signals_event = self.portfolio.generate_signals()
                            if signals_event is not None:
                                self._event_queue.add(signals_event)
    
                    elif event.type == 'SIGNALS': 
                        if self.print: print("Generating orders")
                        orders_event = self.portfolio.generate_orders_from_signals(event)
                        
                        if orders_event is not None:
                            self._event_queue.add(orders_event)


                    elif event.type == 'ORDERS':
                        if self.print: print("processing orders")
                        trades_event, cancelled_orders_event = self.broker.process_orders(self.portfolio, event) # Might get no fills or cancelled orders
                        
                        if trades_event is not None:
                            self._event_queue.add(trades_event)
                        
                        if cancelled_orders_event is not None:
                            self._event_queue.add(cancelled_orders_event)

                    elif event.type == 'TRADES':
                        # Here the portfolios state with regards to active positions and return calculation can be handeled
                        self.portfolio.handle_trades_event(event) # Don't know what this will do yet. Dont know what it will return

                    elif event.type == 'CANCELLED_ORDERS':
                        self.portfolio.handle_cancelled_orders_event(event) 

            """
            Here the day is over, before ending the day and starting a new one we want to update the margin account according 
            to latest close prices and update the balance and positions if any was liquidated throughout the day.
            Here we also process bankruptices, delistings, dividends, interest payment and interest reception.
            """
            
            if self.market_data.is_business_day():
                if self.print: print("manage active trades")
                margin_account_update_event = self.broker.manage_active_trades(self.portfolio) 

                if margin_account_update_event is not None:
                    # NOTE: # Now the mony from liquidated positions are available to update the margin account
                    if self.print: print("Handle margin account update")
                    self.portfolio.handle_margin_account_update(margin_account_update_event) 

                # NOTE: Process bankruptices and delistings at the open? 
                # I guess I dont want the proceeds of these events until the next day
                # the events happen sometime during the day
                # At the same time, I don't want to trade in bankrupt companies...
                # If a trade is made in a company that is delestied or bankrupt the same day, I just have to deal with this I guess...
                # Conclusion: Process bankruptcies at the end of the day
                if self.print: print("handle corp actions")
                self.broker.handle_corp_actions(self.portfolio)
                
                # NOTE: Dividends are payed at the end of the day
                if self.print: print("handle dividends")
                self.broker.handle_dividends(self.portfolio)



            # NOTE: Pay interest
            if self.print: print("handle interest on short positions")
            self.broker.handle_interest_on_short_positions(self.portfolio)

            # NOTE: Receive interest
            if self.print: print("handle interest on cash and margin accounts")
            self.broker.handle_interest_on_cash_and_margin_accounts(self.portfolio)


            if self.handle_data is not None:
                self.handle_data(self)

            # Notice that this is called after margin_account has been updated and liquidation events have updated the balance and the portfolio state
            self.capture_daily_state()
            
            report_progress(self.market_data.cur_date, self.start, self.end, time0, "Backtest")
            

        self.calculate_statistics()


        if self.analyze is not None:
            self.analyze(self)


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
        # portfolio_value = self.portfolio.calculate_portfolio_value()
        # self.perf.at[self.market_data.cur_date, "portfolio_value"] = portfolio_value        


        self.portfolio.capture_state()
        self.broker.capture_state()
        


    def calculate_statistics(self):
        self.stats["sharpe_ratio"] = None # https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
        
                
                # Cost Related:
        self.stats["total_slippage"] = self.portfolio.costs["slippage"].sum(),
        self.stats["total_commission"] = self.portfolio.costs["commission"].sum(),
        self.stats["total_charged"] = self.portfolio.costs["charged"].sum(),
        self.stats["total_margin_interest"] = self.portfolio.costs["margin_interest"].sum(),
        self.stats["total_account_interest"] = self.portfolio.costs["account_interest"].sum(),
        self.stats["total_short_dividends"] = self.portfolio.costs["short_dividends"].sum(),
        self.stats["total_short_losses"] = self.portfolio.costs["short_losses"].sum(),

        # Received Related:
        self.stats["total_dividends"] = self.portfolio.received["dividends"].sum(),
        self.stats["total_interest"] = self.portfolio.received["interest"].sum(),
        self.stats["total_proceeds"] = self.portfolio.received["proceeds"].sum(),

        # Other Metrics
        self.stats["end_value"] = self.portfolio.calculate_portfolio_value(),
        self.stats["total_return"] = self.portfolio.calculate_return_over_period(self.start, self.end)


        # NOTE: Add all backtest statistic calculates to here. Probably contain calculations in their own functions (under utils.metrics ?)
        self.stats["time_range"] = [self.start, self.end]
        self.stats["average_aum"] = None # Not sure
        self.stats["capacity"] = None # Not sure
        self.stats["maximum_dollar_position_size"] = None # Not sure
        self.stats["frequency_of_bets"] = calculate_frequency_of_bets(self.broker.blotter)
        self.stats["average_holding_period"] = calculate_average_holding_period(self.broker.blotter)
        self.stats["annualized_turnover"] = calculate_annualized_turnover(self.broker.blotter)



        # self.perf.at[self.market_data.cur_date, "max_trades_per_month"] = None
        # self.perf.at[self.market_data.cur_date, "min_trades_per_month"] = None      

        

    def save_state_to_disk_and_return(self):
        """
        This will save a snapshot of all relevant state of the backtest and save it 
        to disk as a pickle.  
        The output is used to generate a dashboard for the backtest.
        From backtester:
        - perf
        - stats
        
        From Portfolio:
        - portfolio signlas -> all signals generated -> with reference to feature data, so it can be retreved
        - portfolio order history -> all orders generated and send to the broker -> can be used with the blotter to see what orders was not filled

        From broker:
        - cancelled_orders
        - Blotter history
        - All trades (as df and Trade[])
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
                "costs": self.portfolio.costs,
                "received": self.portfolio.received,
                "portfolio_value": self.portfolio.portfolio_value,
                "order_history": self.portfolio.order_history_to_df(),
                "signals": self.portfolio.signals_to_df(),
            },
            "broker": {
                "blotter_history": self.broker.blotter_history_to_df(),
                "all_trades": self.broker.all_trades_to_df(),
                "trade_objects": self.broker.all_trades_as_objects(),
                "cancelled_orders": self.broker.cancelled_orders_to_df()
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
        
