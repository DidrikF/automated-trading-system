"""
Backtesting system developed for Didrik Fleischer's master's thesis : "Automated trading systems using machine learning".

The APIs are inspired by Quantopian/Zipline (https://github.com/quantopian/zipline).

"""


# I can read command line options and start the backtest in this file...

# Make it possible to use the cross-validation method for backtesting???

# Import dashboard and all compoentnsto set up a backtest

import pytest
import pandas as pd 
import os
import shutil
import threading
from queue import Queue
import time
from dateutil.relativedelta import *


from backtester import Backtester
# from strategy import BuyAppleStrategy, RandomLongShortStrategy
from portfolio import Portfolio, RandomLongShortStrategy
from broker import Broker
from data_handler import DailyBarsDataHander, MLFeaturesDataHandler
from utils import EquityCommissionModel, EquitySlippageModel
from visualization.visualization import plot_data
from errors import MarketDataNotAvailableError

if __name__ == "__main__":
    start_date = pd.to_datetime("2010-01-01")
    end_date = pd.to_datetime("2010-06-01")


    market_data_handler = DailyBarsDataHander( 
        source_path="../../datasets/testing/sep.csv",
        store_path="./test_bundles",
        file_name_time_data="time_data",
        file_name_ticker_data="ticker_data",
        start=start_date,
        end=end_date
    )

    feature_data_handler = MLFeaturesDataHandler(
        source_path="../../datasets/testing/sep.csv",
        store_path="./test_bundles",
        file_name="feature_data",
    )

    def handle_data(bt): # perf, port, md, cur_date
        portfolio_value = bt.portfolio.calculate_value()
        bt.perf.at[bt.market_data.cur_date, "portfolio_value"] = portfolio_value
        
        bt.perf.at[bt.market_data.cur_date, "AAPL"] = bt.market_data.current_for_ticker("AAPL")["close"] # Shuold succeed allways

    def initialize(bt):
        pass


    def analyze(bt):
        # print(bt.perf.head())
        # plot_data(bt.perf.index, bt.perf["portfolio_value"], xlabel="Time", ylabel="Value ($)", title="Portfolio Value")
        pass


    backtester = Backtester(
        market_data_handler=market_data_handler, 
        feature_data_handler=feature_data_handler, 
        start=start_date,  # REDUNDANT
        end=end_date, # REDUNDANT
        output_path="./backtests",
        initialize_hook=initialize,
        handle_data_hook=handle_data,
        analyze_hook=analyze
        )


    # strategy = BuyAppleStrategy(desc="Buy some apple every day!")
    strategy = RandomLongShortStrategy(desc="Buy or sell randomly stocks from provided list.", tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "BRK.A", "FB", "JPM", "BAC", "JNJ", "XOM"], amount=2)
    strategy.set_order_restrictions(
        max_position_size= 0.10, 
        max_positions= 30, 
        min_positions= 5, 
        max_orders_per_day= 2, # max dollar amount spent 
        max_orders_per_month= 30, 
        max_hold_period= relativedelta(months=1),
    )

    # Turnaround vs costs 

    # make sure to have sufficient balance to exploit big opportunities


    backtester.set_portfolio(Portfolio, balance=100000, strategy=strategy)

    """
    backtester.set_constraints( # and then the backtester configures whatever other objects that need this information?
        max_position_size=1000
    )
    """

    slippage_model = EquitySlippageModel()
    commission_model = EquityCommissionModel()
    backtester.set_broker(Broker,
        slippage_model=slippage_model,
        commission_model=commission_model,
        annual_margin_interest_rate=0.06,
        initial_margin_requirement=0.50,
        maintenance_margin_requirement=0.30
    )
    
    # Run this in thread?

    performance = backtester.run()

    """
    for portfolio in backtester.portfolio.portfolio_history:
        print(portfolio)
    """

    backtest_state = backtester.save_state_to_disk_and_return()

    