"""
Backtesting system developed for Didrik Fleischer's master's thesis : "Automated trading systems using machine learning"

The APIs are inspired by Quantopian/Zipline (https://github.com/quantopian/zipline).
"""

# I can read command line options and start the backtest in this file...
# Import dashboard and all components to set up a backtest

import pandas as pd
from dateutil.relativedelta import *

from automated_trading_system.backtester.backtester import Backtester
from automated_trading_system.backtester.broker import Broker
from automated_trading_system.backtester.data_handler import DailyBarsDataHander, MLFeaturesDataHandler
from automated_trading_system.backtester.portfolio import Portfolio, RandomLongShortStrategy
from automated_trading_system.backtester.utils import EquitySlippageModel, EquityCommissionModel
from automated_trading_system.config.settings import Settings

if __name__ == "__main__":
    settings = Settings(env="TEST", root="../../")

    data_set = settings.get_data_set()
    save_location = settings.get_save_location()
    start_date = pd.to_datetime("2010-01-01")
    end_date = pd.to_datetime("2010-06-01")

    market_data_handler = DailyBarsDataHander(
        source_path=data_set,
        store_path=save_location,
        file_name_time_data="time_data",
        file_name_ticker_data="ticker_data",
        start=start_date,
        end=end_date
    )

    feature_data_handler = MLFeaturesDataHandler(
        source_path=data_set,
        store_path=save_location,
        file_name="feature_data",
    )


    def handle_data(bt):  # perf, port, md, cur_date
        portfolio_value = bt.portfolio.calculate_value()
        bt.perf.at[bt.market_data.cur_date, "portfolio_value"] = portfolio_value

        # Should succeed always
        bt.perf.at[bt.market_data.cur_date, "AAPL"] = bt.market_data.current_for_ticker("AAPL")["close"]


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
        end=end_date,  # REDUNDANT
        output_path=save_location,
        initialize_hook=initialize,
        handle_data_hook=handle_data,
        analyze_hook=analyze
    )

    # strategy = BuyAppleStrategy(desc="Buy some apple every day!")
    strategy = RandomLongShortStrategy(
        desc="Buy or sell randomly stocks from provided list.",
        tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "BRK.A", "FB", "JPM", "BAC", "JNJ", "XOM"],
        amount=2
    )
    strategy.set_order_restrictions(
        max_position_size=0.10,
        max_positions=30,
        min_positions=5,
        max_orders_per_day=2,
        max_orders_per_month=30,
        max_hold_period=relativedelta(months=1),
    )

    backtester.set_portfolio(Portfolio, balance=100000, strategy=strategy)

    """
    backtester.set_constraints( # and then the backtester configures whatever other objects that need this information?
        max_position_size=1000
    )
    """

    slippage_model = EquitySlippageModel()
    commission_model = EquityCommissionModel()
    backtester.set_broker(
        Broker,
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
