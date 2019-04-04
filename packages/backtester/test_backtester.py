import pytest
import pandas as pd 
import os
import shutil

from backtester import Backtester
from strategy import BuyAppleStrategy
from portfolio import Portfolio
from broker import Broker
from data_handler import DailyBarsDataHander, MLFeaturesDataHandler
from utils import EquityCommissionModel, EquitySlippageModel


@pytest.fixture(scope='module', autouse=True)
def setup():

    yield


"""
Notes:
  - Maybe not have different event object, just Event with a type and data property...
"""


@pytest.mark.skip()
def test_daily_bars_data_handler():
    try:
        shutil.rmtree(base_test_dir)
    except:
        pass

    data_handler = DailyBarsDataHander(
        source_path="../../datasets/testing/sep.csv",
        store_path="./test_bundles",
        file_name_time_data="time_data",
        file_name_ticker_data="ticker_data",
        start=pd.to_datetime("2010-01-01"),
        end=pd.to_datetime("2014-12-31")
    )


    i = 0
    for date, df in data_handler.time_data.items():
        assert isinstance(date, pd.datetime)
        
        """
        if i >4: break
        print(date)
        print(df.head())
        i += 1
        """

    i = 0
    for ticker, df in data_handler.ticker_data.items():
        assert len(df.ticker.unique()) == 1
        """
        if i >4: break
        print(ticker)
        print(df.head())
        i += 1
        """

    assert os.path.isfile("./test_bundles/time_data.pickle")
    assert os.path.isfile("./test_bundles/ticker_data.pickle")



    
@pytest.mark.skip()
def test_ml_feature_data_handler():
    try:
        shutil.rmtree(base_test_dir)
    except:
        pass

    data_handler = MLFeaturesDataHandler(
        source_path="../../datasets/testing/sep.csv",
        store_path="./test_bundles",
        file_name="feature_data",
    )

    # Not implemented yet



def test_backtester():

    market_data_handler = DailyBarsDataHander( 
        source_path="../../datasets/testing/sep.csv",
        store_path="./test_bundles",
        file_name_time_data="time_data",
        file_name_ticker_data="ticker_data",
        start=pd.to_datetime("2010-01-01"),
        end=pd.to_datetime("2014-12-31")
    )

    feature_data_handler = MLFeaturesDataHandler(
        source_path="../../datasets/testing/sep.csv",
        store_path="./test_bundles",
        file_name="feature_data",
    )

    def handle_data(backtester):
        print("HANDLE DATA HOOK RUNNING")


    def initialize(backtester):
        print("INITIALIZE HOOK RUNNING")

    def analyze(backtester):
        print("ANALYZE HOOK RUNNING")


    backtester = Backtester(
        market_data_handler=market_data_handler, 
        feature_data_handler=feature_data_handler, 
        start=pd.to_datetime("2010-01-01"), 
        end=pd.to_datetime("2015-01-01"),
        output_path="./performance/perf_test.csv",
        initialize_hook=initialize,
        handle_data_hook=handle_data,
        analyze_hook=analyze
        )


    strategy = BuyAppleStrategy(desc="Buy some apple every day!")
    backtester.set_portfolio(Portfolio, balance=100000, strategy=strategy)

    backtester.portfolio.set_max_position_size(10)

    """
    backtester.set_constraints( # and then the backtester configures whatever other objects that need this information?
        max_position_size=1000
    )
    """

    slippage_model = EquitySlippageModel()
    commission_model = EquityCommissionModel()
    backtester.set_broker(Broker, slippage_model=slippage_model, commission_model=commission_model)
    


    # portfolio = Portfolio(balance=100000, strategy=strategy)
    # broker = Broker(slippage_model, commission_model)

    performance = backtester.run()

    print(performance.head())



"""
Prioritet i ordre boka

Reinvestere dividends er av liten konsekvens uansett

Skatt fra utbytte 25 prosent

Ikke skatt ved kjøp og salg
"""

