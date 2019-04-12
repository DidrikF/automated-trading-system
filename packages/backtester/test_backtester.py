import pytest
import pandas as pd 
import os
import shutil


from backtester import Backtester
# from strategy import BuyAppleStrategy
from portfolio import Portfolio
from broker import Broker
from data_handler import DailyBarsDataHander, MLFeaturesDataHandler
from utils import EquityCommissionModel, EquitySlippageModel
from visualization.visualization import plot_data
from errors import MarketDataNotAvailableError

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

    # data_handler.time_data = data_handler.time_data.sort_index(ascending=True)
    # print(data_handler.time_data.iloc[1])
    # print(data_handler.time_data.head())
    print(next(data_handler.tick))
    print(next(data_handler.tick))
    print(next(data_handler.tick))

    data_handler.current_for_ticker("AAPL")["close"]

    """
    i = 0
    for date, df in data_handler.time_data.items():
        assert isinstance(date, pd.datetime)
        
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

    assert False

    
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
    start_date = pd.to_datetime("2010-01-01")
    end_date = pd.to_datetime("2010-12-31")


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
        portfolio_value = bt.portfolio.get_value()
        bt.perf.at[bt.market_data.cur_date, "portfolio_value"] = portfolio_value
        
        bt.perf.at[bt.market_data.cur_date, "AAPL"] = bt.market_data.current_for_ticker("AAPL")["close"] # Shuold succeed allways

    def initialize(bt):
        pass


    def analyze(bt):
        print(bt.perf.head())
        plot_data(bt.perf.index, bt.perf["portfolio_value"], xlabel="Time", ylabel="Value ($)", title="Portfolio Value")



    backtester = Backtester(
        market_data_handler=market_data_handler, 
        feature_data_handler=feature_data_handler, 
        start=start_date,  # REDUNDANT
        end=end_date, # REDUNDANT
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


"""
NOTES:
This section outlines how I intend to proceed towards a completed system.

Many features remain to be implemented.


1. Visualization and performance reporting
    - This work will prepare many visualizations for the report
    - This work will also substitute testing work
    - It will be awesome to look at.

    - I want to illustrate all aspects of the system, LIVE.
    - This requrie some clever linking do Dash, but once this is done it will be easy to extend
    - I can also implement a loop slowdown parameter, so make it more visible to
    the user of the system what is going on.

    Wanted on the screen: 
        - Piechart of portfolio allocation
        - Line chart of portfolio value
        - Line chart of portfolio return (on different periods)
        - Candlestick chart, where you can select stock
        - Table of executed orders -> with accocated signal, fill, commission, 
          slippage, open, execution price, and stock information (one mega master table),
          portfolio balance

        - Table of top 10 signals for each day
        - Table of non executed orders
        


2. Order handling:
    - Every tick of market data should be used to check whether stop-loss, take-profit or timeout is triggered.
    - On the time out date, the trades should be eliminated at the open, before any other trades are executed.



3. Performance tracking
    - I need a way to calculate portfolio return and take into account slippage, and the 
    actual execution price.


4. Need to have a clever way to donduct some high level tests.



"""





"""
Prioritet i ordre boka

Reinvestere dividends er av liten konsekvens uansett

Skatt fra utbytte 25 prosent

Ikke skatt ved kjøp og salg
"""

"""
Errors when running backtest
dont know what it is exacly
look into how market data is handled
why is there so many dates that are being queried for that does not have data for apple? non-business days should be skipped.
    - You have allready forseen this

also maybe use the length of the index of perf for example when reporting progress
    - perf then need to have its index come from the actual dates in the source dataset.
"""