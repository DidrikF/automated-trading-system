import pytest
import pandas as pd 
import os, sys
import shutil
from queue import Queue
import time
from dateutil.relativedelta import *
import pickle
import datetime

from sklearn.ensemble import RandomForestClassifier

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, ".."))
# sys.path.insert(0, myPath)

from backtester import Backtester
from portfolio import Portfolio
from ml_strategy import MLStrategy
from broker import Broker
from data_handler import DailyBarsDataHander, MLFeaturesDataHandler
from utils.utils import EquityCommissionModel, EquitySlippageModel
from utils.errors import MarketDataNotAvailableError
from ml_strategy_models import features


start_date = pd.to_datetime("2010-01-01")
end_date = pd.to_datetime("2010-12-31")


log_path = "./logs/log_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.mkdir(log_path)

print("Instantiating Market Data Handler")
market_data_handler = DailyBarsDataHander( 
    path_prices="../../dataset_development/datasets/testing/sep.csv",
    path_snp500="../../dataset_development/datasets/macro/snp500.csv",
    path_interest="../../dataset_development/datasets/macro/t_bill_rate_3m.csv",
    path_corp_actions="../../dataset_development/datasets/sharadar/SHARADAR_EVENTS.csv",
    store_path="./test_bundles",
    start=start_date,
    end=end_date,
    rebuild=False,
)

def handle_data(bt):
    pass

def initialize(bt):
    pass

def analyze(bt):
    pass


backtester = Backtester(
    market_data_handler=market_data_handler, 
    start=start_date,  # REDUNDANT
    end=end_date, # REDUNDANT
    log_path=log_path,
    output_path="./backtests",
    initialize_hook=initialize,
    handle_data_hook=handle_data,
    analyze_hook=analyze
)

"""
strategy = RandomLongShortStrategy(desc="Buy or sell randomly stocks from provided list.", tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "BRK.A", "FB", "JPM", "BAC", "JNJ", "XOM"], amount=2)
strategy.set_order_restrictions(
    max_position_size= 0.10, 
    max_positions= 30, 
    min_positions= 5, 
    max_orders_per_day= 2, # max dollar amount spent 
    max_orders_per_month= 30, 
    max_hold_period= relativedelta(months=1),
)
"""

print("Instantiating ML Feature Data Handler")
feature_handler = MLFeaturesDataHandler(
    path_features="../../dataset_development/datasets/testing/ml_dataset.csv",
    store_path="./test_bundles",
    start=start_date,
    end=end_date
)

# NOTE: Need to get from aws compute node or train locally with parameters found
side_classifier = pickle.load(open("../../models/simple_side_classifier.pickle", "rb"))
certainty_classifier = pickle.load(open("../../models/simple_certainty_classifier.pickle", "rb"))

strategy = MLStrategy(
    rebalance_weekday=0, 
    side_classifier=side_classifier, 
    certainty_classifier=certainty_classifier,
    ptSl=[1, -0.8],
    feature_handler=feature_handler,
    features=features,    
    initial_margin_requirement=0.5, 
    log_path=log_path,
    # accepted_signal_age: dt.relativedelta=relativedelta(days=7)
)

strategy.set_order_restrictions(
    max_position_size=0.05, # 0.05
    max_positions=30, # 30
    minimum_balance=100000, # 1% of initial balance?
    max_percent_to_invest_each_period=0.33, # 0.33
    max_orders_per_period=10, # 10
    min_order_size_limit=5000,
    percentage_short=0.30,
    volume_limit=0.1
)

print("Computing predictions...")
strategy.compute_predictions()

slippage_model = EquitySlippageModel()
commission_model = EquityCommissionModel()
backtester.set_broker(Broker,
    slippage_model=slippage_model,
    commission_model=commission_model,
    log_path=log_path,
    annual_margin_interest_rate=0.06,
    initial_margin_requirement=0.50,
    maintenance_margin_requirement=0.30
)

backtester.set_portfolio(Portfolio, log_path=log_path, balance=10000000, strategy=strategy)

backtester.run()

@pytest.fixture(scope='module', autouse=True)
def setup():

    yield


def test_portfolio_get_monthly_returns():
    print(backtester.portfolio.get_monthly_returns())


