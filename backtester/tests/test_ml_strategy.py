import sys, os
import pandas as pd
import pytest
import math
from dateutil.relativedelta import *
import numpy as np

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, ".."))
# sys.path.insert(0, os.path.join(myPath))

from broker import Broker, Trade
from data_handler import DailyBarsDataHander, MLFeaturesDataHandler
from utils.utils import EquityCommissionModel, EquitySlippageModel, Signal
from order import Order
from event import Event
from portfolio import Portfolio
from ml_strategy import MLStrategy, MockSideClassifier, MockCertaintyClassifier

start_date = pd.to_datetime("2010-01-04") # Monday
end_date = pd.to_datetime("2010-01-19") # 2 weeks after, 2 rebalances


market_data = DailyBarsDataHander(
    path_prices="./test_datasets/test_sep.csv", # "../../dataset_development/datasets/testing/sep.csv",
    path_snp500="./test_datasets/test_snp500.csv", # "../../dataset_development/datasets/macro/snp500.csv",
    path_interest="./test_datasets/test_rf_rate.csv", # "../../dataset_development/datasets/macro/t_bill_rate_3m.csv",
    path_corp_actions="./test_datasets/test_corp_actions_no_bankruptcies.csv", # "../../dataset_development/datasets/sharadar/SHARADAR_EVENTS.csv",
    store_path="./test_bundles",
    start=start_date,
    end=end_date,
    rebuild=True
)

print(market_data.corp_actions)

feature_handler = MLFeaturesDataHandler(
    path_features="./test_datasets/test_dataset.csv", 
    store_path="./test_bundles", 
    start=start_date, 
    end=end_date
)

commission_model = EquityCommissionModel()
slippage_model = EquitySlippageModel()

broker = Broker(
    market_data=market_data,
    commission_model=commission_model,
    slippage_model=slippage_model,
    annual_margin_interest_rate=0.06,
    initial_margin_requirement=0.5,
    maintenance_margin_requirement=0.3,
    tax_rate=0.25
)

side_classifier = MockSideClassifier()
certainty_classifier = MockCertaintyClassifier()

strategy = MLStrategy(
    rebalance_weekday=0, 
    side_classifier=side_classifier, 
    certainty_classifier=certainty_classifier,
    ptSl=[1, -1],
    feature_handler=feature_handler,
    features=["mom1m", "mom12m"],    
    initial_margin_requirement=0.5, 
    # accepted_signal_age: dt.relativedelta=relativedelta(days=7)
)

strategy.set_order_restrictions(
    max_position_size=0.3, # 0.05
    max_positions=2, # 30
    minimum_balance=10, # 2% of initial balance?
    max_percent_to_invest_each_period=0.33, # 0.33
    max_orders_per_period=1, # 10
)

start_balance = 100
portfolio = Portfolio(
    market_data=market_data,
    broker=broker,
    strategy=strategy,
    balance=start_balance,
    initial_margin_requirement=0.5,
    maintenance_margin_requirement=0.3
)

@pytest.fixture(scope='module', autouse=True)
def setup():

    yield

"""
@pytest.fixture(autouse=True)
def run_around_tests():
    global market_data, broker, strategy, start_balance, commission_model,slippage_model, portfolio
    market_data.tick = market_data._next_tick_generator()
    portfolio.__init__(
        market_data=market_data,
        broker=broker,
        strategy=strategy,
        balance=start_balance,
        initial_margin_requirement=0.5,
        maintenance_margin_requirement=0.3
    )
    broker.__init__(
        market_data=market_data,
        commission_model=commission_model,
        slippage_model=slippage_model,
        annual_margin_interest_rate=0.06,
        initial_margin_requirement=0.5,
        maintenance_margin_requirement=0.3,
        tax_rate=0.25
    )
    yield
"""

def test_ml_strategy():
    """
    Things to test:
    can generate signals
    can generate orders from signals under given restrictions
    """

    # Market Loop
    last_mondays_trades = []

    end_date = pd.to_datetime("2010-01-18")
    while market_data.cur_date < end_date:
        orders_event = None
        trades_event = None
        cancelled_orders_event = None
        market_data_event = next(market_data.tick)
        print("Loop Date: ", market_data.cur_date, market_data.cur_date.weekday())
        if (market_data_event.is_business_day == True): 
            signals_event = portfolio.generate_signals()
            
            if signals_event is not None:
                print("Signlas: ", [signal.signal_id for signal in signals_event.data])
                orders_event = portfolio.generate_orders_from_signals(signals_event)
                
                # NOTE: Need to test the generated orders...

            if orders_event is not None:
                print("Orders: ", [order.id for order in orders_event.data])
                trades_event, cancelled_orders_event = broker.process_orders(portfolio, orders_event)
            
            if trades_event is not None:
                portfolio.handle_trades_event(trades_event) # Don't know what this will do yet. Dont know what it will return
            
            if cancelled_orders_event is not None:
                portfolio.handle_cancelled_orders_event(cancelled_orders_event) 

            margin_account_update_event = broker.manage_active_trades(portfolio) 

            if margin_account_update_event is not None:
                portfolio.handle_margin_account_update(margin_account_update_event) 
            broker.handle_corp_actions(portfolio)                
            broker.handle_dividends(portfolio)

        broker.handle_interest_on_short_positions(portfolio)
        broker.handle_interest_on_cash_and_margin_accounts(portfolio)


        # Do checks each rebalancing date
        if market_data.cur_date.weekday() == 0:
            last_mondays_trades = broker.blotter.get_active_trades()
            print("Cur Trades (monday): (order_id)", [trade.order_id for trade in last_mondays_trades])
            print(
                "Closed Trades: ", [trade.order_id for trade in broker.blotter.closed_trades], 
                " Closed Trades cause: ", [trade.close_cause for trade in broker.blotter.closed_trades]
            )

        if market_data.cur_date.weekday() == 5:
            print("Trades on Friday: ", [trade.order_id for trade in broker.blotter.active_trades])
            
            if len(broker.blotter.active_trades) > 0:
                assert last_mondays_trades[-1].order_id == broker.blotter.active_trades[-1].order_id
            else:
                assert len(last_mondays_trades) == 0

    """
    Looks good, but I should do more tests:
    Test the parameters for orders...
    1. Number of stock for each trade
    2. Direction of stock is the same as signal
    3. The orders respect limitations (calculate manually from dashboard info...)
    """


"""
date,ticker,ewmstd_2y_monthly,mom1m,mom12m,datekey,timeout
2010-01-03,AAPL,0.15,0.10,0.20,2010-01-04,2010-01-19
2010-01-03,IBM,0.15,0.10,0.20,2010-01-04,2010-01-19
2010-01-06,AAPL,0.15,0.10,0.20,2010-01-04,2010-01-19
2010-01-06,IBM,0.15,0.10,0.20,2010-01-04,2010-01-19
2010-01-17,AAPL,0.15,0.10,0.20,2010-01-04,2010-01-19
2010-01-17,IBM,0.15,0.10,0.20,2010-01-04,2010-01-19
"""
@pytest.mark.skip()
def test_get_top_signals():
    signals = [
        Signal(
            signal_id=1, 
            ticker="AAPL", 
            direction=0.6, 
            certainty=0.70, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        ),Signal(
            signal_id=2, 
            ticker="MSFT", 
            direction=0.6, 
            certainty=0.65, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        ),Signal(
            signal_id=3, 
            ticker="IBM", 
            direction=0.6, 
            certainty=0.60, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        ),Signal(
            signal_id=4, 
            ticker="AAPL", 
            direction=0.6, 
            certainty=0.80, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        )
    ]

    top_signals = strategy._get_top_signals(signals, 2, set())
    print("Top Signals")
    for signal in top_signals:
        print("signal_id: ", signal.signal_id)

    assert len(top_signals) == 2
    assert top_signals[0].signal_id == 4
    assert top_signals[1].signal_id == 2

@pytest.mark.skip()
def test_get_top_signals_taking_into_account_current_positions():
    signals = [
        Signal(
            signal_id=1, 
            ticker="AAPL", 
            direction=0.6, 
            certainty=0.70, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        ),Signal(
            signal_id=2, 
            ticker="MSFT", 
            direction=0.6, 
            certainty=0.65, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        ),Signal(
            signal_id=3, 
            ticker="IBM", 
            direction=0.6, 
            certainty=0.60, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        ),Signal(
            signal_id=4, 
            ticker="AAPL", 
            direction=0.6, 
            certainty=0.80, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        )
    ]

    top_signals = strategy._get_top_signals(signals, 2, ["AAPL"])
    print("Top Signals")
    for signal in top_signals:
        print("signal_id: ", signal.signal_id)

    assert len(top_signals) == 2
    assert top_signals[0].signal_id == 2
    assert top_signals[1].signal_id == 3


@pytest.mark.skip()
def test_get_allocation_between_signlas():
    signals = [
        Signal(
            signal_id=1, 
            ticker="AAPL", 
            direction=0.6, 
            certainty=0.70, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        ),Signal(
            signal_id=2, 
            ticker="MSFT", 
            direction=0.6, 
            certainty=0.65, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        ),Signal(
            signal_id=3, 
            ticker="IBM", 
            direction=0.6, 
            certainty=0.60, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        ),Signal(
            signal_id=4, 
            ticker="AAPL", 
            direction=0.6, 
            certainty=0.80, 
            ewmstd=0.15, 
            timeout=pd.to_datetime("2010-01-01"),
            features_date=pd.to_datetime("2010-01-01")
        )
    ]

    allocations = strategy._get_allocation_between_signals(signals)
    print("Allocations: ", allocations)
    assert allocations[0] == pytest.approx(0.2 / (0.3+0.2+0.15+0.10))
    assert allocations[1] == pytest.approx(0.15 / (0.3+0.2+0.15+0.10))
    assert allocations[2] == pytest.approx(0.10 / (0.3+0.2+0.15+0.10))
    assert allocations[3] == pytest.approx(0.3 / (0.3+0.2+0.15+0.10))

    assert np.array(allocations).sum() == pytest.approx(1)