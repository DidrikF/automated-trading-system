import sys, os
import pandas as pd
import pytest
import math
from dateutil.relativedelta import *
import numpy as np

import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, ".."))
# sys.path.insert(0, os.path.join(myPath))

from broker import Broker, Trade
from data_handler import DailyBarsDataHander
from utils.utils import EquityCommissionModel, EquitySlippageModel, Signal
from order import Order
from event import Event
from portfolio import Portfolio
from simple_strategies import TestStrategy 

start_date = pd.to_datetime("2010-01-01")
end_date = pd.to_datetime("2010-01-04")


market_data = DailyBarsDataHander(
    path_prices="./test_datasets/test_sep.csv", # "../../dataset_development/datasets/testing/sep.csv",
    path_snp500="./test_datasets/test_snp500.csv", # "../../dataset_development/datasets/macro/snp500.csv",
    path_interest="./test_datasets/test_rf_rate.csv", # "../../dataset_development/datasets/macro/t_bill_rate_3m.csv",
    path_corp_actions="./test_datasets/test_corp_actions.csv", # "../../dataset_development/datasets/sharadar/SHARADAR_EVENTS.csv",
    store_path="./test_bundles",
    start=start_date,
    end=end_date,
    rebuild=False
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

strategy = TestStrategy()

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


def test_data_handler_interest():
    print(market_data.rf_rate)

def test_broker():
    global broker
    assert isinstance(broker, Broker)


def test_broker_LONG_process_orders_and__process_order():
    global market_data, commission_model, slippage_model, broker, portfolio, strategy, start_date, end_date, start_balance

    next(broker.market_data.tick)
    order_date = pd.to_datetime("2010-01-01")

    order_1 = Order(
        order_id=1,
        ticker="AAPL",
        amount=1,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=None,
        take_profit=None,
        timeout=None,
    )

    order_2 = Order(
        order_id=2,
        ticker="AAPL",
        amount=2,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=None,
        take_profit=None,
        timeout=None,
    )

    orders_event = Event(event_type="ORDERS", data=[order_1, order_2], date=order_date)


    # print("Portfolio Balance: ", portfolio.balance)
    broker.process_orders(portfolio, orders_event)
    # print("Portfolio Balance: ", portfolio.balance)

    assert broker.blotter.active_trades[0].order == order_1
    assert broker.blotter.active_trades[1].order == order_2

    costs = commission_model.calculate(1, 10) + commission_model.calculate(2, 10) + 10 * 3
    assert portfolio.balance == pytest.approx(start_balance - costs)



def test_broker_SHORT_process_orders_and__process_order():
    # Process short order
    global market_data, commission_model, slippage_model, broker, portfolio, strategy, start_date, end_date


    next(broker.market_data.tick)

    order_date = pd.to_datetime("2010-01-01")

    order_1 = Order(
        order_id=1,
        ticker="AAPL",
        amount=-1,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=None,
        take_profit=None,
        timeout=None,
    )

    order_2 = Order(
        order_id=2,
        ticker="AAPL",
        amount=-2,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=None,
        take_profit=None,
        timeout=None,
    )

    orders_event = Event(event_type="ORDERS", data=[order_1, order_2], date=order_date)

    broker.process_orders(portfolio, orders_event)

    assert broker.blotter.active_trades[0].order == order_1
    assert broker.blotter.active_trades[1].order == order_2


    # check margin account
    margin_requirement_1 = 1 * 10 *(1 + 0.5)
    margin_requirement_2 = 2 * 10 *(1 + 0.5)

    assert portfolio.margin_account == pytest.approx(margin_requirement_1 + margin_requirement_2)

    commission = broker.commission_model.calculate(-1, 10) + broker.commission_model.calculate(-2, 10)
    assert commission > 0
    total_costs = commission + margin_requirement_1 + margin_requirement_2
    assert portfolio.balance == pytest.approx(start_balance - total_costs)


"""
date,ticker,open,high,low,close,dividends
2010-01-01,AAPL,10,10.5,9.5,10,2
2010-01-02,AAPL,10,11,9,10,1
2010-01-03,AAPL,11,11,8,10,0
2010-01-04,AAPL,10,10,10,10,0

2010-01-01,MSFT,10,10.5,9.5,10,2
2010-01-02,MSFT,11,14,11,13,1
2010-01-03,MSFT,12,14,11,13,0

rf_rate
date,rate,      daily
2010-01-01,0.05 0.001
2010-01-02,0.04 0.002
2010-01-03,0.05 0.001
2010-01-04,0.04 0.002

snp500
date,close
2010-01-01,10
2010-01-02,10.5
2010-01-03,11
2010-01-04,11.5

ticker,date,eventcodes
AAPL,2010-01-02,22|91
AAPL,2010-01-03,13|42|98
MSFT,2010-01-02,23|43|14

"""

def test_broker_LONG_manage_active_positions():

    # FIRST ITERATION
    next(broker.market_data.tick)
    order_date = pd.to_datetime("2010-01-01")
    order_1 = Order(
        order_id=1,
        ticker="AAPL",
        amount=2,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=-0.1, # Triggered
        take_profit=0.399999, # Can get 0.399999999 when working with these number, so to make this trigger
        timeout=pd.to_datetime("2010-01-06")
    )

    orders_event = Event(event_type="ORDERS", data=[order_1], date=order_date)
    broker.process_orders(portfolio, orders_event)
    end_date = pd.to_datetime("2010-01-04")
    broker.handle_dividends(portfolio)

    # LATTER ITERATIONS
    while broker.market_data.cur_date < end_date:
        market_data_event = next(broker.market_data.tick)
        margin_acc_update_event = broker.manage_active_trades(portfolio)
        if margin_acc_update_event != None:
            portfolio.handle_margin_account_update(margin_acc_update_event)
        
        broker.handle_dividends(portfolio)


    closed_trade = broker.blotter.closed_trades[0]

    assert closed_trade.close_date == pd.to_datetime("2010-01-03")
    assert closed_trade.close_price == pytest.approx(10*1.399999 - 3) # 3 in dividends, makes the 0.4 take_profit barrier trigger
    assert closed_trade.dividends_per_share == 3
    assert closed_trade.total_dividends == 6

def test_exit_of_short_trade_due_to_dividend_payment():
    # FIRST ITERATION
    next(broker.market_data.tick)
    order_date = pd.to_datetime("2010-01-01")
    order_1 = Order(
        order_id=1,
        ticker="AAPL",
        amount=-2,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=-0.29999, # Will happen the next day due to dividend payment and high of 11 (remember dividends are registered at the very end of a day)
        take_profit=0.2, # Requires a price of 8 (which is hit on 01-03, but due to dividend, it will not be hit)
        timeout=pd.to_datetime("2010-01-06")
    )

    orders_event = Event(event_type="ORDERS", data=[order_1], date=order_date)
    broker.process_orders(portfolio, orders_event)
    broker.handle_dividends(portfolio)

    # Check margin account (money should have been added) and balance
    payed_dividends = 2 * 2
    margin_account_size = 10 * 2 * (1 + 0.5) - payed_dividends
    assert portfolio.margin_account == pytest.approx(margin_account_size)
    assert portfolio.balance == pytest.approx(100 - (10 * 2 * (1 + 0.5)) - broker.commission_model.calculate(-2, 10))

    # LATTER ITERATIONS    
    while broker.market_data.cur_date < pd.to_datetime("2010-01-04"):
        market_data_event = next(broker.market_data.tick)
        margin_acc_update_event = broker.manage_active_trades(portfolio)
        if margin_acc_update_event != None:
            portfolio.handle_margin_account_update(margin_acc_update_event)
        
        broker.handle_corp_actions(portfolio)
        broker.handle_dividends(portfolio)
        # broker.handle_interest_on_short_positions(portfolio)
        # broker.handle_interest_on_cash_and_margin_accounts(portfolio)

    assert len(broker.blotter.active_trades) == 0
    assert len(broker.blotter.closed_trades) == 1
    closed_trade = broker.blotter.closed_trades[0]
    # Check close of trade given the take profit and stop loss limits
    assert closed_trade.close_date == pd.to_datetime("2010-01-02")

    # print("dividends: ", closed_trade.dividends_per_share)
    # print("fill_price: ", closed_trade.fill_price)
    # print("stop_loss: ", closed_trade.stop_loss)
    # print("direction: ", closed_trade.direction)
    # print("return_if_close_price_is 11: ", closed_trade.return_if_close_price_is(11))
    assert closed_trade.close_price == pytest.approx(10*1.29999 - 2)

    assert closed_trade.total_ret == pytest.approx(-0.29999)
    assert closed_trade.ret == pytest.approx((((10*1.29999 - 2)/10) - 1)*(-1))
    assert closed_trade.dividends_per_share == 2
    assert closed_trade.CLOSED == True
    assert closed_trade.close_cause == "STOP_LOSS_REACHED"


    total_commission = broker.commission_model.calculate(-2, 10) + broker.commission_model.calculate(-2, 11)
    dividends_payed = 2 * 2
    trade_dollar_return = (((10*1.29999 - 2)/10 - 1)*(-1)) * 2 * 10
    # Check balance after trade is closed (money should have been added back)
    assert portfolio.balance == pytest.approx(100 - total_commission - dividends_payed + trade_dollar_return)

    # Check margin account (money should have been freed up)
    assert portfolio.margin_account == 0



def test_handle_corp_actions():
    # FIRST ITERATION
    next(broker.market_data.tick)
    order_date = pd.to_datetime("2010-01-01")
    order_1 = Order(
        order_id=1,
        ticker="AAPL",
        amount=2,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=-1, # Wont trigger
        take_profit=1, # Wont trigger
        timeout=pd.to_datetime("2010-01-06")
    )


    orders_event = Event(event_type="ORDERS", data=[order_1], date=order_date)
    broker.process_orders(portfolio, orders_event)
    balance_after_initating_trade = 100 - 2*10 - broker.commission_model.calculate(2, 10)
    assert portfolio.balance == pytest.approx(balance_after_initating_trade)
    broker.handle_dividends(portfolio)
    balance_after_first_dividend = balance_after_initating_trade + 4
    assert portfolio.balance == pytest.approx(balance_after_first_dividend)

    # LATTER ITERATIONS
    end_date = pd.to_datetime("2010-01-04")
    while broker.market_data.cur_date < end_date:
        market_data_event = next(broker.market_data.tick)
        margin_acc_update_event = broker.manage_active_trades(portfolio)
        if margin_acc_update_event != None:
            portfolio.handle_margin_account_update(margin_acc_update_event)
        
        broker.handle_corp_actions(portfolio)
        broker.handle_dividends(portfolio)

    closed_trade = broker.blotter.closed_trades[0]

    assert closed_trade.close_date == pd.to_datetime("2010-01-03")
    assert closed_trade.close_price == pytest.approx(0) # 3 in dividends, makes the 0.4 take_profit barrier trigger
    assert portfolio.balance == pytest.approx(balance_after_first_dividend + 2*1)




def test_handle_interest_on_cash_and_margin_accounts():
    # FIRST ITERATION
    next(broker.market_data.tick)
    order_date = pd.to_datetime("2010-01-01")
    order_1 = Order(
        order_id=1,
        ticker="AAPL",
        amount=-2,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=-1, # Will not happen
        take_profit=1, # Will not happen
        timeout=pd.to_datetime("2010-01-06")
    )

    orders_event = Event(event_type="ORDERS", data=[order_1], date=order_date)
    broker.process_orders(portfolio, orders_event)

    original_margin_account_size = 10 * 2 * (1 + 0.5)
    assert portfolio.margin_account == pytest.approx(original_margin_account_size)
    balance_after_trade = 100 - original_margin_account_size - broker.commission_model.calculate(-2, 10)
    assert portfolio.balance == pytest.approx(balance_after_trade)

    # print("real balance: ", portfolio.balance)
    # print("calculated balance: ", balance_after_trade)
    
    broker.handle_interest_on_cash_and_margin_accounts(portfolio)

    total_account_size = portfolio.margin_account + portfolio.balance
    daily_rf_rate_1 = 0.001
    # print("interest rate given as var: ", daily_rf_rate_1)
    interest_payment_1 = total_account_size * daily_rf_rate_1
    # print("Total interest I think should be received: ", interest_payment_1)

    assert portfolio.balance == pytest.approx(balance_after_trade + interest_payment_1, rel=1e-4)

    # NEW TICK
    market_data_event = next(broker.market_data.tick)
    margin_acc_update_event = broker.manage_active_trades(portfolio)
    if margin_acc_update_event != None:
        # print("MARGIN ACCOUNT UPDATE")
        portfolio.handle_margin_account_update(margin_acc_update_event)
    
    broker.handle_interest_on_cash_and_margin_accounts(portfolio)
    daily_rf_rate_2 = 0.002
    interest_payment_2 = (portfolio.margin_account + portfolio.balance) * daily_rf_rate_2

    assert portfolio.balance == pytest.approx(balance_after_trade + interest_payment_1 + interest_payment_2, rel=1e-4)



def test_handle_interest_on_short_positions():
    # FIRST ITERATION
    next(broker.market_data.tick)
    order_date = pd.to_datetime("2010-01-01")
    order_1 = Order(
        order_id=1,
        ticker="AAPL",
        amount=-2,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=-1, # will not happen
        take_profit=1, # will not happen
        timeout=pd.to_datetime("2010-01-06")
    )

    orders_event = Event(event_type="ORDERS", data=[order_1], date=order_date)
    broker.process_orders(portfolio, orders_event)
    broker.handle_interest_on_short_positions(portfolio)


    next(broker.market_data.tick)
    broker.manage_active_trades(portfolio)
    broker.handle_interest_on_short_positions(portfolio)

    margin_interest_rate = 0.06/360
    assert portfolio.balance == pytest.approx(100 - 2*10*1.5 - broker.commission_model.calculate(-2, 10) - 20*margin_interest_rate - 20*margin_interest_rate)



def test_calculate_required_margin_account_size():

    # FIRST ITERATION
    next(broker.market_data.tick)
    order_date = pd.to_datetime("2010-01-01")
    order_1 = Order(
        order_id=1,
        ticker="MSFT",
        amount=-1,
        date=order_date,
        signal=Signal.from_nothing(),
        stop_loss=-1, # Will not happen
        take_profit=1, # Will not happen
        timeout=pd.to_datetime("2010-01-06")
    )

    orders_event = Event(event_type="ORDERS", data=[order_1], date=order_date)
    broker.process_orders(portfolio, orders_event)

    margin_account0 = broker.calculate_required_margin_account_size("open")
    assert margin_account0 == pytest.approx(10*1.5) 

    margin_account_update_event = broker.manage_active_trades(portfolio) 
    
    margin_account1 = broker.calculate_required_margin_account_size("close")
    assert margin_account1 == pytest.approx(10*1.5) # Initial margin requirement when in the money (even == in the money)


    if margin_account_update_event is not None:
        portfolio.handle_margin_account_update(margin_account_update_event) 
    broker.handle_corp_actions(portfolio)
    broker.handle_dividends(portfolio)
    broker.handle_interest_on_short_positions(portfolio)
    broker.handle_interest_on_cash_and_margin_accounts(portfolio)


    # NEW TICK, where the position gets out of the money
    market_data_event = next(broker.market_data.tick)
    margin_account_update_event = broker.manage_active_trades(portfolio) 

    # Calculte margin requriement
    margin_account2 = broker.calculate_required_margin_account_size("close")
    assert margin_account2 == pytest.approx(13 * 1.3) # Maintenance margin, when not in the money

    if margin_account_update_event is not None:
        portfolio.handle_margin_account_update(margin_account_update_event)

    assert portfolio.margin_account == pytest.approx(margin_account2)    

    broker.handle_corp_actions(portfolio)
    broker.handle_dividends(portfolio)
    broker.handle_interest_on_short_positions(portfolio)
    broker.handle_interest_on_cash_and_margin_accounts(portfolio)



    """
    2010-01-01,MSFT,10,10.5,9.5,10,2
    2010-01-02,MSFT,11,14,11,13,1
    2010-01-03,MSFT,12,14,11,13,0
    """

def test_equity_commission_model():

    def other_fees(amount, price, commission):
        us_clearing_fee_per_share = 0.00020 # 0.0000207
        us_transaction_fees_per_dollar = 0.0000207
        nyse_pass_through_fees_per_commission = 0.000175
        finra_pass_through_fees_per_commission = 0.00056
        finra_trading_activity_fee_per_share = 0.000119
        us_clearing_fees = us_clearing_fee_per_share * abs(amount)
        us_trasaction_fees = us_transaction_fees_per_dollar * abs(amount) * price
        nyse_pass_through_fees = nyse_pass_through_fees_per_commission * commission
        finra_pass_through_fees = finra_pass_through_fees_per_commission * commission
        finra_trading_activity_fee = finra_trading_activity_fee_per_share * abs(amount)

        return us_clearing_fees + us_trasaction_fees + nyse_pass_through_fees + finra_pass_through_fees + finra_trading_activity_fee


    # Test min
    min_commission = 0.35 # > 90 * 0.0035
    assert commission_model.calculate(90, 10) == pytest.approx(min_commission + other_fees(90, 10, min_commission))


    # test max
    max_commission = 0.1 # 1% of trade value, trade value = 10000 * 0.001 = 10 -> max commission = 0.1
    # per share commission = 0.0035 * 10000 = 35 -> so max commission of 0.1.
    assert commission_model.calculate(10000, 0.001) == pytest.approx(max_commission + other_fees(10000, 0.001, max_commission))

    # test in between
    normal_commission = 1000*0.0035
    assert commission_model.calculate(1000, 15) == pytest.approx(normal_commission + other_fees(1000, 15, normal_commission))




def test_dividend_payment_and_handling():
    # Dealt with under test_exit_of_short_trade_due_to_dividend_payment
 
    # Check return calculation

    # Check dividend payment
    pass



def test_trade():
    pass

def test_equity_slippage_model():
    pass


"""
date,ticker,open,high,low,close,dividends
2010-01-01,AAPL,10,10.5,9.5,10,2
2010-01-02,AAPL,10,11,9,10,1
2010-01-03,AAPL,11,11,8,10,0
2010-01-04,AAPL,10,10,10,10,0
"""





"""
AAPL SEP
Ticker  date        open                high                    low                 close
AAPL,   2010-01-04  ,30.49              ,30.643                 ,30.34              ,30.573             ,123432400.0,0.0,214.01,2018-06-19
AAPL,   2010-01-05  ,30.657             ,30.799                 ,30.464             ,30.626             ,150476200.0,0.0,214.38,2018-06-19
AAPL,   2010-01-06  ,30.626             ,30.747                 ,30.107             ,30.139             ,138040000.0,0.0,210.97,2018-06-19
AAPL,   2010-01-07  ,30.25              ,30.286                 ,29.864             ,30.083             ,119282800.0,0.0,210.58,2018-06-19
AAPL,   2010-01-08  ,30.043000000000006 ,30.286                 ,29.866             ,30.283             ,111902700.0,0.0,211.98,2018-06-19
AAPL,   2010-01-11  ,30.4               ,30.429                 ,29.779             ,30.016             ,115557400.0,0.0,210.11,2018-06-19
AAPL,   2010-01-12  ,29.884             ,29.967                 ,29.489             ,29.674             ,148614900.0,0.0,207.72,2018-06-19
AAPL,   2010-01-13  ,29.696             ,30.133000000000006     ,29.157             ,30.093000000000004 ,151473000.0,0.0,210.65,2018-06-19
AAPL,   2010-01-14  ,30.016             ,30.066                 ,29.86              ,29.919             ,108223500.0,0.0,209.43,2018-06-19
AAPL,   2010-01-15  ,30.133000000000006 ,30.229                 ,29.41              ,29.419             ,148516900.0,0.0,205.93,2018-06-19
AAPL,   2010-01-19  ,29.761             ,30.741                 ,29.606             ,30.72              ,182501900.0,0.0,215.04,2018-06-19
AAPL,   2010-01-20  ,30.701             ,30.793000000000006     ,29.929             ,30.246             ,153038200.0,0.0,211.725,2018-06-19
AAPL,   2010-01-21  ,30.297             ,30.473000000000006     ,29.601             ,29.725             ,152038600.0,0.0,208.072,2018-06-19
AAPL,   2010-01-22  ,29.54              ,29.643                 ,28.166             ,28.25              ,220441900.0,0.0,197.75,2018-06-19
AAPL,   2010-01-25  ,28.93              ,29.243                 ,28.599             ,29.011             ,266424900.0,0.0,203.075,2018-06-19
AAPL,   2010-01-26  ,29.421             ,30.53                  ,28.94              ,29.42              ,466777500.0,0.0,205.94,2018-06-19
AAPL,   2010-01-27  ,29.55              ,30.083                 ,28.504             ,29.698             ,430642100.0,0.0,207.884,2018-06-19
AAPL,   2010-01-28  ,29.276             ,29.357                 ,28.386             ,28.47              ,293375600.0,0.0,199.29,2018-06-19
AAPL,   2010-01-29  ,28.726             ,28.886                 ,27.179             ,27.438             ,311488100.0,0.0,192.063,2018-06-19
AAPL,   2010-02-01  ,27.481             ,28.0                   ,27.329             ,27.819000000000006 ,187469100.0,0.0,194.73,2018-06-19
AAPL,   2010-02-02  ,27.987             ,28.046                 ,27.626             ,27.98              ,174585600.0,0.0,195.86,2018-06-19
AAPL,   2010-02-03  ,27.881             ,28.6                   ,27.774             ,28.461             ,153832000.0,0.0,199.23,2018-06-19
AAPL,   2010-02-04  ,28.104             ,28.339                 ,27.367             ,27.436             ,189413000.0,0.0,192.05,2018-06-19


AAPL,   2010-02-05  ,27.518             ,28.0                   ,27.264             ,27.923             ,212576700.0,0.0,195.46,2018-06-19
AAPL,   2010-02-08  ,27.956             ,28.269                 ,27.714             ,27.731             ,119567700.0,0.0,194.12,2018-06-19
AAPL,   2010-02-09  ,28.06              ,28.214                 ,27.821             ,28.027             ,158221700.0,0.0,196.19,2018-06-19
AAPL,   2010-02-10  ,27.984             ,28.086                 ,27.751             ,27.874             ,92590400.0,0.0,195.116,2018-06-19
AAPL,   2010-02-11  ,27.84              ,28.536                 ,27.723000000000006 ,28.381             ,137586400.0,0.0,198.67,2018-06-19
AAPL,   2010-02-12  ,28.301             ,28.806                 ,27.929             ,28.626             ,163867200.0,0.0,200.38,2018-06-19
AAPL,   2010-02-16  ,28.849             ,29.099                 ,28.789             ,29.057             ,135934400.0,0.0,203.4,2018-06-19
AAPL,   2010-02-17  ,29.17              ,29.187                 ,28.694000000000006 ,28.936             ,109099200.0,0.0,202.55,2018-06-19
AAPL,   2010-02-18  ,28.804             ,29.127                 ,28.703000000000007 ,28.99              ,105706300.0,0.0,202.928,2018-06-19
AAPL,   2010-02-19  ,28.837             ,29.029                 ,28.73              ,28.81              ,103867400.0,0.0,201.67,2018-06-19
AAPL,   2010-02-22  ,28.906             ,28.929                 ,28.456             ,28.631             ,97640900.0,0.0,200.416,2018-06-19
AAPL,   2010-02-23  ,28.571             ,28.761                 ,27.959             ,28.151             ,143773700.0,0.0,197.05900000000003,2018-06-19
AAPL,   2010-02-24  ,28.319000000000006 ,28.777                 ,28.263             ,28.665             ,115141600.0,0.0,200.656,2018-06-19
AAPL,   2010-02-25  ,28.197             ,28.98                  ,28.127             ,28.857             ,166281500.0,0.0,202.0,2018-06-19
AAPL,   2010-02-26  ,28.911             ,29.31                  ,28.857             ,29.231             ,126865200.0,0.0,204.62,2018-06-19

"""