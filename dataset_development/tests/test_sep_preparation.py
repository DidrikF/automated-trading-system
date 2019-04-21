import pytest
import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime, timedelta

from packages.multiprocessing.engine import pandas_mp_engine
from sep_features import add_weekly_and_12m_stock_returns,\
        add_equally_weighted_weekly_market_returns,\
        dividend_adjusting_prices_backwards, dividend_adjusting_prices_forwards


sep = None


@pytest.fixture(scope='module', autouse=True)
def setup():
    global sep
    # Will be executed before the first test in the module
    sep = pd.read_csv("./datasets/testing/sep_extended.csv", parse_dates=["date"], index_col="date")
    # sep = sep.drop(columns='index')
    # sep = sep.drop(sep.columns[sep.columns.str.contains('unnamed',case = False)],axis = 1)
    
    yield
    # Will be executed after the last test in the module
    sep.sort_values(by=["ticker", "date"], ascending=True, inplace=True)
    sep.to_csv("./datasets/testing/sep_prepared.csv")



def test_dividend_adjusting_prices_backwards():
    """
    Note that when dividend adjusting close prices as described here:
    https://blog.quandl.com/guide-to-stock-price-calculation?fbclid=IwAR1yzVApBnMk-acsWKSzTRp-rSMdY09HHEz0jl-0f2XGkywZ4zmTcowEWGg
    it is implicit in the adjustment that dividends are reinvested in the stock.
    This must be accounted for when testing.
    """
    global sep
    sep_adjusted = pandas_mp_engine(callback=dividend_adjusting_prices_backwards, atoms=sep, data=None, \
    molecule_key='sep', split_strategy= 'ticker', \
        num_processes=1, molecules_per_process=1)

    sep = sep_adjusted # For later tests to have the adj_close column available.

    sep_fcx = sep_adjusted.loc[sep_adjusted.ticker=="FCX"]

    # date0: 2010-01-04
    # date1: 2010-05-03
    sep_fcx = sep_fcx.loc["2010-01-04":"2010-05-03"]

    return_of_reinvested_dividend = 0
    for date, row in sep_fcx.iterrows():
        stock_appreciation_percentage = (sep_fcx.iloc[-1]["adj_close"] / row["adj_close"]) - 1
        return_of_reinvested_dividend += row["dividends"] * (1 + stock_appreciation_percentage)

    total_return = (sep_fcx.loc["2010-05-03"]["close"] - sep_fcx.loc["2010-01-04"]["close"] + \
        return_of_reinvested_dividend) / sep_fcx.loc["2010-01-04"]["close"]

    total_return_from_adjusted_close = (sep_fcx.loc["2010-05-03"]["adj_close"] / \
        sep_fcx.loc["2010-01-04"]["adj_close"]) - 1

    assert total_return_from_adjusted_close == pytest.approx(total_return)


@pytest.mark.skip()
def test_dividend_adjusting_prices_forwards():
    """
    Note that when dividend adjusting close prices as described here:
    https://blog.quandl.com/guide-to-stock-price-calculation?fbclid=IwAR1yzVApBnMk-acsWKSzTRp-rSMdY09HHEz0jl-0f2XGkywZ4zmTcowEWGg
    it is implicit in the adjustment that dividends are reinvested in the stock.
    This must be accounted for when testing.
    """
    global sep
    sep_adjusted = pandas_mp_engine(callback=dividend_adjusting_prices_forwards, atoms=sep, data=None, \
    molecule_key='sep', split_strategy= 'ticker', \
        num_processes=1, molecules_per_process=1)


    sep_fcx = sep_adjusted.loc[sep_adjusted.ticker=="FCX"]

    # date0: 2010-01-04
    # date1: 2010-05-03
    sep_fcx = sep_fcx.loc["2010-01-04":"2010-05-03"]

    return_of_reinvested_dividend = 0
    for date, row in sep_fcx.iterrows():
        stock_appreciation_percentage = (sep_fcx.iloc[-1]["adj_close_forward"] / row["adj_close_forward"]) - 1
        return_of_reinvested_dividend += row["dividends"] * (1 + stock_appreciation_percentage)

    total_return = (sep_fcx.loc["2010-05-03"]["close"] - sep_fcx.loc["2010-01-04"]["close"] + \
        return_of_reinvested_dividend) / sep_fcx.loc["2010-01-04"]["close"]


    total_return_from_adjusted_close = (sep_fcx.loc["2010-05-03"]["adj_close_forward"] / \
        sep_fcx.loc["2010-01-04"]["adj_close_forward"]) - 1

    assert total_return_from_adjusted_close == pytest.approx(total_return)

def test_add_weekly_and_12m_stock_returns():
    global sep

    # First calculate weekly returns for each ticker for all dates
    sep = pandas_mp_engine(callback=add_weekly_and_12m_stock_returns, atoms=sep, data=None, \
        molecule_key='sep', split_strategy= 'ticker', \
            num_processes=4, molecules_per_process=1)

    sep.sort_values(by=["ticker", "date"], ascending=True, inplace=True)
    sep_aapl = sep.loc[sep["ticker"] == "AAPL"]


    # assert ((sep_aapl.loc["1998-01-07"]["close"] / sep_aapl.loc["1997-12-31"]["close"]) - 1) == pytest.approx(sep_aapl.loc["1998-01-07"]["mom1w"]) # sep_extended does not contain the dates
    assert ((sep_aapl.loc["1999-03-10"]["adj_close"] / sep_aapl.loc["1999-03-3"]["adj_close"]) - 1) == \
            pytest.approx(sep_aapl.loc["1999-03-10"]["mom1w"])


    # date0 = pd.to_datetime("2001-01-30")
    # date0_1m_back = date0 - relativedelta(days=30)
    # date0_12m_back = date0 - relativedelta(days=365)

    # print(date0) # 2001-01-30
    # print(date0_1m_back) # 2000-12-29
    # print(date0_12m_back) # 2000-01-31

    return_12m = (sep_aapl.loc["2001-01-30"]["adj_close"] / sep_aapl.loc["2000-01-31"]["adj_close"]) - 1

    assert sep_aapl.loc["2001-01-30"]["mom12m_actual"] == pytest.approx(return_12m)


def test_add_equally_weighted_weekly_market_returns():
    global sep

    # First calculate weekly returns for each ticker for all dates, done in previous test

    sep = pandas_mp_engine(callback=add_equally_weighted_weekly_market_returns, atoms=sep, data=None, \
        molecule_key='sep', split_strategy= 'date', \
            num_processes=4, molecules_per_process=1)

    # sep = sep.sort_values(by=["ticker", "date"], ascending=True)
    sep_aapl = sep.loc[sep["ticker"] == "AAPL"]

    print(sep_aapl["2003-04"])

    # assert sep.loc[sep.index == "1998-01-07", "mom1w"].mean() == pytest.approx(sep_aapl.loc["1998-01-07"]["ewmm"])
    assert sep.loc[sep.index == "2000-01-07", "mom1w"].mean() == pytest.approx(sep_aapl.loc["2000-01-07"]["mom1w_ewa_market"])

