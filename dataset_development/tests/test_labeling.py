import pandas as pd
import pytest
import math
from dateutil.relativedelta import *

import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from datetime import datetime

from ..labeling import equity_risk_premium_labeling, add_labels_via_triple_barrier_method
from ..multiprocessing.engine import pandas_mp_engine
from ..utils.visualization import visualize_triple_barrier_method, candlestick_chart

"""
Test datasets:
/datasets/testing/...
Complete data for for AAPL, NTK (Consumer Electronics) and FCX (Copper)
sep.csv
sf1_art.csv
sf1_arq.csv
"""

sep_featured = None
sep_labeled = None
sep = None
sep_aapl = pd.DataFrame()

@pytest.fixture(scope='module', autouse=True)
def setup():
    global sep, sep_featured, sep_labeled, sep_aapl
    
    sep_featured = pd.read_csv("../datasets/testing/sep_featured.csv", parse_dates=["date", "datekey", "timeout"], index_col="date", low_memory=False)
    sep = pd.read_csv("../datasets/testing/sep.csv", parse_dates=["date"], index_col="date", low_memory=False)
    
    sep_aapl = sep.loc[sep.ticker == "AAPL"]
    sep_index = pd.date_range(sep.index.min(), sep.index.max())
    sep_aapl = sep_aapl.reindex(sep_index)
    sep_aapl["close"] = sep_aapl["close"].fillna(method="ffill")

    yield
    
    # Will be executed after the last test in the module
    if isinstance(sep_labeled, pd.DataFrame):
        sep_labeled.sort_values(by=["ticker", "date"], inplace=True)
        sep_labeled.to_csv("../datasets/testing/sep_labeled.csv")



def test_equity_risk_premium_labeling():
    global sep_featured, sep_labeled, sep_aapl

    tb_rate = pd.read_csv("../datasets/macro/t_bill_rate_3m.csv", parse_dates=["date"] ,index_col="date", low_memory=False)

    sep_aapl_labeled = pandas_mp_engine(callback=equity_risk_premium_labeling, atoms=sep_featured, \
        data=None, molecule_key='sep_featured', split_strategy='ticker', \
                num_processes=1, molecules_per_process=1, tb_rate=tb_rate)

    date0 = pd.to_datetime("2013-05-24")
    date1 = date0 + relativedelta(days=30)
    
    return_1m = (sep_aapl.loc[date1]["close"] / sep_aapl.loc[date0]["close"]) - 1
    rf_rate = tb_rate.loc[date0]["rate"] / 3

    erp_1m = return_1m - rf_rate

    assert sep_aapl_labeled.loc[date0]["erp_1m"] == erp_1m

    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(sep_aapl_labeled.head(10))
    assert False
    """


def test_add_labels_via_triple_barrier_method():
    global sep, sep_featured, sep_aapl

    sep_triple_barrier = pandas_mp_engine(callback=add_labels_via_triple_barrier_method, atoms=sep_featured, \
        data={'sep': sep}, molecule_key='sep_featured', split_strategy= 'ticker', \
            num_processes=1, molecules_per_process=1, ptSl=[0.8,-0.8], min_ret=None)

    sep_triple_barrier.to_csv("../datasets/testing/sep_triple_barrier.csv")

    # Sample Dates
    date0 = pd.to_datetime("2004-08-05")
    date1 = pd.to_datetime("2004-09-03")
    date2 = pd.to_datetime("2004-10-05")
    date3 = pd.to_datetime("2004-11-05")
    dates = [date0, date1, date2, date3]

    visualize_triple_barrier_method(2, 2, dates, "AAPL", sep_triple_barrier, sep_aapl)


