import pytest
import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime, timedelta

from packages.multiprocessing.engine import pandas_mp_engine
from sep_industry_features import add_indmom

sep = None
completed = False

@pytest.fixture(scope='module', autouse=True)
def setup():
    global sep
    global completed
    # Will be executed before the first test in the module
    sep = pd.read_csv("./datasets/testing/sep_prepared.csv", parse_dates=["date"], index_col="date")
    
    yield
    # Will be executed after the last test in the module
    if completed == True:
        sep.to_csv("./datasets/testing/sep_prepared.csv")


def test_add_indmom():
    global sep
    global completed

    sep = pandas_mp_engine(callback=add_indmom, atoms=sep, data=None, \
        molecule_key='sep', split_strategy= 'industry', \
            num_processes=4, molecules_per_process=1)
    
    completed = True

    sep.sort_values(by=["ticker", "date"], ascending=True, inplace=True)

    industry_sep = sep.loc[sep["industry"] == "Consumer Electronics"]
    sep_aapl = sep.loc[sep["ticker"] == "AAPL"]
    sep_ntk = sep.loc[sep["ticker"] == "NTK"]

    # assert sep.loc[sep.index == "1998-01-07", "mom1w"].mean() == pytest.approx(sep_aapl.loc["1998-01-07"]["ewmm"])
    assert sep_aapl.loc["2000-01-07"]["indmom"] == \
            pytest.approx(industry_sep.loc["2000-01-07", "mom12m_actual"].mean())

    assert sep_ntk.loc["2011-05-04"]["indmom"] == \
            pytest.approx(industry_sep.loc["2011-05-04", "mom12m_actual"].mean())



