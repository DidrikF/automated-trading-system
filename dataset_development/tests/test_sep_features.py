import pytest
import pandas as pd
import math
from dateutil.relativedelta import *

from ..helpers.helpers import get_x_past_months_of_data
from ..sep_features import add_sep_features
from ..multiprocessing.engine import pandas_mp_engine

"""
Each step is performed for each industry separately

Step-by-Step Dataset Construction:
1. Extend the SEP dataset with information usefull for sampling (most recent 10-K filing date, Industry classifications)
2. Use different sampling techniques to get monthly observations
    1. At first use timebars (sampling at a fixed time interval), but try to respect the different fiscal years
3. Calculate the various price and volume based features
4. Add inn SF1 and DAILY data
5. Compute features based on SF1
6. Select the features you want and combine into one ML ready dataset
"""

sep_prepared = None
sep_sampled = None
sep_featured = None
sf1_art = None

@pytest.fixture(scope='module', autouse=True)
def setup():
    global sep_prepared, sep_sampled, sep_featured, sf1_art

    # Will be executed before the first test in the module
    sep_prepared = pd.read_csv("../datasets/testing/sep_prepared.csv",  parse_dates=["date"], index_col="date")
    sep_sampled = pd.read_csv("../datasets/testing/sep_sampled.csv", parse_dates=["date"], index_col="date")
    sf1_art = pd.read_csv("../datasets/testing/sf1_art.csv", parse_dates=["calendardate", "datekey"], index_col="calendardate")

    yield
    # Will be executed after the last test in the module
    sep_featured.sort_values(by=["ticker", "date"], inplace=True)
    sep_featured.to_csv("../datasets/testing/sep_featured.csv")
    # sep_sampled.to_csv("./datasets/testing/sep_sampled.csv")


def test_add_return():
    global sep_prepared
    global sep_sampled
    global sep_featured
    global sf1_art

    print(type(sep_sampled), type(sep_prepared), type(sf1_art))

    sep_featured = pandas_mp_engine(callback=add_sep_features, atoms=sep_sampled, \
        data={'sep': sep_prepared, "sf1_art": sf1_art}, molecule_key='sep_sampled', split_strategy= 'ticker', \
            num_processes=1, molecules_per_process=1)

    sep_featured_aapl = sep_featured.loc[sep_featured.ticker == "AAPL"]
    sep_prepared_aapl = sep_prepared.loc[sep_prepared.ticker == "AAPL"]

    # Date0 : 2001-04-12 (Shifted 30 days)
    # Date0+1m : 2001-05-11
    # Date0+2m : 2001-06-11
    # Date0+3m : 2001-07-11
    # https://planetcalc.com/410/

    assert  sep_featured_aapl.loc["2001-04-12"]["return_1m"] == \
            (sep_prepared_aapl.loc["2001-05-11"]["adj_close"] / sep_prepared_aapl.loc["2001-04-12"]["adj_close"]) - 1
    assert  sep_featured_aapl.loc["2001-04-12"]["return_2m"] == \
            (sep_prepared_aapl.loc["2001-06-11"]["adj_close"] / sep_prepared_aapl.loc["2001-04-12"]["adj_close"]) - 1
    assert  sep_featured_aapl.loc["2001-04-12"]["return_3m"] == \
            (sep_prepared_aapl.loc["2001-07-11"]["adj_close"] / sep_prepared_aapl.loc["2001-04-12"]["adj_close"]) - 1

def test_add_mom1m_mom6m_mom12m_mom24m():
    global sep_prepared
    global sep_sampled
    global sep_featured
    
    sep_featured_aapl = sep_featured.loc[sep_featured.ticker == "AAPL"]
    sep_prepared_aapl = sep_prepared.loc[sep_prepared.ticker == "AAPL"]

    # Date0 : 2001-04-12 (Shifted 30 days)
    # Date0-30d : 2001-03-13 
    # Data0-182d: 2000-10-12 
    assert  sep_featured_aapl.loc["2001-04-12"]["mom6m"] == (sep_prepared_aapl.loc["2001-03-13"]["adj_close"] / sep_prepared_aapl.loc["2000-10-12"]["adj_close"]) - 1


def test_add_beta():
    global sep_featured
    # AAPL 2000-02-01:
    # Using 2 years of data
    # Cov:  0.003480979859486004, Var: 0.00291186561831057, Beta: 1.1943905852713281

    # AAPL 1999-02-08:
    # Using 1 years of data
    # Cov: 0.0032885050928968835, Var: 0.0028698769711752014, Beta: 1.1458697100699253
    # Confirmed manually.

    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "2000-02-01")]\
            .iloc[0]["beta"] == pytest.approx(1.1943905852713281)

def test_add_betasq():
    global sep_featured
    # AAPL 2000-02-01:
    # Using 2 years of data
    # Cov:  0.003480979859486004, Var: 0.00291186561831057, Beta: 1.1943905852713281
    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "2000-02-01")]\
            .iloc[0]["betasq"] == pytest.approx(1.1943905852713281**2)


def test_add_idiovol():
    global sep_featured
    # 1 Year of apple ending 1999-02-08
    # Confirmed STD of diff between stock and market return = 0.045536866012663164
    # This was in fact the output from the code.

    # 2 Years ending 2000-02-01, has std: 0.05365805443425454
    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "2000-02-01")]\
            .iloc[0]["idiovol"] == pytest.approx(0.05365805443425454)


def test_add_chmom():
    global sep_featured
    # Date0 : 2001-04-12
    # Date0-30d : 2001-03-13 
    # Data0-182d: 2000-10-12 
    # Data0-(182+30)d: 2000-09-12
    # Data0-365d: 2000-04-12

    sep_prepared_aapl = sep_prepared.loc[sep_prepared.ticker == "AAPL"]

    chmom_at_2001_04_12 = (sep_prepared_aapl.loc["2001-03-13"]["adj_close"] / sep_prepared_aapl.loc["2000-10-12"]["adj_close"] - 1) - \
            (sep_prepared_aapl.loc["2000-09-12"]["adj_close"] / sep_prepared_aapl.loc["2000-04-12"]["adj_close"] - 1)

    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "2001-04-12")].iloc[0]["chmom"] == chmom_at_2001_04_12


def test_add_illiquidity():
    global sep_featured
    # AAPL 1999-04-08 has confirmed illiquidity: -1.5139424798827458e-11

    # When volume is zero, (absolute return / dollar volume) becomes NaN, which is the case for several dates in the dataset.
    # When calculation the mean, these values are skipped.
    
    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "1999-04-08")].iloc[0]["ill"] == pytest.approx(-1.5139424798827458e-11)


def test_add_dolvol():
    global sep_featured
    # AAPL at 1998-03-09 has confirmed dolvol = 21.807906903155157


    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "1998-03-09")]\
        .iloc[0]["dolvol"] == pytest.approx(21.807906903155157)


def test_add_dy():
    global sep_featured
    # FCX 1999-05-12 has confirmed dy = 1.904597186490487e-11
    assert sep_featured.loc[(sep_featured.ticker == "FCX") & (sep_featured.index == "1999-05-12")]\
        .iloc[0]["dy"] == pytest.approx(1.904597186490487e-11)


def test_add_maxret():
    global sep_featured
    # AAPL at 1998-02-09 has confirmed maxret = 0.04661654135338322
    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "1998-02-09")]\
        .iloc[0]["maxret"] == pytest.approx(0.04661654135338322)


def test_add_mve(): # ln of marketcap
    global sep_featured
    global sf1_art
    # AAPL at 1998-02-09
    sep_aapl_row = sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "1998-02-09")].iloc[0]
    sf1_aapl = sf1_art.loc[sf1_art.ticker == "AAPL"]
    datekey0 = sep_aapl_row["datekey"]
    sf1_art_row = sf1_aapl.loc[sf1_aapl.datekey == datekey0]
    marketcap = sf1_art_row["sharefactor"]*sf1_art_row["sharesbas"]*sep_aapl_row["close"]

    assert sep_aapl_row["mve"] == math.log(marketcap)


def test_add_retvol():
    global sep_featured
    # AAPL at 1998-02-09 has confirmed retvol = 0.023875469881802412
    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "1998-02-09")]\
        .iloc[0]["retvol"] == pytest.approx(0.023875469881802412)


def test_add_std_dolvol():
    global sep_featured
    test_date = pd.to_datetime("1998-02-09")
    sep_prepared_aapl = sep_prepared[(sep_prepared.ticker == "AAPL")]
    sep_past_month = get_x_past_months_of_data(sep_prepared_aapl, test_date, 1)
    print(sep_past_month)
    mean_price = (sep_past_month["close"]+sep_past_month["open"])/2
    print(mean_price)
    std_dolvol = (mean_price*sep_past_month["volume"]).std() * math.sqrt(22)

    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "1998-02-09")].iloc[0]["std_dolvol"] == \
        std_dolvol


def test_add_std_turn():
    global sep_featured
    # AAPL at 1998-02-09 has confirmed std_turn = 0.07019337356877098
    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "1998-02-09")]\
        .iloc[0]["std_turn"] == pytest.approx(0.07019337356877098) 



def test_add_turn():
    global sep_featured
    # AAPL at 1998-04-09 has confirmed turn = 0.5562349274431175
    assert sep_featured.loc[(sep_featured.ticker == "AAPL") & (sep_featured.index == "1998-04-09")]\
        .iloc[0]["turn"] == pytest.approx(0.5562349274431175) 


def test_add_zerotrade():
    global sep_featured
    # NTK at 2011-05-12 has confirmed zero_trade = 8.639755267801805
    assert sep_featured.loc[(sep_featured.ticker == "NTK") & (sep_featured.index == "2011-05-12")]\
        .iloc[0]["zerotrade"] == pytest.approx(8.639755267801805)