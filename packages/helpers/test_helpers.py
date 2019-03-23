import pandas as pd
import pytest
from dateutil.relativedelta import *
import numpy as np
import math
import pytest
from .helpers import get_acceptable_dates, select_row_closes_to_date, get_row_with_closest_date
from . helpers import get_most_up_to_date_10k_filing, get_most_up_to_date_10q_filing,\
    get_calendardate_x_quarters_ago, get_calendardate_index, forward_fill_gaps

sf1_art = None
sf1_arq = None

@pytest.fixture(scope='module', autouse=True)
def setup():
    global sf1_art, sf1_arq
    # Will be executed before the first test in the module
    sf1_art = pd.read_csv("../../datasets/testing/sf1_art.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"], index_col="datekey")
    sf1_art = sf1_art.sort_values(by="datekey", ascending=True)
    
    sf1_art["datekey"] = sf1_art.index # For testing purposes, not needed elsewere

    sf1_arq = pd.read_csv("../../datasets/testing/sf1_arq.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"], index_col="datekey")
    sf1_arq = sf1_arq.sort_values(by="datekey", ascending=True)

    sf1_arq["datekey"] = sf1_arq.index # For testing purposes
    
    yield
    # Will be executed after the last test in the module

def test_get_calendardate_x_quarters_ago():
    cur_reportperiod = pd.to_datetime("2003-03-31")
    
    assert get_calendardate_x_quarters_ago(cur_reportperiod, 5) == pd.to_datetime("2001-12-31")
    assert get_calendardate_x_quarters_ago(cur_reportperiod, 4) == pd.to_datetime("2002-03-31")
    assert get_calendardate_x_quarters_ago(cur_reportperiod, 7) == pd.to_datetime("2001-06-30")
    assert get_calendardate_x_quarters_ago(cur_reportperiod, 12) == pd.to_datetime("2000-03-31")
    
def test_get_most_up_to_date_10k_filing():
    global sf1_art
    date_2012_05_10 = pd.to_datetime("2012-05-10") # report period: 2012-03-31
    date_2012_08_09 = pd.to_datetime("2012-08-09") # report period: 2012-06-30 
    # (10K one year earlier has a correction released 2011-11-14)

    sf1_art_ntk = sf1_art.loc[sf1_art.ticker=="NTK"]
    sf1_art_aapl = sf1_art.loc[sf1_art.ticker=="AAPL"]

    assert get_most_up_to_date_10k_filing(sf1_art_ntk, date_2012_05_10, 1)["datekey"] == \
        pd.to_datetime("2011-05-12")

    filing_correction = get_most_up_to_date_10k_filing(sf1_art_ntk, date_2012_08_09, 1)
    assert filing_correction["datekey"] == pd.to_datetime("2011-11-14")
    assert filing_correction["calendardate"] == pd.to_datetime("2011-06-30")


def test_get_most_up_to_date_10q_filing():
    global sf1_arq
    date_2012_05_10 = pd.to_datetime("2012-05-10") # report period: 2012-03-31
    date_2012_08_09 = pd.to_datetime("2012-08-09") # report period: 2012-06-30 
    # (10K one year earlier has a correction released 2011-11-14)

    sf1_arq_ntk = sf1_arq.loc[sf1_arq.ticker=="NTK"]
    sf1_arq_aapl = sf1_arq.loc[sf1_arq.ticker=="AAPL"]

    assert get_most_up_to_date_10q_filing(sf1_arq_ntk, date_2012_05_10, 4)["datekey"] == \
        pd.to_datetime("2011-05-12")

    filing_correction = get_most_up_to_date_10q_filing(sf1_arq_ntk, date_2012_08_09, 4)
    assert filing_correction["datekey"] == pd.to_datetime("2011-11-14")
    assert filing_correction["calendardate"] == pd.to_datetime("2011-06-30")

    assert get_most_up_to_date_10q_filing(sf1_arq_ntk, date_2012_05_10, 0)["datekey"] == \
        date_2012_05_10

def test_get_calendardate_index():
    start = pd.to_datetime("2012-06-30")
    end = pd.to_datetime("2013-12-31")

    index = get_calendardate_index(start, end)

    desired = [pd.to_datetime("2012-06-30"), pd.to_datetime("2012-09-30"), pd.to_datetime("2012-12-31"), pd.to_datetime("2013-03-31"), pd.to_datetime("2013-06-30"), pd.to_datetime("2013-09-30"), pd.to_datetime("2013-12-31")]

    for date1, date2 in zip(index, desired):
        assert date1 == date2

def test_forward_fill_gaps():
    sf1_missing = pd.read_csv("../../datasets/testing/ffill_sf1_art.csv", parse_dates=["datekey", "calendardate", "reportperiod"], index_col="calendardate")
    # First three quarters of 2000 is missing (should be filled completely)
    # All four quarters of 2007 is missing (should not fill completely)
    sf1_miss_aapl = sf1_missing.loc[sf1_missing.ticker=="AAPL"]
    
    assert math.isnan(sf1_miss_aapl.loc["1998-06-30"]["capex"]) == True
    
    with pytest.raises(KeyError):
        row = sf1_miss_aapl.loc["2000-03-31"]
        row = sf1_miss_aapl.loc["2007-06-30"]

    
    sf1_filled = forward_fill_gaps(sf1_miss_aapl, 3)

    try:
        sf1_filled.loc["2000-03-31"]
        sf1_filled.loc["2000-06-30"]
        sf1_filled.loc["2000-09-30"]
        sf1_filled.loc["2007-03-31"]
        sf1_filled.loc["2007-03-31"]
        sf1_filled.loc["2007-06-30"]
        sf1_filled.loc["2007-09-30"]
    except KeyError as e:
        pytest.fail("KeyError raised when it should not have been")


    with pytest.raises(KeyError):
        row = sf1_filled.loc["2007-12-31"]

    assert math.isnan(sf1_filled.loc["1998-06-30"]["capex"]) == True


#____________________________________END____________________________________


@pytest.mark.skip(reason="Not used atm")
def test_get_row_with_closest_date():
    global sf1_art
    global sf1_arq

    print(sf1_art)

    date_cur = pd.to_datetime("2012-12-31")

    date_1y_ago = date_cur - relativedelta(years=+1)
    date_2y_ago = date_cur - relativedelta(years=+2)

    art_row_1y_ago = get_row_with_closest_date(sf1_art, date_1y_ago, 30)
    art_row_2y_ago = get_row_with_closest_date(sf1_art, date_2y_ago, 30)
    
    date_1q_ago = date_cur - relativedelta(months=+3)
    date_2q_ago = date_cur - relativedelta(months=+6)
    date_3q_ago = date_cur - relativedelta(months=+9)
    date_4q_ago = date_cur - relativedelta(years=+1)
    date_5q_ago = date_cur - relativedelta(months=+15)
    date_6q_ago = date_cur - relativedelta(months=+18)
    date_7q_ago = date_cur - relativedelta(months=+22)
    date_8q_ago = date_cur - relativedelta(years=+2)

    arq_row_cur = get_row_with_closest_date(sf1_arq, date_cur, 14)
    arq_row_1q_ago = get_row_with_closest_date(sf1_arq, date_1q_ago, 14)
    arq_row_2q_ago = get_row_with_closest_date(sf1_arq, date_2q_ago, 14)
    arq_row_3q_ago = get_row_with_closest_date(sf1_arq, date_3q_ago, 21)
    arq_row_4q_ago = get_row_with_closest_date(sf1_arq, date_4q_ago, 30)
    arq_row_5q_ago = get_row_with_closest_date(sf1_arq, date_5q_ago, 30)
    arq_row_6q_ago = get_row_with_closest_date(sf1_arq, date_6q_ago, 30)
    arq_row_7q_ago = get_row_with_closest_date(sf1_arq, date_7q_ago, 30)
    arq_row_8q_ago = get_row_with_closest_date(sf1_arq, date_8q_ago, 30)



@pytest.mark.skip(reason="Not used atm")
def test_get_acceptable_dates():
    # 2012-12-31
    # 2012-12-31 + 6 days = 2013-01-06
    # 2012-12-31 - 6 days = 2012-12-25
    date = pd.to_datetime("2012-12-31")
    acceptable_dates = get_acceptable_dates(date, 6)
    print(acceptable_dates)
    assert len(acceptable_dates) == 1 + 6*2
    assert acceptable_dates[0] == pd.to_datetime("2012-12-25")
    assert acceptable_dates[-1] == pd.to_datetime("2013-01-06")

@pytest.mark.skip(reason="Not used atm")
def test_select_row_closes_to_date():
    pass




"""
SF1_ART for NTK
calendardate    datekey     reportperiod
2010-12-31	    2011-03-31	2010-12-31
2011-03-31	    2011-05-12	2011-04-02

2011-06-30	    2011-08-09	2011-07-02  
2011-06-30	    2011-11-14	2011-07-02  # UPDATE

2011-09-30	    2011-11-15	2011-10-01
2011-12-31	    2012-03-29	2011-12-31
2012-03-31	    2012-05-10	2012-03-31

2012-06-30	    2012-08-09	2012-06-30
2012-09-30	    2012-11-08	2012-09-29
2012-12-31	    2013-03-15	2012-12-31
2013-03-31	    2013-05-09	2013-03-30
2013-06-30	    2013-08-01	2013-06-29
2013-09-30	    2013-11-07	2013-09-28
2013-12-31	    2014-03-11	2013-12-31
2014-03-31	    2014-05-05	2014-03-29
2014-06-30	    2014-08-04	2014-06-28
2014-09-30	    2014-11-03	2014-09-27
2014-12-31	    2015-03-02	2014-12-31
2015-03-31	    2015-05-04	2015-03-28
2015-06-30	    2015-08-03	2015-06-27
2015-09-30	    2015-11-03	2015-09-26
2015-12-31	    2016-02-29	2015-12-31
2016-03-31	    2016-05-12	2016-04-02

SF1_ARQ for NTK
calendardate    datekey     reportperiod
2010-12-31	    2011-03-31	2010-12-31
2011-03-31	    2011-05-12	2011-04-02

2011-06-30	    2011-08-09	2011-07-02
2011-06-30	    2011-11-14	2011-07-02 # UPDATE

2011-09-30	    2011-11-15	2011-10-01
2011-12-31	    2012-03-29	2011-12-31
2012-03-31	    2012-05-10	2012-03-31

2012-06-30	    2012-08-09	2012-06-30
2012-09-30	    2012-11-08	2012-09-29
2012-12-31	    2013-03-15	2012-12-31
2013-03-31	    2013-05-09	2013-03-30
2013-06-30	    2013-08-01	2013-06-29
2013-09-30	    2013-11-07	2013-09-28
2013-12-31	    2014-03-11	2013-12-31
2014-03-31	    2014-05-05	2014-03-29
2014-06-30	    2014-08-04	2014-06-28
2014-09-30	    2014-11-03	2014-09-27
2014-12-31	    2015-03-02	2014-12-31
2015-03-31	    2015-05-04	2015-03-28
2015-06-30	    2015-08-03	2015-06-27
2015-09-30	    2015-11-03	2015-09-26
2015-12-31	    2016-02-29	2015-12-31
2016-03-31	    2016-05-12	2016-04-02
2016-06-30	    2016-08-08	2016-07-02


"""