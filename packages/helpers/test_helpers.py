import pandas as pd
import pytest
from dateutil.relativedelta import *
from .helpers import get_row_with_closest_date, get_acceptable_dates, select_row_closes_to_date
from . helpers import get_most_up_to_date_10k_filing, get_most_up_to_date_10q_filing, get_report_period_x_quarters_ago

sf1_art = None
sf1_arq = None

@pytest.fixture(scope='module', autouse=True)
def setup():
    global sf1_art_featured
    # Will be executed before the first test in the module
    sf1_art = pd.read_csv("../../datasets/testing/sf1_art.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"], index_col="datekey")
    sf1_arq = pd.read_csv("../../datasets/testing/sf1_arq.csv", parse_dates=["datekey"], index_col="datekey")
    yield
    # Will be executed after the last test in the module

def test_get_report_period_x_quarters_ago():
    cur_reportperiod = pd.to_datetime("2003-03-31")
    # CONTINUE HERE...

def test_get_most_up_to_date_10k_filing():
    global sf1_art
    cur_date = pd.to_datetime("2012-05-10")

    filing_1y_ago = get_most_up_to_date_10k_filing(sf1_art, cur_date, 1)


def test_get_most_up_to_date_10q_filing():
    global sf1_arq


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