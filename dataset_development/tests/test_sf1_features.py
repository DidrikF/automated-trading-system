import pandas as pd
import pytest
import numpy as np
import math

from ..sf1_features import add_sf1_features
from ..processing.engine import pandas_mp_engine
from ..helpers.helpers import forward_fill_gaps, get_most_up_to_date_10q_filing, get_most_up_to_date_10k_filing


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

sf1_art_featured = None
sf1_art = None
sf1_arq = None
metadata = None


sf1_art_aapl_filled = None # Needed to conduct tests
sf1_arq_aapl_filled = None # Needed to conduct tests
sf1_art_aapl_index_snapshot = None # Needed to conduct tests
sf1_art_aapl = None
sf1_arq_aapl = None
sf1_art_aapl_featured = None
sf1_art_ntk_featured = None
sf1_art_aapl_done = None

art_row_cur = None
art_row_1y_ago = None

arq_row_cur = None
arq_row_1q_ago = None
arq_row_2q_ago = None
arq_row_3q_ago = None
arq_row_4q_ago = None
arq_row_5q_ago = None
arq_row_6q_ago = None
arq_row_7q_ago = None

check_row = None

features = ["roaq", "chtx", "rsup", "sue", "cinvest", "nincr", "roavol", "cashpr", "cash", "bm", "currat", "depr", "ep", "lev", "quick",\
    "rd_sale", "roic", "salecash", "saleinv", "salerec", "sp", "tb", "sin", "tang", "debtc_sale", "eqt_marketcap", "dep_ppne", "tangibles_marketcap", \
        "agr", "cashdebt", "chcsho", "chinv", "egr", "gma", "invest", "lgr", "operprof", "pchcurrat", "pchdepr", "pchgm_pchsale", "pchquick", "pchsale_pchinvt", \
            "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv", "rd", "roeq", "sgr", "grcapx", "chtl_lagat", "chlt_laginvcap", "chlct_lagat", "chint_lagat",\
                "chinvt_lagsale", "chint_lagsgna", "chltc_laginvcap", "chint_laglt", "chdebtnc_lagat", "chinvt_lagcor", "chppne_laglt", "chpay_lagact",\
                     "chint_laginvcap", "chinvt_lagact", "pchppne", "pchlt", "pchint", "chdebtnc_ppne", "chdebtc_sale", "age", "ipo", "profitmargin",\
                          "chprofitmargin", "industry", "change_sales", "ps"]

@pytest.fixture(scope='module', autouse=True)
def setup():

    global sf1_art_featured, sf1_art, sf1_arq, metadata, sf1_art_aapl, sf1_arq_aapl

    sf1_art = pd.read_csv("../datasets/testing/lacking_sf1_art.csv", parse_dates=["calendardate", "datekey", "reportperiod"],\
        index_col="calendardate", low_memory=False)
    sf1_arq = pd.read_csv("../datasets/testing/lacking_sf1_arq.csv", parse_dates=["calendardate", "datekey", "reportperiod"],\
        index_col="calendardate", low_memory=False)
    metadata = pd.read_csv("../datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", parse_dates=["firstpricedate"], low_memory=False)

    sf1_art_aapl = sf1_art.loc[sf1_art.ticker=="AAPL"]
    sf1_arq_aapl = sf1_arq.loc[sf1_arq.ticker=="AAPL"]

    yield
    # Will be executed after the last test in the module

    # sf1_art_featured.to_csv("./datasets/testing/sf1_art_featured.csv", index=False)


def test_running_add_sf1_features():
    global sf1_art_featured, sf1_art, sf1_arq, metadata, sf1_art_aapl_featured, sf1_art_ntk_featured

    sf1_art_featured = pandas_mp_engine(callback=add_sf1_features, atoms=sf1_art, \
        data={"sf1_arq": sf1_arq, 'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'ticker', \
            num_processes=1, molecules_per_process=1)

    sf1_art_aapl_featured = sf1_art_featured.loc[sf1_art_featured.ticker=="AAPL"]
    sf1_art_ntk_featured = sf1_art_featured.loc[sf1_art_featured.ticker=="NTK"]

    sf1_art_featured = sf1_art_featured.sort_values(by=["ticker", "calendardate", "datekey"])
    
    sf1_art_aapl_featured = sf1_art_aapl_featured.sort_values(by=["calendardate", "datekey"])
    
    # Only temporary...
    cols = list(set(features).intersection(set(sf1_art_aapl_featured.columns)))
    sf1_art_aapl_only_features = sf1_art_aapl_featured[cols]


    sf1_art_featured.to_csv("../datasets/testing/sf1_art_featured_snapshot.csv") # Need to work some more with the testing and then I remove this line to not compremise the correct snapshot...
    sf1_art_aapl_featured.to_csv("../datasets/testing/sf1_art_aapl_featured_snapshot.csv")
    sf1_art_aapl_only_features.to_csv("../datasets/testing/sf1_art_aapl_only_features.csv")



""" MAKE NEW SETUP TEST """

# I can have multiple setup functions
def test_setup_aapl_2001_03_31():
    """ This setup function is very important for later tests to be accurate! """
    global sf1_art_aapl_featured, sf1_arq_aapl_filled, sf1_art_aapl, sf1_arq_aapl
    global arq_row_cur, arq_row_1q_ago, arq_row_2q_ago, arq_row_3q_ago, arq_row_4q_ago, arq_row_5q_ago,\
        arq_row_6q_ago, arq_row_7q_ago
    global art_row_cur, art_row_1y_ago
    global check_row
    
    sf1_art_aapl_filled = forward_fill_gaps(sf1_art_aapl, 3)
    sf1_art_aapl_done = sf1_art_aapl_filled.reset_index() # this should be the same dataframe used in add_sf1_features()
    sf1_arq_aapl_filled = forward_fill_gaps(sf1_arq_aapl, 3)
    
    sf1_art_aapl_done["caldate"] = sf1_art_aapl_done.calendardate
    sf1_arq_aapl_filled["caldate"] = sf1_arq_aapl_filled.index # NOT WORKING!!!! ????

    # Remove when completely sure they are correct. Then make a test comparing against these
    sf1_art_aapl_done.to_csv("../datasets/testing/sf1_art_aapl_done_snapshot.csv") # This look good
    sf1_arq_aapl_filled.to_csv("../datasets/testing/sf1_arq_aapl_filled_snapshot.csv") # This look good

    # Select some rows and do some assertions to verify things are selected as they are supposed to

    print(sf1_art_aapl_featured.loc[sf1_art_aapl_featured.index == "2001-03-31"])
    """
    AAPL,ARQ,1997-09-30,1997-12-05 
    AAPL,ARQ,1997-12-31,1998-02-09 
    AAPL,ARQ,1998-03-31,1998-05-11 
    AAPL,ARQ,1998-06-30,1998-08-10 
    AAPL,ARQ,1998-09-30,1998-12-23 
    AAPL,ARQ,1998-12-31,1999-02-08 
    AAPL,ARQ,1999-03-31,1999-05-11 
    AAPL,ARQ,1999-06-30,1999-08-06 
    AAPL,ARQ,1999-09-30,1999-12-22 
    AAPL,ARQ,1999-12-31,2000-02-01 
    AAPL,ARQ,2000-12-31,2001-02-12 
    AAPL,ARQ,2001-03-31,2001-05-14 
    AAPL,ARQ,2001-06-30,2001-08-13 
    AAPL,ARQ,2001-09-30,2001-12-21 
    AAPL,ARQ,2001-12-31,2002-02-11 
    AAPL,ARQ,2002-03-31,2002-05-14 
    AAPL,ARQ,2002-06-30,2002-08-09 
    AAPL,ARQ,2002-09-30,2002-12-19 
    AAPL,ARQ,2002-12-31,2003-02-10 
    """
    check_row = sf1_art_aapl_featured.loc[sf1_art_aapl_featured.index == "2001-03-31"].iloc[-1]
    
    art_row_cur = sf1_art_aapl_done.loc[sf1_art_aapl_done.calendardate == "2001-03-31"].iloc[-1] # You can swap out this date, and it should work
    caldate = art_row_cur["caldate"]
    datekey = art_row_cur["datekey"]
    
    art_row_1y_ago = get_most_up_to_date_10k_filing(sf1_art_aapl_done, caldate, datekey, 1)

    arq_row_cur = get_most_up_to_date_10q_filing(sf1_arq_aapl_filled, caldate, datekey, 0)
    arq_row_1q_ago = get_most_up_to_date_10q_filing(sf1_arq_aapl_filled, caldate, datekey, 1)
    arq_row_2q_ago = get_most_up_to_date_10q_filing(sf1_arq_aapl_filled, caldate, datekey, 2)
    arq_row_3q_ago = get_most_up_to_date_10q_filing(sf1_arq_aapl_filled, caldate, datekey, 3)    
    arq_row_4q_ago = get_most_up_to_date_10q_filing(sf1_arq_aapl_filled, caldate, datekey, 4)
    arq_row_5q_ago = get_most_up_to_date_10q_filing(sf1_arq_aapl_filled, caldate, datekey, 5)
    arq_row_6q_ago = get_most_up_to_date_10q_filing(sf1_arq_aapl_filled, caldate, datekey, 6)
    arq_row_7q_ago = get_most_up_to_date_10q_filing(sf1_arq_aapl_filled, caldate, datekey, 7)

    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(sf1_art_aapl_done)
        print(check_row)
        print(art_row_cur, '\n')
        print(arq_row_cur, '\n')
        print(arq_row_1q_ago, '\n')
    """

    # With respect to 2001-03-31 for AAPL
    assert arq_row_cur["datekey"] == pd.to_datetime("2001-05-14")
    assert art_row_1y_ago["datekey"] == pd.to_datetime("2000-02-01")

    assert arq_row_1q_ago["datekey"] == pd.to_datetime("2001-02-12")
    assert arq_row_2q_ago["datekey"] == pd.to_datetime("2000-02-01")
    assert arq_row_3q_ago["datekey"] == pd.to_datetime("2000-02-01")
    assert arq_row_4q_ago["datekey"] == pd.to_datetime("2000-02-01")
    assert arq_row_5q_ago["datekey"] == pd.to_datetime("2000-02-01")
    assert arq_row_6q_ago["datekey"] == pd.to_datetime("1999-12-22")
    assert arq_row_7q_ago["datekey"] == pd.to_datetime("1999-08-06")
    

def test_add_sf1_features_roaq():
    # Select rows
    global sf1_art_aapl_featured # Assert against this
    # global sf1_art_aapl_done # Use this can re-calclate features
    global arq_row_cur, arq_row_1q_ago
    global check_row # check aginst this
    # Do calc and compare
    roaq = arq_row_cur["netinc"] / arq_row_1q_ago["assets"]

    print(arq_row_cur["netinc"], arq_row_1q_ago["assets"], roaq)

    assert check_row["roaq"] == roaq


def test_add_sf1_features_chtx():
    global arq_row_cur, arq_row_4q_ago, check_row

    chtx = (arq_row_cur["taxexp"] / arq_row_4q_ago["taxexp"]) - 1
    assert check_row["chtx"] == chtx


def test_add_sf1_features_rsup():
    global arq_row_cur, arq_row_4q_ago, check_row

    rsup = (arq_row_cur["revenueusd"] - arq_row_4q_ago["revenueusd"]) / arq_row_cur["marketcap"]
    assert check_row["rsup"] == rsup

def test_add_sf1_features_cash_sue():    
    global arq_row_cur, arq_row_4q_ago, check_row
    
    sue = (arq_row_cur["netinc"] - arq_row_4q_ago["netinc"]) / arq_row_cur["marketcap"]
    assert check_row["sue"] == sue

def test_add_sf1_features_cinvest():
    global arq_row_cur, arq_row_1q_ago, arq_row_2q_ago, arq_row_3q_ago, arq_row_4q_ago

    # arq_row_cur["revenueusd"] = np.nan

    if arq_row_cur["revenueusd"] != 0:
        chppne_sales_cur = (arq_row_cur["ppnenet"] - arq_row_1q_ago["ppnenet"]) / arq_row_cur["revenueusd"]
    else:
        chppne_sales_cur = (arq_row_cur["ppnenet"] - arq_row_1q_ago["ppnenet"]) * 0.01
        print("SCALED 1")
    
    # Previous three quarters of chppne/sales
    if arq_row_1q_ago["revenueusd"] != 0:
        chppne_sales_q_1 = (arq_row_1q_ago["ppnenet"] - arq_row_2q_ago["ppnenet"]) / arq_row_1q_ago["revenueusd"]
    else:
        chppne_sales_q_1 = (arq_row_1q_ago["ppnenet"] - arq_row_2q_ago["ppnenet"]) * 0.01
        print("SCALED 2")

    if arq_row_2q_ago["revenueusd"] != 0:
        chppne_sales_q_2 = (arq_row_2q_ago["ppnenet"] - arq_row_3q_ago["ppnenet"]) / arq_row_2q_ago["revenueusd"]
    else:
        chppne_sales_q_2 = (arq_row_2q_ago["ppnenet"] - arq_row_3q_ago["ppnenet"]) * 0.01
        print("SCALED 3")

    if arq_row_3q_ago["revenueusd"] != 0:
        chppne_sales_q_3 = (arq_row_3q_ago["ppnenet"] - arq_row_4q_ago["ppnenet"]) / arq_row_3q_ago["revenueusd"]
    else:
        chppne_sales_q_3 = (arq_row_3q_ago["ppnenet"] - arq_row_4q_ago["ppnenet"]) * 0.01
        print("SCALED 4")

    cinvest = chppne_sales_cur - ( (chppne_sales_q_1 + chppne_sales_q_2 + chppne_sales_q_3) / 3 )

    print(cinvest)

    assert check_row["cinvest"] == cinvest


def test_nincr_calculation():
    arq_rows = [100, 90, 80, 70, 20, 10, 5, 15]
    nr_of_earnings_increases = 0
    for i in range(4):
        cur_netinc = arq_rows[i]
        prev_netinc = arq_rows[i+4]
        if cur_netinc > prev_netinc:
            nr_of_earnings_increases += 1
        else:
            break
    
    assert nr_of_earnings_increases == 4

def test_add_sf1_features_nincr():
    global arq_row_cur, arq_row_1q_ago, arq_row_2q_ago, arq_row_3q_ago, arq_row_4q_ago, arq_row_5q_ago, arq_row_6q_ago, arq_row_7q_ago
    arq_rows = [arq_row_cur, arq_row_1q_ago, arq_row_2q_ago, arq_row_3q_ago, arq_row_4q_ago, arq_row_5q_ago, arq_row_6q_ago, arq_row_7q_ago]
    global sf1_arq_aapl, sf1_arq_aapl_filled
    global check_row

    print_df = sf1_arq_aapl[["datekey", "netinc"]]
    print_df = print_df.loc[print_df.index <= art_row_cur["calendardate"]]    

    print_df_2 = sf1_arq_aapl_filled[["datekey", "netinc"]]
    print_df_2 = print_df_2.loc[print_df_2.index <= art_row_cur["calendardate"]]   
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(print_df)
        print(print_df_2) 

    nr_of_earnings_increases = 0
    for i in range(4):
        cur_row = arq_rows[i]
        prev_row = arq_rows[i+4]
        if (not cur_row.empty) and (not prev_row.empty):
            cur_netinc = cur_row["netinc"]
            prev_netinc = prev_row["netinc"]
            if cur_netinc > prev_netinc:
                nr_of_earnings_increases += 1
            else:
                break
        else:
            break
    
    assert check_row["nincr"] == nr_of_earnings_increases


def test_add_sf1_features_roavol():
    global arq_row_cur, arq_row_1q_ago, arq_row_2q_ago, arq_row_3q_ago, arq_row_4q_ago, arq_row_5q_ago, arq_row_6q_ago, arq_row_7q_ago
    arq_rows = [arq_row_cur, arq_row_1q_ago, arq_row_2q_ago, arq_row_3q_ago, arq_row_4q_ago, arq_row_5q_ago, arq_row_6q_ago, arq_row_7q_ago]
    global check_row

    print(check_row["roavol"])

    assert check_row["roavol"] == pytest.approx(0.020400037081686622)


#__________________YEARLY FILING BASED FEATURES START_________________

def test_add_sf1_features_cashpr():
    global art_row_cur, check_row

    cashpr = (art_row_cur["marketcap"] + art_row_cur["debtnc"] - art_row_cur["assets"]) / art_row_cur["cashneq"]

    assert check_row["cashpr"] == cashpr


def test_add_sf1_features_cash():
    global art_row_cur, check_row

    cash = art_row_cur["cashnequsd"] / art_row_cur["assetsavg"]
    
    assert check_row["cash"] == cash

def test_add_sf1_features_bm():
    # Book to market (bm), Formula: SF1[equityusd]t-1 / SF1[marketcap]t-1
    global art_row_cur, check_row

    bm = art_row_cur["equityusd"] / art_row_cur["marketcap"]

    assert check_row["bm"] == bm


def test_add_sf1_features_cfp():
    global art_row_cur, check_row

    cfp = art_row_cur["ncfo"] / art_row_cur["marketcap"]

    assert check_row["cfp"] == cfp


def test_add_sf1_features_currat():
    global art_row_cur, check_row

    currat = art_row_cur["assetsc"] / art_row_cur["liabilitiesc"]

    assert check_row["currat"] == currat


def test_add_sf1_features_depr():
    global art_row_cur, check_row

    depr = art_row_cur["depamor"] / art_row_cur["ppnenet"]

    
    assert check_row["depr"] == depr


def test_add_sf1_features_ep():
    global art_row_cur, check_row

    ep = art_row_cur["netinc"] / art_row_cur["marketcap"]

    assert check_row["ep"] == ep

def test_add_sf1_features_lev():
    global art_row_cur, check_row

    lev = art_row_cur["liabilities"] / art_row_cur["marketcap"]

    assert check_row["lev"] == lev


def test_add_sf1_features_quick():
    global art_row_cur, check_row

    quick = (art_row_cur["assetsc"] - art_row_cur["inventory"]) / art_row_cur["liabilitiesc"]

    assert check_row["quick"] == quick


def test_add_sf1_features_rd_mve():
    global art_row_cur, check_row

    rd_mve = art_row_cur["rnd"] / art_row_cur["marketcap"]

    assert check_row["rd_mve"] == rd_mve

def test_add_sf1_features_rd_sale():
    global art_row_cur, check_row

    rd_sale = art_row_cur["rnd"] / art_row_cur["revenueusd"]

    assert check_row["rd_sale"] == rd_sale

def test_add_sf1_features_roic():
    global art_row_cur, check_row
    
    nopic_t_1 = art_row_cur["revenueusd"] - art_row_cur["cor"] - art_row_cur["opinc"]
    roic = (art_row_cur["ebit"] - nopic_t_1) / (art_row_cur["equity"] + art_row_cur["liabilities"] + art_row_cur["cashneq"] - art_row_cur["investmentsc"])

    assert check_row["roic"] == roic


def test_add_sf1_features_salecash():
    global art_row_cur, check_row

    salecash = art_row_cur["revenueusd"] / art_row_cur["cashneq"]

    assert check_row["salecash"] == salecash

def test_add_sf1_features_saleinv():
    global art_row_cur, check_row

    saleinv = art_row_cur["revenueusd"] / art_row_cur["inventory"]

    assert check_row["saleinv"] == saleinv



def test_add_sf1_features_salerec():
    global art_row_cur, check_row

    salerec = art_row_cur["revenueusd"] / art_row_cur["receivables"]

    assert check_row["salerec"] == salerec

def test_add_sf1_features_sp():
    global art_row_cur, check_row

    sp = art_row_cur["revenueusd"] / art_row_cur["marketcap"]

    assert check_row["sp"] == sp

def test_add_sf1_features_tb():
    global art_row_cur, check_row

    tb = art_row_cur["taxexp"] / art_row_cur["netinc"]

    assert check_row["tb"] == tb

def test_add_sf1_features_sin():
    global art_row_cur, check_row, metadata

    ticker = art_row_cur["ticker"]

    assert metadata.loc[metadata["ticker"] == ticker].iloc[-1]["industry"] == "Consumer Electronics"

    industry_cur = "Beverages - Brewers"

    if industry_cur in ["Beverages - Brewers", "Beverages - Wineries & Distilleries", "Gambling", "Tobacco"]: # "Electronic Gaming & Multimedia"
        sin = 1
    else:
        sin = 0

    assert sin == 1

    assert check_row["sin"] == 0

def test_add_sf1_features_tang():
    global art_row_cur, check_row

    tang = (art_row_cur["cashnequsd"] + 0.715*art_row_cur["receivables"] + 0.547*art_row_cur["inventory"] + \
        0.535*art_row_cur["ppnenet"]) / art_row_cur["assets"]

    assert check_row["tang"] == tang

def test_add_sf1_features_debtc_sale():
    global art_row_cur, check_row

    debtc_sale = art_row_cur["debtc"] / art_row_cur["revenueusd"]

    assert check_row["debtc_sale"] == debtc_sale


def test_add_sf1_features_eqt_marketcap():
    global art_row_cur, check_row

    eqt_marketcap = ( art_row_cur["equityusd"] - art_row_cur["intangibles"]) / art_row_cur["marketcap"]

    assert check_row["eqt_marketcap"] == eqt_marketcap

def test_add_sf1_features_dep_ppne():
    global art_row_cur, check_row

    dep_ppne = art_row_cur["depamor"] / art_row_cur["ppnenet"]

    assert check_row["dep_ppne"] == dep_ppne


def test_add_sf1_features_tangibles_marketcap():
    global art_row_cur, check_row

    tangibles_marketcap = art_row_cur["tangibles"] / art_row_cur["marketcap"]

    assert check_row["tangibles_marketcap"] == tangibles_marketcap


def test_add_sf1_features_agr():
    global art_row_cur, art_row_1y_ago, check_row

    agr = (art_row_cur["assets"] / art_row_1y_ago["assets"] ) - 1

    assert check_row["agr"] == agr


def test_add_sf1_features_cashdebt():
    global art_row_cur, art_row_1y_ago, check_row 

    cashdebt = (art_row_cur["revenueusd"] + art_row_cur["depamor"]) /  ((art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) / 2)

    assert check_row["cashdebt"] == cashdebt



def test_add_sf1_features_chcsho():
    global art_row_cur, art_row_1y_ago, check_row 

    chcsho = (art_row_cur["sharesbas"] / art_row_1y_ago["sharesbas"]) - 1

    assert check_row["chcsho"] == chcsho



def test_add_sf1_features_chinv():
    global art_row_cur, art_row_1y_ago, check_row 

    chinv = ( art_row_cur["inventory"] - art_row_1y_ago["inventory"] ) / art_row_cur["assetsavg"]

    assert check_row["chinv"] == chinv

def test_add_sf1_features_egr():
    global art_row_cur, art_row_1y_ago, check_row 

    egr = (art_row_cur["equityusd"] / art_row_1y_ago["equityusd"]) - 1

    assert check_row["egr"] == egr


def test_add_sf1_features_gma():
    global art_row_cur, art_row_1y_ago, check_row 

    gma = (art_row_cur["revenueusd"] - art_row_cur["cor"]) / art_row_1y_ago["assets"]

    assert check_row["gma"] == gma



def test_add_sf1_features_invest():
    global art_row_cur, art_row_1y_ago, check_row 
    
    invest = ((art_row_cur["ppnenet"] - art_row_1y_ago["ppnenet"]) + (art_row_cur["inventory"] - art_row_1y_ago["inventory"])) / art_row_1y_ago["assets"]

    assert check_row["invest"] == invest


def test_add_sf1_features_lgr():
    global art_row_cur, art_row_1y_ago, check_row 

    lgr = ( art_row_cur["liabilities"] / art_row_1y_ago["liabilities"] ) - 1

    assert check_row["lgr"] == lgr


def test_add_sf1_features_operprof():
    global art_row_cur, art_row_1y_ago, check_row 

    operprof = ( art_row_cur["revenueusd"] - art_row_cur["cor"] - art_row_cur["sgna"] - art_row_cur["intexp"] ) / art_row_1y_ago["equityusd"]

    assert check_row["operprof"] == operprof


def test_add_sf1_features_pchcurrat():
    global art_row_cur, art_row_1y_ago, check_row

    pchcurrat =  ( (art_row_cur["assetsc"] / art_row_cur["liabilitiesc"]) / (art_row_1y_ago["assetsc"] / art_row_1y_ago["liabilitiesc"]) ) - 1

    assert check_row["pchcurrat"] == pchcurrat


def test_add_sf1_features_pchdepr():
    global art_row_cur, art_row_1y_ago, check_row

    pchdepr = ( (art_row_cur["depamor"]/ art_row_cur["ppnenet"]) / (art_row_1y_ago["depamor"] / art_row_1y_ago["ppnenet"]) ) - 1

    assert check_row["pchdepr"] == pchdepr


def test_add_sf1_features_pchgm_pchsale():
    global art_row_cur, art_row_1y_ago, check_row

    gross_margin_t_1 = (art_row_cur["revenueusd"] - art_row_cur["cor"]) / art_row_cur["revenueusd"]
    gross_margin_t_2 = (art_row_1y_ago["revenueusd"] - art_row_1y_ago["cor"]) / art_row_1y_ago["revenueusd"]

    pchgm_pchsale = ((gross_margin_t_1 / gross_margin_t_2) - 1) - ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1)

    assert check_row["pchgm_pchsale"] == pchgm_pchsale


def test_add_sf1_features_pchquick():
    global art_row_cur, art_row_1y_ago, check_row

    quick_ratio_cur = ( art_row_cur["assetsc"] - art_row_cur["inventory"] ) / art_row_cur["liabilitiesc"]
    quick_ratio_1y_ago = ( art_row_1y_ago["assetsc"] - art_row_1y_ago["inventory"] ) / art_row_1y_ago["liabilitiesc"]

    pchquick = (quick_ratio_cur / quick_ratio_1y_ago) - 1

    assert check_row["pchquick"] == pchquick


def test_add_sf1_features_pchsale_pchinvt():
    global art_row_cur, art_row_1y_ago, check_row

    pchsale_pchinvt =  ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1) - ((art_row_cur["inventory"] / art_row_1y_ago["inventory"]) - 1)

    assert check_row["pchsale_pchinvt"] == pchsale_pchinvt


def test_add_sf1_features_pchsale_pchrect():
    global art_row_cur, art_row_1y_ago, check_row

    pchsale_pchrect = ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1) - ((art_row_cur["receivables"] / art_row_1y_ago["receivables"]) - 1)

    assert check_row["pchsale_pchrect"] == pchsale_pchrect

def test_add_sf1_features_pchsale_pchxsga():
    global art_row_cur, art_row_1y_ago, check_row

    pchsale_pchxsga = ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1) - ((art_row_cur["sgna"] / art_row_1y_ago["sgna"]) - 1)

    assert check_row["pchsale_pchxsga"] == pchsale_pchxsga

def test_add_sf1_features_pchsaleinv():
    global art_row_cur, art_row_1y_ago, check_row

    pchsaleinv = ((art_row_cur["revenueusd"] / art_row_cur["inventory"]) / (art_row_1y_ago["revenueusd"] / art_row_1y_ago["inventory"])) - 1

    assert check_row["pchsaleinv"] == pchsaleinv


def test_add_sf1_features_rd():
    global art_row_cur, art_row_1y_ago, check_row

    rd_cur = (art_row_cur["rnd"] / art_row_cur["assets"])

    rd_1y_ago = (art_row_1y_ago["rnd"] / art_row_1y_ago["assets"])

    pch_rd = rd_cur/rd_1y_ago - 1

    if pch_rd > 0.05:
        rd = 1
    else:
        rd = 0

    # print(art_row_cur["rnd"], art_row_cur["assets"])
    # print(art_row_1y_ago["rnd"], art_row_1y_ago["assets"])
    # print(rd_cur, rd_1y_ago, pch_rd, rd)

    assert check_row["rd"] == rd


def test_add_sf1_features_roeq():
    global art_row_cur, art_row_1y_ago, check_row

    roeq = art_row_cur["netinc"] / art_row_1y_ago["equityusd"]

    assert check_row["roeq"] == roeq


def test_add_sf1_features_sgr():
    global art_row_cur, art_row_1y_ago, check_row
    
    sgr = (art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1

    assert check_row["sgr"] == sgr


def test_add_sf1_features_grcapx():
    global art_row_cur, art_row_1y_ago, check_row
    
    grcapx = (art_row_cur["capex"] / art_row_1y_ago["capex"]) - 1

    assert check_row["grcapx"] == grcapx

def test_add_sf1_features_chtl_lagat():
    global art_row_cur, art_row_1y_ago, check_row

    chtl_lagat = (art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) / art_row_1y_ago["assets"]

    assert check_row["chtl_lagat"] == chtl_lagat

def test_add_sf1_features_invcap():
    global art_row_cur, art_row_1y_ago, check_row

    chlt_laginvcap = (art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) / art_row_1y_ago["invcap"]

    assert check_row["chlt_laginvcap"] == chlt_laginvcap


def test_add_sf1_features_chlct_lagat():
    global art_row_cur, art_row_1y_ago, check_row

    chlct_lagat = (art_row_cur["liabilitiesc"] - art_row_1y_ago["liabilitiesc"]) / art_row_1y_ago["assets"]

    assert check_row["chlct_lagat"] == chlct_lagat


def test_add_sf1_features_chint_lagat():
    global art_row_cur, art_row_1y_ago, check_row

    chint_lagat = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["assets"]

    assert check_row["chint_lagat"] == chint_lagat



def test_add_sf1_features_chinvt_lagsale():
    global art_row_cur, art_row_1y_ago, check_row

    chinvt_lagsale = (art_row_cur["inventory"] - art_row_1y_ago["inventory"]) / art_row_1y_ago["revenueusd"]

    assert check_row["chinvt_lagsale"] == chinvt_lagsale


def test_add_sf1_features_chint_lagsgna():
    global art_row_cur, art_row_1y_ago, check_row

    chint_lagsgna = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["sgna"]

    assert check_row["chint_lagsgna"] == chint_lagsgna

def test_add_sf1_features_chltc_laginvcap():
    global art_row_cur, art_row_1y_ago, check_row

    chltc_laginvcap = (art_row_cur["liabilitiesc"] - art_row_1y_ago["liabilitiesc"]) / art_row_1y_ago["invcap"]

    assert check_row["chltc_laginvcap"] == chltc_laginvcap

def test_add_sf1_features_chdebtnc_lagat():
    global art_row_cur, art_row_1y_ago, check_row

    chdebtnc_lagat = (art_row_cur["debtnc"] - art_row_1y_ago["debtnc"]) / art_row_1y_ago["assets"]

    assert check_row["chdebtnc_lagat"] == chdebtnc_lagat



def test_add_sf1_features_chinvt_lagcor():
    global art_row_cur, art_row_1y_ago, check_row 

    chinvt_lagcor = (art_row_cur["inventory"] - art_row_1y_ago["inventory"]) / art_row_1y_ago["cor"]

    assert check_row["chinvt_lagcor"] == chinvt_lagcor


def test_add_sf1_features_chppne_laglt():
    global art_row_cur, art_row_1y_ago, check_row 

    chppne_laglt = (art_row_cur["ppnenet"] - art_row_1y_ago["ppnenet"]) / art_row_1y_ago["liabilities"]

    assert check_row["chppne_laglt"] == chppne_laglt

def test_add_sf1_features_chpay_lagact():
    global art_row_cur, art_row_1y_ago, check_row 

    chpay_lagact = (art_row_cur["payables"] - art_row_1y_ago["payables"]) / art_row_1y_ago["assetsc"]

    assert check_row["chpay_lagact"] == chpay_lagact


def test_add_sf1_features_chint_laginvcap():
    global art_row_cur, art_row_1y_ago, check_row 

    chint_laginvcap = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["invcap"]

    assert check_row["chint_laginvcap"] == chint_laginvcap


def test_add_sf1_features_chinvt_lagact():
    global art_row_cur, art_row_1y_ago, check_row 

    chinvt_lagact = (art_row_cur["inventory"] - art_row_1y_ago["inventory"]) / art_row_1y_ago["assetsc"]

    assert check_row["chinvt_lagact"] == chinvt_lagact

def test_add_sf1_features_pchppne():
    global art_row_cur, art_row_1y_ago, check_row     

    pchppne = (art_row_cur["ppnenet"] / art_row_1y_ago["ppnenet"]) - 1

    assert check_row["pchppne"] == pchppne


def test_add_sf1_features_pchlt():
    global art_row_cur, art_row_1y_ago, check_row     

    pchlt = (art_row_cur["liabilities"] / art_row_1y_ago["liabilities"]) - 1

    assert check_row["pchlt"] == pchlt


def test_add_sf1_features_pchint():
    global art_row_cur, art_row_1y_ago, check_row     

    print(art_row_cur["intexp"], art_row_1y_ago["intexp"], check_row["pchint"])

    if art_row_1y_ago["intexp"] != 0:
        pchint = (art_row_cur["intexp"] / art_row_1y_ago["intexp"]) - 1
    else:
        pchint = math.nan

    if math.isnan(pchint) == True:
        assert math.isnan(check_row["pchint"])
    else:
        assert check_row["pchint"] == pchint


def test_add_sf1_features_chdebtnc_ppne():
    global art_row_cur, art_row_1y_ago, check_row     

    chdebtnc_ppne = (art_row_cur["debtnc"] - art_row_1y_ago["debtnc"]) / art_row_cur["ppnenet"]

    assert check_row["chdebtnc_ppne"] == chdebtnc_ppne

def test_add_sf1_features_chdebtc_sale():
    global art_row_cur, art_row_1y_ago, check_row     

    chdebtc_sale = (art_row_cur["debtc"] - art_row_1y_ago["debtc"]) / art_row_cur["revenueusd"]

    assert check_row["chdebtc_sale"] == chdebtc_sale



def test_add_sf1_features_age():
    global art_row_cur, art_row_1y_ago, check_row     

    ticker = art_row_cur["ticker"]
    metadata_ticker = metadata.loc[metadata.ticker==ticker].iloc[0]

    age = round((art_row_cur["datekey"] - metadata_ticker["firstpricedate"]).days / 365)

    # print(art_row_cur["datekey"],  metadata_ticker["firstpricedate"])
    # print(age)

    assert check_row["age"] == age


def test_add_sf1_features_ipo():
    global art_row_cur, art_row_1y_ago, check_row     

    ticker = art_row_cur["ticker"]
    metadata_ticker = metadata.loc[metadata.ticker==ticker].iloc[0]

    days_since_ipo = (art_row_cur["datekey"] - metadata_ticker["firstpricedate"]).days

    if days_since_ipo <= 365:
        ipo = 1
    else:
        ipo = 0

    # print(art_row_cur["datekey"], metadata_ticker["firstpricedate"])
    # print(days_since_ipo, ipo)

    assert check_row["ipo"] == ipo


def test_add_sf1_features_ps():
    global art_row_cur, art_row_1y_ago, check_row

    if (art_row_cur["assetsavg"] != 0) and (art_row_1y_ago["assetsavg"] != 0) and (art_row_cur["liabilitiesc"] != 0) \
        and (art_row_1y_ago["liabilitiesc"] != 0) and (art_row_cur["revenueusd"] != 0) and (art_row_1y_ago["revenueusd"] != 0):
        i1_positive_netinc = 1 if (art_row_cur["netinc"] > 0) else 0

        i2_positive_roa = 1 if ( (art_row_cur["netinc"] / art_row_cur["assetsavg"]) > 0 ) else 0

        i3_positive_operating_cash_flow = 1 if (art_row_cur["ncfo"] > 0) else 0

        i4_ncfo_exceeds_netinc = 1 if (art_row_cur["ncfo"] > art_row_cur["netinc"]) else 0

        i5_lower_long_term_debt_to_assets = 1 if ((art_row_cur["debtnc"] / art_row_cur["assetsavg"]) < (art_row_1y_ago["debtnc"] / art_row_1y_ago["assetsavg"])) else 0
        
        i6_higher_current_ratio = 1 if ((art_row_cur["assetsc"] / art_row_cur["liabilitiesc"]) > (art_row_1y_ago["assetsc"] / art_row_1y_ago["liabilitiesc"])) else 0
       
        i7_no_new_shares = 1 if (art_row_cur["sharesbas"] <= art_row_1y_ago["sharesbas"]) else 0
        
        i8_higher_gross_margin =  1 if ( ((art_row_cur["revenueusd"] - art_row_cur["cor"]) / art_row_cur["revenueusd"]) > ((art_row_1y_ago["revenueusd"] - art_row_1y_ago["cor"] ) / art_row_1y_ago["revenueusd"]) ) else 0
        
        i9_higher_asset_turnover_ratio =  1 if ((art_row_cur["revenueusd"] / art_row_cur["assetsavg"]) > (art_row_1y_ago["revenueusd"] / art_row_1y_ago["assetsavg"])) else 0

        ps = i1_positive_netinc + i2_positive_roa + i3_positive_operating_cash_flow + i4_ncfo_exceeds_netinc + i5_lower_long_term_debt_to_assets + i6_higher_current_ratio + i7_no_new_shares + i8_higher_gross_margin + i9_higher_asset_turnover_ratio
    else:
        ps = np.nan
    
    """
    print("i1_positive_netinc", i1_positive_netinc, art_row_cur["netinc"])
    print("i2_positive_roa", i2_positive_roa, art_row_cur["netinc"] / art_row_cur["assetsavg"])
    print("i3_positive_operating_cash_flow",i3_positive_operating_cash_flow, art_row_cur["ncfo"])

    print("i4_ncfo_exceeds_netinc", i4_ncfo_exceeds_netinc, art_row_cur["ncfo"], ">", art_row_cur["netinc"])
    print("i5_lower_long_term_debt_to_assets", i5_lower_long_term_debt_to_assets, (art_row_cur["debtnc"] / \
        art_row_cur["assetsavg"]), "<", (art_row_1y_ago["debtnc"] / art_row_1y_ago["assetsavg"]))
    print("i6_higher_current_ratio", i6_higher_current_ratio)
    print("i7_no_new_shares", i7_no_new_shares)
    print("i8_higher_gross_margin", i8_higher_gross_margin)
    print("i9_higher_asset_turnover_ratio", i9_higher_asset_turnover_ratio)
    print(ps)
    """
    assert check_row["ps"] == ps

#_________________________PREP INDUSTRY CALCULATIONS_______________________
def test_add_sf1_features_profitmargin():
    global art_row_cur, art_row_1y_ago, check_row

    profitmargin = art_row_cur["netinc"] / art_row_cur["revenueusd"]

    assert check_row["profitmargin"] == profitmargin


def test_add_sf1_features_chprofitmargin():
    global art_row_cur, art_row_1y_ago, check_row

    chprofitmargin = (art_row_cur["netinc"] - art_row_cur["revenueusd"]) - (art_row_1y_ago["netinc"] / art_row_1y_ago["revenueusd"])

    assert check_row["chprofitmargin"] == chprofitmargin

def test_add_sf1_features_industry():
    global check_row, metadata, art_row_cur

    ticker = art_row_cur["ticker"]
    industry = metadata.loc[metadata.ticker==ticker].iloc[0]["industry"]

    assert check_row["industry"] == industry



def test_add_sf1_features_change_sales():
    global art_row_cur, art_row_1y_ago, check_row

    change_sales = art_row_cur["revenueusd"] - art_row_1y_ago["revenueusd"]

    assert check_row["change_sales"] == change_sales




#______________________END PREP INDUSTRY CALCULATIONS____________________

@pytest.mark.skip()
def test_shape(): 
    """Test shape of result to see that features have been calculated for "all" rows"""
    global sf1_art_featured, features

    assert (sf1_art_featured.shape[1] - sf1_art.shape[1]) == len(features) + 1 # Don't know exactly...


@pytest.mark.skip()
def test_feature_coverage():
    """Test to check that features have a minimum coverage in the dataset."""
    global sf1_art_featured
    
    for ticker in list(sf1_art_featured.ticker.unique()):
        sf1_art_featured_ticker = sf1_art_featured.loc[sf1_art_featured.ticker == ticker]
        sf1_art_featured_selected = sf1_art_featured_ticker[features]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(ticker)
            print(len(sf1_art_featured_selected) - sf1_art_featured_selected.count())





"""
ticker,dimension,calendardate,datekey,reportperiod,lastupdated,accoci,assets,assetsavg,assetsc,assetsnc,assetturnover,bvps,capex,cashneq,cashnequsd,cor,consolinc,currentratio,de,debt,debtc,debtnc,debtusd,deferredrev,depamor,deposits,divyield,dps,ebit,ebitda,ebitdamargin,ebitdausd,ebitusd,ebt,eps,epsdil,epsusd,equity,equityavg,equityusd,ev,evebit,evebitda,fcf,fcfps,fxusd,gp,grossmargin,intangibles,intexp,invcap,invcapavg,inventory,investments,investmentsc,investmentsnc,liabilities,liabilitiesc,liabilitiesnc,marketcap,ncf,ncfbus,ncfcommon,ncfdebt,ncfdiv,ncff,ncfi,ncfinv,ncfo,ncfx,netinc,netinccmn,netinccmnusd,netincdis,netincnci,netmargin,opex,opinc,payables,payoutratio,pb,pe,pe1,ppnenet,prefdivis,price,ps,ps1,receivables,retearn,revenue,revenueusd,rnd,roa,roe,roic,ros,sbcomp,sgna,sharefactor,sharesbas,shareswa,shareswadil,sps,tangibles,taxassets,taxexp,taxliabilities,tbvps,workingcapital
AAPL,ART,1997-09-30,1997-12-05,1997-09-26,2019-01-30,0.0,4233000000.0,,3424000000.0,809000000.0,,0.34,-53000000.0,1230000000.0,1230000000.0,5713000000.0,-1045000000.0,1.883,2.527,976000000.0,25000000.0,951000000.0,976000000.0,0.0,118000000.0,0.0,0.0,0.0,-1045000000.0,-927000000.0,-0.131,-927000000.0,-1045000000.0,-1045000000.0,-0.29600000000000004,-0.29600000000000004,-0.29600000000000004,1200000000.0,,1200000000.0,1769575844.0,-2.0,-1.909,135000000.0,0.038,1.0,1368000000.0,0.193,0.0,0.0,2161000000.0,,437000000.0,229000000.0,229000000.0,0.0,3033000000.0,1818000000.0,1215000000.0,2023575844.0,-322000000.0,-384000000.0,34000000.0,-161000000.0,0.0,23000000.0,-533000000.0,-36000000.0,188000000.0,0.0,-1045000000.0,-1045000000.0,-1045000000.0,0.0,0.0,-0.14800000000000002,2438000000.0,-1070000000.0,685000000.0,0.0,1.686,-1.936,-1.908,486000000.0,0.0,0.565,0.28600000000000003,0.281,1035000000.0,589000000.0,7081000000.0,7081000000.0,485000000.0,,,,-0.14800000000000002,0.0,1286000000.0,1.0,3583815536.0,3529736000.0,3529736000.0,2.006,4233000000.0,259000000.0,0.0,264000000.0,1.199,1606000000.0
AAPL,ART,1998-09-30,1998-12-23,1998-09-25,2019-01-30,0.0,4289000000.0,4104750000.0,3698000000.0,591000000.0,1.4469999999999998,0.434,43000000.0,1481000000.0,1481000000.0,4462000000.0,309000000.0,2.4330000000000003,1.612,954000000.0,0.0,954000000.0,954000000.0,0.0,111000000.0,0.0,0.0,0.0,329000000.0,440000000.0,0.07400000000000001,440000000.0,329000000.0,329000000.0,0.084,0.075,0.084,1642000000.0,1440000000.0,1642000000.0,4860080402.0,15.0,11.046,818000000.0,0.221,1.0,1479000000.0,0.249,0.0,0.0,2242000000.0,2288500000.0,78000000.0,819000000.0,819000000.0,0.0,2647000000.0,1520000000.0,1127000000.0,5387080402.0,251000000.0,0.0,41000000.0,-22000000.0,0.0,19000000.0,-543000000.0,-566000000.0,775000000.0,0.0,309000000.0,309000000.0,309000000.0,0.0,0.0,0.052000000000000005,1218000000.0,261000000.0,719000000.0,0.0,3.281,17.434,16.926,348000000.0,0.0,1.422,0.907,0.884,955000000.0,898000000.0,5941000000.0,5941000000.0,303000000.0,0.075,0.215,0.14400000000000002,0.055,0.0,908000000.0,1.0,3788953812.0,3695272000.0,4701676000.0,1.608,4289000000.0,182000000.0,20000000.0,173000000.0,1.135,2178000000.0
"""
