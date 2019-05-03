import pandas as pd
import pytest
import numpy as np

from ..processing.engine import pandas_mp_engine
from ..sf1_industry_features import add_industry_sf1_features
from ..helpers.helpers import get_most_up_to_date_10k_filing


sf1_art_featured = None
industry_sf1_art_featured = None
metadata = None
sf1_aapl = None
check_row = None
sf1_ce_industry = None

features = ["bm_ia", "cfp_ia", "chatoia", "mve_ia", "pchcapex_ia", "chpmia", "herf", "ms"]


@pytest.fixture(scope='module', autouse=True)
def setup():
    global industry_sf1_art_featured, sf1_art_featured, metadata
    # Will be executed before the first test in the module
    sf1_art_featured = pd.read_csv("../datasets/testing/sf1_art_featured_snapshot.csv", parse_dates=["calendardate", "datekey", "reportperiod"],\
        index_col="calendardate", low_memory=False)
    metadata = pd.read_csv("../datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", parse_dates=["firstpricedate"], low_memory=False)
    
    yield
    
    # Will be executed after the last test in the module
    industry_sf1_art_featured.to_csv("../datasets/testing/industry_sf1_art_featured.csv", index=False)


def test_running_add_industry_sf1_features():
    global industry_sf1_art_featured, sf1_art_featured, metadata, sf1_aapl, sf1_ce_industry

    industry_sf1_art_featured = pandas_mp_engine(callback=add_industry_sf1_features, atoms=sf1_art_featured, \
        data={'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'industry', \
            num_processes=1, molecules_per_process=1)
    
    industry_sf1_art_featured = industry_sf1_art_featured.sort_values(by=["ticker", "calendardate", "datekey"])


    industry_sf1_art_featured_aapl = industry_sf1_art_featured.loc[industry_sf1_art_featured.ticker=="AAPL"]


    # Only temporary...
    cols = list(set(features).intersection(set(industry_sf1_art_featured.columns)))
    industry_sf1_aapl_only_features = industry_sf1_art_featured_aapl[cols]

    industry_sf1_art_featured.to_csv("../datasets/testing/industry_sf1_art_featured_snapshot.csv")
    industry_sf1_aapl_only_features.to_csv("../datasets/testing/industry_sf1_art_aapl_only_features.csv")

    # assert False

def test_setup_aapl_2001_03_31():
    global check_row, sf1_aapl, art_row_cur, art_row_1y_ago, sf1_art_featured, sf1_ce_industry


    # Not touched by add_industry_sf1_features:    
    sf1_aapl = sf1_art_featured.loc[sf1_art_featured.ticker == "AAPL"]
    sf1_aapl["caldate"] = sf1_aapl.index # not optimal, but ok for testing i guess...
    sf1_aapl = sf1_aapl.reset_index() # this should be the same dataframe used in add_industry_sf1_features()

    sf1_ce_industry = sf1_art_featured.loc[sf1_art_featured.industry == "Consumer Electronics"]

    art_row_cur = sf1_aapl.loc[sf1_aapl.calendardate == "2015-03-31"].iloc[-1]
    caldate = art_row_cur["caldate"]
    datekey = art_row_cur["datekey"]


    """ I will for now assume I can get around needing two years of data for calculating industry features """

    # art_row_1y_ago = get_most_up_to_date_10k_filing(sf1_aapl, caldate, datekey, 1)
    # print(art_row_1y_ago) # Not there because it needs to be forward filled

    # The row to verify has the correct values (touched by add_industry_sf1_features)
    check_row = industry_sf1_art_featured.loc[(industry_sf1_art_featured.ticker=="AAPL") & \
        (industry_sf1_art_featured.index == "2015-03-31")].iloc[-1]

    assert (np.array(sf1_ce_industry.ticker.unique()) == np.array(["AAPL", "NTK"])).all()

    assert art_row_cur["datekey"] == pd.to_datetime("2015-04-28")
    # assert art_row_1y_ago["datekey"] == pd.to_datetime("2000-02-01") # I need to forward fill...

    assert check_row["datekey"] == pd.to_datetime("2015-04-28")


def test_add_industry_adjusted_book_to_market():
    global check_row, art_row_cur, sf1_ce_industry

    sf1_art_for_date = sf1_ce_industry.loc[sf1_ce_industry.index == "2015-03-31"]

    industry_mean_bm = (sf1_art_for_date["equityusd"] / sf1_art_for_date["marketcap"]).mean()

    bm_ia = (art_row_cur["equityusd"] / art_row_cur["marketcap"]) - industry_mean_bm

    assert check_row["bm_ia"] == bm_ia


def test_add_industry_adjusted_cfp():
    global check_row, art_row_cur, sf1_ce_industry

    sf1_art_for_date = sf1_ce_industry.loc[sf1_ce_industry.index == "2015-03-31"]

    industry_mean_cfp =  (sf1_art_for_date["ncfo"] / sf1_art_for_date["marketcap"]).mean()

    cfp_ia =   (art_row_cur["ncfo"] / art_row_cur["marketcap"]) - industry_mean_cfp

    assert check_row["cfp_ia"] == cfp_ia


def test_add_chatoia():
    global check_row, art_row_cur, sf1_ce_industry

    sf1_art_for_date = sf1_ce_industry.loc[sf1_ce_industry.index == "2015-03-31"]

    industry_mean_asset_turnover =  (sf1_art_for_date["change_sales"] / sf1_art_for_date["assetsavg"]).mean()

    chatoia = (art_row_cur["change_sales"]  / art_row_cur["assetsavg"]) - industry_mean_asset_turnover

    assert check_row["chatoia"] == chatoia


def test_add_mve_ia():
    global check_row, art_row_cur, sf1_ce_industry

    sf1_art_for_date = sf1_ce_industry.loc[sf1_ce_industry.index == "2015-03-31"]

    industry_mean_marketcap = sf1_art_for_date["marketcap"].mean()

    mve_ia =  art_row_cur["marketcap"] - industry_mean_marketcap

    assert check_row["mve_ia"] == mve_ia


def test_add_pchcapex_ia():
    global check_row, art_row_cur, sf1_ce_industry

    sf1_art_for_date = sf1_ce_industry.loc[sf1_ce_industry.index == "2015-03-31"]
    
    industry_mean_percent_change_capex = sf1_art_for_date["grcapx"].mean()

    pchcapex_ia = art_row_cur["grcapx"] - industry_mean_percent_change_capex

    assert check_row["pchcapex_ia"] == pchcapex_ia



def test_add_chpmia():
    global check_row, art_row_cur, sf1_ce_industry

    sf1_art_for_date = sf1_ce_industry.loc[sf1_ce_industry.index == "2015-03-31"]

    industry_mean_change_profit_margin = sf1_art_for_date["chprofitmargin"].mean()

    chpmia = art_row_cur["chprofitmargin"] - industry_mean_change_profit_margin

    assert check_row["chpmia"] == chpmia


def test_add_herf():
    global check_row, art_row_cur, sf1_ce_industry

    sf1_art_for_date = sf1_ce_industry.loc[sf1_ce_industry.index == "2015-03-31"]

    sum_industry_revenue = sf1_art_for_date["revenueusd"].sum()

    sum_sqrd_percent_of_revenue = 0

    for i, company_row in sf1_art_for_date.iterrows():
        sum_sqrd_percent_of_revenue += (company_row["revenueusd"] / sum_industry_revenue)**2

    assert check_row["herf"] == sum_sqrd_percent_of_revenue



def test_add_ms():
    global check_row, art_row_cur, sf1_ce_industry

    sf1_art_for_date = sf1_ce_industry.loc[sf1_ce_industry.index == "2015-03-31"]

    industry_mean_return_on_assets = (sf1_art_for_date["netinc"] / sf1_art_for_date["assetsavg"]).mean()
    industry_mean_cash_flow_return_on_assets = (sf1_art_for_date["ncfo"] / sf1_art_for_date["assetsavg"]).mean()
    industry_mean_rnd_intensity = (sf1_art_for_date["rnd"] / sf1_art_for_date["assetsavg"]).mean()
    industry_mean_capex_intensity = (sf1_art_for_date["capex"] / sf1_art_for_date["assetsavg"]).mean()
    industry_mean_advertising_intensity = (sf1_art_for_date["sgna"] / sf1_art_for_date["assetsavg"]).mean()


    if (art_row_cur["assetsavg"] != 0) and (art_row_cur["netinc"] != 0):
        i1_roa_above_avg = 1 if ((art_row_cur["netinc"] / art_row_cur["assetsavg"]) > industry_mean_return_on_assets) else 0
        
        i2_cf_roa_above_avg = 1 if ((art_row_cur["ncfo"] / art_row_cur["assetsavg"]) > industry_mean_cash_flow_return_on_assets) else 0
        
        i3_ncfo_exceeds_netinc = 1 if (art_row_cur["ncfo"] > art_row_cur["netinc"]) else 0
        
        i6_rnd_intensity = 1 if ((art_row_cur["rnd"] / art_row_cur["assetsavg"]) > industry_mean_rnd_intensity) else 0
        
        i7_capex_indensity = 1 if ((art_row_cur["capex"] / art_row_cur["assetsavg"]) > industry_mean_capex_intensity) else 0
        
        i8_advertising_intensity = 1 if ((art_row_cur["sgna"] / art_row_cur["assetsavg"]) > industry_mean_advertising_intensity) else 0


    ms = i1_roa_above_avg + i2_cf_roa_above_avg + i3_ncfo_exceeds_netinc + i6_rnd_intensity + i7_capex_indensity + i8_advertising_intensity

    """
    print(i1_roa_above_avg, i2_cf_roa_above_avg, i3_ncfo_exceeds_netinc, i6_rnd_intensity, i7_capex_indensity, i8_advertising_intensity)
    print(art_row_cur["capex"] / art_row_cur["assetsavg"])
    print(industry_mean_capex_intensity)
    print(ms)
    """

    assert check_row["ms"] == ms

