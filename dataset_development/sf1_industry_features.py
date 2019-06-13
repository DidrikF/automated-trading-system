import pandas as pd
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import numpy as np


from processing.engine import pandas_mp_engine
from helpers.helpers import get_most_up_to_date_10k_filing, get_calendardate_x_quarters_ago


def add_industry_sf1_features(sf1_art, metadata):
    """
    Returns dataframe with features requiring calculations over whole industries:
    bm_ia, cfp_ia, chatoia, mve_ia, pchcapex_ia, chpmia, herf, ms and ps

    NOTE: sf1_art is forward filled and has a calendardate index
    NOTE: Requires sf1_features.py to be executed first, and that its output is given to this function.
    """

    if isinstance(metadata, pd.DataFrame) == True:
        metadata = metadata.iloc[0]

    sf1_art = sf1_art.reset_index() # I need this, because there are several companies in sf1_art (duplicate calendardates in index)
    
    industry_means = pd.DataFrame()

    for index_cur, art_row_cur in sf1_art.iterrows():
        """ Iterating over multiple tickers ! """
        caldate_cur = art_row_cur["calendardate"] 
        datekey_cur = art_row_cur["datekey"]
        ticker = art_row_cur["ticker"]
        
        # profitmargin, chprofitmargin are not used... maybe I will end up needing them...
        sf1_art_for_date = sf1_art.loc[sf1_art["calendardate"] == caldate_cur]
        sf1_art_for_date = sf1_art_for_date.sort_values(by=["datekey"])

        # If not filtering out duplicate filings for the same company, on this calendardate, some companies will be dispropotianally weighted.
        sf1_art_for_date = sf1_art_for_date.drop_duplicates(subset="ticker", keep="first") # Keeping the first gets out of any lookahead bias

        # ____________________________ CALCULATE INDUSTRY MEANS _____________________________
        
        if caldate_cur not in industry_means.index:
            
            industry_mean_bm = (sf1_art_for_date["equityusd"] / sf1_art_for_date["marketcap"]).mean() # I think it automatically excludes rows which leads to zero in the denominator.
            industry_mean_cfp = (sf1_art_for_date["ncfo"] / sf1_art_for_date["marketcap"]).mean() # I think it automatically excludes rows which leads to zero in the denominator.
            industry_mean_asset_turnover = (sf1_art_for_date["change_sales"] / sf1_art_for_date["assetsavg"]).mean() # I think it automatically excludes rows which leads to zero in the denominator.
            industry_mean_marketcap = sf1_art_for_date["marketcap"].mean()
            industry_mean_percent_change_capex = sf1_art_for_date["grcapx"].mean()
            industry_mean_change_profit_margin = sf1_art_for_date["chprofitmargin"].mean()


            industry_mean_return_on_assets = (sf1_art_for_date["netinc"] / sf1_art_for_date["assetsavg"]).mean()
            industry_mean_cash_flow_return_on_assets = (sf1_art_for_date["ncfo"] / sf1_art_for_date["assetsavg"]).mean()
            industry_mean_rnd_intensity = (sf1_art_for_date["rnd"] / sf1_art_for_date["assetsavg"]).mean()
            industry_mean_capex_intensity = (sf1_art_for_date["capex"] / sf1_art_for_date["assetsavg"]).mean()
            industry_mean_advertising_intensity = (sf1_art_for_date["sgna"] / sf1_art_for_date["assetsavg"]).mean()



            industry_means.at[caldate_cur, "industry_mean_bm"] = industry_mean_bm
            industry_means.at[caldate_cur, "industry_mean_cfp"] = industry_mean_cfp
            industry_means.at[caldate_cur, "industry_mean_asset_turnover"] = industry_mean_asset_turnover
            industry_means.at[caldate_cur, "industry_mean_marketcap"] = industry_mean_marketcap
            industry_means.at[caldate_cur, "industry_mean_percent_change_capex"] = industry_mean_percent_change_capex
            industry_means.at[caldate_cur, "industry_mean_change_profit_margin"] = industry_mean_change_profit_margin
        

            industry_means.at[caldate_cur, "industry_mean_return_on_assets"] = industry_mean_return_on_assets
            industry_means.at[caldate_cur, "industry_mean_cash_flow_return_on_assets"] = industry_mean_cash_flow_return_on_assets
            industry_means.at[caldate_cur, "industry_mean_rnd_intensity"] = industry_mean_rnd_intensity
            industry_means.at[caldate_cur, "industry_mean_capex_intensity"] = industry_mean_capex_intensity
            industry_means.at[caldate_cur, "industry_mean_advertising_intensity"] = industry_mean_advertising_intensity


        #____________________________ DONE CALCULATE INDUSTRY MEANS _____________________________


        #__________________________CALCULATE INDUSTRY ADJUSTED FEATURES__________________________
       
    
        sf1_art_ticker = sf1_art.loc[sf1_art["ticker"] == ticker] # Not very efficient...
        sf1_art_for_ticker = sf1_art.loc[sf1_art.ticker == ticker]
        art_row_1y_ago = get_most_up_to_date_10k_filing(sf1_art_for_ticker, caldate_cur, datekey_cur, 1)

        #______________________________REQUIRING ONLY CURRENT ROW___________________________

        # Industry-adjusted book-to-market (bm_ia), Formula: bm - industry_mean(bm)
        # Industry adjusted book-to-market ratio.
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[index_cur, "bm_ia"] = (art_row_cur["equityusd"] / art_row_cur["marketcap"]) - industry_means.at[caldate_cur, "industry_mean_bm"]

        # Industry-adjusted cash flow to price ratio (cfp_ia), Formula: cfp - indutry_mean(cfp)
        # cfp = SF1[ncfo]t-1 / SF1[marketcap]t-1
        # Industry adjusted cfp.
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[index_cur, "cfp_ia"] = (art_row_cur["ncfo"] / art_row_cur["marketcap"]) - industry_means.at[caldate_cur, "industry_mean_cfp"]

        # Industyr-adjusted change in asset turnover (chatoia), 
        # Formula: ((SF1[revenueusd]t-1 - SF1[revenueusd]t-2) / SF1[assetsavg]t-1) - industry_mean((SF1[revenueusd]t-1 - SF1[revenueusd]t-2) / SF1[assetsavg]t-1))
        # 2-digit SIC - fiscal-year mean adjusted change in sales (sale) divided by average total assets (at)
        if art_row_cur["assetsavg"] != 0:
            sf1_art.at[index_cur, "chatoia"] = (art_row_cur["change_sales"]  / art_row_cur["assetsavg"]) - industry_means.at[caldate_cur, "industry_mean_asset_turnover"]


        # Industry-adjusted size (mve_ia), Formula: SF1[marketcap]t-1 - industry_mean(SF1[marketcap]t-1)
        # 2-digit SIC industry-adjusted fiscal year-end market capitalization.
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[index_cur, "mve_ia"] = art_row_cur["marketcap"] - industry_means.at[caldate_cur, "industry_mean_marketcap"]


        # Industry-adjusted % change in capital expenditure (pchcapex_ia), Formula: ((SF1[capex]t-1 / SF1[capex]2-1) - 1) - industry_mean((SF1[capex]t-1 / SF1[capex]2-1))
        # 2-digit SIC - fiscal-year mean adjusted percent change in capital expenditures (capex).
        sf1_art.at[index_cur, "pchcapex_ia"] = art_row_cur["grcapx"]- industry_means.at[caldate_cur, "industry_mean_percent_change_capex"]
        
        # Industry-adjusted change in profit margin	Soliman (chpmia), 
        # Formula: (SF1[netinc]t-1 / SF1[revenueusd]t-1) - (SF1[netinc]t-2 / SF1[revenueusd]t-2) - industry_mean((SF1[netinc]t-1 / SF1[revenueusd]t-1) - (SF1[netinc]t-2 / SF1[revenueusd]t-2))  --> [chprofitmargin]t-1 - industry_mean([chprofitmargin]t-1)
        # 2-digit SIC - fiscal-year mean adjusted change in income before extraordinary items (ib) divided by sales (sale).
        sf1_art.at[index_cur, "chpmia"] = art_row_cur["chprofitmargin"] - industry_means.at[caldate_cur, "industry_mean_change_profit_margin"]


        # Industry sales concentration (herf), Formula: SF1[revenueusd] is used to proxy market share
        herf = get_herf(sf1_art_for_date) # CACHE IT MAYBE
        sf1_art.at[index_cur, "herf"] = herf


        # Finantial statement score (ms): Fromula: Sum of 8 (6) indicator variables for fundamental performance. (My own type, excluding too demanding indicators)
        ms = get_ms(art_row_cur, industry_means, caldate_cur)
        sf1_art.at[index_cur, "ms"] = ms        

    # Reset index
    sf1_art = sf1_art.set_index("calendardate")

    return sf1_art # This is still forward filled, but its ok, When i merge this with sep_sampled, I only take out the correct rows


def get_herf(sf1_art_for_date):
    # Industry sales concentration (herf), Formula: SF1[revenueusd] is used to proxy market share
    # 2-digit SIC - fiscal-year sales concentration (sum of squared percent of sales in industry for each company)
    # this calculation is done each year for each industry, and then average the values over the past 3 years (or two years).

    sum_industry_revenue = sf1_art_for_date["revenueusd"].sum()

    sum_sqrd_percent_of_revenue = 0

    for i, company_row in sf1_art_for_date.iterrows():
        sum_sqrd_percent_of_revenue += (company_row["revenueusd"] / sum_industry_revenue)**2
    
    return sum_sqrd_percent_of_revenue



def get_ms(art_row_cur: pd.Series, industry_means: pd.DataFrame, caldate_cur: pd.datetime) -> int:
    # Finantial statement score (ms): Fromula: Sum of 8 (6) indicator variables for fundamental performance. (My own type, excluding too demanding indicators)
    # Excluding ROA variability and sales growth variability
    # Advertising expenses is not availabe, so I use Selling, General and Administrative expenses (sgna)
    # I assume they mean G-Score, https://www.businessinsider.com/glamour-stocks-avoiding-falling-stars-using-mohanrams-g-score-2011-7?r=US&IR=T&IR=T
    if (art_row_cur["assetsavg"] != 0) and (art_row_cur["netinc"] != 0):
        i1_roa_above_avg = 1 if ((art_row_cur["netinc"] / art_row_cur["assetsavg"]) > industry_means.at[caldate_cur, "industry_mean_return_on_assets"]) else 0
        
        i2_cf_roa_above_avg = 1 if ((art_row_cur["ncfo"] / art_row_cur["assetsavg"]) > industry_means.at[caldate_cur, "industry_mean_cash_flow_return_on_assets"]) else 0
        
        i3_ncfo_exceeds_netinc = 1 if (art_row_cur["ncfo"] > art_row_cur["netinc"]) else 0
        
        i6_rnd_intensity = 1 if ((art_row_cur["rnd"] / art_row_cur["assetsavg"]) > industry_means.at[caldate_cur, "industry_mean_rnd_intensity"]) else 0
        
        i7_capex_indensity = 1 if ((art_row_cur["capex"] / art_row_cur["assetsavg"]) > industry_means.at[caldate_cur, "industry_mean_capex_intensity"]) else 0
        
        i8_advertising_intensity = 1 if ((art_row_cur["sgna"] / art_row_cur["assetsavg"]) > industry_means.at[caldate_cur, "industry_mean_advertising_intensity"]) else 0

        ms = i1_roa_above_avg + i2_cf_roa_above_avg + i3_ncfo_exceeds_netinc + i6_rnd_intensity + i7_capex_indensity + i8_advertising_intensity
    
        return ms

    else:
         return np.nan




if __name__ == "__main__":    
    sf1_art = pd.read_csv("./datasets/testing/sf1_art.csv", index_col="datekey", parse_dates=["datekey", "calendardate"])
    metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", index_col="ticker", parse_dates=["firstpricedate"])

    sf1_art["datekey"] = pd.to_datetime(sf1_art["datekey"])
    sf1_art["calendardate"] = pd.to_datetime(sf1_art["calendardate"])
    
    metadata["firstpricedate"] = pd.to_datetime(metadata["firstpricedate"])

    sep = pandas_mp_engine(callback=add_equally_weighted_weekly_market_returns, atoms=sep, data=None, \
        molecule_key='sep', split_strategy= 'date', \
            num_processes=4, molecules_per_process=1)

