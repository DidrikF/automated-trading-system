import pandas as pd
from packages.helpers.helpers import print_exception_info
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import numpy as np
from packages.multiprocessing.engine import pandas_mp_engine

"""
Each step is performed for each industry separately

Step-by-Step Dataset Construction:
1. Extend the SEP dataset with information usefull for sampling (most recent 10-K filing date, Industry classifications)
2. Use different sampling techniques to get monthly observations
    1. At first use timebars (sampling at a fixed time interval), but try to respect the different fiscal years
3. Calculate the various price and volume based features
4. Compute features based on SF1
5. Add inn SF1 and DAILY data
6. Select the features you want and combine into one ML ready dataset
"""

# Rewrite for multiprocessing engine

def add_industry_sf1_features(sf1_art, metadata):

    industry_means = pd.DataFrame()

    for index, art_row_cur in sf1_art.iterrows():
        date_cur = art_row_cur["datekey"]

        date_1y_ago = date_cur - relativedelta(years=+1)
        date_2y_ago = date_cur - relativedelta(years=+2)
        ticker = art_row_cur["ticker"]

        sf1_art_ticker = sf1_art.loc[sf1_art["ticker"] == ticker]
        # print("TICKER AFTER INDUSTRY CLACULATIONS: ", ticker, date_cur)
        art_row_1y_ago = get_row_with_closest_date(sf1_art_ticker, date_1y_ago, 30)
        art_row_2y_ago = get_row_with_closest_date(sf1_art_ticker, date_2y_ago, 30)
        # print("AM I PRINTED", type(art_row_1y_ago), art_row_1y_ago.empty)


        # ____________________________ CALCULATE INDUSTRY NUMBERS _____________________________
        
        if date_cur not in industry_means.index:
            calendardate_cur = art_row_cur["calendardate"]
            calendardate_1y_ago = art_row_cur["calendardate"] - relativedelta(years=+1) # Needs to be:, 03-31, 06-30, 09-30, 12-31, seems to be so.
            
            sf1_art_for_date = sf1_art.loc[sf1_art["calendardate"] == calendardate_cur]
            sf1_art_for_date = sf1_art_for_date.drop_duplicates(subset="ticker", keep="first")

            industry_mean_bm = (sf1_art_for_date["equityusd"] / sf1_art_for_date["marketcap"]).mean() # I think it automatically excludes rows which leads to zero in the denominator.
            industry_mean_cfp = (sf1_art_for_date["ncfo"] / sf1_art_for_date["marketcap"]).mean() # I think it automatically excludes rows which leads to zero in the denominator.
            industry_mean_asset_turnover = (sf1_art_for_date["revenueusd"] / sf1_art_for_date["assetsavg"]).mean() # I think it automatically excludes rows which leads to zero in the denominator.
            industry_mean_marketcap = sf1_art_for_date["marketcap"].mean()

            industry_mean_return_on_assets = (sf1_art_for_date["netinc"] / sf1_art_for_date["assetsavg"]).mean()
            industry_mean_cash_flow_return_on_assets = (sf1_art_for_date["ncfo"] / sf1_art_for_date["assetsavg"]).mean()
            industry_mean_rnd_intensity = (sf1_art_for_date["rnd"] / sf1_art_for_date["assetsavg"]).mean()
            industry_mean_capex_intensity = (sf1_art_for_date["capex"] / sf1_art_for_date["assetsavg"]).mean()
            industry_mean_advertising_intensity = (sf1_art_for_date["sgna"] / sf1_art_for_date["assetsavg"]).mean()

            
            # INDUSTRY NUMBERS REQUIRING DATA FROM LAST TWO YEARLY FILINGS 
            industry_mean_change_profit_margin, industry_mean_percent_change_capex = get_chpmia_pchcapex_ia(sf1_art, sf1_art_for_date, calendardate_1y_ago)
            

            industry_means.at[calendardate_cur, "industry_mean_bm"] = industry_mean_bm
            industry_means.at[calendardate_cur, "industry_mean_cfp"] = industry_mean_cfp
            industry_means.at[calendardate_cur, "industry_mean_asset_turnover"] = industry_mean_asset_turnover
            industry_means.at[calendardate_cur, "industry_mean_marketcap"] = industry_mean_marketcap

            industry_means.at[calendardate_cur, "industry_mean_return_on_assets"] = industry_mean_return_on_assets
            industry_means.at[calendardate_cur, "industry_mean_cash_flow_return_on_assets"] = industry_mean_cash_flow_return_on_assets
            industry_means.at[calendardate_cur, "industry_mean_rnd_intensity"] = industry_mean_rnd_intensity
            industry_means.at[calendardate_cur, "industry_mean_capex_intensity"] = industry_mean_capex_intensity
            industry_means.at[calendardate_cur, "industry_mean_advertising_intensity"] = industry_mean_advertising_intensity

            industry_means.at[calendardate_cur, "industry_mean_change_profit_margin"] = industry_mean_change_profit_margin
            industry_means.at[calendardate_cur, "industry_mean_percent_change_capex"] = industry_mean_percent_change_capex
        
        # CONT: GET DATES FROM industry_means

        # print("sum_percent_change_capex: ", sum_percent_change_capex)
        # print("industry_mean_percent_change_capex" , industry_mean_percent_change_capex)
        # print("Ticker: ", art_row_cur["ticker"], "calendardate: ", art_row_cur["calendardate"])
        # print("sum_change_profit_margin: ", sum_change_profit_margin)
        # print("industry_mean_change_profit_margin" , industry_mean_change_profit_margin)

        # ____________________________ DONE CALCULATE INDUSTRY NUMBERS _____________________________


        # DONE CALCULATING INDUSTRY MEAN VALUES THE CURRENT DATE (calendardate), PROCEEDING TO CALCULATE INDUSTRY ADJUSTED FEATURES:
       


        # Industry-adjusted book-to-market (bm_ia), Formula: bm - industry_mean(bm)
        # Industry adjusted book-to-market ratio.
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[index, "bm_ia"] = (art_row_cur["equityusd"] / art_row_cur["marketcap"]) - industry_means.at[date_cur, "industry_mean_bm"]

        # Industry-adjusted cash flow to price ratio (cfp_ia), Formula: cfp - indutry_mean(cfp)
        # cfp = SF1[ncfo]t-1 / SF1[marketcap]t-1
        # Industry adjusted cfp.
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[index, "cfp_ia"] = (art_row_cur["ncfo"] / art_row_cur["marketcap"]) - industry_means.at[date_cur, "industry_mean_cfp"]


        # Industyr-adjusted change in asset turnover (chatoia), Formula: ((SF1[revenueusd]t-1/ SF1[assetsavg]t-1) - industry_mean(SF1[revenueusd]t-1 / SF1[assetsavg]t-1))
        # 2-digit SIC - fiscal-year mean adjusted change in sales (sale) divided by average total assets (at)
        if art_row_cur["assetsavg"] != 0:
            sf1_art.at[index, "chatoia"] = (art_row_cur["revenueusd"] / art_row_cur["assetsavg"]) - industry_means.at[date_cur, "industry_mean_asset_turnover"]

        # Industry-adjusted size (mve_ia), Formula: SF1[marketcap]t-1 - industry_mean(SF1[marketcap]t-1)
        # 2-digit SIC industry-adjusted fiscal year-end market capitalization.
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[index, "mve_ia"] = art_row_cur["marketcap"] - industry_means.at[date_cur, "industry_mean_marketcap"]

        if not art_row_1y_ago.empty:
            # Industry-adjusted % change in capital expenditure (pchcapex_ia), Formula: ((SF1[capex]t-1 / SF1[capex]2-1) - 1) - industry_mean((SF1[capex]t-1 / SF1[capex]2-1))
            # 2-digit SIC - fiscal-year mean adjusted percent change in capital expenditures (capex).
            if art_row_1y_ago["capex"] != 0:
                # print("Percent change capex: ", ((art_row_cur["capex"] / art_row_1y_ago["capex"]) - 1))
                sf1_art.at[index, "pchcapex_ia"] = ((art_row_cur["capex"] / art_row_1y_ago["capex"]) - 1) - industry_means.at[date_cur, "industry_mean_percent_change_capex"]
                # print("pchcapex_ia right after assignment: ", sf1_art.iloc[index]["ticker"], sf1_art.iloc[index]["calendardate"], sf1_art.iloc[index]["pchcapex_ia"])
            
            
            
            # Industry-adjusted change in profit margin	Soliman (chpmia), 
            # Formula: (SF1[netinc]t-1 / SF1[revenueusd]t-1) - (SF1[netinc]t-2 / SF1[revenueusd]t-2) - industry_mean((SF1[netinc]t-1 / SF1[revenueusd]t-1) - (SF1[netinc]t-2 / SF1[revenueusd]t-2))  --> [chprofitmargin]t-1 - industry_mean([chprofitmargin]t-1)
            # 2-digit SIC - fiscal-year mean adjusted change in income before extraordinary items (ib) divided by sales (sale).
            if art_row_cur["revenueusd"] != 0 and art_row_1y_ago["revenueusd"] != 0:
                sf1_art.at[index, "chpmia"] = ((art_row_cur["netinc"] / art_row_cur["revenueusd"]) - (art_row_1y_ago["netinc"] / art_row_1y_ago["revenueusd"])) - industry_means.at[date_cur, "industry_mean_change_profit_margin"]



        # Industry sales concentration (herf), Formula: SF1[revenueusd] is used to proxy market share
        # 2-digit SIC - fiscal-year sales concentration (sum of squared percent of sales in industry for each company)
        # this calculation is done each year for each industry, and then average the values over the past 3 years (or two years).
        herf = get_herf(sf1_art, sf1_art_for_date, calendardate_1y_ago) # Maybe cache this...
        sf1_art.at[index, "herf"] = herf
        

        # Finantial statement score (ms): Fromula: Sum of 8 (6) indicator variables for fundamental performance. (My own type, excluding too demanding indicators)
        # Excluding ROA variability and sales growth variability
        # Advertising expenses is not availabe, so I use Selling, General and Administrative expenses (sgna)
        # I assume they mean G-Score, https://www.businessinsider.com/glamour-stocks-avoiding-falling-stars-using-mohanrams-g-score-2011-7?r=US&IR=T&IR=T
        if (art_row_cur["assetsavg"] != 0) and (art_row_cur["netinc"] != 0):
            i1_roa_above_avg = 1 if ((art_row_cur["netinc"] / art_row_cur["assetsavg"]) > industry_means.at[date_cur, "industry_mean_return_on_assets"]) else 0
            i2_cf_roa_above_avg = 1 if ((art_row_cur["ncfo"] / art_row_cur["assetsavg"]) > industry_means.at[date_cur, "industry_mean_cash_flow_return_on_assets"]) else 0
            i3_ncfo_exceeds_netinc = 1 if (art_row_cur["ncfo"] > art_row_cur["netinc"]) else 0
            i6_rnd_intensity = 1 if ((art_row_cur["rnd"] / art_row_cur["assetsavg"]) > industry_means.at[date_cur, "industry_mean_rnd_intensity"]) else 0
            i7_capex_indensity = 1 if ((art_row_cur["capex"] / art_row_cur["assetsavg"]) > industry_means.at[date_cur, "industry_mean_capex_intensity"]) else 0
            i8_advertising_intensity = 1 if ((art_row_cur["sgna"] / art_row_cur["assetsavg"]) > industry_means.at[date_cur, "industry_mean_advertising_intensity"]) else 0
            
            ms = i1_roa_above_avg + i2_cf_roa_above_avg + i3_ncfo_exceeds_netinc + i6_rnd_intensity + i7_capex_indensity + i8_advertising_intensity
            sf1_art.at[index, "ms"] = ms


        # Financial statements score (ps): Piotroski 	2000, JAR 	Sum of 9 indicator variables to form fundamental health score.	See link in notes
        # Link: https://www.investopedia.com/terms/p/piotroski-score.asp

        if (not art_row_1y_ago.empty) and (art_row_cur["assetsavg"] != 0) and (art_row_1y_ago["assetsavg"] != 0) and (art_row_cur["liabilitiesc"] != 0) and (art_row_1y_ago["liabilitiesc"] != 0):
            i1_positive_netinc = 1 if (art_row_cur["netinc"] > 0) else 0
            i2_positive_roa = 1 if ( (art_row_cur["netinc"] / art_row_cur["assetsavg"]) > 0 ) else 0
            i3_ncfo_exceeds_netinc = 1 if (art_row_cur["ncfo"] > art_row_cur["netinc"]) else 0
            i4_lower_long_term_debt_to_assets = 1 if ((art_row_cur["debtnc"] / art_row_cur["assetsavg"]) < (art_row_1y_ago["debtnc"] / art_row_1y_ago["assetsavg"])) else 0
            i5_higher_current_ratio = 1 if ((art_row_cur["assetsc"] / art_row_cur["liabilitiesc"]) > (art_row_1y_ago["assetsc"] / art_row_1y_ago["liabilitiesc"])) else 0
            i6_no_new_shares = 1 if (art_row_cur["sharesbas"] <= art_row_1y_ago["sharesbas"]) else 0
            i7_higher_gross_margin =  1 if ( ((art_row_cur["revenueusd"] - art_row_cur["cor"]) / art_row_cur["revenueusd"]) > ((art_row_1y_ago["revenueusd"] - art_row_1y_ago["cor"] ) / art_row_1y_ago["revenueusd"]) ) else 0
            i8_higher_asset_turnover_ratio =  1 if ((art_row_cur["revenueusd"] / art_row_cur["assetsavg"]) > (art_row_1y_ago["revenueusd"] / art_row_1y_ago["assetsavg"])) else 0

            ps = i1_positive_netinc + i2_positive_roa + i3_ncfo_exceeds_netinc + i4_lower_long_term_debt_to_assets + i5_higher_current_ratio + i6_no_new_shares + i7_higher_gross_margin + i8_higher_asset_turnover_ratio
            sf1_art.at[index, "ps"] = ps

    return sf1_art

def get_row_with_closest_date(df, date, margin):
    """
    Returns the row in the df closest to the date as long as it is within the margin.
    If no such date exist it returns None.
    Assumes only one ticker in the dataframe.
    """
    acceptable_dates = get_acceptable_dates(date, margin)
    candidates = df.loc[df["datekey"].isin(acceptable_dates)]
    if len(candidates.index) == 0:
        return pd.DataFrame()
    
    best_row = select_row_closes_to_date(candidates, date)
    return best_row


def get_acceptable_dates(date, margin):
    dates = [(date + timedelta(days=x)).isoformat() for x in range(-margin, +margin)]
    dates.insert(0, date.isoformat())
    return dates

def select_row_closes_to_date(candidates, desired_date):
    candidate_dates = candidates.loc[:,"datekey"].tolist()
    
    best_date = min(candidate_dates, key=lambda candidate_date: abs(desired_date - candidate_date))
    best_row = candidates.loc[candidates["datekey"] == best_date].iloc[0]

    return best_row



def get_chpmia_pchcapex_ia(sf1_art, sf1_art_for_date, calendardate_1y_ago):
    sum_percent_change_capex = 0
    sum_change_profit_margin = 0
    number_of_observations_capex = 0
    number_of_observations_profit_margin = 0
    industry_mean_percent_change_capex = 0
    industry_mean_change_profit_margin = 0

    for _, sf1_art_row_for_date in sf1_art_for_date.iterrows():
        sf1_art_row_for_date_1y_ago = sf1_art.loc[(sf1_art["ticker"] == sf1_art_row_for_date["ticker"]) & (sf1_art["calendardate"] == calendardate_1y_ago)] # sf1_art_ticker[sf1_art_ticker["calendardate"] == calendardate_1y_ago]

        if not sf1_art_row_for_date_1y_ago.empty:
            sf1_art_row_for_date_1y_ago = sf1_art_row_for_date_1y_ago.iloc[-1]

            cur_capex = sf1_art_row_for_date["capex"] # might be nan
            prev_capex = sf1_art_row_for_date_1y_ago["capex"] # might be nan
            
            # print("Date: ", calendardate_cur)
            # print(type(cur_capex), type(prev_capex), ((not np.isnan(cur_capex)) and (not np.isnan(prev_capex)) and (prev_capex != 0)))
            # print("cur_capex: ", cur_capex, type(cur_capex), np.isnan(cur_capex))
            # print("prev_capex: ", prev_capex, type(prev_capex), np.isnan(prev_capex))
            # print("Is all good to add to sum_percent_change_capex? ", ((not np.isnan(cur_capex)) and (not np.isnan(prev_capex)) and (prev_capex != 0)) )
            
            if (not np.isnan(cur_capex)) and (not np.isnan(prev_capex)) and (prev_capex != 0):
                sum_percent_change_capex += (cur_capex / prev_capex) - 1
                number_of_observations_capex += 1

            # Industry mean change in profit margin: industry_mean((SF1[netinc]t-1 / SF1[revenueusd]t-1) - (SF1[netinc]t-2 / SF1[revenueusd]t-2))
            if (sf1_art_row_for_date["revenueusd"] != 0) and (sf1_art_row_for_date_1y_ago["revenueusd"] != 0):
                cur_profit_margin = sf1_art_row_for_date["netinc"] / sf1_art_row_for_date["revenueusd"]
                prev_profit_margin = sf1_art_row_for_date_1y_ago["netinc"] / sf1_art_row_for_date_1y_ago["revenueusd"]
                sum_change_profit_margin += cur_profit_margin - prev_profit_margin
                number_of_observations_profit_margin += 1

    if number_of_observations_capex != 0:
        industry_mean_percent_change_capex = sum_percent_change_capex / number_of_observations_capex # SO THEY MIGHT NOT BE AVIALABLE..
    if number_of_observations_profit_margin != 0:
        industry_mean_change_profit_margin = sum_change_profit_margin / number_of_observations_profit_margin


    return (industry_mean_change_profit_margin, industry_mean_percent_change_capex)




def get_herf(sf1_art, sf1_art_for_date, calendardate_cur, calendardate_1y_ago):
    sum_industry_revenue = sf1_art_for_date["revenueusd"].sum()
    sum_sqrd_percent_of_revenue = 0

    sf1_art_for_date_1y_ago = sf1_art.loc[sf1_art["calendardate_1y_ago"] == calendardate_1y_ago].drop_duplicates(subset="ticker", keep="first")

    for i, row_cur in sf1_art_for_date.iterrows():
        sum_sqrd_percent_of_revenue_cur += (row_cur["revenueusd"] / sum_industry_revenue)**2
    
    for i, row_cur in sf1_art_for_date_1y_ago.iterrows():
        sum_sqrd_percent_of_revenue_1y_ago += (row_cur["revenueusd"] / sum_industry_revenue)**2
    
    return (sum_sqrd_percent_of_revenue_cur + sum_sqrd_percent_of_revenue_1y_ago) / 2


if __name__ == "__main__":
    
    # import mp engine, read metadata and sf1 and set datekey as index
    sf1_art = pd.read_csv("./datasets/testing/sf1_art.csv", index_col="datekey", parse_dates=["datekey", "calendardate"])
    metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", index_col="ticker", parse_dates=["firstpricedate"])

    # call mp engine and split on ticker

    # Drop unused columns

    # update all code to use the index

    # update code where reindexing, filling and shifting is a better and simpler solution

    # write function to detect missing data (and maybe fix it, may be simpler and less brittle code if I do so)


    sf1_art["datekey"] = pd.to_datetime(sf1_art["datekey"])
    sf1_art["calendardate"] = pd.to_datetime(sf1_art["calendardate"])
    
    metadata["firstpricedate"] = pd.to_datetime(metadata["firstpricedate"])

    sep = pandas_mp_engine(callback=add_equally_weighted_weekly_market_returns, atoms=sep, data=None, \
        molecule_key='sep', split_strategy= 'date', \
            num_processes=4, molecules_per_process=1)