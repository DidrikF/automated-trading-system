import pandas as pd
from packages.helpers.helpers import print_exception_info
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta

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

def add_sep_features(index_filename, testing):

    if testing == True:
        try:
            sep_extended = pd.read_csv("./datasets/testing/sep_extended.csv", index_col=1)
            sep_sampled = pd.read_csv("./datasets/testing/sep_sampled.csv", index_col=1)
        except Exception as e:
            print_exception_info(e)
            sys.exit()

    else:
        # Need to adapt to take filename and use with multiprocesseing
        pass
    
    sep_extended["date"] = pd.to_datetime(sep_extended["date"])
    sep_extended["datekey"] = pd.to_datetime(sep_extended["datekey"])
    sep_sampled["date"] = pd.to_datetime(sep_extended["date"])
    sep_sampled["datekey"] = pd.to_datetime(sep_extended["datekey"])


    tickers = list(sep_sampled.ticker.unique())

    for ticker in tickers:
        sep_sampled_ticker = sep_sampled.loc[sep_sampled["ticker"] == ticker]
        sep_extended_ticker = sep_extended.loc[sep_extended["ticker"] == ticker]

        # Iterating sampled rows
        for index, cur_row in sep_sampled_ticker.iterrows():
            date = cur_row["date"]
            
            # Get rows from sep_extended
            date_1m_ahead = date + relativedelta(months=+1)
            date_1m_ago = date - relativedelta(months=+1)
            date_6m_ago = date - relativedelta(months=+6)
            date_12m_ago = date - relativedelta(months=+12)
            date_24m_ago = date - relativedelta(months=+24)

            row_1m_ahead = get_row_with_closest_date(sep_extended_ticker, date_1m_ahead, 3)
            row_1m_ago = get_row_with_closest_date(sep_extended_ticker, date_1m_ago, 3)
            row_6m_ago = get_row_with_closest_date(sep_extended_ticker, date_6m_ago, 7)
            row_12m_ago = get_row_with_closest_date(sep_extended_ticker, date_12m_ago, 14)
            row_24m_ago = get_row_with_closest_date(sep_extended_ticker, date_24m_ago, 14)
            
            # print(row_1m_ahead)

            # Calculate and add new features
            if not row_1m_ahead.empty:
                sep_sampled.at[index, "return"] = ( row_1m_ahead["close"] / cur_row["close"] ) - 1
                # sep_sampled.at[index, "total_return"] = 
            if not row_1m_ago.empty:
                sep_sampled.at[index, "mom1m"] = ( cur_row["close"] / row_1m_ago["close"] ) - 1

            if not row_6m_ago.empty:
                sep_sampled.at[index, "mom6m"] = ( cur_row["close"] / row_6m_ago["close"] ) - 1 
                print("cur date: {}, 6m ago date: {}".format(cur_row["date"], row_6m_ago["date"]))
            if not row_12m_ago.empty:
                sep_sampled.at[index, "mom12m"] = ( cur_row["close"] / row_12m_ago["close"] ) - 1 
                print("cur date: {}, 12m ago date: {}".format(cur_row["date"], row_12m_ago["date"]))

            if not row_24m_ago.empty:
                sep_sampled.at[index, "mom24m"] = ( cur_row["close"] / row_24m_ago["close"] ) - 1
                # print("cur date: {}, 24m ago date: {}".format(cur_row["date"], row_24m_ago["date"]))


            # Beta
            """
            Estimated market beta from weekly returns and equal weighted market returns for 3
            years ending month t-1 with at least 52 weeks of returns.	
            Cov(Ri, Rm)/Var(Rm), where Ri, Rm is weekly measurements and Rm is equal 
            weighted market returns. 52 weeks to 3 years of data is used.
            """
            # Get weekly returns for the last 52 weeks for the Ticker


            # Get weekly equal weighted market returns for the last 52 weeks 
            # NEED TO FIND A WAY TO GET THIS DATA; external source approximated by S&P500 equal weighted, 
            # or calculated manually using my list of S&P500 constituents


    if testing == True:
        return sep_sampled

    # Save
    # OBS need to update the filepath
    sep_sampled.to_csv("./datasets/testing/sep_featured.csv")


def get_row_with_closest_date(df, date, margin):
    """
    Returns the row in the df closest to the date as long as it is within the margin.
    If no such date exist it returns None.
    Assumes only one ticker in the dataframe.
    """
    acceptable_dates = get_acceptable_dates(date, margin)
    candidates = df.loc[df["date"].isin(acceptable_dates)]
    if len(candidates.index) == 0:
        return pd.DataFrame()
    
    best_row = select_row_closes_to_date(candidates, date)
    return best_row


def get_acceptable_dates(date, margin):
    dates = [(date + timedelta(days=x)).isoformat() for x in range(-margin, +margin)]
    dates.insert(0, date.isoformat())
    return dates

def select_row_closes_to_date(candidates, desired_date):
    candidate_dates = candidates.loc[:,"date"].tolist()
    
    best_date = min(candidate_dates, key=lambda candidate_date: abs(desired_date - candidate_date))
    best_row = candidates.loc[candidates["date"] == best_date].iloc[0]

    return best_row