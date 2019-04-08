import pandas as pd
from packages.helpers.helpers import print_exception_info
import sys
import math
from dateutil.relativedelta import *
from datetime import datetime, timedelta
from packages.multiprocessing.engine import pandas_mp_engine

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

# NEED SF1[marketcap]t-1, SF1[sharefactor]t-1, SF1[sharesbas]t-1 passed to this function
def add_sep_features(sep_sampled, sep, sf1_art):
    # sep_sampled and sep contains data for one ticker
    # sep contains basic and market wide features added in sep_preparation.py

    sep_empty = True if (len(sep) == 0) else False

    if sep_empty == True:
        print("Sep was empty for some ticker in add_sep_features. Don't know why.")
        return sep_sampled


    pd.options.mode.chained_assignment = None  # default='warn'

    date_index = pd.date_range(sep.index.min(), sep.index.max())
    sep_filled = sep.reindex(date_index)
    sep_filled["adj_close"] = sep_filled["adj_close"].fillna(method="ffill")

    # Maybe i can to a combination or reindex, resample and shift to calculate features

    sep_filled_1m_ahead = sep_filled.shift(periods=-30)
    sep_filled_2m_ahead = sep_filled.shift(periods=-60)
    sep_filled_3m_ahead = sep_filled.shift(periods=-90)
    
    sep_filled_1m_ago = sep_filled.shift(periods=30)
    sep_filled_6m_ago = sep_filled.shift(periods=182)
    sep_filled_12m_ago = sep_filled.shift(periods=365)
    sep_filled_24m_ago = sep_filled.shift(periods=2*365)
    

    """ CALCULATE FEATURES IN NEED OF SEP_FILLED """
    # Labels
    sep_filled["return_1m"] = (sep_filled_1m_ahead["adj_close"] / sep_filled["adj_close"]) - 1
    sep_filled["return_2m"] = (sep_filled_2m_ahead["adj_close"] / sep_filled["adj_close"]) - 1
    sep_filled["return_3m"] = (sep_filled_3m_ahead["adj_close"] / sep_filled["adj_close"]) - 1


    # Momentum features
    sep_filled["mom1m"] = (sep_filled["adj_close"] / sep_filled_1m_ago["adj_close"]) - 1
    sep_filled["mom6m"] = (sep_filled_1m_ago["adj_close"] / sep_filled_6m_ago["adj_close"]) - 1 # 5-month cumulative returns ending one month before month end.
    sep_filled["mom12m"] = (sep_filled_1m_ago["adj_close"] / sep_filled_12m_ago["adj_close"]) - 1 # 11-month cumulative returns ending one month before month end.
    sep_filled["mom24m"] = (sep_filled["adj_close"] / sep_filled_24m_ago["adj_close"]) - 1 # MY OWN ...
    
    # In preparation for chmom --> Cumulative returns from months t-6 to t-1 (mom6m) minus months t-12 to t-7
    sep_filled_7m_ago = sep_filled.shift(periods=182+30)
    sep_filled["mom12m_to_7m"] = (sep_filled_7m_ago["adj_close"] / sep_filled_12m_ago["adj_close"]) - 1
    
    # Change in 6 month momentum (chmom): ((SEP[close]u-1 / SEP[close]u-6) - 1) - ((SEP[close]u-7 / SEP[close]u-12) -1) --> Cumulative returns from months t-6 to t-1 minus months t-12 to t-7
    sep_filled["chmom"] = sep_filled["mom6m"] - sep_filled["mom12m_to_7m"]



    def custom_resample(array_like):
        return array_like[0]

    # Flags to enable printing of the first good results from the below loop. Used during testing and debugging.
    print_first_good_result_1 = False
    print_first_good_result_2 = False
    print_first_good_result_3 = True

    first_date = None

    sf1_art_empty = True if (len(sf1_art) == 0) else False

    """ CALCULATE FEATURES ONLY FOR SAMPLES """
    for date, row in sep_sampled.iterrows():
        if first_date is None:
            first_date = date
        
        if sf1_art_empty == True:
            print("No sf1_art data for ticker {} in add_sep_features".format(row["ticker"]))
            break

        sf1_row = sf1_art.loc[sf1_art.datekey == row["datekey"]].iloc[-1]
        sharesbas = sf1_row["sharesbas"]
        sharefactor = sf1_row["sharefactor"]
        price = row["close"]
        marketcap = sharesbas*sharefactor*price

        date_2y_ago = date - relativedelta(years=2)
        date_1y_ago = date - relativedelta(years=1)
        date_3m_ago = date - relativedelta(months=3)
        date_2m_ago = date - relativedelta(months=2)
        date_1m_ago = date - relativedelta(months=1)
        
        # Using sep_filled
        sep_past_2years = sep_filled.loc[(sep_filled.index <= date) & (sep_filled.index >= date_2y_ago)]
        
        # Using sep / sep_prepared
        sep_past_year = sep.loc[(sep.index <= date) & (sep.index >= date_1y_ago)]
        sep_past_2months = sep.loc[(sep.index <= date) & (sep.index >= date_2m_ago)]
        sep_past_1month = sep.loc[(sep.index <= date) & (sep.index >= date_1m_ago)]

        # Size: (mve or mvel1): ln(SEP[close]m-1 * SF1[sharefactor]t-1 * SF1[sharesbas]t-1)
        if marketcap != 0:
            sep_sampled.at[date, "mve"] = math.log(marketcap)

        # Take Weekly Samples every monday
        if date >= (first_date + relativedelta(years=2)):
            # NOTE: sep_past_2years is forward-filled!
            sep_past_2years_reduced = sep_past_2years[["mom1w", "mom1w_ewa_market"]]
            weekly_samples = sep_past_2years_reduced.resample("W-MON").apply(custom_resample)
            weekly_samples = weekly_samples.loc[weekly_samples.index <= date]
        
            # Beta (beta): Cov(Ri, Rm)/Var(Rm), where Ri, Rm is weekly measurements and Rm is equial weighted market returns. 52 weeks to 3 years of data is used.        
            covariance = weekly_samples.cov().iloc[0][1]
            variance_market = weekly_samples["mom1w_ewa_market"].var()
            if variance_market != 0:
                beta = covariance / variance_market
                sep_sampled.at[date, "beta"] = beta
            
            # Beta Squared (betasq): beta^2
            sep_sampled.at[date, "betasq"] = beta**2

            # Idiosyncratinc return vol (idiovol): std(SEP[weekly_return], Equal weighted weekly market returns) using minimum 1 year, maximum 3 years of data
            # Standard deviation of residuals of weekly returns on weekly equal weighted market returns for 3 years prior to month end
            weekly_samples["diff"] = weekly_samples["mom1w"] - weekly_samples["mom1w_ewa_market"]
            std_stock_less_market_return = weekly_samples["diff"].std()
            sep_sampled.at[date, "idiovol"] = std_stock_less_market_return

            """
            # Printing for testing and debugging
            if print_first_good_result_1 == True:
                print(row["ticker"], date)
                print(weekly_samples)
                print("cov: ", covariance, "var: ", variance_market, "beta: ", beta)
                print("std diff: ", std_stock_less_market_return)
                print_first_good_result_1 = False    
            """
        
        if date >= (first_date + relativedelta(years=1)):
            # NOTE: sep_past_year is not forward-filled

            # Illiquidity (ill): avg(SEP[close]-SEP[open] / ( (SEP[close]+SEP[open]) / 2 )*SEP[volume]) for the past year
            # Average of daily (absolute return / dollar volume).

            illiquidity_df = pd.DataFrame()
            illiquidity_df["return"] = (sep_past_year["close"] / sep_past_year["open"] - 1)
            illiquidity_df["dollar_vol"] = (((sep_past_year["open"] + sep_past_year["close"]) / 2)*sep_past_year["volume"])
            illiquidity_df["return_over_dollar_vol"] = illiquidity_df["return"] / illiquidity_df["dollar_vol"]
            illiquidity = illiquidity_df["return_over_dollar_vol"].mean()
            sep_sampled.at[date, "ill"] = illiquidity

            
            # Dividend to price (dy): (sum(SEP[dividend]) the past year at t-1) / SF1[marketcap]t-1
            if marketcap != 0:
                sep_sampled.at[date, "dy"] = sep_past_year["dividends"].sum() / marketcap

            """
            if print_first_good_result_2 == True:
                print(row["ticker"], date)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    # print(illiquidity_df)
                    print(sep_past_year["dividends"])
                # print(illiquidity)
                print(row["datekey"], sharesbas, sharefactor, price, marketcap)
                print(sep_sampled.at[date, "dy"])
                print_first_good_result_2 = False
            """

        if date >= (first_date + relativedelta(months=3)):
            # ONLY DO FOR SAMPLES, NEED TO RUN SEP_PREPARATION
            # Share turnover (turn): avg(SEP[volume)m-1, SEP[volume]m-2, SEP[volume]m-3) / SF1[sharesbas]t-1
            # Average monthly trading volume for most recent 3 months scaled by number of shares outstanding in current month.
            sep_2m_ago_to_1m_ago = sep.loc[(sep.index < date_1m_ago) & (sep.index >= date_2m_ago)]
            sep_3m_ago_to_2m_ago = sep.loc[(sep.index < date_2m_ago) & (sep.index >= date_3m_ago)]

            volume_past_1m = sep_past_1month["volume"].sum()
            volume_2m_ago_to_1m_ago = sep_2m_ago_to_1m_ago["volume"].sum()
            volume_3m_ago_to_2m_ago = sep_3m_ago_to_2m_ago["volume"].sum()
            avg_monthly_volume = (volume_past_1m + volume_2m_ago_to_1m_ago + volume_3m_ago_to_2m_ago) / 3
            if sharesbas != 0:
                turn = avg_monthly_volume / sharesbas
                sep_sampled.at[date, "turn"] = turn


            """
            if print_first_good_result_3 == True:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(row["ticker"], date)
                    print(sep_past_1month["volume"])
                    print(sep_2m_ago_to_1m_ago["volume"])
                    print(sep_3m_ago_to_2m_ago["volume"])
                    print("sharesbas: ", sharesbas)
                    print(sep_sampled.at[date, "turn"])
                print_first_good_result_3 = False
            """

        if date >= (first_date + relativedelta(months=2)):
            # Dollar trading volume (dolvol): ln(sum(SEP[close]*SEP[volume]) for all days the past two months)
            sum_close_volume = (sep_past_2months["close"]*sep_past_2months["volume"]).sum()
            if sum_close_volume != 0:
                dolvol = math.log(sum_close_volume)
                sep_sampled.at[date, "dolvol"] = dolvol
                
            """
            if print_first_good_result_3 == True:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(row["ticker"], date)
                    print(sep_past_2months["close"]*sep_past_2months["volume"])
                    print(dolvol)
                print_first_good_result_3 = False
            """

    
        if date >= (first_date + relativedelta(months=1)):
            sep_past_1month_daily_returns = (sep_past_1month["close"]/sep_past_1month["open"]) - 1

            # Maximum daily return (maxret): max(SEP[open]d - SEP[close]d) last 1 month
            sep_sampled.at[date, "maxret"] = sep_past_1month_daily_returns.max()

            # Return volatility (retvol): std(SEP[open]d/sep[close]d - 1) the last 1 month
            sep_sampled.at[date, "retvol"] = sep_past_1month_daily_returns.std()

            # Volatility of liquidity (dollar trading volume) (sdt_dolvol): 
            # std((SEP[open]d - SEP[close]d)/2 * SEP[volume]d) using 1 month of data * sqrt(22)
            sep_sampled.at[date, "std_dolvol"] = (((sep_past_1month["close"]+sep_past_1month["open"]) / 2) * \
                sep_past_1month["volume"]).std() * math.sqrt(22)


            # Volatility of liquidity (share turnover) (std_turn): std([daily_turnover]d) using 1 month of data * sqrt(22)
            # share turnover = shares traded over the period / avg number of outstanding shares
            sep_sampled.at[date, "std_turn"] = (sep_past_1month["volume"] / sep_past_1month["sharesbas"]).std() * math.sqrt(22)

            # Zero trading days (zerotrade): count(SEP[volume]d == 0) for 1 month of data / [monthly_turnover]m-1
            # Turnover weighted number of zero trading days for most recent 1 month.
            num_zero_trading_days_the_past_month = len(sep_past_1month.loc[sep_past_1month["volume"] == 0])
            total_volume_past_month = sep_past_1month["volume"].sum()
            avg_number_of_shares_outstanding = sep_past_1month["sharesbas"].mean()
            monthly_turnover = total_volume_past_month / avg_number_of_shares_outstanding
            deflator = 11000/12 # Liu selected deflator of 11000 for 12-month zerotrade in 2006, might not be optimal for todays market.
            number_of_trading_days_past_month = len(sep_past_1month)
            if (number_of_trading_days_past_month != 0) and (monthly_turnover != 0):
                sep_sampled.at[date, "zerotrade"] = (num_zero_trading_days_the_past_month + (1/monthly_turnover)/deflator) * 21/number_of_trading_days_past_month
            
            """
            if (print_first_good_result_3 == True) and (num_zero_trading_days_the_past_month > 0):
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(row["ticker"], date)
                    print(sep_past_1month[["volume", "sharesbas"]])
                    print("num_zero_trading_days_the_past_month: ", num_zero_trading_days_the_past_month)
                    print("number_of_trading_days_past_month: ", number_of_trading_days_past_month)
                    print(sep_sampled.at[date, "zerotrade"])
                print_first_good_result_3 = False
            """
    
    # Downsample and move values over to sep_sampled (which is what gets returned in the end)
    sep_filled = sep_filled.loc[sep_sampled.index]
    # Remember to copy over any features added in sep_industry_features.py
    sep_sampled[["return_1m", "return_2m", "return_3m", "mom1m", "mom6m", "mom12m", "mom24m", "chmom", "indmom"]] = \
         sep_filled[["return_1m", "return_2m", "return_3m", "mom1m", "mom6m", "mom12m", "mom24m", "chmom", "indmom"]]


    return sep_sampled



if __name__ == "__main__":

    # load data, import multiprocessing engine, split and run code with engine...
    sep_df = pd.read_csv("./datasets/testing/sep.csv", parse_dates=["date"], index_col="date")
    sep_sampled_df = pd.read_csv("./datasets/testing/sep_sampled.csv", parse_dates=["date"], index_col="date")

    # sep_df.index = pd.to_datetime(sep_df.index)
    # sep_sampled_df.index = pd.to_datetime(sep_sampled_df.index)

    sep_sampled_df = sep_sampled_df.drop(columns='index')
    sep_sampled_df = sep_sampled_df.drop(sep_sampled_df.columns[sep_sampled_df.columns.str.contains('unnamed',case = False)],axis = 1)
    

    date_index = pd.date_range(sep_df.index[0], sep_df.index[-1])

    # print(len(sep_df.index))

    # sep_df = sep_df[~sep_df.index.duplicated(keep="first")]

    # print(len(sep_df.index))

    # print(sep_df.duplicated())

    # sep_6m_old = sep[sep_sampled.index - relativedelta(months=+6)]
    # print(sep_6m_old.head())

    # print(sep_sampled_df.head())

    result = pandas_mp_engine(callback=add_sep_features, atoms=sep_sampled_df, \
        data={'sep': sep_df}, molecule_key='sep_sampled', split_strategy= 'ticker', \
            num_processes=1, molecules_per_process=1)
