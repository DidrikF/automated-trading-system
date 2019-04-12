import sys
from automated_trading_system.dataset_builder.dataset import Dataset, merge_datasets_simple
from automated_trading_system.logger.logger import Logger
from automated_trading_system.helpers.helpers import print_exception_info

from automated_trading_system.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from automated_trading_system.helpers.custom_exceptions import FeatureError
import pandas as pd
import numpy as np
import logging
import datetime
from dateutil.relativedelta import *
from os import listdir
from os.path import isfile, join

"""
Each step is performed for each industry separately

Step-by-Step Dataset Construction:
1. Extend the SEP dataset with information usefull for sampling (most recent 10-K filing date, Industry classifications)
2. Use different sampling techniques to get monthly observations
    1. At first use timebars (sampling at a fixed time interval), but try to respect the different fiscal years
(3). Calculate the various price and volume based features
4. Add inn SF1 and DAILY data
5. Compute features based on SF1
6. Select the features you want and combine into one ML ready dataset
"""

if __name__ == "__main__":

    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = "./logs/price_volume_features_" + date_time + ".log"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[logging.FileHandler(log_filename, mode="a"), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()


    # THE REST OF THE SCRIPT SHOULD BE DONE ONCE PER FILE
    samples_path = "./datasets/timebar_samples/"
    filenames = [f for f in listdir(samples_path) if isfile(join(samples_path, f))]
    filename = ["Advertising Agencies.csv"]
    # filenames = ["Building Materials.csv"]
    total_tickers = 0

    for filename in filenames:
        try:
            file_path = samples_path + filename
            samples = pd.read_csv(file_path)
        except Exception as e:
            print(e)
            logger.error("# Failed to read csv file with path: {}, skipping this file and trying the next.".format(file_path))
            continue

        samples["date"] = pd.to_datetime(samples["date"], format="%Y-%m-%d")
        samples["datekey"] = pd.to_datetime(samples["datekey"], format="%Y-%m-%d")

        # Maybe treat each company in isolation, might be easier to reason about...
        tickers = list(samples.ticker.unique())

        # print("Working on tickers: ", tickers)

        drop_indexes = list()
        last_ticker = None

        columns = list(samples.columns.values)
        # print(columns)
        result = pd.DataFrame(columns=columns)

        tickers_with_timegaps = set()

        # Verify absence of gaps in the data (this is timebar_samples btw)
        """
        Before starting to calculate price and volume (etc.) features, it should be safe to assume that 
        samples are available at a monthly frequency for all companies. This is verifies by the below code.
        Any discovered gaps must be eliminated before commenting out this section and running the rest of
        the script.
        """
        for ticker in tickers:
            ticker_samples = samples.loc[samples["ticker"] == ticker]
            last_row = None
            time_gaps_in_ticker = False
            gaps = list()
            for t_index, t_row in ticker_samples.iterrows():
                if last_row is None:
                    last_row = t_row
                    continue

                if ( (t_row["date"] - relativedelta(days=+35)) < last_row["date"] ) and ( last_row["date"] > (t_row["date"] - relativedelta(days=+25)) ):
                   gaps.append(t_row["date"])
                   tickers_with_timegaps.add(ticker)
                   time_gaps_in_ticker = True 

                last_row = t_row

            if time_gaps_in_ticker:
                logger.error("# Ticker {} has timegaps at: {}".format(ticker, gaps))

        total_tickers += len(tickers)
        logger.debug("# File: {} - Tickers with time gaps ({}/{}): {}".format(filename, len(tickers_with_timegaps), len(tickers), tickers_with_timegaps))
        logger.debug("Total tickers: {}".format(total_tickers))

        continue

        # From here on you can assume that all companies have monthly samples without gaps (safe to use shift()...)
        for ticker in tickers:
            ticker_samples = samples.loc[samples["ticker"] == ticker]
            # logger.debug("Length of ticker_observations for {}: {}".format(ticker, len(ticker_samples)))

            ticker_samples['return'] = ( ticker_samples['close'].shift(-1) / ticker_samples['close'] ) - 1

            ticker_samples['mom1m'] = ( ticker_samples['close'] / ticker_samples['close'].shift(1) ) - 1

            ticker_samples['mom6m'] = ( ticker_samples['close'] / ticker_samples['close'].shift(6) ) - 1
            
            ticker_samples['mom12m'] = ( ticker_samples['close'] / ticker_samples['close'].shift(12) ) - 1

            ticker_samples['chmom6m'] = ( ( ticker_samples['close'].shift(1) / ticker_samples['close'].shift(6) ) - 1 ) - ( (ticker_samples['close'].shift(7) / ticker_samples['close'].shift(12)) - 1 )


            i = -1
            last_index = ticker_samples.iloc[-1].index
            # Add date of last 10-K filing and age of sf1_art data (Dont think I need to, dont know what I was thinking when writing this)
            for cur_index, row in ticker_samples.iterrows():
                """
                i += 1
                if i < 12:
                    # drop_indexes.append(cur_index) don't think I nees this because I will drop all rows with N/A values
                    continue;                
                """

                #row_1_month_ago = ticker_samples.iloc[cur_index-1]
                #row_6_months_ago = ticker_samples.iloc[cur_index-6]
                #row_12_months_ago = ticker_samples.iloc[cur_index-12]

                if cur_index <= (last_index - 1):
                    # This index might be out of bounds!!!
                    row_next_month = ticker_samples.iloc[cur_index+1] # I might here trust the order of the indexes in the samples dataframe

                    # Return including dividends Calculation
                    ticker_samples.at[cur_index, 'return_including_dividends'] = ( (row_next_month["close"] - row["close"]) + row_next_month["last_months_dividends"]) / row["close"]


            # NOTE: Not sure everything works out with the indexes...
            result = result.append(ticker_samples, sort=True)

        # Drop indexes with less than 1 year of history behind them
        result = result.dropna(axis=0)

        # Ordering the columns
        columns.extend(["return", "mom1m", "mom6m", "mom12m", "chmom6m"])
        result = result[columns] 

        # Sort values by ticker, then date. Not sure if I need to, but...
        result = result.sort_values(by=["ticker", "date"]) 

        # Save
        result_dataset = Dataset.from_df(result)
        save_path = "./datasets/timebar_samples_with_features/" + filename
        result_dataset.to_csv(save_path)

        logger.debug("# Completed adding features for file: {}".format(filename))


#1: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
#2: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28
#3: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
#4: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

