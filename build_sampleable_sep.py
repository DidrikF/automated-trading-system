import sys
from packages.dataset_builder.dataset import Dataset, merge_datasets_simple
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError
import pandas as pd
import numpy as np
import logging
import time

"""
I dont know the answer to all the below steps, so I need to write code that can easily be manipulated to add a feature at any step.

Each step is performed for each industry (first "Banks - Regional - US" with 2394 firms) separately. This is to limit lookup times in large dataframes.

Step by step revised:
1. Extend the SEP dataset with information usefull for sampling (most recent 10-K filing, Industry classification, MarketCap)
2. Use different sampling techniques to get monthly observations
3. Calculate the various price and volume based features
    1. I will end up with maybe 5-10 GB of data at that point
4. Add inn SF1 and DAILY data
5. Compute features based 
6. Sample data (from prices)


"""

if __name__ == "__main__":

    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[logging.FileHandler("./logs/build_sampleable_sep.log", mode="a"), logging.StreamHandler(sys.stdout)])
    

    try:
        # logger = Logger('./logs')
        logger = logging.getLogger()
    except Exception as e:
        print_exception_info(e)
        sys.exit()

    try:
        # sep = Dataset("./datasets/sharadar/PURGED_SEP.csv", None, "./datasets/sharadar/SHARADAR_INDICATORS.csv")
        # sf1_art = Dataset("./datasets/sharadar/SHARADAR_SF1_ART.csv", None, "./datasets/sharadar/SHARADAR_INDICATORS.csv")
        # tickers_metadata = Dataset("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", None, "./datasets/sharadar/SHARADAR_INDICATORS.csv")
        sep = pd.read_csv("./datasets/sharadar/PURGED_SEP.csv", low_memory=False) # , nrows=100000
        bank_tickers_df = pd.read_csv("./datasets/industry_tickers/Banks - Regional - US.csv")
        bank_tickers = bank_tickers_df["ticker"]
        sep = sep.loc[sep["ticker"].isin(bank_tickers)]
        print("Length of sep: ", len(sep))
        sf1_art = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ART.csv", low_memory=False)
        sf1_art = sf1_art.loc[sf1_art["ticker"].isin(bank_tickers)]

        tickers_metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", low_memory=False)
    except Exception as e:
        print_exception_info(e)
        sys.exit()

    sep["date"] = pd.to_datetime(sep["date"], format="%Y-%m-%d")
    sf1_art["datekey"] = pd.to_datetime(sf1_art["datekey"], format="%Y-%m-%d")

    sep.sort_values(by=["ticker", "date"], inplace=True)
    # sf1_art.sort(by=["ticker", "datekey"])
    # tickers_metadata.sort(by="ticker")

    # Add columns
    for col in ['datekey', 'age', 'industry', 'sector', 'siccode']:
        sep[col] = None



    # Add date date of last 10-K filing and age of sf1_art data
    """for index, row in sep.interrows():
        if i > 10:
            break

        sep_ticker = row["ticker"]
        sep_date = row["date"]
    """
    for sep_index, row in sep.iterrows():
        # sep_ticker = sep.iloc[sep_index]["ticker"]
        # sep_date = sep.iloc[sep_index]["date"]
        sep_ticker = row["ticker"]
        sep_date = row["date"]


        # print("ticker: ", sep_ticker)
        # print("price date: ", sep_date)
        # sys.exit()

        """
        Get the row in SF1_ART with 10-K filing date (datekey) as close in the past as possible
            1. Get candidate rows
            2. Select best by choosing the closet past observation (based on datekey)
        """

        t0 = time.time()

        sf1_art_for_ticker = sf1_art.loc[sf1_art["ticker"] == sep_ticker]
        
        # Extract past dates
        past_sf1_art_for_ticker = sf1_art_for_ticker.loc[sf1_art_for_ticker["datekey"] <= sep_date]
        
        # Get date of latest 10-K form filing
        date_of_latest_filing = past_sf1_art_for_ticker["datekey"].max()

        sep.at[sep_index, "datekey"] = date_of_latest_filing
        sep.at[sep_index, "age"] = (date_of_latest_filing - sep_date)

        t1 = time.time()

        total = t1-t0
        print("Time to get and set datekey and age: ", total) # Takes 0.003 seconds for the larges industry when sampling down to only those tickers for both sep and sf1 datasets

        """
        # Get SF1 row associated with latest date of 10-K form filing        
        # I WANT THIS TO BE A SERIES; HOPEFULLY I CEN THEN GET THE VALUE BY THE COLUMN
        latest_sf1_row = sf1_art_for_ticker.loc[sf1_art_for_ticker["datekey"] == date_of_latest_filing]
        """

        # Add industry classification
        metadata = tickers_metadata.loc[tickers_metadata["ticker"] == sep_ticker]

        if metadata.empty:
            message = "# Metadata was not available for ticker: " + sep_ticker
            logger.warning(message)
            continue

        sep.at[sep_index, "industry"] = metadata.iloc[-1]["industry"]
        sep.at[sep_index, "sector"] = metadata.iloc[-1]["sector"]
        sep.at[sep_index, "siccode"] = metadata.iloc[-1]["siccode"]
        
        if sep_index % 10000 == 0:
            print()
            logger.debug("# Done {} iterations out of {}".format(sep_index, len(sep.index)))




    """
    daily_prices_df = merge_datasets_simple(daily.data, prices.data, on=['ticker', 'date'], suffixes=('_daily', '_prices'))
    daily_prices_dataset = Dataset.from_df(daily_prices_df)


    daily_prices_dataset.to_csv("./datasets/sharadar_compiled/set_1.csv")


    prices_daily_df = merge_datasets_simple(prices.data, daily.data, on=['ticker', 'date'], suffixes=('_daily', '_prices'))
    prices_daily_dataset = Dataset.from_df(prices_daily_df)


    prices_daily_dataset.to_csv("./datasets/sharadar_compiled/prices_daily_set_2.csv")
    """

    # sep = sep.astype({"siccode": int})

    sep_dataset = Dataset.from_df(sep)
    sep_dataset.to_csv("./datasets/sharadar_compiled/PURGED_SAMPLEABLE_SEP.csv")

    logger.close()



"""
Used:
industry
sector
siccode

Not Used:
sicsector
famaindustry
famasector
sicindustry
"""