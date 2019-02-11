import sys
from packages.dataset_builder.dataset import Dataset, merge_datasets_simple
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError
import pandas as pd
import numpy as np
import logging

"""
I dont know the answer to all the below steps, so I need to write code that can easily be manipulated to add a feature at any step.

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
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, filemode='a', handlers=[logging.FileHandler("./logs/build_sampleable_sep.log"), logging.StreamHandler(sys.stdout)])
    

    try:
        # logger = Logger('./logs')
        logger = logging.getLogger()
    except Exception as e:
        print_exception_info(e)
        sys.exit()

    try:
        observations = pd.read_csv("./datasets/sharadar_compiled/PURGED_SAMPLEABLE_SEP.csv", low_memory=False) # , nrows=100000
    except Exception as e:
        print_exception_info(e)
        logger.close()
        sys.exit()

    observations["date"] = pd.to_datetime(observations["date"], format="%Y-%m-%d")
    observations["datekey"] = pd.to_datetime(observations["datekey"], format="%Y-%m-%d")



    observations.sort_values(by=["ticker", "date"], inplace=True)


    # Add date date of last 10-K filing and age of sf1_art data
    for i, obs_index in enumerate(observations.index):

        """
        Sampling strategy: Monthly best effort:
        Description: The purpose of this strategy is to sample observations so that observations are captured as close to the filing of 10-K forms as possible.
        The months after the filing of a 10-K form will be sampled at the closest date to the same day-number in the following month.
        Release at 15. jan of 10-K form will cause a sample, if 15. feb is a sunday a new sample will happen the 16. feb.

        """

        sep_ticker = sep.iloc[sep_index]["ticker"]
        sep_date = sep.iloc[sep_index]["date"]
        
        """
        Get the row in SF1_ART with 10-K filing date (datekey) as close in the past as possible
            1. Get candidate rows
            2. Select best by choosing the closet past observation (based on datekey)
        """    
        sf1_art_for_ticker = sf1_art.loc[sf1_art["ticker"] == sep_ticker, :]
        
        # Extract past dates
        past_sf1_art_for_ticker = sf1_art_for_ticker.loc[sf1_art_for_ticker["datekey"] <= sep_date]
        
        # Get date of latest 10-K form filing
        date_of_latest_filing = past_sf1_art_for_ticker["datekey"].max()

        sep.at[sep_index, "datekey"] = date_of_latest_filing
        sep.at[sep_index, "age"] = (date_of_latest_filing - sep_date)

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
        
        if i % 10000 == 0:
            logger.debug("# Done {} iterations out of {}".format(i, len(sep.index)))




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
