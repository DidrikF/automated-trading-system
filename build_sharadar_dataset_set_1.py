import sys
from packages.dataset_builder.dataset import Dataset, merge_datasets_simple
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError

"""
### Step by step instructions for creating dataset of daily observations from SHARADAR tables:
1. Use SHARADAR_DAILY as a base and merge it with SHARADAR_SEP on date and ticker
    1. Keep all columns (except redundant once, including: ticker and date)
    2. How to deal with missing data?
2. Find SHARADAR_SF1_ART most resent observation based on "datekey" (which is data or form 10 filing to the SEC) for each ticker and date and merge the SF1 data into that row
    1. Remember to keep the "datekey", it will be important for later sampling
3. Add in data associated with each ticker in each row from SHARADAR_TICKERS_METADATA


Not perfect overlap in supported tickers and dates complicates things, need to find the best supported dataset and use it as a base to marge towards...

"""

if __name__ == "__main__":

    try:
        logger = Logger('./logs')
    except Exception as e:
        print_exception_info(e)
        sys.exit()

    try:
        daily = Dataset("./datasets/daily/set_2.csv", None, "./datasets/sharadar/SHARADAR_INDICATORS.csv")
        prices = Dataset("./datasets/prices/set_2.csv", None, "./datasets/sharadar/SHARADAR_INDICATORS.csv")
        sf1_art = Dataset("./datasets/sf1_art/set_1.csv", None, "./datasets/sharadar/SHARADAR_INDICATORS.csv")
        tickers_metadata = Dataset("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", None, "./datasets/sharadar/SHARADAR_INDICATORS.csv")
    except Exception as e:
        print_exception_info(e)
        sys.exit()

    


    daily.sort(by=['ticker', 'date'])
    prices.sort(by=["ticker", "date"])
    sf1_art.sort(by=["ticker", "datekey"])
    tickers_metadata.sort(by="ticker")

    """
    daily_prices_df = merge_datasets_simple(daily.data, prices.data, on=['ticker', 'date'], suffixes=('_daily', '_prices'))
    daily_prices_dataset = Dataset.from_df(daily_prices_df)

    daily_prices_df.describe()
    daily_prices_df.head()

    daily_prices_dataset.to_csv("./datasets/sharadar_compiled/set_1.csv")
    """

    prices_daily_df = merge_datasets_simple(prices.data, daily.data, on=['ticker', 'date'], suffixes=('_daily', '_prices'))
    prices_daily_dataset = Dataset.from_df(prices_daily_df)

    prices_daily_df.describe()
    prices_daily_df.head()

    prices_daily_dataset.to_csv("./datasets/sharadar_compiled/prices_daily_set_2.csv")

    logger.close()


    # I have more prices than daily metrics (in the first 3.3M rows there are 734K only available in prices and 1300 only available in daily)
    # Pretty much the same for set_2
    
    # Can i create daily metrics for the missing dates?

    # 1202 tickers daily are all also in prices 
    # prices have 1415 tickers (212 more than daily), I think this is the case for all 10 sets approximately

    # There are also some dates present in prices that are not present in daily (and the other way around possibly)

    # All these issues must be resolved

    # Can I make daily metrics where they are missing

    # I need to cross reference ticker availability with the SF1 database also!!!