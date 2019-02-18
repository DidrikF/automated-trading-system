import sys
from packages.dataset_builder.dataset import Dataset
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError
import pandas as pd

if __name__ == "__main__":

    try:
        # tickers = Dataset("./datasets/sharadar/SHARADAR_TICKERS.csv")
        
        # I need to split tickers into industries as well!
        
        tickers = pd.read_csv("./datasets/tickers/all_usable.csv", low_memory=False)
        metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", low_memory=False)


    except Exception as e:
        print_exception_info(e)
        sys.exit()
    
    metadata = metadata[["ticker", "industry"]]

    # Add industry to each ticker
    ticker_industry_df = tickers.merge(metadata, on=["ticker"], how="left")
    ticker_industry_df = ticker_industry_df.drop_duplicates(subset="ticker")

    ticker_industry = Dataset.from_df(ticker_industry_df)
    
    # Sort by industry, then ticker
    ticker_industry.sort(by=["industry", "ticker"])



    datasets = ticker_industry.split_on_industry()

    for i, dataset in enumerate(datasets):
        industry = dataset.data.iloc[0]["industry"]
        path = "./datasets/industry_tickers/" + industry +  ".csv"
        dataset.to_csv(path)
