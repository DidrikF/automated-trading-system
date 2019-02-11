import sys
from packages.dataset_builder.dataset import Dataset
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError
import pandas as pd

if __name__ == "__main__":

    try:

        purged_sep = pd.read_csv("./datasets/sharadar/PURGED_SEP.csv", low_memory=False)
        metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", low_memory=False)

    except Exception as e:
        print_exception_info(e)
        sys.exit()
    


    metadata_tickers = set(metadata.ticker.unique())

    tickers_sep = set(purged_sep["ticker"].unique())
    print("Tickers in purged sep: ", len(tickers_sep))

    usable_tickers = metadata_tickers.intersection(tickers_sep)
    print("Usable tickers len: ", len(usable_tickers))
    


    tickers_df = pd.DataFrame(list(usable_tickers), columns=["ticker"])
    
    ticker_dataset = Dataset.from_df(tickers_df)

    ticker_dataset.sort(by="ticker")

    ticker_dataset.to_csv("./datasets/tickers/all_usable.csv")

    datasets = ticker_dataset.split_on_ticker(into=10)

    for i, dataset in enumerate(datasets):
        path = "./datasets/tickers/set_" + str(i + 1) +  ".csv"
        dataset.to_csv(path)

