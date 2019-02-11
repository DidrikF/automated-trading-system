import sys
from packages.dataset_builder.dataset import Dataset
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError
import pandas as pd
from os import listdir
from os.path import isfile, join


if __name__ == "__main__":

    try:
    
        daily_df = pd.read_csv("./datasets/sharadar/PURGED_DAILY.csv", low_memory=False)

        ticker_path = "./datasets/industry_tickers/"

        onlyfiles = [f for f in listdir(ticker_path) if isfile(join(ticker_path, f))]

        for filename in onlyfiles:
            path = ticker_path + filename
            tickers = pd.read_csv(path, low_memory=False)["ticker"]
            daily_chunk = daily_df.loc[daily_df["ticker"].isin(tickers)]
            daily_dataset = Dataset.from_df(daily_chunk)
            daily_dataset.sort(by=["ticker", "date"])

            save_path = "./datasets/industry_daily/" + filename

            daily_dataset.to_csv(save_path)


    except Exception as e:
        print_exception_info(e)
        sys.exit()
    