import sys
from automated_trading_system.dataset_builder.dataset import Dataset
from automated_trading_system.logger.logger import Logger
from automated_trading_system.helpers.helpers import print_exception_info

from automated_trading_system.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from automated_trading_system.helpers.custom_exceptions import FeatureError
import pandas as pd
from os import listdir
from os.path import isfile, join


if __name__ == "__main__":

    try:
    
        sf1_art_df = pd.read_csv("./datasets/sharadar/PURGED_SF1_ART.csv", low_memory=False)

        ticker_path = "./datasets/industry_tickers/"

        filenames = [f for f in listdir(ticker_path) if isfile(join(ticker_path, f))]
        filenames.remove("None.csv")
        filenames.remove("Infrastructure Operations.csv")

        for filename in filenames:
            path = ticker_path + filename
            tickers = pd.read_csv(path, low_memory=False)["ticker"]
            sf1_art_chunk = sf1_art_df.loc[sf1_art_df["ticker"].isin(tickers)]
            sf1_art_dataset = Dataset.from_df(sf1_art_chunk)
            sf1_art_dataset.sort(by=["ticker", "datekey"])

            save_path = "./datasets/industry_sf1_art/" + filename

            sf1_art_dataset.to_csv(save_path)
            print("Done with file: {}".format(filename))


    except Exception as e:
        print_exception_info(e)
        sys.exit()