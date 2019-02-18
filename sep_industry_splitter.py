import sys
from os import listdir
from os.path import isfile, join

import pandas as pd

from packages.dataset_builder.dataset import Dataset
from packages.dataset_builder.feature_builders import (book_to_market,
                                                       book_value,
                                                       cash_holdings)
from packages.helpers.custom_exceptions import FeatureError
from packages.helpers.helpers import print_exception_info
from packages.logger.logger import Logger

if __name__ == "__main__":

    try:
    
        sep = pd.read_csv("./datasets/sharadar/PURGED_SEP.csv", low_memory=False)

        ticker_path = "./datasets/industry_tickers/"

        filenames = [f for f in listdir(ticker_path) if isfile(join(ticker_path, f))]
        filenames.remove("None.csv")
        filenames.remove("Infrastructure Operations.csv")


        print("Filenames: ", filenames)

        for filename in filenames:
            path = ticker_path + filename
            tickers = pd.read_csv(path, low_memory=False)["ticker"]
            sep_chunk = sep.loc[sep["ticker"].isin(tickers)]
            sep_dataset = Dataset.from_df(sep_chunk)
            sep_dataset.sort(by=["ticker", "date"])

            save_path = "./datasets/industry_sep/" + filename

            sep_dataset.to_csv(save_path)
            print("Done with file: {}: ".format(filename))


    except Exception as e:
        print_exception_info(e)
        sys.exit()

