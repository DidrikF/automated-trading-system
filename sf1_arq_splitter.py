import sys
from packages.dataset_builder.dataset import Dataset
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError
import pandas as pd

if __name__ == "__main__":

    ticker_datasets = list()

    for i in range(10):
        set_nr = i + 1
        path = "./datasets/tickers/set_" + str(set_nr) + ".csv"
        try:
            ticker_dataset = Dataset(path)
        except Exception as e:
            print_exception_info(e)
            sys.exit()
        
        ticker_datasets.append(ticker_dataset)



    try:
        df = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ARQ.csv")
        print(df.head())
        sf1_arq_dataset = Dataset.from_df(df)
        # sf1_arq_dataset = Dataset("./datasets/sharadar/SHARADAR_SEP.csv", None, "./datasets/sharadar/indicators_demo.csv")
    except Exception as e:
        print_exception_info(e)
        sys.exit()

    for index, ticker_dataset in enumerate(ticker_datasets):
        tickers = ticker_dataset.data["ticker"].values.tolist()
        # print(ticker_dataset.data["ticker"].values.tolist())
        prices_df = sf1_arq_dataset.data.loc[sf1_arq_dataset.data["ticker"].isin(tickers)].copy()
        sf1_arq_dataset_chunk = Dataset.from_df(prices_df)
        sf1_arq_dataset_chunk.sort(["ticker", "datekey"])
        # print(prices_df.head())

        set_nr = index + 1
        path = "./datasets/SF1_ARQ/set_" + str(set_nr) + ".csv"
        sf1_arq_dataset_chunk.to_csv(path)
    
