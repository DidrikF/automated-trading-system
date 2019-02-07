import sys
from packages.dataset_builder.dataset import Dataset
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError

if __name__ == "__main__":

    try:
        tickers = Dataset("./datasets/sharadar/SHARADAR_TICKERS.csv")
    except Exception as e:
        print_exception_info(e)
        sys.exit()
    
    """
    try:
        prices_dataset = Dataset("./datasets/sharadar/SHARADAR_SEP.csv", None, "./datasets/sharadar/indicators_demo.csv")
    except Exception as e:
        print_exception_info(e)
        sys.exit()

    prices_dataset.info()
    """

    datasets = tickers.split_simple(10)

    for i, dataset in enumerate(datasets):
        path = "./datasets/tickers/set_" + str(i + 1) +  ".csv"
        dataset.to_csv(path)


    