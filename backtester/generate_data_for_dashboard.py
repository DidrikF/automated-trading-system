import pandas as pd

if __name__ == "__main__":
    date = pd.to_datetime("2010-01-01")

    dataset = pd.read_csv("../dataset_development/datasets/completed/ml_dataset.csv", low_memory=True, parse_dates=["date"], index_col="date")
    dataset = dataset.loc[dataset.index >= date]
    dataset = dataset.sort_values(by=["date", "ticker"])

    sep = pd.read_csv("../dataset_development/datasets/sharadar/SEP_PURGED_ADJUSTED.csv", parse_dates=["date"], index_col="date")
    sep = sep.loc[sep.index >= date]
    sep = sep.sort_values(by=["date", "ticker"])

    dataset.to_csv("./datasets/backtested_dataset.csv")
    sep.to_csv("./datasets/backtested_sep.csv")