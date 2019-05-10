import pandas as pd

if __name__ == "__main__":
    dataset = pd.read_csv("../datasets/completed/ml_dataset.csv", low_memory=True)

    test_dataset = dataset.loc[dataset.ticker.isin(["AAPL", "NTK", "FCX"])]

    test_dataset.to_csv("../datasets/testing/ml_dataset.csv", index=False)

    
