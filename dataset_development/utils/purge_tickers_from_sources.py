import pandas as pd


if __name__ == "__main__":

    sf1_art = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ART.csv", low_memory=False)
    
    sep = pd.read_csv("./datasets/sharadar/SHARADAR_SEP.csv", low_memory=False)
    metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", low_memory=False)

    print("sf1_art length: ", len(sf1_art))
    print("sep length: ", len(sep))
    print("metadata length: ", len(metadata))


    wanted_tickers = list(sf1_art.ticker.unique())
    print("Wanded tickers length: ", len(wanted_tickers))

    purged_sep = sep.loc[sep.ticker.isin(wanted_tickers)]
    purged_metadata = metadata[metadata.ticker.isin(wanted_tickers) & (metadata.table=="SF1")]


    purged_sep.to_csv("./datasets/sharadar/SEP_PURGED.csv", index=False)
    purged_metadata.to_csv("./datasets/sharadar/METADATA_PURGED.csv", index=False)


    print("sf1_art length: ", len(sf1_art))
    print("Purged sep length: ", len(purged_sep))
    print("Purged metadata length: ", len(purged_metadata))