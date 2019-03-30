import pandas as pd    


if __name__ == "__main__": 
    metadata = pd.read_csv("./datasets/sharadar/METADATA_PURGED.csv")
    grouped_molecules = metadata.groupby('industry')

    dfs = {}

    for industry, molecule in grouped_molecules:
        dfs[industry] = molecule

    
    print(metadata.values[0])