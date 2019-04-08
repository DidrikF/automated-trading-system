"""
Meta labeling requires the primary model to have produced predictions on side first. These predictions
are use when running triple barrier search with asymmetric barrieres for labeling the dataset for
training of the secondary model (informing the sizing decision).
"""

import pandas as pd    


if __name__ == "__main__": 
    metadata = pd.read_csv("./datasets/sharadar/METADATA_PURGED.csv")
    grouped_molecules = metadata.groupby('industry')

    dfs = {}

    for industry, molecule in grouped_molecules:
        dfs[industry] = molecule

    
    print(metadata.values[0])