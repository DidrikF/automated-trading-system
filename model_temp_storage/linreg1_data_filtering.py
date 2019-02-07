from sklearn.linear_model import LinearRegression
from dataset_builder.dataset import Dataset
from datetime import datetime, timedelta
import pandas as pd
import sys
from math import isnan

if __name__ == "__main__":
    fundamentals = Dataset('./datasets/fundamentals_processed_split_adjusted.csv')

    fundamentals.print_cols()

    fundamentals.drop_cols_except([6,7,8]) # Misc Stocks: 35
    
    fundamentals.info()

    drop_list = []

    for index, row in enumerate(fundamentals.data.iterrows()):
        eps = float(row[1][0])
        price = float(row[1][1])
        bvps = float(row[1][2])
        # print(eps, price, bvps)

        remove = False

        if not isinstance(eps, (int, float)) or not isinstance(price, (int, float)) or not isinstance(bvps, (int, float)):
            remove = True
        
        if eps == 0 or price == 0 or bvps == 0:
            remove = True 
        
        if isnan(eps) or isnan(price) or isnan(bvps):
            remove = True 
        
        if remove == True:
            drop_list.append(index)

    
    # print("Drop list: ", drop_list)
    
    print(len(fundamentals.data))

    fundamentals.data.drop(fundamentals.data.index[drop_list], inplace=True)

    print(len(fundamentals.data))

    fundamentals.data.to_csv("./datasets/fundamentals_final.csv")

    # DATASET FILTERING TO REMOVE BAD ROWS
    
    # Save dataset

    print("Done!")

