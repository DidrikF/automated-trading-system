from sklearn.linear_model import LinearRegression
from dataset_builder.dataset import Dataset
from dataset_builder.helpers import merge_datasets, df_filter_rows, add_derived_column
from datetime import datetime, timedelta
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import pandas as pd
import sys

if __name__ == "__main__":
    """
    fundamentals = Dataset('C:/datasets/nyse/fundamentals.csv')
    fundamentals.parse_dates(["Period Ending"])
    
    prices = Dataset('C:/datasets/nyse/prices-split-adjusted.csv')
    prices.drop_cols_except([0,1,3])
    prices.parse_dates(["date"])

    # fundamentals.print_cols()
    # prices.print_cols()
    

   
    def possible_dates(date):
        dates = [(date + timedelta(days=x)).isoformat() for x in range(-7, 7)]
        dates.insert(0, date.isoformat())
        return dates


    merge_on = [
        {
            "mapping": ("Ticker Symbol", "symbol"),
            "possible_values": lambda value: [value.lower(), value.upper()], # need to return a list
        },
        {
            "mapping": ("Period Ending", "date"),
            "possible_values": possible_dates, # need to return a list
        }
    ]

    columns_to_add = ["close"]

    def select_best(rows: pd.DataFrame, col_val_to_match_in_df2: dict):
        dates = rows.loc[:,"date"].tolist()
        
        if type(col_val_to_match_in_df2["date"]) == list:
            desired_date = col_val_to_match_in_df2["date"][0] # First element is the desired value
        else:
            desired_date = col_val_to_match_in_df2["date"]

        desired_date = parse(desired_date)
        
        result = min(dates, key=lambda date_from_list: abs(desired_date - date_from_list))

        return rows.index[rows["date"] == result].tolist()[0]


    dataset = merge_datasets(fundamentals.data, prices.data, merge_on, select_best, columns_to_add)

    dataset.to_csv("./datasets/experimentation/fundamentals_complete.csv")

    """

    # Add return column to prices dataset and merge that into fundimentals

    prices = Dataset('C:/datasets/nyse/prices-split-adjusted.csv')
    prices.drop_cols_except([0,1,3])
    prices.parse_dates(["date"])

    dataset = Dataset("./test_df.csv")
    
    def monthly_stock_return_calculation(row: pd.Series, df: pd.DataFrame) -> float:
        row_index = row[0]
        ticker = row[1]["symbol"]
        current_date =  row[1]["date"]
        date_next_month = current_date + relativedelta(months=+1)
        # print(current_date, date_next_month)

        current_price = row[1]["close"]
        
        match_df = df.loc[(df["date"] == date_next_month) & (df["symbol"] == ticker)]


        # How the flying fuck to get the value, had the same problem yesterday.
        # Also should add in some ambiguity regarding the exact date one month in the future
        # If no price is aviable one month ahead, set the value to be None, and filter/etc. it out later
        print(match_df.iloc[0]["close"])
        price_next_month = match_df.iloc[0]["close"]

        
        # print(current_price)

        #print(price_next_month)

        # print(row_index)



    # Should probably be a class method
    prices.data = add_derived_column(prices.data, "Return Comming Month", monthly_stock_return_calculation)


    sys.exit()





    dataset.drop_cols_by_index([0,1,81])
    dataset.print_cols()
    
    print(dataset.data.head())







    # DATASET FILTERING TO REMOVE BAD ROWS
    def filter_func(row: pd.Series, df: pd.DataFrame) -> bool:
        for val in row:
            if val in ["None", None, ""]:
                return False
            else:
                return True
    

    # Feature Scaling



    
    # Save dataset

    print("Done!")

