from sklearn.linear_model import LinearRegression
from dataset_builder.dataset import Dataset
from datetime import datetime, timedelta
import pandas as pd
import sys

if __name__ == "__main__":
    fundamentals = Dataset('C:/datasets/nyse/fundamentals.csv')
    prices = Dataset('C:/datasets/nyse/prices-split-adjusted.csv')

   #  fundamentals.print_cols()

    fundamentals.drop_cols_except([1,2, 12, 68, 72, 77]) # Misc Stocks: 35
    prices.drop_cols_except([0,1,3])
    
    # fundamentals.info()
    # prices.info()

    # fundamentals.print_cols()
    # print(fundamentals.data.head())

    symbols_in_prices = set(prices.data["symbol"].tolist())
    symbols_in_fundamentals = set(fundamentals.data["Ticker Symbol"].tolist())
    common_symbols = symbols_in_prices.intersection(symbols_in_fundamentals)

    fundamentals.data["Period Ending"] = pd.to_datetime(fundamentals.data["Period Ending"])
    prices.data["date"] = pd.to_datetime(prices.data["date"])

    #print(prices.data["date"][0].month)
    #print(prices.data["date"][0].year)
    #print(prices.data["date"][0].day)

    fundamentals_unique_years = set()
    prices_unique_years = set()

    for date in fundamentals.data["Period Ending"].tolist():
        fundamentals_unique_years.add(date.year)

    for date in prices.data["date"].tolist():
        prices_unique_years.add(date.year)

    common_years = fundamentals_unique_years.intersection(prices_unique_years)
    
    print("Fundamentals years: ", fundamentals_unique_years)
    print("Prices years: ", prices_unique_years)
    print("Common years: ", common_years)

    def date_in_nearest_future(dates: list, date):
        return min(dates, key=lambda date_from_list: abs(date - date_from_list))

    fundamentals.data["Price"] = None
    print(fundamentals.data.head())
    
    for index, row in enumerate(fundamentals.data.iterrows()):
        # print("Row: ")
        # print(row)
        fundimental_ticker = row[1][0]
        fundimental_date = row[1][1]

        # Get all rows in prices with ticker "fundimental_ticker".
        prices_for_ticker = prices.data.loc[prices.data['symbol'] == fundimental_ticker]

        # Get closest date in the future to fundimental_date from prices_for_ticker 
        dates_of_prices_for_ticker = prices_for_ticker["date"].tolist()
        

        # print(len(dates_of_prices_for_ticker))
        if len(dates_of_prices_for_ticker) > 0:
            closest_date = date_in_nearest_future(dates_of_prices_for_ticker, fundimental_date)
        else:
            continue
        # print("Closest date: ", closest_date)

        price_of_closest_date = float(prices_for_ticker.loc[prices_for_ticker["date"] == closest_date]["close"])
        # print("Price of closest date: ", price_of_closest_date)

        # df.loc[row(s), col(s)]
        fundamentals.data.loc[index, "Price"] = price_of_closest_date

        # print(fundamentals.data.head())

        # price_rows = prices.data.loc[prices.data['symbol'].isin(list(common_symbols))]


    print(fundamentals.data.head())


    # Adding price associated with each row in fundamentals, this is uniqely desided by the Ticker+Date
    # Where the date is as close as possible to the one in fundamentals


    # Adding book value per share as a column
    # book_value = total_assets - total_liabilities

    fundamentals.data["Book Value per Share"] = None

    for index, row in enumerate(fundamentals.data.iterrows()):
        # print(row)
        total_assets = row[1][3]
        total_liabilities = row[1][4]
        common_stock = row[1][2]
        # print(total_assets, total_liabilities, common_stock)
        
        if total_assets == 0 or total_liabilities == 0 or common_stock == 0:
            fundamentals.data.loc[index, "Book Value per Share"] = None
            continue

        fundamentals.data.loc[index, "Book Value per Share"] = (total_assets - total_liabilities) / common_stock



    fundamentals.data.to_csv("./datasets/fundamentals_processed_split_adjusted.csv")

    # DATASET FILTERING TO REMOVE BAD ROWS
    
    # Save dataset

    print("Done!")

