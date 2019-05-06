import sys, os
import pickle
import pandas as pd
from abc import ABC, abstractmethod
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import math

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, ".."))


from event import Event, MarketDataEvent
from utils.errors import MarketDataNotAvailableError
from utils.logger import Logger
from dataset_development.processing.engine import pandas_mp_engine

"""
Due to memory not being an issue, I think it is most effective have data in memory
split both by ticker and date.
time_data = {
"2012-01-04": df(columns: ["ticker", "open", "high", ...], index_col="ticker"),
"2012-01-05": df(columns: ["ticker", "open", "high", ...], index_col="date"),
...
}
ticker_data = {
"AAPL": df(columns: ["date", "open", "high", ...]),
"MSFT: df(columns: ["date", "open", "high", ...]),
...
}

"""

class DataHandler(ABC):
    def ingest(self, parse_type):
        pass

    def get_data(self):
        pass

    def current(self):
        pass

"""
I want all data to have a complete datetime index. But I need to be able to distinguish between business day and non-business days.
Options:
- Don't forward fill data..
- Have a column indicating if it is a business day (or better, if the price was reported in the source in the given date) 
    (then I can ffill and still know if i can trade...)
- I also need to know if it is a business day, which means that all sort of code is supposed to run.

What if data is messing for a company on a day, then it may to be safe to assume it can be traded.

Also, now I end up with a lot of different dataframes, would be nice to just work with one.

OHLC are not dividend adjusted
sep:
open (int) - high (int) - low (int) - close (int) - close_adjusted (int) - business_day (bool) - can_trade (bool)


Ticker data df conduces me. It is best if I can trust that the same data is in both time and ticker data dataframes.

For this reason I should to all wanted transformations before "dividing" the data...

I can have -> Then I only have 2 dataframes (!!!)
ticker_data["snp_500"] # split and dividend adjsuted, so I can compare directly with the portfolio
ticker_data["rf_3m"]
ticker_data["rf_1m"]
ticker_data["rf_1d"]

"""

class DailyBarsDataHander(DataHandler):
    def __init__(
        self, 
        path_prices: str, 
        path_snp500: str, 
        path_interest: str,
        store_path: str,
        start: pd.datetime, 
        end: pd.datetime
    ):
        self.path_prices = path_prices
        self.path_snp500 = path_snp500
        self.path_interest = path_interest

        self.file_name_bundle = "data_bundle" # Store all data needed for backtesting here

        self.store_path = store_path

        self.start = start
        self.end = end

        # VERY IMPORTANT THAT THIS IS MANIPULATED CORRECTLY
        self.cur_date = start # Maintens the current date of the system (is this a good solution???)

        self.end_of_data_reached = False

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        # Make and store time and ticker data together
        full_path_bundle_data = self.store_path + "/" + self.file_name_bundle + ".pickle"
        if os.path.isfile(full_path_bundle_data) == True:
            data_file = open(full_path_bundle_data, 'rb')
            bundle = pickle.load(data_file)
            data_file.close()
        else:
            # Make bundle
            bundle = self.ingest_data()

            # Store bundle
            full_store_path = self.store_path + '/' + self.file_name_bundle + ".pickle"
            outfile = open(full_store_path, 'wb')
            pickle.dump(bundle, outfile)
            outfile.close()


        self.time_data = bundle["time_data"]
        self.ticker_data = bundle["ticker_data"]
        self.rf_rate = bundle["rf_rate"]


        # print(self.time_data)

        # Set up data iterator
        dates = self.time_data.index.get_level_values(0).drop_duplicates(keep='first').to_frame()
        self.date_index_to_iterate  = dates.loc[(dates.index >= self.start) & (dates.index <= self.end)].index

        self.tick = self._next_tick_generator()



    def ingest_data(self):
        data: pd.DataFrame = pd.read_csv(self.path_prices, parse_dates=["date"], index_col="date", low_memory=False)
        
        data = self.ingest_snp500(data)

        # Reindex all stocks, ffill and add business_day and can_trade columns
        data = pandas_mp_engine(
            callback=reindex_and_ffill,
            atoms=data,
            data=None,
            molecule_key="sep",
            split_strategy="ticker",
            num_processes=6,
            molecules_per_process=1
        )
        
        """
        grouped_data = data.groupby("ticker")
        dfs = pd.DataFrame(columns=data.columns)
        for ticker, df in grouped_data:
            df = reindex_and_ffill(df)
            dfs = dfs.append(df, sort=True)

        print(dfs)
        """

        
        data = data.reset_index()

        data = data.rename(index=str, columns={"index": "date"})

        time_data = {}
        split_df = data.groupby("date")
        for date, df in split_df:
            df = df.sort_values(by=["ticker"])
            time_data[date] = df.set_index("ticker").sort_index()
            if any(time_data[date].index.duplicated()):
                print("duplicate data for one or more tickers on date: ", date)
                # drop duplicates probably

        time_data = pd.concat(time_data)
        time_data = time_data.sort_index()


        ticker_data = {}
        split_df = data.groupby("ticker")
        for ticker, df in split_df:
            df = df.sort_values(by="date")
            ticker_data[ticker] = df.set_index("date").sort_index()
            if any(ticker_data[ticker].index.duplicated()):
                print("duplicate data for ticker ", ticker, " on one or more dates")
                # drop duplicates probably
        ticker_data = pd.concat(ticker_data)
        ticker_data = ticker_data.sort_index()



        rf_rate = self.ingest_interest()

        return {
            "time_data": time_data,
            "ticker_data": ticker_data,
            "rf_rate": rf_rate
        }
            


    def ingest_snp500(self, data):
        """
        Ingest S&P500 data and merge it into the data bundle as another instrument.
        """
        dateparse = lambda x: pd.datetime.strptime(x, "%d.%m.%Y")
        snp500: pd.DataFrame = pd.read_csv(self.path_snp500, parse_dates=["date"], date_parser=dateparse, index_col="date")

        # new_index: pd.DatetimeIndex = pd.date_range(snp500.index.min(), snp500.index.max())
        # snp500 = snp500.reindex(new_index)
        # snp500 = snp500.fillna(method="ffill", axis=0)
        
        snp500["ticker"] = "snp500"
        
        data = data.append(snp500, sort=True)

        return data

        
    def ingest_interest(self):
        """
        Ingest Interest rate data and calculate daily and weekly rates and return as a dataframe.
        """

        rf_rate: pd.DataFrame = pd.read_csv(self.path_interest, parse_dates=["date"], index_col="date")

        # Make daily, weekly, and 3m (not modified)
        rf_rate["daily"] = rf_rate["rate"] / 91 # Assumes 3-month T-bill reaches maturity after 13 weeks (13*7 = 91 days)
        rf_rate["weekly"] = rf_rate["rate"] / 13
        rf_rate["3_month"] = rf_rate["rate"]

        rf_rate = rf_rate.drop(columns=["rate"])
    
        return rf_rate



    # ____________________________________________________________________________________________________________
    # NEED TO UPDATE THE BELOW: to take into account: business days, can trade
    # This should make things easier as prices is allways available even though it might not be possible to trade


    def _next_tick_generator(self):
        """
        Maybe this should indicate whether it is a business day or not.
        """

        for date in self.date_index_to_iterate:
            self.cur_date = date
            tick_data = self.time_data.loc[date]
            interest_data = self.rf_rate[date]
            is_business_day = self.is_business_day()

            yield MarketDataEvent(event_type="DAILY_MARKET_DATA", data=tick_data, date=date, interest=interest_data, business_day=is_business_day)
            
            # yield MarketDataEvent("DAILY_MARKET_DATA", tick_data, date, interest_data, is_business_day)

    def can_trade(self, ticker, date):
        """
        Use can_trade column the date being outside min/max date index for the ticker.
        Important to check this for all tickers the strategy/portfolio/broker that should be traded.
        """
        try:
            res = self.ticker_data.loc[ticker, date]["can_trade"]
        except:
            res = False

        return res
        

    def is_business_day(self, date=None):
        """
        It is a business day if data is available for one or more tickers (firms).
        """
        if date is None:
            date = self.cur_date

        data = self.time_data.loc[date]

        for ticker, row in data.iterrows():
            if (ticker != "snp500") and (row["can_trade"] == True):
                return True

        return False



    # The final method, update_bars, is the second abstract method from DataHandler. 
    # It simply generates a MarketEvent that gets added to the queue as it appends the latest bars 
    # to the latest_symbol_data:
    def next_tick_old(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.

        OBS: may be more stable to maintain the index we are looking up and then get the date from that...
        """
        while True:
            if self.cur_date > self.end:
                self.end_of_data_reached = True
                raise IndexError("The current date is beyond the end date of the backtest. DataHandler cannot return next_tick.")

            try:
                daily_data = self.get_data_for_date(self.cur_date)
                break
            except KeyError:   
                self.cur_date = self.cur_date + relativedelta(days=1) # Skip non-business days, holidays, etc.


        self.cur_date = self.cur_date + relativedelta(days=1) # increment cur_date for before next call to next_tick
        
        return Event(event_type="DAILY_MARKET_DATA", data=daily_data, date=self.cur_date)


    def current_tick(self): # Should not be possible to fail...
        """
        Returns the market data for the current tick.
        This does not say whether or not you can trade....
        """
        return self.get_data_for_date(self.cur_date)


    def get_data_for_date(self, date: pd.datetime):
        try:
            daily_data = self.time_data.loc[date] # I thnk .loc
        except:
            raise KeyError("No daily data for date {}".format(date))
        else:
            return daily_data

    def current_for_ticker(self, ticker):
        try:
            data = self.ticker_data.loc[ticker, self.cur_date] # may need to update, df style
        except Exception as e:
            raise MarketDataNotAvailableError("No market data for ticker {} on date {}".format(ticker, self.cur_date))
        else:
            return data

    def last_for_ticker(self, ticker):
        """
        Returns the most recent SEP row for the ticker.
        """
        # Need to implement
        return math.nan

    def continue_backtest(self):
        """Checks if there are more ticks to be processed"""
        if not self.end_of_data_reached:
            return True
        else:
            return False


    
    def get_ticker_data(self, ticker):
        """
        Return data for ticker
        """
        
        try:
            data = self.ticker_data.loc[ticker][self.start:self.end]
        except Exception as e:
            return pd.DataFrame(columns=self.ticker_data.iloc[[0]].columns)
            # print(e)
            # raise MarketDataNotAvailableError("No market data for ticker {} from {} to {}".format(ticker, start, end))
        else:
            return data



    

def dividend_adjusting_prices_backwards(sep: pd.DataFrame) -> pd.DataFrame: 
    """
    Split strategy: ticker
    Adds dividend adjusted close prices to dataframe.
    """
    sep = sep.sort_values(by="date", ascending=False)
    adjustment_factor = 1

    # Looping backwards in time...
    for date, row in sep.iterrows():
        # At each date we want to adjust the price according to the accumulated 
        # adjustment factor from future dates, not the current date.
        sep.at[date, "adj_open"] = row["open"] / adjustment_factor
        sep.at[date, "adj_high"] = row["high"] / adjustment_factor
        sep.at[date, "adj_low"] = row["low"] / adjustment_factor
        sep.at[date, "adj_close"] = row["close"] / adjustment_factor
        
        # All the earlier dates than the current need to be adjusted according 
        # to the current accumulated adjustment factor, taking into account
        # any new dividend on the current date.
        adjustment_factor_update = (row["close"] + row["dividends"]) / row["close"]
        adjustment_factor = adjustment_factor * adjustment_factor_update

    return sep


def reindex_and_ffill(sep: pd.DataFrame) -> pd.DataFrame:
    """
    Use this with multiprocessing
    """

    # Create complete index
    new_index = pd.date_range(sep.index.min(), sep.index.max())
    sep = sep.reindex(new_index)
    sep["can_trade"] = ~sep["close"].isnull()

    sep = sep.fillna(method="ffill")


    return sep




# IMPLEMENTATION OF THIS CAN BE DEFERRED TO LATER, WHEN BUILDING THE ML STRATEGY
class MLFeaturesDataHandler(DataHandler):
    def __init__(self, source_path: str, store_path: str, file_name):
        self.source_path = source_path
        self.store_path = store_path
        self.file_name = file_name


        full_store_path = self.store_path + '/' + self.file_name + ".pickle"
        if os.path.isfile(full_store_path):
            data_file = open(full_store_path,'rb')
            self.data = pickle.load(data_file)
            data_file.close()
        else:
            self.ingest()

    def ingest(self):
        pass

    def get_data_for_day(self, date):
        pass

    def get_data(self, ticker, start, end):
        pass
    
    def current(self, current_date) -> Event:
        # event.date = event.date - realativedelta(days=1)
        # make sure to only give new data that was released the previous business day.
        
        return Event(event_type="FEATURE_DATA", data=pd.DataFrame(), date=current_date) # Will contain feature data for last business day before current_date










"""The below is all about moving data fast in and out of python in a desired format"""

# I'M NOT SURE HOW IMPORTANT EFFICIENT DATA READING AND WRITING IS FOR ME. CAN I NOT JUST USE PANDAS?

# Data API

# Data writers


# Data readers
# All sort of query methods implemented by zipline to get wanted data fast.
