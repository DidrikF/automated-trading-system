import os
import pickle
import pandas as pd
from abc import ABC, abstractmethod
from dateutil.relativedelta import *
from datetime import datetime, timedelta

from event import Event
from utils.errors import MarketDataNotAvailableError
from utils.logger import Logger

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


class DailyBarsDataHander(DataHandler):
    def __init__(self, source_path, store_path: str, file_name_time_data, file_name_ticker_data, start: pd.datetime, end: pd.datetime):
        self.source_path = source_path
        self.store_path = store_path
        self.file_name_time_data = file_name_time_data
        self.file_name_ticker_data = file_name_ticker_data
        
        self.start = start
        self.end = end

        # VERY IMPORTANT THAT THIS IS MANIPULATED CORRECTLY
        self.cur_date = start # Maintens the current date of the system (is this a good solution???)

        self.end_of_data_reached = False

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        full_path_time_data = self.store_path + "/" + self.file_name_time_data + ".pickle"
        if os.path.isfile(full_path_time_data) == True:
            data_file = open(full_path_time_data, 'rb')
            self.time_data = pickle.load(data_file)
            data_file.close()
        else:
            self.ingest(parse_type="time")
        
        # data = self.time_data.loc[(self.time_data.index >= self.start) & (self.time_data.index <= self.end)]
        # self.date_index_to_iterate = data.index.get_level_values(0)
        # date_index_to_iterate = date_index[self.start:(self.end + relativedelta(days=1))]
        # print(self.time_data)

        # print(type(self.time_data))

        dates = self.time_data.index.get_level_values(0).drop_duplicates(keep='first').to_frame()
        self.date_index_to_iterate  = dates.loc[(dates.index >= self.start) & (dates.index <= self.end)].index

        self.tick = self._next_tick_generator()

        
        full_path_ticker_data = self.store_path + "/" + self.file_name_ticker_data + ".pickle"
        if os.path.isfile(full_path_ticker_data) == True:
            data_file = open(full_path_ticker_data, 'rb')
            self.ticker_data = pickle.load(data_file)
            data_file.close()
        else:
            self.ingest(parse_type="ticker")


    def ingest(self, parse_type="time"):
        """
        Parse data into desired format for backtesting, save it and make set it on the time_data
        or ticker_data attributes.
        """
        print("ingest called")
        source_df = pd.read_csv(self.source_path, parse_dates=["date"], low_memory=False)

        if parse_type == "time":
            print("Parsing data to time format from source_path{}".format(self.source_path))
            self.time_data = {}
            split_df = source_df.groupby("date")
            for date, df in split_df:
                df = df.sort_values(by=["ticker"])
                self.time_data[date] = df.set_index("ticker")
                if any(self.time_data[date].index.duplicated()):
                    print("duplicate data for one or more tickers on date: ", date)
                    # drop duplicates probably

            self.time_data = pd.concat(self.time_data)
            self.time_data = self.time_data.sort_index()

            full_store_path = self.store_path + '/' + self.file_name_time_data + ".pickle"
            outfile = open(full_store_path, 'wb')
            pickle.dump(self.time_data, outfile)
            outfile.close()


        elif parse_type == "ticker":
            print("Parsing data to ticker format from source_path{}".format(self.source_path))
            self.ticker_data = {}
            split_df = source_df.groupby("ticker")
            for ticker, df in split_df:
                df = df.sort_values(by="date")
                self.ticker_data[ticker] = df.set_index("date")
                if any(self.ticker_data[ticker].index.duplicated()):
                    print("duplicate data for ticker ", ticker, " on one or more dates")
                    # drop duplicates probably

            self.ticker_data = pd.concat(self.ticker_data)
            self.ticker_data = self.ticker_data.sort_index()

            full_store_path = self.store_path + '/' + self.file_name_ticker_data + ".pickle"
            outfile = open(full_store_path, 'wb')
            pickle.dump(self.ticker_data, outfile)
            outfile.close()

    
    
    def _next_tick_generator(self):

        for date in self.date_index_to_iterate:
            self.cur_date = date
            tick_data = self.time_data.loc[date]
            
            yield Event(event_type="DAILY_MARKET_DATA", data=tick_data, date=date)

    
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



    def can_trade_ticker(self, ticker, date):
        """ Checks if the there is price data beyond the $date for the $ticker """
        # Use max() on dateindex
        pass

    def history(self, ticker): # REDUNDANT
        """
        Returns a window of data for the given assets and fields.
        This data is adjusted for splits, dividends, and mergers as of the current algorithm time.
        """
        pass
    
    def get_index(self):
        """
        return DateTimeIndex that contains all dates with data from self.time_data
        """



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


class DataLoader():
    """Class for loading and parsing data into a format suited for running backtest on."""
    pass



class Bundle(): # Dont think this is appropriate for me
    def __init__(self):
        pass
    def ingest(self):
        pass


