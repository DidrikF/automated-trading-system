import sys, os
import pickle
import pandas as pd
from abc import ABC, abstractmethod
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import math

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, ".."))
# sys.path.insert(0, os.path.join(myPath))

from event import Event, MarketDataEvent
from utils.errors import MarketDataNotAvailableError
from dataset_development.processing.engine import pandas_mp_engine


class DataHandler(ABC):

    def ingest_data(self):
        pass

    def current(self):
        pass


class DailyBarsDataHander(DataHandler):
    def __init__(
        self, 
        path_prices: str, 
        path_snp500: str, 
        path_interest: str,
        path_corp_actions: str,
        store_path: str,
        start: pd.datetime, 
        end: pd.datetime,
        rebuild: bool=False,
    ):
        self.path_prices = path_prices
        self.path_snp500 = path_snp500
        self.path_interest = path_interest
        self.path_corp_actions = path_corp_actions

        self.file_name_bundle = "data_bundle" # Store all data needed for backtesting here

        self.store_path = store_path

        self.start = start
        self.end = end

        # VERY IMPORTANT THAT THIS IS MANIPULATED CORRECTLY
        self.cur_date = start # Maintens the current date of the system

        self.end_of_data_reached = False

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        # Make and store time and ticker data together
        full_path_bundle_data = self.store_path + "/" + self.file_name_bundle + ".pickle"
        if (os.path.isfile(full_path_bundle_data) == True) and (rebuild == False):
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
        self.corp_actions = bundle["corp_actions"]

        # Set up data iterator
        dates = self.time_data.index.get_level_values(0).drop_duplicates(keep='first').to_frame()
        self.date_index_to_iterate  = dates.loc[(dates.index >= self.start) & (dates.index <= self.end)].index

        self.tick = self._next_tick_generator()

    def ingest_data(self):
        print("Reading SEP")
        data: pd.DataFrame = pd.read_csv(self.path_prices, parse_dates=["date"], index_col="date", low_memory=False)
        data = data.loc[data.index >= self.start]
        
        print("Ingesting SNP500")
        data = self.ingest_snp500(data)

        # Reindex all stocks, ffill and add business_day and can_trade columns
        print("Reindex and Forward Fill")
        data = pandas_mp_engine(
            callback=reindex_and_ffill,
            atoms=data,
            data=None,
            molecule_key="sep",
            split_strategy="ticker_new",
            num_processes=6,
            molecules_per_process=1
        )
        

        data = data.reset_index()
        data = data.rename(index=str, columns={"index": "date"})

        print("Making time_data")
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

        print("Making ticker_data")
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


        print("Ingesting Interest Rates")
        rf_rate = self.ingest_interest()
        date_index = time_data.index.get_level_values(0).drop_duplicates(keep='first')
        print("Ingesting Coporate Actions")
        corp_actions = self.ingest_corporate_actions(data.set_index("date"), date_index)

        return {
            "time_data": time_data,
            "ticker_data": ticker_data,
            "rf_rate": rf_rate,
            "corp_actions": corp_actions
        }
            


    def ingest_snp500(self, data):
        """
        Ingest S&P500 data and merge it into the data bundle as another instrument.
        """
        dateparse = lambda x: pd.datetime.strptime(x, "%d.%m.%Y")
        snp500: pd.DataFrame = pd.read_csv(self.path_snp500, parse_dates=["date"], date_parser=dateparse, index_col="date")
        snp500 = snp500.loc[snp500.index >= self.start]
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
        rf_rate = rf_rate.loc[rf_rate.index >= self.start]

        # Make daily, weekly, and 3m (not modified) - compounding interest
        rf_rate["daily"] = ((1+rf_rate["rate"])**(1/91))-1 # Assumes 3-month T-bill reaches maturity after 13 weeks (13*7 = 91 days)
        rf_rate["monthly"] = ((1+rf_rate["rate"])**(1/3))-1
        rf_rate["3_month"] = rf_rate["rate"]

        rf_rate = rf_rate.drop(columns=["rate"])
    
        return rf_rate

    def ingest_corporate_actions(self, sep, date_index, actions=None):
        """
        Ingest Sharadars Corporate Events dataset and extract reports of bankruptices.
        Also get the dates and tickers of delistings, which is defined to be the last date 
        of ticker data in sep, before the last date in sep.
        """

        if actions is None:
            actions = pd.read_csv(self.path_corp_actions, parse_dates=["date"])

        corp_actions = pd.DataFrame(columns=["date", "ticker", "action"])
        ca_index = 0

        for index, row in actions.iterrows():
            codes = [int(num_string) for num_string in str(row["eventcodes"]).split("|")]
            if 13 in codes:
                corp_actions.loc[ca_index] = [row["date"], row["ticker"], "bankruptcy"]
                ca_index += 1


        bankrupt_tickers = corp_actions["ticker"].unique()

        sep_grouped = sep.groupby("ticker")
        last_date_of_sep = date_index.max()
        for ticker, ticker_df in sep_grouped:
            last_date_of_ticker = ticker_df.index.max()
            if last_date_of_ticker < last_date_of_sep:
                # Company was delisted
                corp_actions.loc[ca_index] =  [last_date_of_ticker, ticker, "delisted"]
                ca_index += 1


        # If a delesting happens on the same day or before a bankruptcy, then the bankruptcy takes presidence.
        for ticker in bankrupt_tickers: # It can only be a conflict if the ticker went bankrupt
            ticker_actions: pd.DataFrame = corp_actions.loc[corp_actions.ticker == ticker]
            
            if ticker_actions.shape[0] > 1: # Multiple bankruptices or delestings or both a bankruptcy and delesting
                # Prioritize bankruptcy by make all action's bankruptcy if one of them are
                if "bankruptcy" in (ticker_actions["action"].unique()):
                    for index in ticker_actions.index:
                        corp_actions.loc[index, "action"] = "bankruptcy"

                # Remove all but the earliest bankruptcy action from corp_actions
                earliest_ticker_event_date = ticker_actions["date"].min()
                ticker_actions_less_earliset_date = ticker_actions.loc[ticker_actions.date != earliest_ticker_event_date]
                indexes_to_drop = ticker_actions_less_earliset_date.index
                corp_actions = corp_actions.drop(indexes_to_drop, axis=0)

        # Drop any remaining duplicates (a result of bankruptcy being reported multiple times for the same stock on the same date)
        corp_actions = corp_actions.drop_duplicates(subset=["ticker"]) # Must be updated if more corporate actions are to be supported
        corp_actions = corp_actions.sort_values(['date', "ticker"])

        return corp_actions


    # ____________________________________________________________________________________________________________


    def _next_tick_generator(self):
        """
        Generator function yielding market data events.
        """

        for date in self.date_index_to_iterate:
            self.cur_date = date
            tick_data = self.time_data.loc[date]
            interest_data = self.rf_rate.loc[date]
            is_business_day = self.is_business_day(self.cur_date)

            yield MarketDataEvent(event_type="DAILY_MARKET_DATA", data=tick_data, date=date, interest=interest_data, is_business_day=is_business_day)
            

    def can_trade(self, ticker, date=None):
        """
        Returns true if the ticker can be traded at the provided date (current date).
        
        Use can_trade column the date being outside min/max date index for the ticker.
        Important to check this for all tickers the strategy/portfolio/broker that should be traded.
        """
        res = True

        if date is None:
            date = self.cur_date

        if self.is_bankrupt_or_delisted(ticker, date):
            return "Bankrupt or Delisted"

        try:
            res = self.ticker_data.loc[ticker, date]["can_trade"]
        except:
            res = "Can Trade False (no sep data)"

        return res

    def is_bankrupt_or_delisted(self, ticker, date=None):
        if date is None:
            date = self.cur_date

        # NOTE: can fail
        passed_ticker_actions = self.corp_actions.loc[(self.corp_actions.ticker == ticker) & (self.corp_actions.date <= date)]["action"].unique()
        
        if ("bankruptcy" in passed_ticker_actions) or ("delisted" in passed_ticker_actions):
            return True
        
        return False

    def current_corp_actions(self) -> pd.DataFrame:
        # NOTE: can fail
        return self.corp_actions.loc[self.corp_actions.date == self.cur_date]

    def get_dividends(self, date: pd.datetime):
        cur_sep = self.time_data.loc[date]
        sep_with_dividends = cur_sep.loc[cur_sep.dividends != 0]
        sep_with_dividends = sep_with_dividends.dropna(axis=0, subset=["dividends"])
        return sep_with_dividends

    def current_dividends(self):
        cur_sep = self.time_data.loc[self.cur_date]
        sep_with_dividends = cur_sep.loc[cur_sep.dividends != 0]
        sep_with_dividends = sep_with_dividends.dropna(axis=0, subset=["dividends"])        
        return sep_with_dividends

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


    def current_tick(self): # Should not be possible to fail...
        """
        Returns the market data for the current tick. Because marketdata is forward filled, this method will allways return 
        the last available prices (also on holidays, weekends, etc.).
        """
        return self.get_data_for_date(self.cur_date)


    def get_data_for_date(self, date: pd.datetime):
        try:
            daily_data = self.time_data.loc[date]
        except:
            raise KeyError("No daily data for date {}".format(date))
        else:
            return daily_data

    def current_for_ticker(self, ticker): # NOTE: I want this to fail if sebsequent code depends on this being provided.
        """
        Returns the current sep row for the provided ticker.
        """
        try:
            data = self.ticker_data.loc[ticker, self.cur_date] # may need to update, df style
        except Exception as e:
            raise MarketDataNotAvailableError("No market data for ticker {} on date {}".format(ticker, self.cur_date))
        else:
            return data

    def prev_for_ticker(self, ticker):
        try:
            prev_date = self.cur_date - relativedelta(days=1)
            data = self.ticker_data.loc[ticker, prev_date]
        except:
            raise MarketDataNotAvailableError("No market data for ticker {} on date {}".format(ticker, prev_date))
        else:
            return data

    def continue_backtest(self):
        """Checks if there are more ticks to be processed"""
        if not self.end_of_data_reached:
            return True
        else:
            return False


    def get_ticker_data(self, ticker):
        """
        Return data for ticker from the start to the end date.
        """
        start = self.start.strftime("%Y-%m-%d")
        end = self.end.strftime("%Y-%m-%d")
        try:
            data = self.ticker_data.loc[ticker][start:end]
        except Exception as e:
            print("Get Ticker data error: ", e)
            return pd.DataFrame(columns=self.ticker_data.iloc[[0]].columns)
        else:
            return data


    def get_daily_interest_rate(self):
        return self.rf_rate.loc[self.cur_date]["daily"]


class MLFeaturesDataHandler(DataHandler):
    def __init__(
        self, 
        path_features: str, 
        store_path: str, 
        start: pd.datetime, 
        end: pd.datetime
    ):
        self.path_features = path_features
        self.store_path = store_path
        self.file_name = "feature_bundle"
        self.start = start
        self.end = end

        self.emitted_until_date = start


        full_store_path = self.store_path + '/' + self.file_name + ".pickle"
        
        if os.path.isfile(full_store_path):
            data_file = open(full_store_path,'rb')
            feature_data = pickle.load(data_file)
            data_file.close()
        else:
            feature_data = self.ingest_data()

            outfile = open(full_store_path, 'wb')
            pickle.dump(feature_data, outfile)
            outfile.close()

        self.feature_data = feature_data

    def ingest_data(self):
        feature_data: pd.DataFrame = pd.read_csv(self.path_features, parse_dates=["date", "datekey", "timeout"], index_col="date", low_memory=False)

        feature_data = feature_data.sort_index()
        date = self.start - relativedelta(days=7)
        feature_data = feature_data.loc[date:]

        return feature_data


    def get_range(self, start, stop) -> Event:
        # make sure to only give new data that was released available the previous day.
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(stop, str):
            stop = pd.to_datetime(stop)
    
        # NOTE: think the exception handling should be better, but maybe its fine
        try:
            data = self.feature_data.loc[start:stop]
        except: 
            data = pd.DataFrame()

        return data


    def next_batch(self, date):
        """
        Returns feature data from self.emitted_until_date to the day before the provided date.
        The user must decide what features are relevant. (can return 10 days of data, 14 days, 7 days... 
        it all depends on how frequent the function is called and business days (non-holiday, non-extreme events, non-weekend))
        """

        # make sure to only give new data that was released available the previous day.
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        date = date - relativedelta(days=1)
        
        try:
            data = self.feature_data.loc[self.emitted_until_date:date]
            self.emitted_until_date = date
        except Exception as e:
            data = pd.DataFrame()

        return Event(event_type="FEATURE_DATA", data=data, date=date)


    def get_data_for_day(self, date):
        pass

    def get_data(self, ticker, start, end):
        pass
    


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
    Reinded and forward fill a dataframe of prices. Also adds a column to indicate whether or
    not a row was forward filled or not.

    Can be uses with pandas multiprocessing engines.
    """

    # Create complete index
    new_index = pd.date_range(sep.index.min(), sep.index.max())
    sep = sep.reindex(new_index)
    sep["can_trade"] = ~sep["close"].isnull()

    sep = sep.fillna(method="ffill")

    return sep


""" Interesting Event Codes:

[
    "EVENTCODES",
    "13",
    "N",
    "N",
    "Bankruptcy or Receivership",
    "EventCode to Title mapping for EVENTS table.",
    "text"
],
[
    "EVENTCODES",
    "31",
    "N",
    "N",
    "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing",
    "EventCode to Title mapping for EVENTS table.",
    "text"
],
[
    "EVENTCODES",
    "25",
    "N",
    "N",
    "Cost Associated with Exit or Disposal Activities",
    "EventCode to Title mapping for EVENTS table.",
    "text"
],
            [
    "EVENTCODES",
    "51",
    "N",
    "N",
    "Changes in Control of Registrant",
    "EventCode to Title mapping for EVENTS table.",
    "text"
],
"""
