import os
import pickle

from abc import ABCMeta, abstractmethod

class DataHandler():
  def __init__(self):
    # ERROR

  __metaclass__ = ABCMeta

  @abstractmethod()
  def ingest():
    raise NotImplementedError("Should implement ingest()")

  @abstractmethod()
  def get_data():
    raise NotImplementedError("Should implement get_data()")

  @abstractmethod()
  def current():
    raise NotImplementedError("Should implement current()")


class DailyBarsDataHander(DataHandler):
  def __init__(self, source_path: str, store_path: str, file_name): -> None
    self.source_path = source_path
    self.store_path = store_path
    self.file_name = file_name

    full_path = self.store_path + "/" + self.file_name
    if os.isfile(full_path):
      data_file = open(full_path,'rb')
      self.data = pickle.load(data_file)
      data_file.close()
    else:
      self.ingest()

  def ingest(self):
    """
    Parse data into desired format for backtesting, save it and set it as self.data on
    this instance of DataHandler.
    """
    df = pd.read_csv(source_path, parse_dates=["dates"], low_memory=False)
    # The datastructure should make it easy to get data from each company, as well as
    # data for all companies at a specific date and a date range.
    
    # GET BACK TO THIS WHEN YOU HAVE A STRINGER NOTION OF HOW THE DATA SHOULD BE STRUCTURED.

    data = None

    self.data = data

    full_store_path = self.store_path + '/' + self.file_name
    outfile = open(full_store_path,'wb')
    pickle.dump(data,outfile)
    outfile.close()
  
   def get_data(self, ticker, start, end):
    """
    For the given asset or iterable of assets, returns true if all of the following are true: 1) the asset is alive for the session of the current simulation time

    (if current simulation time is not a market minute, we use the next session)
    (if we are in minute mode) the asset’s exchange is open at the
    current simulation time or at the simulation calendar’s next market minute
    there is a known last price for the asset.
    """
    pass

  def current(ticker): # require a notion of time
    """
    Returns the current value of the given assets for the given fields at the current simulation time. 
    Current values are the as-traded price and are usually not adjusted for events like splits or dividends.
    """
    pass
  
  def can_trade(self, ticker, date):
    pass
  
  def history(self, ticker): # REDUNDANT
    """
    Returns a window of data for the given assets and fields.
    This data is adjusted for splits, dividends, and mergers as of the current algorithm time.
    """
    pass

  
  # The final method, update_bars, is the second abstract method from DataHandler. 
  # It simply generates a MarketEvent that gets added to the queue as it appends the latest bars 
  # to the latest_symbol_data:
  def update_bars(self):
      """
      Pushes the latest bar to the latest_symbol_data structure
      for all symbols in the symbol list.
      """
      for s in self.symbol_list:
          try:
              bar = self._get_new_bar(s).next()
          except StopIteration:
              self.continue_backtest = False
          else:
              if bar is not None:
                  self.latest_symbol_data[s].append(bar)
      self.events.put(MarketEvent())




class MLFeaturesDataHandler(DataHandler):
  def __init__(self, source_path: str, store_path: str, file_name): -> None
    self.source_path = source_path
    self.store_path = store_path
    self.file_name = file_name
    full_store_path = self.store_path + '/' + self.file_name
    if os.isfile(full_store_path):
      data_file = open(full_path,'rb')
      self.data = pickle.load(data_file)
      data_file.close()
    else:
      self.ingest()

  def ingest():

  def get_data():

  def current():


 
# Do I need a notion of time?
# month start, end, day of the week?
# Count business days for example
# No need for intra day time (hours, minutes)




class DataLoader():
  """Class for loading and parsing data into a format suited for running backtest on."""
  

"""The below is all about moving data fast in and out of python in a desired format"""

# I'M NOT SURE HOW IMPORTANT EFFICIENT DATA READING AND WRITING IS FOR ME. CAN I NOT JUST USE PANDAS?

# Data API

# Data writers


# Data readers
# All sort of query methods implemented by zipline to get wanted data fast.




class Bundle(): # Dont think this is appropriate for me
  def __init__():
    pass
  def ingest():
    pass


