import pandas as pd
import math
import Event
import Strategy
import CommissionModel
import SlippageModel
from event import EventQueue


# NOT SURE ABOUT THIS, WHERE TO PUT METHODS ETC
class Backtest():
  def __init__(self, data_handler: DataHandler, feature_handler: DataHandler):
    self.data_handler = data_handler # set this through method?
    self.feature_handler = feature_handler # set this through method
    self.event_queue = EventQueue()
    self.start = start
    self.end = end
    # ..

  def run():
    # Check configuration

    while True: # This loop generates new "ticks" until the backtest is completed.
      if self.data_handler.continue_backtest() == True:
        self.data_handler.next_tick()
        market_data = self.data_handler.get_next() # Adds event I guess
      else:
        break

      while True: # This is executed until all events for the tick has been processed
        try:
            event = self.event_queue.get(False)
        except Queue.Empty:
            break
        else:
          if event is not None:
            if event.type == 'DAILY_MARKET_DATA':
              # need this to add the feature data to the queue, so it can be "processed" (used to generate predictions)
              # event.date = event.date - realativedelta(days=1)
              self.feature_handler.get_features(event) # the function need access to the event queue

            elif event.type == "FEATURE_DATA"
              
              strategy.calculate_signals(event)
              
              port.update_timeindex(event) # WHAT?


            elif event.type == 'SIGNAL':
              port.update_signal(event)

            elif event.type == 'ORDER':
              broker.execute_order(event)

            elif event.type == 'FILL':
              port.update_fill(event)


  def get_info(): 
    """Get initial setting of the backtest."""
    pass
  
  def set_benchmark():
    """Set the benchmark asset."""
    # Interesting, how does this work?
    # Another account, using a different strategy.

  def set_commission():
    """Set commission model to use."""


  def set_slippage(self, ):
  """Set slippage model to use."""

    if not isinstance()


class OrderCancelationPolicy(): 
  pass




class CommissionModel(): # Too much un needed complexity..
  """
  Abstract commission model interface.
  Commission models are responsible for accepting order/transaction pairs and 
  calculating how much commission should be charged to an algorithm’s account 
  on each transaction.
  """

class EquityCommissionModel():
  pass


# CommissionPerShare

# CommissionPerTrade

# CommissionPerDollar


class SlippageModel(): # Too much un needed complexity..
  pass

# FixedSlippage

# VolumeShareSlippage



