import pandas as pd
import math

from strategy import some_strat
# import CommissionModel
# import SlippageModel
from event import EventQueue, Event
from data_handler import DataHandler, DailyBarsDataHander, MLFeaturesDataHandler
from broker import Broker
from utils import CommissionModel, SlippageModel

# NOT SURE ABOUT THIS, WHERE TO PUT METHODS ETC
class Backtester():
  def __init__(self, data_handler: DataHandler, feature_handler: DataHandler, start, end):
    self.data_handler = data_handler # set this through method?
    self.feature_handler = feature_handler # set this through method
    self.event_queue = EventQueue()
    self.start = start
    self.end = end
    # ..

    def set_benchmark(self):
        """Set the benchmark asset."""
        # Interesting, how does this work?
        # Another account, using a different strategy.

    def set_commission(self, commission_model: CommissionModel):
        """Set commission model to use."""


    def set_slippage(self, slippage_model: SlippageModel):
        """Set slippage model to use."""

    def set_broker(self, broker: Broker):
        """ Set broker instance that will execute orders. """

    def run(self):
        # Check configuration

        """
        The whole point of doing it in an event driven way is to make it easier to reason about
        """

        while True: # This loop generates new "ticks" until the backtest is completed.
            if self.data_handler.continue_backtest() == True:
                self.event_queue.append(self.data_handler.next_tick())
                # market_data = self.data_handler.get_next() # Adds event I guess
            else:
                break

            while True: # This is executed until all events for the tick has been processed
                try:
                    event = self.event_queue.get(False)
                except: # queue is empty
                    break
                else:
                    if event is not None:
                        if event.type == 'DAILY_MARKET_DATA':
                            # need this to add the feature data to the queue, so it can be "processed" (used to generate predictions)
                            # event.date = event.date - realativedelta(days=1)
                            self.feature_handler.get_features(event) # the function need access to the event queue

                        elif event.type == "FEATURE_DATA":
                        
                            strategy.calculate_signals(event)
                            
                            port.update_timeindex(event) # WHAT?


                        elif event.type == 'SIGNAL':
                            port.update_signal(event)

                        elif event.type == 'ORDER':
                            broker.execute_order(event)

                        elif event.type == 'FILL':
                            port.update_fill(event)


    def get_info(self): 
        """Get initial setting of the backtest."""
        pass
    


        