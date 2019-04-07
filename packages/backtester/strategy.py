"""
Maybe just have an abstract base class here, setting the interface that the automated trading system
must adhere to.
"""
from abc import ABC, abstractmethod

from event import Event


class Strategy(ABC):
    def generate_signals(self):
        pass


class BuyAppleStrategy():

    def __init__(self, desc):
        self.description = desc
        self.signal_id_seed = 0

    def generate_signals(self, feature_data_event: Event):
        signals = []
        
        # make predictions based on the features in $feature_data_event
        signals.append(Signal(signal_id=self.get_signal_id(), ticker="AAPL", direction=1, certainty=1))
        signals.append(Signal(signal_id=self.get_signal_id(), ticker="FCX", direction=1, certainty=0.5))

        return Event(event_type="SIGNALS", data=signals, date=feature_data_event.date)

    def get_signal_id(self):
        self.signal_id_seed += 1
        return self.signal_id_seed

class Signal():
    def __init__(self, signal_id, ticker, direction, certainty):
        self.signal_id = signal_id
        self.ticker = ticker
        self.direction = direction
        self.certainty = certainty

        self.ewastd = None # The variablity measure behind the barriers, may also be relevant
        self.barriers = (None, None, None) # Relevant for calculations of stop-loss, take-profit and also relevent for rebalancing

        self.feature_data_index = None # Need to take this as arguments
        self.feature_data_date = None # Need to take this as arguments


    @classmethod
    def from_nothing(cls):
        return cls("NONE", "NONE", "NONE")