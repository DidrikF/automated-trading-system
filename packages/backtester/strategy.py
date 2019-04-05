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


    def generate_signals(self, feature_data_event: Event):
        signals = []
        
        # make predictions based on the features in $feature_data_event
        signals.append(Signal(ticker="AAPL", direction=1, certainty=1))

        return Event(event_type="SIGNALS", data=signals, date=feature_data_event.date)


class Signal():
    def __init__(self, ticker, direction, certainty):
        self.ticker = ticker
        self.direction = direction
        self.certainty = certainty

    @classmethod
    def from_nothing(cls):
        return cls("NONE", "NONE", "NONE")