
from abc import ABCMeta, abstractmethod

class Event():
  """
  Base event class, can be extended with any wanted functionality.
  """
  def __init__(self, event_type, data, date=None):
      self.type = event_type
      self.data = data
      self.date = date



class EventQueue():
    def __init__(self):
        self.queue = []

    def add(self, evt: Event):
        if isinstance(evt, Event):
            self.queue.append(evt)
        else:
            raise TypeError("The queue only accepts objects of class/subclass Event.")

    
    def get(self):
        return self.queue.pop()



# Don't know if this is needed.

class MarketDataEvent(Event):
    def __init__(self):
        self.type = "MARKET"


class FillEvent(Event):
    def __init__(self):
        self.type = "FILL"




