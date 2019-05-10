
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


class MarketDataEvent(Event):
    def __init__(self, event_type, data, date, interest, is_business_day):
        
        super().__init__(event_type, data, date)

        self.interest = interest
        self.is_business_day = is_business_day    



class FillEvent(Event):
    def __init__(self):
        self.type = "FILL"




