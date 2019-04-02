
from abc import ABCMeta, abstractmethod

class Event():
  """
  Base event class, can be extended with any wanted functionality.
  """
  pass


class MarketUpdateEvent(Event):
  def __init__(self)
    self.type = "MARKET"


class FillEvent(Event):
  def __init__(self):
    self.type = "FILL"



class EventQueue():
  def __init__(self):
    self.queue = []

  
  def add(evt: Event):
    if isinstance(evt, Event):
    
    else:
      raise TypeError("The queue only accepts objects of class/subclass Event.")

  
  def pop(self):
    return self.queue.pop()
