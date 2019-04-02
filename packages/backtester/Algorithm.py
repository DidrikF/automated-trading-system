"""
Maybe just have an abstract base class here, setting the interface that the automated trading system
must adhere to.
"""
from abc import ABCMeta, abstractmethod

from utils import Order
from event import Event, MarketEvent

class Algorithm():
  def __init__(self):
    # ERROR

  __metaclass__ = ABCMeta

  @abscractmethod()
  def generate_signal():
    raise NotImplementedError("Should implement generate_signal()")


class BuyAppleAlgorithm():

  def __init__(self, desc):
    self.description = desc


  def generate_signal(self, some_sort_of_event: Event): -> Signal
    pass
