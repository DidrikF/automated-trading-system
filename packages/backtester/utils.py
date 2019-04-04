from abc import ABC, abstractmethod


def show_progress():
    """
    Print progression statistics for the backtest to stdout.
    """
    pass



class CommissionModel(ABC):
    def __init__(self):
        pass
    def calculate(self, order):
        pass

class SlippageModel(ABC):
    def __init__(self):
        pass

    def calculate(self, order):
        pass


class EquityCommissionModel(CommissionModel):
    """
    The commission model are responsible for accepting order/transaction pairs and 
    calculating how much commission should be charged to an algorithm’s account 
    on each transaction.
    """
    def __init__(self):
        pass
    def calculate(self, order):
        return 0.0



class EquitySlippageModel(SlippageModel):
    """
    The slippage model is responsible for calculating the slippage associated with an order.
    """
    def __init__(self):
        pass
    def calculate(self, order):
        return 0.0






class Transaction():
    """
    The transaction being processed. A single order may generate multiple transactions if there isn’t enough 
    volume in a given bar to fill the full amount requested in the order.
    """
    pass



class Asset():
    """Class representing a stock."""
    def __init__(self):
        pass

    def first_traded(self):
        pass
    def security_name(self):
        pass
    def security_end_date(self):
        pass
    def security_start_date(self):
        pass

    def to_dict(self):
        pass

class Blotter(): # Is this the thing that "execute" orders?
    def __init__(self):
        self.record = {}

    def batch_order(self):
        pass


    def write(self, order):
        pass
  
    def read(self):
        pass



class Ledger():
    """The ledger tracks all orders and transactions as well as the current state of the portfolio and positions."""

    pass


class PostionTracker():
    """The current state of the positions held."""
    pass

class PositionStats():
    """Computed values from the current positions."""
    pass



class Account(): # Dont know if this is appropriate for me 
    pass



class CachedObject():
    pass






# CommissionPerShare

# CommissionPerTrade

# CommissionPerDollar

# FixedSlippage

# VolumeShareSlippage


