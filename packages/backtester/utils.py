from abc import ABC, abstractmethod
import pandas as pd
import time
import datetime as dt
import sys


def report_progress(cur_date: pd.datetime, start_date: pd.datetime, end_date: pd.datetime, time0, task):
    """
    Print progression statistics for the backtest to stdout.
    """

    total_days = end_date - start_date
    days_completed = cur_date - start_date

    ratio_of_jobs_completed = float(days_completed.days/total_days.days)
    minutes_elapsed = (time.time()-time0)/60
    minutes_remaining = minutes_elapsed*(1/ratio_of_jobs_completed - 1)
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = time_stamp + " " + str(round(ratio_of_jobs_completed*100, 2)) + "% " + " ratio of days completed " + str(days_completed.days) + "/" + str(total_days.days) + \
        " - " + task + " done after " + str(round(minutes_elapsed, 2)) + " minutes. Remaining " + \
            str(round(minutes_remaining, 2)) + ' minutes.'
    
    if cur_date < end_date: 
        sys.stderr.write(msg + '\r') # override previous line
    else: 
        sys.stderr.write(msg + '\n')
    return



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
        return 10.0
        # https://www.interactivebrokers.co.uk/en/index.php?f=39753&p=stocks2


class EquitySlippageModel(SlippageModel):
    """
    The slippage model is responsible for calculating the slippage associated with an order.
    """
    def __init__(self):
        pass
    def calculate(self, order):
        return 1.0
        # https://arxiv.org/pdf/1103.2214.pdf






class Transaction():
    """
    The transaction being processed. A single order may generate multiple transactions if there isn’t enough 
    volume in a given bar to fill the full amount requested in the order.
    
    An assumption I make is that the whole order is filled or not at all. Multiple transactions will not occure
    and this class therefore becomes redundant and unnecessary.
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


