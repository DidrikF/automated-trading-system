
import pandas as pd
import time
import datetime as dt
import sys
from .types import CommissionModel, SlippageModel

class Signal():
    def __init__(
        self, 
        signal_id: int, 
        ticker: str, 
        direction: int, 
        certainty: float, 
        ewmstd: float, 
        timeout: pd.datetime,
        features_date: pd.datetime,
    ):
        self.signal_id = signal_id
        self.ticker = ticker
        self.direction = direction
        self.certainty = certainty
        self.ewmstd = ewmstd
        self.timeout = timeout
        self.features_date = features_date

    @classmethod
    def from_nothing(cls):
        return cls(-1, "NONE", 1, 0.5, 0.15, pd.to_datetime("2020-01-01"), pd.to_datetime("1995-01-01"))


class EquityCommissionModel(CommissionModel):
    """
    The commission model are responsible for accepting order/transaction pairs and 
    calculating how much commission should be charged to an algorithm’s account 
    on each transaction.
    Link: https://www.interactivebrokers.co.uk/en/index.php?f=39753&p=stocks2
    """
    def __init__(self):
        # Commission
        self.minimum_per_order = 0.35
        self.per_share = 0.0035 # To make it simple, i wont trade over 300 000 shares in a month anyway, which is required to get lower price on Interactive Brokers
        self.maximum_in_percent_of_order_value = 0.01

        # Other fees
        self.us_clearing_fee_per_share = 0.00020 # 0.0000207

        self.us_transaction_fees_per_dollar = 0.0000207
        self.nyse_pass_through_fees_per_commission = 0.000175
        self.finra_pass_through_fees_per_commission = 0.00056
        self.finra_trading_activity_fee_per_share = 0.000119

    def calculate(self, amount: int, price: float, info=None):
        if price < 0:
            if info is not None:
                for key, val in info.items():
                    print(key, ": ", val)
            raise ValueError("Cannot calculate commission with a negative price. Price given: {}".format(price))
        
        commission = min(max(self.minimum_per_order, self.per_share*abs(amount)), self.maximum_in_percent_of_order_value*abs(amount)*price)
        
        us_clearing_fees = self.us_clearing_fee_per_share * abs(amount)
        us_trasaction_fees = self.us_transaction_fees_per_dollar * abs(amount) * price
        nyse_pass_through_fees = self.nyse_pass_through_fees_per_commission * commission
        finra_pass_through_fees = self.finra_pass_through_fees_per_commission * commission
        finra_trading_activity_fee = self.finra_trading_activity_fee_per_share * abs(amount)

        return commission + us_clearing_fees + us_trasaction_fees + nyse_pass_through_fees + finra_pass_through_fees + finra_trading_activity_fee




class EquitySlippageModel(SlippageModel):
    """
    The slippage model is responsible for calculating the slippage associated with an order.
    """
    def __init__(self):
        pass        

    def calculate(self):
        """
        Returns slippage per share.
        """
        return 0



def report_progress(cur_date: pd.datetime, start_date: pd.datetime, end_date: pd.datetime, time0, task):
    """
    Print progression statistics for the backtest to stdout.
    """

    total_days = end_date - start_date
    days_completed = cur_date - start_date

    ratio_of_jobs_completed = float((days_completed.days+1)/total_days.days)
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

