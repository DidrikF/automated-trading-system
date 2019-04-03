


def show_progress():
  pass


class OrderCancelationPolicy(): 
  pass




class CommissionModel(): # Too much un needed complexity..
    """
    Abstract commission model interface.
    Commission models are responsible for accepting order/transaction pairs and 
    calculating how much commission should be charged to an algorithm’s account 
    on each transaction.
    """

class EquityCommissionModel():
    pass


# CommissionPerShare

# CommissionPerTrade

# CommissionPerDollar


class SlippageModel(): # Too much un needed complexity..
    pass


# FixedSlippage

# VolumeShareSlippage


class EquitySlippageModel():
  pass




class Order():
  """
  Parameters:	
asset (Asset) – The asset that this order is for.
amount (int) – The amount of shares to order. If amount is positive, this is the number of shares to buy or cover. If amount is negative, this is the number of shares to sell or short.
limit_price (float, optional) – The limit price for the order.
stop_price (float, optional) – The stop price for the order.
style (ExecutionStyle, optional) – The execution style for the order.
Returns:	
order_id – The unique identifier for this order, or None if no order was placed.
  """
  def __init__(self, ticker, amount):
    self.ticker = ticker
    self.amount = amount

  @classmethod
  def order_value(self, ticker, value):
    pass

  @classmethod
  def order_percent(self, ticker, value):
    pass

  def get_limit_price():
    pass

  def get_stop_price():
    pass

  # Order target methods, probably not needed

class MarketOrder(Order):
  pass

class LimitStopOrder(Order):
  def __init__():
    pass



class Transaction():
  """
  The transaction being processed. A single order may generate multiple transactions if there isn’t enough 
  volume in a given bar to fill the full amount requested in the order.
  """


class Asset():
  """Class representing a stock."""
  def __init__():
    pass

  def first_traded():
    pass
  def security_name():
    pass
  def security_end_date():
    pass
  def security_start_date():
    pass

  def to_dict():
    pass

class Blotter(): # Is this the thing that "execute" orders?
  def __init__(self):
    self.record = {}

  def batch_order():
    pass


  def write(order):
    pass
  
  def read(self):
    pass


def Cache():
  pass


class Ledger():
  """The ledger tracks all orders and transactions as well as the current state of the portfolio and positions."""

  pass

class Portfolio():
  pass

class PostionTracker():
  """The current state of the positions held."""
  pass

class PositionStats():
  """Computed values from the current positions."""
  pass



class Account(): # Dont know if this is appropriate for me 
  pass




def get_order():
  """Lookup an order based on the order id returned from one of the order functions."""

def get_open_orders():
  """Retrieve all of the current open orders."""

def cancel_order():
  """Cancel an open order."""

def set_cancel_policy():
  """Sets the order cancellation policy for the simulation."""


def NeverCancel():
  """Orders are never automatically canceled."""


def set_do_not_order_list():
  pass

def set_long_only():
  pass


def set_max_leverage(): # Dont think I will have leverage
  pass

def set_max_order_count():
  pass

def set_max_order_size():
  """
  Set a limit on the number of shares and/or dollar value of any single order placed for sid. 
  Limits are treated as absolute values and are enforced at the time that the algo attempts to place an order for sid.
  If an algorithm attempts to place an order that would result in exceeding one of these limits, 
  raise a TradingControlException.
  """

def set_max_position_size():
  """
  Set a limit on the number of shares and/or dollar value held for the given sid. Limits are treated 
  as absolute values and are enforced at the time that the algo attempts to place an order for sid. 
  This means that it’s possible to end up with more than the max number of shares due to splits/dividends, 
  and more than the max notional due to price improvement.
  If an algorithm attempts to place an order that would result in increasing the absolute value of shares/dollar 
  value exceeding one of these limits, raise a TradingControlException.
  """



class CachedObject():
  pass