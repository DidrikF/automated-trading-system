from utils import CommissionModel, SlippageModel
from event import Event
from backtester import logr


class Broker():
    """
    Execute orders from Strategy and creates Fill events.
    """
    def __init__(self, market_data, commission_model, slippage_model):
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.market_data = market_data

        self.unprocessed_orders = []
        self.active_orders = [] # active_positions ?

        # Should the blotter have its own class?
        self.blotter = [] # record of trades (fills, and fills holds the order behind them)

    def process_orders(self, portfolio, orders_event):
        orders = orders_event.data

        fill_objects = []

        for order in orders:
            try:
                fill_object = self._process_order(portfolio, order)
            except OrderProcessingError as e:
                logr.warning("Failed processing order with error: {}".format(e))
                # Try later of disgard?
                # Default behavior for now will be to disregard the order
                # Should I inform the portfolio?
                self.unprocessed_orders.append(order)
            else:
                fill_objects.append(fill_object)

        return Event(event_type="FILLS", data=fill_objects, date=orders_event.date)

    def _process_order(self, portfolio, order):
        # Do checks on volume vs amount and other checks. Or maybe most of this is done by the portfolio object
        validation_errors = self.validate_order(order)
        if validation_errors is not None :
            logr.warning("Order failed validation with validation errors: {}".format(validation_errors))
            raise OrderProcessingError("Cannot complete order, with validation errors: {}".format(validation_errors))
        
        try:
            stock_price = self.market_data.ticker_current(order.ticker)["open"]
        except MarketDataNotAvailableError as e:
            logr.warning("When processing an order; market data was not available for ticker {} on date {}.".format(order.ticker, self.market_data.cur_date))
            raise OrderProcessingError("Cannot complete order, with error: {}".format(e))
        
        cost = order.amount * stock_price
        commission = self.commission_model.calculate(order) # Might what a different interface
        slippage = self.slippage_model.calculate(order)
        
        # Charge portfolio etc. and if successfull, "complete" the order by appending to active_orders

        try:
            portfolio.charge(amount)
            portfolio.charge_commission(commission)
        except BalanceTooLowError as e:
            logr.warning("Balance too low! Balance: {}, wanted to charge {} for the stock and {} in commission".format(portfolio.balance, cost, commission))
            raise OrderProcessingError("Cannot complete order, with error: {}".format(e))

        fill = Fill(
            order=order,
            date=self.market_data.cur_date,
            price=stock_price-slippage
            commission=commission
            slippage=slippage
        )
        
        self.blotter.append(fill)

        return fill

    def validate_order(self, order):
        # IMPLEMENT VALIDATION LOGIC!
        
        return None 

    def get_order(self, order_id):
        """
        Lookup an order based on the order id.
        """
        # Search each 

    def get_open_orders(self):
        """
        Retrieve all of the current open orders.
        
        """

    def cancel_order(self):
        """
        Cancel an open order.
        Cancel order by ID!
        """

    def set_cancel_policy(self):
        """
        Sets the order cancellation policy for the simulation.
        If an order has not been filled after some number of ticks the order is abandoned (for example)
        
        should this be handled by the portfolio, and the broker just has basic functionality, and the portfolio must do the heavy lifting?
        """


    def NeverCancel(self):
        """
        Orders are never automatically canceled.
        Call this function to make the broker continue forever to fill order, unless it gets canceled by the outside (for example)
        """



    def set_commission_model(self, commission_model: CommissionModel):
        """
        Set commission model to use.
        The commission model is responsible for modeling the costs associated with executing orders. 
        """
        if not isinstance(commission_model, CommissionModel):
            raise TypeError("Must be instance of CommissionModel")

        self._commission_model = commission_model

    def set_slippage_model(self, slippage_model: SlippageModel):
        """
        Set slippage model to use. The slippage model is responsible for modeling the effect your order has on the stock price.
        Generally the stock price will move against you when submitting an order to the market.
        """
        if not isinstance(slippage_model, SlippageModel):
            raise TypeError("Must be instance of SlippageModel")

        self._slippage_model = slippage_model


class Fill():
    def __init__(self, order, date, price, commission, slippage):
        self.order_id = order.id
        self.ticker = order.ticker
        self.order = order
        self.date = date
        self.price = price # slippage subtracted
        self.commission = commission
        self.slippage = slippage


class OrderCancellationPolicy(): 
    pass


