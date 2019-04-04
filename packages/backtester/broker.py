from utils import CommissionModel, SlippageModel
from event import Event

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

    def process_orders(self, portfolio, orders_event):
        orders = orders_event.data

        # Add to unprocessed ???
        
        fill_objects = []

        for order in orders:
            fill_object = self._process_order(portfolio, order)
            fill_objects.append(fill_object)

        return Event(event_type="FILLS", data=fill_objects, date=orders_event.date)

    def _process_order(self, portfolio, order):
        # Do checks on volume vs amount and other checks. Or maybe most of this is done by the portfolio object


        commission = self.commission_model.calculate(order) # Might what a different interface
        slippage = self.slippage_model.calculate(order)
        
        # Charge portfolio etc. and if successfull, "complete" the order by appending to active_orders

        try:
            portfolio.charge(commission)
        except:
            # Cannot charge commission ? Maybe allways allow commission to be charged? IDK
            pass

        self.active_orders.append({ # Fill objects maybe?
            "order": order,
            "commission": commission,
            "slippage": slippage,
        })

        return {} # Like a fill object, recept, transaction.... IDK





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

    




class OrderCancellationPolicy(): 
    pass


