import pandas as pd

from utils import CommissionModel, SlippageModel
from event import Event
from logger import Logger
from errors import OrderProcessingError, MarketDataNotAvailableError, BalanceTooLowError
from portfolio import Order, CancelledOrder

class Broker():
    """
    Execute orders from Strategy and creates Fill events.
    """
    def __init__(self, market_data, commission_model, slippage_model):
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.market_data = market_data

        # The broker gets orders, if it gets filled a Fill object is created and added to the blotter
        # if the order is cancelled/fails then a CancelledOrder object is created and appended to cancelled_orders
        self.cancelled_orders = [] # This is more relevant for the portfolio maybe?
        
        # The broker needs to have a record of active position, because he needs to check this for exit rules
        self.active_positions = []

        self.active_positions_history = []

        # Should the blotter have its own class?
        self.blotter = [] # record of trades (fills, and fills holds the order behind them)

    def process_orders(self, portfolio, orders_event):
        orders = orders_event.data

        fill_objects = []
        cancelled_orders = []

        for order in orders:
            try:
                fill_object = self._process_order(portfolio, order)
            except OrderProcessingError as e:
                Logger.logr.warning("Failed processing order with error: {}".format(e))

                cancelled_order = CancelledOrder(order, e)

                cancelled_orders.append(cancelled_order)
            else:
                fill_objects.append(fill_object)
                self.blotter.append(fill_object)
                self.active_positions.append(fill_object)


        self.cancelled_orders.extend(cancelled_orders)

        if len(cancelled_orders) > 0:
            return (
                Event(event_type="FILLS", data=fill_objects, date=orders_event.date),
                Event(event_type="CANCELLED_ORDERS", data=cancelled_orders, date=orders_event.date)
            )
        else:
            return (Event(event_type="FILLS", data=fill_objects, date=orders_event.date), None)



    def _process_order(self, portfolio, order):
        # Do checks on volume vs amount and other checks. Or maybe most of this is done by the portfolio object
        validation_errors = self.validate_order(order)
        if len(validation_errors) != 0:
            Logger.logr.warning("Order failed validation with validation errors: {}".format(validation_errors))
            raise OrderProcessingError("Cannot complete order, with validation errors: {}".format(validation_errors))
        
        try:
            stock_price = self.market_data.current_for_ticker(order.ticker)["open"]
        except MarketDataNotAvailableError as e:
            Logger.logr.warning("When processing an order; market data was not available for ticker {} on date {}.".format(order.ticker, self.market_data.cur_date))
            raise OrderProcessingError("Cannot complete order, with error: {}".format(e))
        
        cost = order.amount * stock_price
        commission = self.commission_model.calculate(order) # Might what a different interface
        slippage = self.slippage_model.calculate(order)
        
        # Charge portfolio etc. and if successfull, "complete" the order by appending to active_orders

        try:
            portfolio.charge(cost)
            portfolio.charge_commission(commission)
        except BalanceTooLowError as e:
            Logger.logr.warning("Balance too low! Balance: {}, wanted to charge {} for the stock and {} in commission".format(portfolio.balance, cost, commission))
            raise OrderProcessingError("Cannot complete order, with error: {}".format(e))

        fill = Fill(
            order=order,
            date=self.market_data.cur_date,
            price=stock_price-slippage,
            commission=commission,
            slippage=slippage,
        )
        
        return fill

    def manage_active_positions(self):
        """
        Active positions are represented by fill objects (contains assocated order) where the resulting position have not yet been
        liquidated. Every day the active positions must be checked to see if a stop-loss, take-profit or timeout would trigger.
        
        This is somewhat different than the state of the portfolio, because each order must be treated by the broker independently (not
        summed together like is done with portfolio.portfolio)
        """
        return None # TypeError: '<=' not supported between instances of 'float' and 'NoneType'
        position_liquidations = []

        for index, fill in enumerate(self.active_positions):
            order = fill.order
            ticker = fill.ticker
            ticker_data = self.market_data.current_for_ticker(ticker)

            # If price went up over the day, assume the low was hit before the high.
            # If the price when down over the day, assume the high was hit first.
            # This being an issue is highly doubtfull, but still
            price_direction_for_the_day = 1 if (ticker_data["open"] <= ticker_data["close"]) else -1

            if order.direction == 1:
                # Long position
                
                if price_direction_for_the_day == 1:
                    # Prices rose over the day, low was hit first.
                
                    if ticker_data["low"] <= order.stop_loss:
                        # sell out of position

                        # pop out fill from active_positions

                        continue

                    elif ticker_data["high"] >= order.take_profit:
                        # Sell out of position

                        # pop out fill

                        continue
                elif price_direction_for_the_day == -1:
                    # price declined over the day, high was hit first.

                    if ticker_data["high"] >= order.take.profit:
                        # sell out

                        # pop fill

                        continue

                    elif ticker_data["low"] >= order.stop_loss:
                        # sell out

                        # pop fill
                        
                        continue

            elif order.direction == -1:
                # Short position: make this logic later
                pass
        
        if len(position_liquidations) > 0:
            return Event(event_type="POSITION_LIQUIDATIONS", data=position_liquidations, date=self.market_data.cur_date)
        else:
            return None


    def validate_order(self, order): # Maybe the portfolio should handle all this, and the broker just executes orders
        # IMPLEMENT VALIDATION LOGIC!
        # Maybe this is expressed through the order cancellation policy
        validation_errors = []

        return validation_errors


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
        Cancel an open order by ID.
        
        Currently orders are either completed "immediatly" or digarded. So this will never be used.
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

    def capture_active_positions_state(self):
        """
        Adds the the current state of active positions to the active_positions_history list with some identifying attributes.
        """
        pass

    def cancelled_orders_to_df(self):
        df = pd.DataFrame(columns=["order_id", "date", "ticker", "amount", "error"])
        for index, c_ord in enumerate(self.cancelled_orders):
            df.at[index, "order_id"] = c_ord.order_id
            df.at[index, "date"] = c_ord.date
            df.at[index, "ticker"] = c_ord.ticker
            df.at[index, "amount"] = c_ord.order.amount
            df.at[index, "error"] = str(c_ord.error)
            # Might want to extend this, don't know atm

        df = df.set_index("order_id")
        df = df.sort_index()

        return df

    def active_positions_history_to_df(self):
        df = pd.DataFrame(columns=["order_id", "ticker", "direction", "amount", "date", "price", "commission", "slippage"])

        for index, fill in enumerate(self.active_positions_history):
            df.at[index, "order_id"] = fill.order_id
            df.at[index, "ticker"] = fill.ticker
            df.at[index, "direction"] = fill.direction
            df.at[index, "amount"] = fill.amount
            df.at[index, "date"] = fill.date
            df.at[index, "price"] = fill.price
            df.at[index, "commission"] = fill.commission
            df.at[index, "slippage"] = fill.slippage
            # May want more of the order properties here, but not sure
        
        df = df.set_index("order_id")
        df = df.sort_index() # This may not be the order things are executed? (or will it be? I decide the order the orders are processed, so I guess, the order should be filled in order.)

        return df


    def blotter_to_df(self):
        df = pd.DataFrame(columns=["order_id", "ticker", "direction", "amount", "date", "price", "commission", "slippage"])

        for index, fill in enumerate(self.blotter):
            df.at[index, "order_id"] = fill.order_id
            df.at[index, "ticker"] = fill.ticker
            df.at[index, "direction"] = fill.direction
            df.at[index, "amount"] = fill.amount
            df.at[index, "date"] = fill.date
            df.at[index, "price"] = fill.price
            df.at[index, "commission"] = fill.commission
            df.at[index, "slippage"] = fill.slippage
            # May want more of the order properties here, but not sure
        
        df = df.set_index("order_id")
        df = df.sort_index() # This may not be the order things are executed? (or will it be? I decide the order the orders are processed, so I guess, the order should be filled in order.)

        return df


class Fill():
    def __init__(self, order, date, price, commission, slippage):
        self.order_id = order.id
        self.ticker = order.ticker
        self.direction = order.direction # Strictly unnecessary, but makes this information easily accessable.
        self.amount = order.amount

        self.date = date
        self.price = price # slippage subtracted
        self.commission = commission
        self.slippage = slippage

        self.order = order


# I guess this should be an event, and be a handler for it in the portfolio. But none of this is implemented atm.
class PositionLiquidation():
    def __init__(self, order, price, date):
        self.order = order
        self.ticker = order.ticker
        self.liquidation_price = price
        self.date = date


class TerminatedPosition():
    def __init__(self, fill: Fill, reason):
        # Any data I want to expose directly....
        self.fill = fill
        self.reason = reason


class OrderCancellationPolicy(): 
    pass


