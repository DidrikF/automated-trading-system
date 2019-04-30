import pandas as pd
import copy

from utils import CommissionModel, SlippageModel
from event import Event
from logger import Logger
from errors import OrderProcessingError, MarketDataNotAvailableError, BalanceTooLowError
from portfolio import Order, CancelledOrder, Portfolio

class Account():
    def __init__(self):
        self.active_trades = []
        self.closed_trades = []


class Trade():
    def __init__(self, fill):
        self.initial_fill = fill
        self.fill_updates = []
        self.cur_amount = fill.amount
        self.ticker = fill.ticker

    def calculate_pnl(self):
        pass




class Broker():
    """
    Execute orders from Strategy and creates Fill events.
    """
    def __init__(self, market_data, commission_model, slippage_model, annual_margin_interest_rate, \
        initial_margin_requirement, maintenance_margin_requirement):
        
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.market_data = market_data

        self.annual_margin_interest_rate = annual_margin_interest_rate
        self.initial_margin_requirement = initial_margin_requirement
        self.maintenance_margin_requirement = maintenance_margin_requirement
        
        # The broker gets orders, if it gets filled a Fill object is created and added to the blotter
        # if the order is cancelled/fails then a CancelledOrder object is created and appended to cancelled_orders
        self.cancelled_orders = [] # This is more relevant for the portfolio maybe?
        
        # The broker needs to have a record of active position, because he needs to check this for exit rules
        self.account = Account()


        self.active_positions_history = []


        self.blotter = [] # Fills

    def process_orders(self, portfolio, orders_event):
        orders = orders_event.data

        fill_objects = []
        cancelled_orders = []

        for order in orders:
            try:
                fill_object = self._process_order(portfolio, order)
            except OrderProcessingError as e:
                Logger.logr.warning("Failed processing order with error: {}".format(e)) # Show log from backtest in the dashboard...

                cancelled_order = CancelledOrder(order, e)
                cancelled_orders.append(cancelled_order)
            else:
                fill_objects.append(fill_object)

                self.blotter.append(fill_object)
                self.active_positions.append(fill_object)

        self.cancelled_orders.extend(cancelled_orders)

        if len(cancelled_orders) == 0:
            cancelled_orders_event = None
        else:
            cancelled_orders_event = Event(event_type="CANCELLED_ORDERS", data=cancelled_orders, date=orders_event.date)

        if len(fill_objects) == 0:
            fills_event = None
        else:
            fills_event =  Event(event_type="FILLS", data=fill_objects, date=orders_event.date)


        return (fills_event, cancelled_orders_event)

    def _process_order(self, portfolio, order):

        try:
            stock_price = self.market_data.current_for_ticker(order.ticker)["open"]
        except MarketDataNotAvailableError as e:
            Logger.logr.warning("When processing an order; market data was not available for ticker {} on date {}.".format(order.ticker, self.market_data.cur_date))
            raise OrderProcessingError("Cannot complete order, with error: {}".format(e))
    

        if order.direction == 1:
            slippage = self.slippage_model.calculate(order)
            fill_price = stock_price + slippage
            cost = order.amount * fill_price
            commission = self.commission_model.calculate(order, fill_price) # The commission is also based on the fill_price
            
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
                price=fill_price,
                commission=commission,
                slippage=slippage,
            )
        
        elif order.direction == -1:
            # calculate initial margin requriement
            initial_margin_requirement = self.calculate_initial_margin_requirement(order)
            slippage = self.slippage_model.calculate(order)
            fill_price = stock_price - slippage
            commission = self.commission_model.calculate(order, fill_price)

            # make sure portfolio has the funds to deposit into the margin account
            try:
                # move funds to portfolio margin account
                portfolio.increase_margin_account(initial_margin_requirement)
                portfolio.charge_commission(commission)
            except:
                Logger.logr.warning("Balance too low! Balance: {}, wanted to charge {} for the stock and {} in commission".format(portfolio.balance, cost, commission))
                raise OrderProcessingError("Cannot complete order, with error: {}".format(e))

            fill = Fill(
                order=order,
                date=self.market_data.cur_date,
                price=fill_price,
                commission=commission,
                slippage=slippage,
            )
        
        return fill


    def handle_bankruptcies(self, bankruptcies_event: Event):
        pass



    def handle_corporate_actions(self, corporate_actions_event: Event):
        """
        Dividends is one type of corporate action.
        Short positions do not receive dividends!
        """

        """
        actions = corporate_actions_event.data
        for action in actoins:
            if action.type == "DIVIDEND":
                ticker_data = self.market_data.current_for_ticker(action.ticker)
                dividend_per_share = ticker_data["dividend"]

                if ticker in self.active_positions:
                    if position.direction == -1:
                        total_dividend = dividend_per_share * position.amount
                        portfolio.charge(total_dividend)
        

        "EVENTCODES",
            "13",
            "N",
            "N",
            "Bankruptcy or Receivership",
            "EventCode to Title mapping for EVENTS table.",
            "text"
        """


    def manage_active_positions(self, portfolio: Portfolio):
        """
        Active positions are represented by fill objects (contains associated order) where the resulting position have not yet been
        liquidated. Every day the active positions must be checked to see if a stop-loss, take-profit or timeout would trigger an exit.
        
        Also as the price of the stocks in the portfolio changes the margin account size must be updated so meet the requirements
        for the short positions.
        
        This is somewhat different than the state of the portfolio, because each order must be treated by the broker independently (not
        summed together like is done with portfolio.portfolio)
        """
        
        # return None # TypeError: '<=' not supported between instances of 'float' and 'NoneType'
        
        liquidated_positions = []

        for index, fill in enumerate(self.active_positions):
            order = fill.order
            ticker = fill.ticker
            try:
                ticker_data = self.market_data.current_for_ticker(ticker)
            except MarketDataNotAvailableError as e:
                Logger.logr.warning("Failed to manage active position, because market data not available for ticker {} on date {}".format(ticker, self.market_data.cur_date))
                continue
            # If price went up over the day, assume the low was hit before the high.
            # If the price when down over the day, assume the high was hit first.
            # This being an issue is highly doubtfull, but still...
            price_direction_for_the_day = 1 if (ticker_data["open"] <= ticker_data["close"]) else -1

            if order.direction == 1:
                # Long position
                if price_direction_for_the_day == 1:
                    # Prices rose over the day, low was hit first.
                
                    if ticker_data["low"] <= order.stop_loss:
                        # Calculate and charge commission
                        commission = self.commission_model.calculate(order=order, fill_price=order.stop_loss) # Do I need to change the signature?
                        portfolio.charge_commission(commission)
                        # sell out of position
                        exit_price = order.stop_loss - self.slippage_model.calculate(order) # SIGN?
                        liquidated_positions.append(PositionLiquidation(fill, exit_price, self.market_data.cur_date))

                        # pop out fill from active_positions
                        self.active_positions.remove(fill)
                        continue

                    elif ticker_data["high"] >= order.take_profit:
                        # Calculate and charge commission
                        commission = self.commission_model.calculate(order=order, fill_price=order.take_profit) # Do I need to change the signature?
                        portfolio.charge_commission(commission)
                        # Sell out of position
                        exit_price = order.take_profit - self.slippage_model.calculate(order) # SIGN?
                        liquidated_positions.append(PositionLiquidation(fill, exit_price, self.market_data.cur_date))
                        # pop out fill
                        self.active_positions.remove(fill)
                        continue
                    
                elif price_direction_for_the_day == -1:
                    # price declined over the day, high was hit first.
                    if ticker_data["high"] >= order.take_profit:
                        # Calculate and charge commission
                        commission = self.commission_model.calculate(order=order, fill_price=order.take_profit) # Do I need to change the signature?
                        portfolio.charge_commission(commission)
                        # Sell out of position
                        exit_price = order.take_profit - self.slippage_model.calculate(order) # SIGN?
                        liquidated_positions.append(PositionLiquidation(fill, order.take_profit, self.market_data.cur_date))
                        # pop out fill
                        self.active_positions.remove(fill)
                        continue

                    elif ticker_data["low"] <= order.stop_loss:
                        # Calculate and charge commission
                        commission = self.commission_model.calculate(order=order, fill_price=order.stop_loss) # Do I need to change the signature?
                        portfolio.charge_commission(commission)
                        # sell out of position
                        exit_price = order.stop_loss - self.slippage_model.calculate(order) # SIGN?
                        liquidated_positions.append(PositionLiquidation(fill, exit_price, self.market_data.cur_date))

                        # pop out fill from active_positions
                        self.active_positions.remove(fill)
                        continue

                # if noe of the above has triggered we check if the timeout has been reached, and give the PositionLiquidation a price equal to the closing price
                if self.market_data.cur_date >= order.time_out:
                    # Calculate and charge commission
                    commission = self.commission_model.calculate(order=order, fill_price=ticker_data["close"]) # Do I need to change the signature?
                    portfolio.charge_commission(commission)

                    exit_price = ticker_data["close"] - self.slippage_model.calculate(order)
                    liquidated_positions.append(PositionLiquidation(fill, exit_price, self.market_data.cur_date))    
                    
                    self.active_positions.remove(fill)
                    continue

            elif order.direction == -1:
                # Short position
                if price_direction_for_the_day == 1:
                    # Prices rose over the day, low was hit first
                    if ticker_data["low"] <= order.take_profit:
                        # Calculate and charge commission
                        commission = self.commission_model.calculate(order=order, fill_price=order.take_profit) # Do I need to change the signature?
                        portfolio.charge_commission(commission)

                        exit_price = order.take_profit + self.slippage_model.calculate(order) # SIGN?
                        liquidated_positions.append(PositionLiquidation(fill, exit_price, self.market_data.cur_date))

                        self.active_positions.remove(fill)                        
                        continue
                    elif ticker_data["high"] >= order.stop_loss: # For short positions, stop loss wil be higher than the fill price
                        # Calculate and charge commission
                        commission = self.commission_model.calculate(order=order, fill_price=order.stop_loss) # Do I need to change the signature?
                        portfolio.charge_commission(commission)

                        exit_price = order.stop_loss + self.slippage_model.calculate(order) # SIGN?
                        liquidated_positions.append(PositionLiquidation(fill, exit_price, self.market_data.cur_date))

                        self.active_positions.remove(fill)                        
                        continue
                    
                elif price_direction_for_the_day == -1:
                    # price declined over the day, high was hit first.
                    if ticker_data["high"] >= order.stop_loss: # For short positions, stop loss wil be higher than the fill price
                        # Calculate and charge commission
                        commission = self.commission_model.calculate(order=order, fill_price=order.stop_loss) # Do I need to change the signature?
                        portfolio.charge_commission(commission)

                        exit_price = order.stop_loss + self.slippage_model.calculate(order) # SIGN?
                        liquidated_positions.append(PositionLiquidation(fill, exit_price, self.market_data.cur_date))

                        self.active_positions.remove(fill) 
                        continue
                    elif ticker_data["low"] <= order.take_profit:
                        # Calculate and charge commission
                        commission = self.commission_model.calculate(order=order, fill_price=order.take_profit) # Do I need to change the signature?
                        portfolio.charge_commission(commission)

                        exit_price = order.take_profit + self.slippage_model.calculate(order) # SIGN?
                        liquidated_positions.append(PositionLiquidation(fill, exit_price, self.market_data.cur_date))

                        self.active_positions.remove(fill)                        
                        continue

                # if noe of the above has triggered we check if the timeout has been reached, and give the PositionLiquidation a price equal to the closing price
                if self.market_data.cur_date >= order.timeout:
                    # Calculate and charge commission
                    commission = self.commission_model.calculate(order=order, fill_price=ticker_data["close"]) # Do I need to change the signature?
                    portfolio.charge_commission(commission)

                    exit_price = order.take_profit + self.slippage_model.calculate() # SIGN?
                    liquidated_positions.append(PositionLiquidation(fill, exit_price, self.market_data.cur_date))    
                    
                    self.active_positions.remove(fill)
                    continue
            
            # Don't do anything if no exit condition was met

        # Margin account size
        required_margin_account_size = self.calculate_required_margin_account_size() # Update based on current positions
        margin_interest = self.calculate_margin_interest()

        if margin_interest != 0:
            portfolio.charge_margin_interest(margin_interest) # I guess I allways want this to succeed


        if len(liquidated_positions) > 0:
            position_liquidations_event = Event(event_type="POSITION_LIQUIDATIONS", data=liquidated_positions, date=self.market_data.cur_date)
        else:
            position_liquidations_event = None

        margin_account_update_event = Event(event_type="MARGIN_ACCOUNT_UPDATE", data=required_margin_account_size, date=self.market_data.cur_date)
        
        return (position_liquidations_event, margin_account_update_event)



    def get_most_updated_close_for_ticker(self, ticker):
        try:
            ticker_data = self.market_data.current_for_ticker(ticker)
        except:
            try:
                ticker_data = self.market_data.last_available_for_ticker(ticker) # NOT IMPLEMENTED
            except:
                ticker_data = None

        if ticker_data is None:
            cur_price = None
        else:
            cur_price = ticker_data["close"]

        return cur_price

    def calculate_initial_margin_requirement(self, order):
        cur_price = self.get_most_updated_close_for_ticker(order.ticker) # the price should allways be available at this point...
        if cur_price is None:
            raise MarketDataNotAvailableError("Failed to calculate initial margin requirement, because data not avialable for ticker {} on {}".format(order.ticker, self.market_data.cur_date))

        if order.direction != -1:
            raise ValueError("Cannot calculate initial margin requirement for order with positive direction.")

        short_position_value = cur_price * abs(order.amount)
        additional_margin_requirement = short_position_value * self.initial_margin_requirement 
        total_margin_requirement = short_position_value + additional_margin_requirement

        return total_margin_requirement

    def calculate_required_margin_account_size(self):
        """
        Calculate the required contribution to the margin account for each short position in the portfolio.

        I assume that there will be an initial margin requirement and a minimum margin requirement associated with each stock.
        The margin requirement for each stock is added together to yield the portfolio margin requirement.
        It is only when the short position is first taken that the margin account must meet the initial margin requirement.
        So only the maintenance requirement is used in the calculations here.

        https://www.investopedia.com/ask/answers/05/shortmarginrequirements.asp

        Margin accounts must maintain a certain margin ratio at all times. If the account value falls 
        below this limit, the client is issued a margin call, which is a demand for deposit of more cash
        or securities to bring the account value back within the limits. The client can add new cash to
        his account or sell some of his holdings to raise the cash.
        
        Whenever the price falls, investors are still required to have meet the initial margin requirement.
        When the price increases the maintenance margin is used -> need to see if the position is inn or out of profit...
        """


        required_margin_account_size = 0

        for fill in self.active_positions:
            if fill.order.direction == -1:
                cur_price = self.get_most_updated_close_for_ticker(fill.ticker)
                if cur_price is None:
                    cur_price = fill.price

                # In the money?
                in_the_money = True if (cur_price < fill.price) else False

                short_position_value = cur_price * fill.amount
                
                if in_the_money:
                    additional_margin_requirement = short_position_value * self.initial_margin_requirement 
                    ticker_margin_requirement = short_position_value + additional_margin_requirement
                else:
                    additional_margin_requirement = short_position_value * self.maintenance_margin_requirement 
                    ticker_margin_requirement = short_position_value + additional_margin_requirement

                required_margin_account_size += ticker_margin_requirement

                # less his margin interest charges over that period of time

        return required_margin_account_size

    # OBS: pay interest on non-business days...
    def calculate_margin_interest(self): 
        """
        Calculate the daily interest expense to charge the portfolio for any borrowed funds for short positions.

        I assume the interest rate are on the money loaned to originally get a hold of the share that was sold and will
        later be bought back to terminate the position.
        """
        amount_borrowed = 0

        for fill in self.active_positions:
            if fill.direction == -1:
                amount_borrowed += abs(fill.amount * fill.price)

        annual_payment = amount_borrowed * self.annual_margin_interest_rate
        daily_payment = annual_payment / 360

        return daily_payment


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



    def capture_state(self):
        """
        Adds the the current state of active positions to the active_positions_history list with some identifying attributes.
        """
        # Capture active positions
        history_obj = {
            "date": self.market_data.cur_date,
            "active_positions": copy.deepcopy(self.active_positions),
        }
        self.active_positions_history.append(history_obj)


    def cancelled_orders_to_df(self):
        df = pd.DataFrame(columns=["order_id", "date", "ticker", "amount", "error"])
        for index, c_ord in enumerate(self.cancelled_orders):
            df.at[index, "order_id"] = c_ord.order_id
            df.at[index, "date"] = c_ord.date
            df.at[index, "ticker"] = c_ord.ticker
            df.at[index, "amount"] = c_ord.order.amount
            df.at[index, "error"] = str(c_ord.error)
            # Might want to extend this, don't know atm

        df = df.sort_values(by=["order_id"])

        return df

    def active_positions_history_to_df(self):
        df = pd.DataFrame(columns=["date", "ticker", "direction", "amount", "order_id", "fill_date"])

        index = 0

        for i, history_obj in enumerate(self.active_positions_history):
            for j, fill in enumerate(history_obj["active_positions"]):
                df.at[index, "date"] = history_obj["date"]
                df.at[index, "ticker"] = fill.ticker
                df.at[index, "direction"] = fill.direction
                df.at[index, "amount"] = fill.amount
                df.at[index, "order_id"] = fill.order_id
                df.at[index, "fill_date"] = fill.date
                """
                # May want more of the order properties here, but not sure
                df.at[index, "price"] = fill.price
                df.at[index, "commission"] = fill.commission
                df.at[index, "slippage"] = fill.slippage
                """
                index += 1
        
        df = df.sort_values(by=["date"]) # This may not be the order things are executed? (or will it be? I decide the order the orders are processed, so I guess, the order should be filled in order.)

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
        
        df = df.sort_values(by=["order_id"]) # This may not be the order things are executed? (or will it be? I decide the order the orders are processed, so I guess, the order should be filled in order.)

        return df



    # ___________ NOT USED ATM ____________________
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
    def __init__(self, fill, price, date):
        order = fill.order

        self.order = order
        self.fill = fill
        self.ticker = order.ticker
        self.amount = order.amount # Only supports selling all shares associated with an order, somewhat complicated to move away from this now...
        self.direction = order.direction
        self.fill_price = fill.price
        self.liquidation_price = price
        self.date = date



class TerminatedPosition():
    def __init__(self, fill: Fill, reason):
        # Any data I want to expose directly....
        self.fill = fill
        self.reason = reason

"""
class OrderCancellationPolicy(): 
    pass
"""

