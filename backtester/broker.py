import pandas as pd
import copy
import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, ".."))


from event import Event
from utils.logger import Logger
from utils.errors import OrderProcessingError, MarketDataNotAvailableError, BalanceTooLowError
from order import Order, CancelledOrder
from utils.types import PortfolioBase, CommissionModel, SlippageModel
from data_handler import DataHandler

class Trade():
    def __init__(
        self, 
        order: Order, 
        date: pd.datetime, 
        fill_price: float, 
        commission: float, 
        slippage: float
    ):
        self.order = order
        self.order_id = order.id
        
        self.ticker = order.ticker
        self.direction = order.direction # Strictly unnecessary, but makes this information easily accessable.
        self.amount = order.amount
        self.stop_loss = order.stop_loss # This needs to be a price that I can compare the open/close to
        self.take_profit = order.take_profit
        self.timeout = order.timeout

        self.date = date
        self.fill_price = fill_price
        self.commission = commission
        self.slippage = slippage
        
        self.interest_expenses = 0
        self.dividends_per_share = 0

        self.cur_value = None

        # Close related
        self.CLOSED = False
        self.close_price = None
        self.close_date = None
        self.close_cause = ""
        self.ret = None
        self.total_ret = None
        self.total_dividends = None
        
    def close(self, close_price: float, close_date: pd.datetime, close_cause: str=""):
        self.close_price = close_price
        self.close_date = close_date
        self.close_cause = close_cause

        self.ret = ((self.close_price / self.fill_price) - 1) * self.direction
        self.total_ret = (((self.close_price + self.dividends_per_share) / self.fill_price) - 1) * self.direction
        self.total_dividends = self.dividends_per_share * self.amount
        
        self.CLOSED = True

    def return_if_close_price_is(self, close_price: float):
        """
        Returns the trades total return if the close price is as provided.
        NOTE: Why not just return total return including dividends, when comparing to exit rules, these are allready derived from price
        series that included dividends.
        """
        # NOTE: I guess I can disguard the direction when writing the formulas like this.
        if self.direction == 1:
            return (((close_price + self.dividends_per_share) / self.fill_price) - 1) * self.direction

        elif self.direction == -1:
            return (((close_price + self.dividends_per_share) / self.fill_price) - 1) * self.direction

            # 9 / 10  - 1  * -1 = 0.9 -1 * -1 = -0.1 * -1 = 0.1
            # 1 / 10 * (-1)  = -0.1
            # 9 + 1 / 10 - 1 *-1 = 1 - 1 * (-1) = 0

    def get_proceeds(self):
        if not self.CLOSED:
            raise ValueError("Trade must be closed to calculate proceeds from closing the position")

        if self.direction == 1:
            return self.amount * self.close_price
        elif self.direction == -1:
            dollar_size_of_original_position = self.fill_price * abs(self.amount)
            return (((self.close_price/self.fill_price) - 1) * self.direction) * dollar_size_of_original_position

    def price_at_exit(self, exit_rule_hit: str):
        if self.direction == 1:
            if exit_rule_hit == "take_profit":
                return self.fill_price * (1 + self.take_profit) - self.dividends_per_share
            elif exit_rule_hit == "stop_loss":
                return self.fill_price * (1 + self.stop_loss) - self.dividends_per_share

        elif self.direction == -1:
            if exit_rule_hit == "take_profit":
                return self.fill_price * (1 + (self.take_profit * self.direction)) - self.dividends_per_share
                #       10 *(1+(0.1 *-1)) = 10 * 0.9 = 9
                #       10 * (1 - 0.1) + dps = 10 => dps must be - 1 => - self.dividends_per_share

            elif exit_rule_hit == "stop_loss":
                return self.fill_price * (1 + (self.stop_loss * self.direction)) - self.dividends_per_share
                #       10 *(1+ (+0.3)) = 13 

    def calculate_pnl(self):
        if self.CLOSED == False:
            raise ValueError("Trade must be closed to calculate PnL")

        return self.total_ret * self.fill_price * self.amount

    def get_total_slippage(self):
        return self.slippage * abs(self.amount)


class Blotter():
    def __init__(self):
        self.active_trades = []
        self.closed_trades = []
        self.cancelled_orders = []
    
    # NOTE: Needs improvement, no not
    def close(self, trade: Trade, close_price: float, close_date: pd.datetime, close_cause: str=""):
        trade.close(close_price, close_date, close_cause)
        try:
            self.active_trades.remove(trade)
            self.closed_trades.append(trade)
        except ValueError as e:
            raise e
        
    def get_active_trades(self) -> list:
        return copy.deepcopy(self.active_trades)

    def get_closed_trades(self) -> list:
        return copy.deepcopy(self.closed_trades)




class Broker():
    """
    Execute orders from a portfolio/strategy and maintain active trades.
    """
    def __init__(
        self, 
        market_data: DataHandler, 
        commission_model: CommissionModel,
        slippage_model: SlippageModel, 
        log_path: str,
        annual_margin_interest_rate: float,
        initial_margin_requirement: float, 
        maintenance_margin_requirement: float,
        tax_rate: float=0.25, # Tax rate on dividends, depends on the tax-bracket of the individual (assumed to be 25%)
    ):
        
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.market_data = market_data
        self.log_path = log_path
        self.logger = Logger("BROKER", log_path + "/broker.log")

        self.annual_margin_interest_rate = annual_margin_interest_rate
        self.initial_margin_requirement = initial_margin_requirement
        self.maintenance_margin_requirement = maintenance_margin_requirement
        self.tax_rate = tax_rate
        
        # The broker gets orders, if it gets filled a Trade object is created and added to the Blotter
        # if the order is cancelled/fails then a CancelledOrder object is created and appended to cancelled_orders
        self.cancelled_orders = [] # This is more relevant for the portfolio maybe?
        
        # The broker needs to have a record of active position, because he needs to check this for exit rules
        self.blotter = Blotter()

        self.blotter_history = [] # Blotter.active_trades at each date...


    def process_orders(self, portfolio, orders_event):
        orders = orders_event.data

        trade_objects = []
        cancelled_orders = []

        for order in orders:
            try:
                trade_object = self._process_order(portfolio, order)
            except OrderProcessingError as e:
                Logger.logr.warning("Failed processing order with error: {}".format(e)) # Show log from backtest in the dashboard...

                cancelled_order = CancelledOrder(order, e)
                cancelled_orders.append(cancelled_order)
            else:
                trade_objects.append(trade_object)

                self.blotter.active_trades.append(trade_object)

        self.blotter.cancelled_orders.extend(cancelled_orders)

        if len(cancelled_orders) == 0:
            cancelled_orders_event = None
        else:
            cancelled_orders_event = Event(event_type="CANCELLED_ORDERS", data=cancelled_orders, date=orders_event.date)

        if len(trade_objects) == 0:
            trades_event = None
        else:
            trades_event =  Event(event_type="TRADES", data=trade_objects, date=orders_event.date)

        return (trades_event, cancelled_orders_event)


    def _process_order(self, portfolio, order):
        if not self.market_data.cur_date == order.date:
            Logger.logr.warning("When processing an order; it could not be completed, because order.date == market_data.cur_date.")
            raise OrderProcessingError("Cannot complete order, because order.date == market_data.cur_date.")

        # To process an order we must be able to trade the stock, which means data is available on the date and not bankrupt or delisted.
        if not self.market_data.can_trade(order.ticker):
            Logger.logr.warning("When processing an order; it could not be completed, because market_data.can_trade() returned False for ticker {} on date {}.".format(order.ticker, self.market_data.cur_date))
            raise OrderProcessingError("Cannot complete order, because market_data.can_trade() returned False.")

        try: 
            stock_price = self.market_data.current_for_ticker(order.ticker)["open"]
        except:
            # NOTE: Should  not be possible if can_trade returns true...but still for consistency I code it this way.
            Logger.logr.warning("When processing an order; market data was not available for ticker {} on date {}.".format(order.ticker, self.market_data.cur_date))
            raise OrderProcessingError("Cannot complete order, because market data is not available.")


        if order.direction == 1:
            slippage = self.slippage_model.calculate()
            fill_price = stock_price + slippage
            cost = order.amount * fill_price
            commission = self.commission_model.calculate(order.amount, fill_price) # The commission is also based on the fill_price
            
            # Charge portfolio etc. and if successfull, "complete" the order by appending to active_orders
            
            # NOTE: Need to roll back if failing in two step procedures

            try:
                portfolio.charge(cost)
            except BalanceTooLowError as e:
                Logger.logr.warning("Balance too low! Balance: {}, wanted to charge {} for the stock".format(portfolio.balance, cost))
                raise OrderProcessingError("Cannot complete order, with error: {}".format(e))

            portfolio.charge_commission(commission) # Does not fail
            
            return Trade(
                order=order,
                date=self.market_data.cur_date,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage,
            )
        

        elif order.direction == -1:
            new_required_margin_account_size = self.calculate_required_margin_account_size("open", [order])
            slippage = self.slippage_model.calculate()
            fill_price = stock_price - slippage
            commission = self.commission_model.calculate(order.amount, fill_price)

            try:
                portfolio.update_margin_account(new_required_margin_account_size)
            except BalanceTooLowError as e:
                Logger.logr.warning("Balance too low! Balance: {}, Margin Account: {}, wanted to update margin account to {}".format(
                        portfolio.balance, portfolio.margin_account, new_required_margin_account_size
                    )
                )
                raise OrderProcessingError("Cannot complete order, with error: {}".format(e))

            portfolio.charge_commission(commission) # Does not fail

            return Trade(
                order=order,
                date=self.market_data.cur_date,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage,
            )
        
    def manage_active_trades(self, portfolio: PortfolioBase):
        """
        Active positions are represented by fill objects (contains associated order) where the resulting position have not yet been
        liquidated. Every day the active positions must be checked to see if a stop-loss, take-profit or timeout would trigger an exit.
        
        Also as the price of the stocks in the portfolio changes the margin account size must be updated so meet the requirements
        for the short positions.
        
        This is somewhat different than the state of the portfolio, because each order must be treated by the broker independently (not
        summed together like is done with portfolio.portfolio)
        """
        
        # NOTE: Deal with bankruptices and delistings and dividends?
        
        closed_trades = []

        for _, trade in enumerate(self.blotter.active_trades):
            order = trade.order
            ticker = trade.ticker

            try:
                ticker_data = self.market_data.current_for_ticker(ticker) # NOTE: need to decide how to manage when we cannot deal in a stock (can_trade...)
            except MarketDataNotAvailableError:
                Logger.logr.warning("Failed to manage active position, because market data not available for ticker {} on date {}".format(ticker, self.market_data.cur_date))
                continue

            price_direction_for_the_day = 1 if (ticker_data["open"] <= ticker_data["close"]) else -1

            # Long position
            if trade.direction == 1:
                # If price went up over the day, assume the low was hit before the high.
                exited = False
                if price_direction_for_the_day == 1:
                    if trade.return_if_close_price_is(ticker_data["low"]) <= trade.stop_loss: # How is total return calculated given long/short and dividends, how is the sign set on exit limits? I just need to decide
                        close_price = trade.price_at_exit("stop_loss") - self.slippage_model.calculate()
                        close_cause = "STOP_LOSS_REACHED"                        
                        exited = True
                    elif trade.return_if_close_price_is(ticker_data["high"]) >= trade.take_profit:
                        close_price = trade.price_at_exit("take_profit") - self.slippage_model.calculate()
                        close_cause = "TAKE_PROFIT_REACHED"
                        exited = True
                    
                elif price_direction_for_the_day == -1:
                    # price declined over the day, high was hit first.
                    if trade.return_if_close_price_is(ticker_data["high"]) >= trade.take_profit:
                        close_price = trade.price_at_exit("take_profit") - self.slippage_model.calculate()
                        close_cause = "TAKE_PROFIT_REACHED"
                        exited = True
                        
                    elif trade.return_if_close_price_is(ticker_data["low"]) <= trade.stop_loss:
                        close_price = trade.price_at_exit("stop_loss") - self.slippage_model.calculate()                        
                        close_cause = "STOP_LOSS_REACHED"                        
                        exited = True
                
                # Check if the timeout has been reached, and give the trade an close_price a price equal to the closing price
                if self.market_data.cur_date >= trade.timeout:
                    close_price = ticker_data["close"] - self.slippage_model.calculate()
                    close_cause = "TIMEOUT_REACHED"
                    exited = True


                if exited:
                    commission = self.commission_model.calculate(amount=order.amount, price=close_price)
                    self.blotter.close(trade, close_price, self.market_data.cur_date, close_cause)
                    portfolio.charge_commission(commission)
                    portfolio.receive_proceeds(trade.get_proceeds())
                    continue


            # Short position
            elif trade.direction == -1:
                # Prices rose over the day, low was hit first
                exited = False
                if price_direction_for_the_day == 1:
                    if trade.return_if_close_price_is(ticker_data["low"]) >= trade.take_profit:
                        close_price = trade.price_at_exit("take_profit") - self.slippage_model.calculate()
                        close_cause = "TAKE_PROFIT_REACHED" 
                        exited = True                
                    elif trade.return_if_close_price_is(ticker_data["high"]) <= trade.stop_loss:
                        close_price = trade.price_at_exit("stop_loss") - self.slippage_model.calculate()
                        close_cause = "STOP_LOSS_REACHED"               
                        exited = True
                    
                # Prices declined over the day, high was hit first
                elif price_direction_for_the_day == -1:
                    if trade.return_if_close_price_is(ticker_data["high"]) <= trade.stop_loss:
                        close_price = trade.price_at_exit("stop_loss") - self.slippage_model.calculate()
                        close_cause = "STOP_LOSS_REACHED"           
                        exited = True
                    elif trade.return_if_close_price_is(ticker_data["low"]) >= trade.take_profit:
                        close_price = trade.price_at_exit("take_profit") - self.slippage_model.calculate()
                        close_cause = "TAKE_PROFIT_REACHED"
                        exited = True

                if self.market_data.cur_date >= trade.timeout:
                    close_price = ticker_data["close"] + self.slippage_model.calculate()
                    close_cause = "TIMEOUT_REACHED"
                    exited = True

                if exited:
                    commission = self.commission_model.calculate(amount=trade.amount, price=close_price, info={
                        "ticker": trade.ticker,
                        "fill_price": trade.fill_price,
                        "stop_loss": trade.stop_loss,
                        "take_profit": trade.take_profit,
                        "price_at_exit_take_profit": trade.price_at_exit("take_profit"),
                        "price_at_exit_stop_loss": trade.price_at_exit("stop_loss"),
                    })
                    portfolio.charge_commission(commission)
                    self.blotter.close(trade, close_price, self.market_data.cur_date, close_cause)
                    proceeds = trade.get_proceeds()
                    # if in profit -> give return to balance
                    if proceeds >= 0:
                        portfolio.receive_proceeds(proceeds)
                    # If in loss -> cover losses with margin account
                    if proceeds < 0:
                        portfolio.charge_margin_account(abs(proceeds))
                    continue
        # After all trades have been checked for any exit conditions and the active trades have been updated, the margin account size is 
        # updated. This is done every business day regardless of the active trades changing.
        # NOTE: When exiting a short, you cannot fail to adjust the margin account.
        required_margin_account_size = self.calculate_required_margin_account_size("close")

        return Event(event_type="MARGIN_ACCOUNT_UPDATE", data=required_margin_account_size, date=self.market_data.cur_date)

        

    def handle_corp_actions(self, portfolio: PortfolioBase):
        """
        Handles bankruptices and delistings.
        """
        corp_actions: pd.DataFrame = self.market_data.current_corp_actions()

        for _, row in corp_actions.iterrows():
            for trade in self.blotter.active_trades:
                if trade.ticker == row["ticker"]:
                    # Ticker was delited or went bankrupt
                    if row["action"] == "bankruptcy":
                        close_price = 0
                        self.blotter.close(trade, close_price, self.market_data.cur_date, "BANKRUPT")
                    elif row["action"] == "delisted":
                        close_price = self.market_data.current_for_ticker(trade.ticker)["close"]
                        self.blotter.close(trade, close_price, self.market_data.cur_date, "DELISTED")
                        

    def handle_dividends(self, portfolio: PortfolioBase):
        """
        Handle dividends for both short and long trades.
        Long trades results in dividends being payed to the portfolios balance.
        Short trades results in the portfolio having to pay the dividend amount.
        """
        dividends = self.market_data.current_dividends()

        # print("dividends: ", dividends)
        for ticker, row in dividends.iterrows():
            for trade in self.blotter.active_trades:
                if ticker == trade.ticker:
                    trade.dividends_per_share += row["dividends"]
                    dividend_amount = row["dividends"] * abs(trade.amount) * (1 - self.tax_rate)
                    # print("dividend amount: ", dividend_amount)
                    if trade.direction == 1:
                        portfolio.receive_dividends(dividend_amount)
                    elif trade.direction == -1:
                        portfolio.charge_for_dividends(dividend_amount)


    def handle_interest_on_short_positions(self, portfolio: PortfolioBase):
        margin_interest = self.calculate_margin_interest()
        if margin_interest != 0:
            portfolio.charge_margin_interest(margin_interest) # NOTE: I allways want this to succeed


    def handle_interest_on_cash_and_margin_accounts(self, portfolio: PortfolioBase):
        daily_rate = self.market_data.get_daily_interest_rate()
        total = (portfolio.balance + portfolio.margin_account) * daily_rate
        if total < 0:
            # NOTE: This should not be happen often
            portfolio.charge_interest(abs(total)) # If accounts have negative net balance, the portfolio must pay interest at the risk free rate
        if total > 0:
            portfolio.receive_interest(total)


    # ___________ MARGIN ACCOUNT METHODS _________________

    def calculate_required_margin_account_size(self, time_of_day="open", new_orders=[]):
        """
        Calculate the required contribution to the margin account for each short position in the portfolio.

        I assume that there will be an initial margin requirement and a minimum margin requirement associated with each stock.
        The margin requirement for each stock is added together to yield the portfolio margin requirement.

        Margin accounts must maintain a certain margin ratio at all times. If the account value falls 
        below this limit, the client is issued a margin call, which is a demand for deposit of more cash
        or securities to bring the account value back within the limits. The client can add new cash to
        his account or sell some of his holdings to raise the cash.
        
        Whenever the price falls, investors are still required to have meet the initial margin requirement.
        When the price increases the maintenance margin is used -> need to see if the position is inn or out of profit.

        Formulas are inspired by the following article:
        https://www.investopedia.com/ask/answers/05/shortmarginrequirements.asp
        """

        required_margin_account_size = 0

        # Add margin requirement for new trades
        for order in new_orders:
            if order.direction == -1:
                cur_price = self.market_data.current_for_ticker(order.ticker)[time_of_day] # NOTE: Do I need to revise?
                if cur_price is None:
                    raise MarketDataNotAvailableError("Failed to calculate order margin requirement, because data not available for ticker {} on {}".format(order.ticker, self.market_data.cur_date))
                
                order_margin_requirement = cur_price * abs(order.amount) * (1 + self.initial_margin_requirement)
                
                required_margin_account_size += order_margin_requirement

        # Add margin requirement for active trades
        for trade in self.blotter.active_trades:
            if trade.direction == -1:
                cur_price = self.market_data.current_for_ticker(trade.ticker)[time_of_day] # This can fail, but should not with forward filled price data
                if cur_price is None:
                    cur_price = trade.fill_price

                # In the money?
                in_the_money = True if (cur_price <= trade.fill_price) else False # NOTE: Does not take into account dividends, but I think that is OK

                short_position_value = cur_price * abs(trade.amount)
                
                if in_the_money:
                    trade_margin_requirement = short_position_value * (1 + self.initial_margin_requirement)
                else:
                    trade_margin_requirement = short_position_value * (1 + self.maintenance_margin_requirement)

                required_margin_account_size += trade_margin_requirement


        return required_margin_account_size


    def calculate_margin_interest(self): 
        """
        Calculate the daily interest expense to charge the portfolio for any borrowed funds for short positions.
        I assume the interest rate are on the money loaned to originally get a hold of the share that was sold and will
        later be bought back to close the trade.
        """
        amount_borrowed = 0
        daily_rate = self.annual_margin_interest_rate / 360

        for trade in self.blotter.active_trades:
            if trade.direction == -1:
                amount_borrowed += abs(trade.amount) * trade.fill_price
                trade.interest_expenses += abs(trade.amount) * trade.fill_price * daily_rate

        return amount_borrowed * daily_rate


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


    # __________ METHOD TO CAPTURE STATE AND RECORD FINAL STATE ________________

    def capture_state(self):
        """
        Adds the the current state of active positions to the active_positions_history list with some identifying attributes.
        Call at the end of each rebalancing date.
        """

        active_trades = self.blotter.get_active_trades()
        for trade in active_trades:
            # Get price
            price = self.market_data.current_for_ticker(trade.ticker)["close"]
            # Add value to trade
            if trade.direction == 1:
                trade.cur_value = trade.amount * price
            elif trade.direction == -1:
                trade.cur_value = abs(trade.amount) * (price - trade.fill_price)


        history_obj = {
            "date": self.market_data.cur_date,
            "active_trades": active_trades,
        }
        self.blotter_history.append(history_obj)



    def blotter_history_to_df(self):
        df = pd.DataFrame(columns=["order_id", "ticker", "direction", "amount", "stop_loss", "take_profit", "timeout", "date", "fill_price",\
            "commission", "slippage", "interest_expenses", "dividends_per_share", "CLOSED", "close_price", "close_date", "close_cause", \
                "ret", "total_ret", "total_dividends"])
        index = 0
        for _, history_obj in enumerate(self.blotter_history):
            for j, trade in enumerate(history_obj["active_trades"]):
                df.at[index, "date"] = history_obj["date"]
                df.at[index, "order_id"] = trade.order_id
                df.at[index, "ticker"] = trade.ticker
                df.at[index, "direction"] = trade.direction
                df.at[index, "amount"] = trade.amount
                df.at[index, "stop_loss"] = trade.stop_loss
                df.at[index, "take_profit"] = trade.take_profit
                df.at[index, "timeout"] = trade.timeout
                df.at[index, "trade_date"] = trade.date
                df.at[index, "fill_price"] = trade.fill_price
                df.at[index, "commission"] = trade.commission
                df.at[index, "slippage"] = trade.slippage
                df.at[index, "interest_expenses"] = trade.interest_expenses
                df.at[index, "dividends_per_share"] = trade.dividends_per_share
                df.at[index, "cur_value"] = trade.cur_value # Added when capturing state
                df.at[index, "CLOSED"] = trade.CLOSED
                df.at[index, "close_price"] = trade.close_price
                df.at[index, "close_date"] = trade.close_date
                df.at[index, "close_cause"] = trade.close_cause
                df.at[index, "ret"] = trade.ret
                df.at[index, "total_ret"] = trade.total_ret
                df.at[index, "total_dividends"] = trade.total_dividends

                index += 1
        
        df = df.sort_values(by=["date", "trade_date"]) 
        return df
    
    def all_trades_to_df(self):
        trades = self.blotter.get_active_trades()
        trades.extend(self.blotter.get_closed_trades())
        df = pd.DataFrame(columns=["order_id", "ticker", "direction", "amount", "stop_loss", "take_profit", "timeout", "date", "fill_price",\
            "commission", "slippage", "interest_expenses", "dividends_per_share", "CLOSED", "close_price", "close_date", "close_cause", \
                "ret", "total_ret", "total_dividends"])
        index = 0
        for trade in trades:
            df.at[index, "order_id"] = trade.order_id
            df.at[index, "ticker"] = trade.ticker
            df.at[index, "direction"] = trade.direction
            df.at[index, "amount"] = trade.amount
            df.at[index, "stop_loss"] = trade.stop_loss
            df.at[index, "take_profit"] = trade.take_profit
            df.at[index, "timeout"] = trade.timeout
            df.at[index, "trade_date"] = trade.date
            df.at[index, "fill_price"] = trade.fill_price
            df.at[index, "commission"] = trade.commission
            df.at[index, "slippage"] = trade.slippage
            df.at[index, "interest_expenses"] = trade.interest_expenses
            df.at[index, "dividends_per_share"] = trade.dividends_per_share
            df.at[index, "CLOSED"] = trade.CLOSED
            df.at[index, "close_price"] = trade.close_price
            df.at[index, "close_date"] = trade.close_date
            df.at[index, "close_cause"] = trade.close_cause
            df.at[index, "ret"] = trade.ret
            df.at[index, "total_ret"] = trade.total_ret
            df.at[index, "total_dividends"] = trade.total_dividends

            index += 1

        df = df.sort_values(by=["date"])
        return df

    def all_trades_as_objects(self):
        trades = self.blotter.get_active_trades()
        trades.extend(self.blotter.get_closed_trades())

        return trades

    def cancelled_orders_to_df(self):
        df = pd.DataFrame(columns=["order_id", "ticker", "date", "error", "amount", "direction", "stop_loss", "take_profit", "timeout", "type"])
        
        for index, c_ord in enumerate(self.cancelled_orders):
            df.at[index, "order_id"] = c_ord.order_id
            df.at[index, "date"] = c_ord.date
            df.at[index, "ticker"] = c_ord.ticker
            df.at[index, "amount"] = c_ord.order.amount
            df.at[index, "error"] = str(c_ord.error)
            df.at[index, "order_direction"] = c_ord.order.direction
            df.at[index, "order_stop_loss"] = c_ord.order.stop_loss
            df.at[index, "order_take_profit"] = c_ord.order.take_profit
            df.at[index, "order_timeout"] = c_ord.order.timeout
            df.at[index, "order_type"] = c_ord.order.type

        df = df.sort_values(by=["order_id"])
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



""" # Dont think this is needed
class OrderCancellationPolicy(): 
    pass
"""


"""
def calculate_initial_margin_requirement(self, order):
    # cur_price = self.get_most_updated_close_for_ticker(order.ticker) # the price should allways be available at this point...
    cur_price = self.market_data.current_for_ticker(order.ticker)["open"] # Trades are always initiated at the open
    if cur_price is None:
        raise MarketDataNotAvailableError("Failed to calculate initial margin requirement, because data not available for ticker {} on {}".format(order.ticker, self.market_data.cur_date))

    if order.direction != -1:
        raise ValueError("Cannot calculate initial margin requirement for order with positive direction.")

    short_position_value = cur_price * abs(order.amount)
    return short_position_value * (1 + self.initial_margin_requirement)
"""