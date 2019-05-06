from abc import ABC, abstractmethod
import math
import pandas as pd
import copy
from dateutil.relativedelta import *
import random

from helpers import sign
# from strategy import Strategy, Signal
from event import Event
from errors import MarketDataNotAvailableError, BalanceTooLowError
from logger import Logger
from utils.utils import Signal, Strategy


class Portfolio():
    def __init__(self, market_data, broker, balance: float, strategy: Strategy, initial_margin_requirement=0.5, maintenance_margin_requirement=0.3):
        self.market_data = market_data # Shared with backtester and broker
        self.broker = broker # this makes it possible for the portfolio to query the broker for active trades.

        # self.account = Account(balance)

        self.balance = balance
        # The margin_account size changes dynamically as stock prices fluctuate...
        self.margin_account = 0 # cash
        
        # Maybe this is best enforced by the broker, but the portfolio must be aware of it to properly generate orders and deposit money to the margin account
        self.initial_margin_requirement = initial_margin_requirement
        self.maintenance_margin_requirement = maintenance_margin_requirement

        # Dont really need this atm, but I can see that it might be needed for order generation
        # self.portfolio = {} # ticker -> position is the wording used in the project
    
        # self.active_positions = [] # fills, same as active positions in the broker
        # self.active_positions_history = []

        self.signals = [] # All signals ever produced
        self.blotter = [] # Fills

        self.strategy = strategy
        self.order_history = [] # All orders ever created (Not all gets sent to the broker necessarily (but most will I envision, why not all?))
        self.cancelled_orders = [] # Cancelled orders
        
        # self.generate_orders_from_signals = strategy.generate_orders_from_signals

        # self.portfolio_history = [] # To be able to play through how the portfolio developed over time.

        self.commissions_charged = pd.DataFrame(index=self.market_data.date_index_to_iterate) # Not implemented
        self.commissions_charged["amount"] = 0
        self.slippage_suffered = pd.DataFrame(index=self.market_data.date_index_to_iterate)
        self.slippage_suffered["amount"] = 0
        
        # Can the above just be merged with the portfolio_value dataframe?
        self.portfolio_value = pd.DataFrame(index=self.market_data.date_index_to_iterate) # This is the basis for portfolio return calculations
        self.portfolio_value["balance"] = pd.NaT
        self.portfolio_value["margin_account"] = pd.NaT
        self.portfolio_value["market_value"] = pd.NaT
        

        self.total_charged_for_stock_purchases = 0
        self.total_commission = 0
        self.total_margin_interest = 0



    def generate_signals(self, feature_data_event):
        signals_event =  self.strategy.generate_signals(feature_data_event)

        self.signals.extend(signals_event.data)

        return signals_event

    def generate_orders_from_signals(self, signals_event):
        orders_event = self.strategy.generate_orders_from_signals(signals_event)
        
        self.order_history.extend(orders_event.data)

        return orders_event


    def handle_fill_event(self, fills_event):
        """
        maybe only the broker needs to see this and the portfolio is simply informed about what the new
        state of the account is.

        Fills tell the portfolio what price the orders was filled at. (one fill per order?)
        The commission, slippage and whatever other information associated with the processing of an order
        is also captured Fill objects. 
        Fill objects are important for the portfolio to track its active positions.
        Knowing exactly how its position changed at at what tiem
        The fill objects are important for tracking portfolio performance, value, etc.
        # I NEED TO READ MORE ABOUT THIS
        """

        fills = fills_event.data
        for fill in fills:
            self.commissions_charged.at[fill.date, "amount"] += fill.commission
            self.slippage_suffered.at[fill.date, "amount"] += self.calculate_dollar_slippage_for_fill(fill)

        """
            # Update portfolio (active positions)
            if fill.ticker not in self.portfolio:
                if fill.direction == 1:
                    self.portfolio[fill.ticker] = {
                        "short_position": 0,
                        "long_position": fill.amount,
                        "net_position": fill.amount,
                    }
                elif fill.direction == -1:
                    self.portfolio[fill.ticker] = {
                        "short_position": abs(fill.amount),
                        "long_position": 0,
                        "net_position": fill.amount, # will be negative for a short
                    }
            else:
                if fill.direction == 1:
                    self.portfolio[fill.ticker]["long_position"] += fill.amount
                    self.portfolio[fill.ticker]["net_position"] += fill.amount
                elif fill.direction == -1:
                    self.portfolio[fill.ticker]["short_position"] += abs(fill.amount)
                    self.portfolio[fill.ticker]["net_position"] += fill.amount # will be negative for a short

        self.active_positions.extend(fills)
        self.blotter.extend(fills)
        """

    def handle_cancelled_orders_event(self, event): # Maybe the strategy should handle this?
        cancelled_orders = event.data
        for cancelled_order in cancelled_orders:
            # Do whatever with the cancelled order. Don't know what...
            pass

        self.cancelled_orders.extend(cancelled_orders)


    def handle_position_liquidations_event(self, position_liquidations_event):
        """
        I only need this information here if the portfolio should act on it...

        A position liquidation will conclude a trade and the balance must be updated and effects on portfolio
        return must be dealt with (this may be as simple as capturing the market value of the portfolio at the close)

        In the creation of the liquidation event the portfolio was charged etc. Here it is simply a matter of updating
        the balance and portfolio state.
        """

        # portfolios balance should get the proceeds of the position exit.
        # liquidations are often the result of atomatic action by the broker: sell on exit rules being met, company going bankrupt or stock going private
        # liquidations could have a reason associated with them

        """
        position_liquidations = position_liquidations_event.data
        
        for position_liquidation in position_liquidations:
            
            if position_liquidation.direction == 1:
                # Update balance
                self.balance += position_liquidation.amount * position_liquidation.liquidation_price
                # Update portfolio
                self.portfolio[position_liquidation.ticker]["long_position"] -= position_liquidation.amount
                self.portfolio[position_liquidation.ticker]["net_position"] -= position_liquidation.amount


            elif position_liquidation.direction == -1:
                # update balance
                self.balance += (position_liquidation.fill_price - position_liquidation.liquidation_price) * position_liquidation.amount

                # update portfolio
                self.portfolio[position_liquidation.ticker]["short_position"] -= abs(position_liquidation.amount)
                self.portfolio[position_liquidation.ticker]["net_position"] += abs(position_liquidation.amount)


            # Remove from active_positions:
            for active_position in self.active_positions:
                if active_position.order_id == position_liquidation.order.id:
                    self.active_positions.remove(active_position)
        """


    def handle_margin_account_update(self, margin_account_update_event):
        required_margin_account_size = margin_account_update_event.data

        if self.margin_account < required_margin_account_size:
            money_to_move = required_margin_account_size - self.margin_account
            self.balance -= money_to_move
            self.margin_account += money_to_move

        elif self.margin_account > required_margin_account_size:
            money_to_move = self.margin_account - required_margin_account_size
            self.margin_account -= money_to_move
            self.balance += money_to_move

        # If the margin account is the same size as required, do nothing.


    def handle_corporate_actions(self, corporate_actions_event):
        """
        Need to find out about...

        corporate_actions = corporate_actions_event.data

        for corporate_action in corporate_actions:
            if corporate_action.type == "dividends":
                self.balance += corporate_action.dividend     
        """ 
        pass



    def charge(self, amount):
        """
        Amount must be a positive value to deduct from the portfolios balance.
        """
        if self.balance >= amount:
            self.total_charged_for_stock_purchases += amount
            self.balance -= amount
        else:
            raise BalanceTooLowError("Cannot charge portfolio because balance is {} and wanted to charge {}".format(self.balance, amount))


    def charge_commission(self, commission):
        """
        Charge the portfolio the commission.
        To avoid senarios where the portfolios balance is too low to even exit its current positions, this will not
        cause errors
        """
        self.total_commission += commission

        self.balance -= commission

    def charge_margin_interest(self, margin_interest):
        """
        Since I dont want the portfolio to get margin calls where additional funds must be added because it cannot meet payments, I allways
        assume margin interest can be payed. This can cause the portfolio to end up with a negative balance. This will trigger sales to get back 
        to positive balance the following trading day. 
        """
        self.total_margin_interest += margin_interest

        self.balance -= margin_interest


    def increase_margin_account(self, amount):
        """
        Move money from the balance to the margin_account. There must be enough funds.
        
        NOTE: This process can fail due to insufficient funds, so when generating orders, the strategy must take
        into account the margin requirements of the short positions.
        """
        try:
            self.charge(amount)
        except BalanceTooLowError as e:
            raise BalanceTooLowError("Cannot increase margin account, because balance is too low. Balance is {} and wanted to move {}".format(self.balance, amount))
        else: 
            self.margin_account += amount



    """The below methods are used to capture the state of the portfolio at the end of the backtest"""
    def calculate_value(self):
        """
        Calculates are returns portfolio value at the end of the current day for both long and short positions.
        This is called after all the days orders have been placed and processed, the margin account has been updated
        and liquidated positions have updated the balance and portfolio state

        Short positions need fill price and current price to calculate value. (An exit price is never relevant, because exited position will have
        its value added to the balance)
        """

        # USE THE BROKER TO GET CURRENT STATE OF ACCOUNT

        portfolio_value = 0


        for fill in self.active_positions:
            try:
                daily_data = self.market_data.current_for_ticker(fill.ticker)
            except MarketDataNotAvailableError:
                return None

            if fill.direction == 1:
                position_value = fill.amount * daily_data["close"]

            elif fill.direction == -1:
                position_value = abs(fill.amount) * (fill.price - daily_data["close"])


            portfolio_value += position_value


        total_value = portfolio_value + self.balance + self.margin_account

        return total_value

    
    
    def calculate_return_over_period(self, start_date, end_date):
        """
        Complicated.
        Get back to this later

        Once daily returns are calculated precicely, is simply becomes a matter of multiplying
        the daily returns over the period together (1+return!)
        """
        if start_date < self.market_data.start or (end_date > self.market_data.cur_date):
            raise ValueError("Attempted to calculate return over period where the start date was lower\
                 than backtest's start date or the end date was higher than the current date.")

        start_row = self.portfolio_value[start_date]
        end_row = self.portfolio_value[end_date]

        start_val = start_row["balance"] + start_row["market_value"]
        end_val = end_row["balance"] + end_row["market_value"]

        return (end_val / start_val) - 1


    def calculate_market_value_of_trades(self): # RENAMED from calculate_market_value()
        """
        Calculate market value of portfolio.
        Not including cash and margin account.
        """
        portfolio_value = 0

        for fill in self.active_positions:
            try:
                daily_data = self.market_data.current_for_ticker(fill.ticker)
            except MarketDataNotAvailableError:
                return None

            if fill.direction == 1:
                position_value = fill.amount * daily_data["close"]

            elif fill.direction == -1:
                position_value = abs(fill.amount) * (fill.price - daily_data["close"])


            portfolio_value += position_value

        return portfolio_value



    def calculate_total_commission(self):
        """ Return total commission charged from fill objects """
        total_commission = 0
        for data, row in self.commissions_charged.iterrows():
            total_commission += row["amount"]

        return total_commission

    def calculate_total_slippage(self):
        """ Return total slippage suffered from fill objects. """
        total_slippage = 0
        for date, row in self.slippage_suffered.iterrows():
            total_commission += row["amount"]

        return total_slippage

    def calculate_dollar_slippage_for_fill(self, fill):
        # fill.slippage: price difference between open and fill (can be negative and positive)
        # fill.amount: number of stocks bought or sold

        # positive slippage when buying is a cost, negative slippage when buying is profit
        # dollar slippage per stock * number of stocks bought/sold (+/-) = slippage cost/reward (desired sign)
        # -10 * -10 = 100 (-)
        # -10 * 10 = -100 (+)
        # 10 * 10 = 100 (-)
        # 10 * -10 = (+)
        # Conclusion: Sign must be inverted when multiplying slippage and amount
        dollar_slippage = (fill.slippage * fill.amount) * -1
        return dollar_slippage


    # This got complicated, I think I can make it simpler by calculating everything from active_positions
    def capture_state(self):
        """
        Capture the state of the portfolio and append it to portfolio_history

        # move this entirely to the broker
        """
        positions = copy.deepcopy(self.portfolio)
        portfolio = {}
        short_values = {}


        for fill in self.active_positions:
            try:
                daily_data = self.market_data.current_for_ticker(fill.ticker)
            except MarketDataNotAvailableError:
                short_values[fill.ticker] = 0 # None, None will cause errors, must find a better way
                continue

            
            if fill.direction == -1:
                if fill.ticker not in short_values:
                    short_values[fill.ticker] = 0
                
                position_value = abs(fill.amount) * (fill.price - daily_data["close"]) # can be negative
                short_values[fill.ticker] += position_value


        for ticker in positions:
            try:
               close = self.market_data.current_for_ticker(ticker)["close"] # Might fail
            except:
                close = math.nan

            try:
                short_value = short_values[ticker] # this will be none if no market data
            except:
                short_value = 0 # 

            long_value = close * positions[ticker]["long_position"]

            positions[ticker]["close"] = close
            positions[ticker]["short_value"] = short_value # OBS: This does not support short positions!
            positions[ticker]["long_value"] = long_value
            positions[ticker]["net_value"] = short_value + long_value

        positions["Cash"] = {
            "close": math.nan,
            "short_position": math.nan,
            "long_position": math.nan,
            "net_position": math.nan,
            "short_value": math.nan,
            "long_value": self.balance,
            "net_value": self.balance,
        }

        positions["Margin Account"] = {
            "close": math.nan,
            "short_position": math.nan,
            "long_position": math.nan,
            "net_position": math.nan,
            "short_value": math.nan,
            "long_value": self.margin_account,
            "net_value": self.margin_account,
        }

        portfolio["positions"] = positions
        portfolio["cur_date"] = self.market_data.cur_date
        portfolio["cur_balance"] = self.balance # Would be nice to see this on the portfolio allocation chart as well
        cur_market_value = self.calculate_market_value()
        portfolio["cur_market_value"] = cur_market_value

        self.portfolio_history.append(portfolio)

        # This is used in return calculations and graphing portfolio value and portfolio return. 
        # Not really related to the portfolio_history list
        self.portfolio_value.loc[self.market_data.cur_date, "balance"] = self.balance
        self.portfolio_value.loc[self.market_data.cur_date, "margin_account"] = self.margin_account
        self.portfolio_value.loc[self.market_data.cur_date, "market_value"] = cur_market_value # This does not support short positions.

        # Capture active positions
        history_obj = {
            "date": self.market_data.cur_date,
            "active_positions": copy.deepcopy(self.active_positions),
        }
        self.active_positions_history.append(history_obj)





    def order_history_to_df(self):
        df = pd.DataFrame(columns=["order_id", "date", "ticker", "amount", "direction", "stop_loss", "take_profit", "time_out", "type", "signal_id"]) # SET COLUMNS, so sorting and set index does not fail
        for index, order in enumerate(self.order_history):
            df.at[index, "order_id"] = order.id
            df.at[index, "date"] = order.date
            df.at[index, "ticker"] = order.ticker
            df.at[index, "amount"] = order.amount
            df.at[index, "direction"] = order.direction
            df.at[index, "stop_loss"] = order.stop_loss
            df.at[index, "take_profit"] = order.take_profit
            df.at[index, "time_out"] = order.time_out
            df.at[index, "type"] = order.type
            df.at[index, "signal_id"] = order.signal.signal_id

        df = df.sort_values(by=["order_id"])

        return df

    """
    # Not needed when this is handled by the broker
    def portfolio_history_to_df(self):
        df = pd.DataFrame(columns=["date", "ticker", "close", "short_position", "long_position", "net_position", "short_value", "long_value", "net_value" ]) # index=self.market_data.date_index_to_iterate
        
        # df["date"] = self.market_data.date_index_to_iterate
        # df = df.set_index(["date", "ticker"])
        for portfolio in self.portfolio_history:
            port_df = pd.DataFrame()
            date = portfolio["cur_date"]
            # df.at[date, "cur_balance"] = portfolio["cur_balance"] # But I have this in portfolio_value
            # df.at[date, "cur_market_value"] = portfolio["cur_market_value"] # But I have this in portfolio_value
            #  print(portfolio["positions"])
            # port_df.at[date, "cur_balance"] = portfolio["cur_balance"]
            for index, ticker in enumerate(portfolio["positions"]):
                position = portfolio["positions"][ticker]
                port_df.at[index, "date"] = date
                port_df.at[index, "ticker"] = ticker
                port_df.at[index, "close"] = position["close"]
                port_df.at[index, "short_position"] = position["short_position"]
                port_df.at[index, "long_position"] = position["long_position"]
                port_df.at[index, "net_position"] = position["net_position"]
                port_df.at[index, "short_value"] = position["short_value"]
                port_df.at[index, "long_value"] = position["long_value"]
                port_df.at[index, "net_value"] = position["net_value"]


            df = df.append(port_df, sort=True)

        df = df.sort_values(by=["date", "ticker"])

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
            
                # May want more of the order properties here, but not sure
                # df.at[index, "price"] = fill.price
                # df.at[index, "commission"] = fill.commission
                # df.at[index, "slippage"] = fill.slippage
                index += 1
        
        df = df.sort_values(by=["date"])
        return df

    """


    def cancelled_orders_to_df(self):
        df = pd.DataFrame(columns=["order_id", "date", "ticker", "amount", "error"])
        for index, c_ord in enumerate(self.cancelled_orders):
            df.at[index, "order_id"] = c_ord.id
            df.at[index, "date"] = c_ord.date
            df.at[index, "ticker"] = c_ord.ticker
            df.at[index, "amount"] = c_ord.order.amount
            df.at[index, "error"] = str(c_ord.error)
            # Might want to extend this, don't know atm

        df = df.sort_values(by=["order_id"])

        return df

    def signals_to_df(self):
        df = pd.DataFrame(columns=["signal_id", "ticker", "direction", "certainty", "ewmstd", "upper_barrier", "lower_barrier", "vertical_barrier", "feature_data_index", "feature_data_date"])
        for index, signal in enumerate(self.signals):
            df.at[index, "signal_id"] = signal.signal_id
            df.at[index, "ticker"] = signal.ticker
            df.at[index, "direction"] = signal.direction
            df.at[index, "certainty"] = signal.certainty
            df.at[index, "ewmstd"] = signal.ewmstd
            df.at[index, "ptSl_0"] = signal.ptSl[0]
            df.at[index, "ptSl_1"] = signal.ptSl[1]          
            # df.at[index, "upper_barrier"] = signal.barriers[0] # I guess I dont know this before the order is generated
            # df.at[index, "lower_barrier"] = signal.barriers[1]
            # df.at[index, "vertical_barrier"] = signal.barriers[2]
            df.at[index, "feature_data_index"] = signal.feature_data_index
            df.at[index, "feature_data_date"] = signal.feature_data_date

        # df = df.set_index("signal_id")
        df = df.sort_values(by=["signal_id"])

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
        
        df = df.sort_values(by=["order_id"])

        return df

    """
    def set_order_validators(self):
        
        # Set list of validators that must be passed for an order to be valid.
        
        pass

    def set_do_not_order_list(self):
        pass

    def set_long_only(self):
        pass

    def set_max_leverage(self): # Dont think I will have leverage
        pass

    def set_max_order_count(self):
        pass

    def set_max_order_size(self):
        # Set a limit on the number of shares and/or dollar value of any single order placed for sid. 
        # Limits are treated as absolute values and are enforced at the time that the algo attempts to place an order for sid.
        # If an algorithm attempts to place an order that would result in exceeding one of these limits, 
        # raise a TradingControlException.
        

    def validate_orders(self):
        # This is better to have here, the broker does not need to know about the portfolios restrictions
        pass
    """




