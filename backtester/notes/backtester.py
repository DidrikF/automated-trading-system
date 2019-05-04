from processing import Process, Queue
import logging
import numpy as np
import pandas as pd

sep_path = "./datasets/prices/set_1.csv"

class Backtester:
    def __init__(self, datafeed):
        self.datafeed = datafeed
        
        self.portfolio = Portfolio()
        self.algorithm = Algorithm()
        self.strategy = None

    def start(self):
        try:
            while True:
                if not queue.empty():
                    o = queue.get()
                    controller._logger.debug(o)

                    if o == 'POISON':
                        # Poison Pill!
                        break

                    timestamp = o[0]
                    ticker = o[1]
                    price = o[2]

                    # Update pricing
                    controller.process_pricing(ticker = ticker, price = price)

                    # Generate Orders
                    orders = controller._algorithm \
                            .generate_orders(timestamp, controller._portfolio)

                    # Process orders
                    if len(orders) > 0:
                        # Randomize the order execution
                        final_orders = [orders[k] for k in np.random.choice(len(orders), 
                                                                            replace=False, 
                                                                            size=len(orders))]

                        for order in final_orders:
                            controller.process_order(order)

                        controller._logger.info(controller._portfolio.value_summary(timestamp))

        except Exception as e:
            print(e)
        finally:
            controller._logger.info(controller._portfolio.value_summary(timestamp))


    def placeOrder(self):
        pass


class DataSource:
    """
    Data source for the backtester. Must implement a "get_data" function 
    which streams data from the data source.
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    @classmethod
    def process(cls, queue, source=None):
        source = cls() if source is None else source
        while True:
            data = source.get_data()
            if data is not None:
                queue.put(data) # Does this block until read on the other end?
                if data == 'POISON':
                    break

    def set_source(self, source_path, tickers, start, end):
        try:
            self._logger.info("Loading data...")
            data = pd.read_csv(source_path, low_memory=False)
        except Exception as e:
            self._logger.error(e)
            pass
        
        # Need to produce "events" from the input data
        events = list()
        
        self._source = events

        self._logger.info("Loaded data!")

    
    def get_data(self):
        try:
            return self._source.pop(0)
        except IndexError as e:
            return 'POISON'




class Order():
    def __init__(self):
        pass



class OrderApi:
    def __init__(self):
        self._slippage_std = 0.01
        self._prob_of_failure = 0.0001
        self._fee = 0.02
        self._fixed_fee = 10
        self._calculate_fee = lambda x: self._fee*abs(x) + self._fixed_fee

    def process_order(self, order):
        slippage = np.random.normal(0, self._slippage_std, size=1)[0]

        if np.random.choice([False, True], p=[self._prob_of_failure, 1 -self._prob_of_failure],size=1)[0]:
            trade_fee = self._fee*order[1]*(1+slippage)*order[2]
            return (order[0], order[1]*(1+slippage), order[2], self._calculate_fee(trade_fee))

class Portfolio():
    pass

class Algorithm():
    """
    Must implement a "generate_order" function which returns a list of orders and an update function.
    Each order is a tuple of the form:
        (Stock Ticker str, Current Price float, Order Amount in shares float)
    """
    def __init__(self):
        self._averages = dict()
        self._price_window = 10 # Number of days to use in moving avg calculation?

    def update(self, stock, price):
        if stock in self._averages:
            self.add_price(stock, price)
        else: 
            length = self._price_window
            self._averages[stock] = {'History' : np.zeros(length), 'Index' : 0, 'Length' : length}
            data = self._averages[stock]['History']
            data[0] = price

    def add_price(self, stock, price):
        pass

    def generate_orders(self, timestamp, portfolio):
        orders = list()

        return orders

    def determine_if_trading(self, data, portfolio_values, cash_balance):
        pass

"""
class Strategy:
    def __init__(self):
        self.algo = None

    def get_order(self, feature_vector) -> Order:
        pass
"""


if __name__ == "__main__":
    q = Queue()
    p = Process(target=DataSource.process, args=((q,)))
    
    p.start()
    
    p.join()