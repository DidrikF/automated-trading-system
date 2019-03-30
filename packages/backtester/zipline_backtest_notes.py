from zipline.api import order, record, symbol


def initialize(context):
    pass



# This function is executed once of every day in the dataset.
def handle_data(context, data):

    # we can access the current price data of the AAPL stock in the data event frame


    # After the call of the order() function, zipline enters the ordered stock and amount in the order book
    # The style=OrderType keyword argument has the following options.
    # style=MarketOrder(exchange)
    # style=StopOrder(stop_price, exchange)
    # style=LimitOrder(limit_price, exchange)
    # style=StopLimitOrder(limit_price=price1, stop_price=price2, exchange)
    order(symbol('AAPL'), 10) # Security object, number of stocks, can be negative for selling
    
    # If the trading volume is high enough for this stock, the order is executed after adding the commission and applying the slippage model

    # The record function places the price of apple at each date in the result/output df
    record(AAPL=data.current(symbol('AAPL'), 'price')) 

    """
    How to set a limit for the required volume to trade an amount?
    """