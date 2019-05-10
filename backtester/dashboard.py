import datetime
import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import pickle
import time
import glob
import os

import dash
import dash_table
from dash.dependencies import Input, Output
import plotly
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html

from data_handler import DailyBarsDataHander, MLFeaturesDataHandler

"""
Reference:

trades/blotter_history = pd.DataFrame(columns=["order_id", "ticker", "direction", "amount", "stop_loss", "take_profit", "timeout", "date", "fill_price",\
            "commission", "slippage", "interest_expenses", "dividends_per_share", "CLOSED", "close_price", "close_date", "close_cause", \
                "ret", "total_ret", "total_dividends"])
canelled_orders = pd.DataFrame(columns=["order_id", "ticker", "date", "error", "amount", "direction", "stop_loss", "take_profit", "timeout", "type"])
trade_objects = Trade[]
"""

"""
Dashboard Layout:



"""

if __name__ == "__main__":


    list_of_files = glob.glob('./backtests/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    # Load data:
    pickle_path = latest_file # "./backtests/backtest_state_20190408-001506.pickle"
    pickle_in = open(pickle_path,"rb") 
    
    backtest = pickle.load(pickle_in) 

    start_date = backtest["settings"]["start"]
    end_date = backtest["settings"]["end"]

    # NOTE: TEST DATA, NEED TO UPDATE LATER
    market_data = DailyBarsDataHander( 
        path_prices="../dataset_development/datasets/testing/sep.csv",
        path_snp500="../dataset_development/datasets/macro/snp500.csv",
        path_interest="../dataset_development/datasets/macro/t_bill_rate_3m.csv",
        path_corp_actions="../dataset_development/datasets/sharadar/SHARADAR_EVENTS.csv",
        store_path="./tests/test_bundles",
        start=start_date,
        end=end_date
    )

    # NOTE: TEST DATA, NEED TO UPDATE LATER
    feature_data = MLFeaturesDataHandler(
        path_features="../dataset_development/datasets/testing/ml_dataset.csv",
        store_path="./tests/test_bundles",
        start=pd.to_datetime("2001-02-12"),
        end=pd.to_datetime("2002-05-14")
    )
    

    app = dash.Dash(__name__)

    # Dashboard Configuration:
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    figure_height = 350

    # CONFIG
    # All in markdown

    # SUMMARY STATISTICS
    costs_slippage_figure = go.Figure(
        data=[go.Bar(
            x=backtest["portfolio"]["costs"].index,
            y=backtest["portfolio"]["costs"]["slippage"],
        )],
        layout=go.Layout(
            # plot_bgcolor=colors['background'],
            # paper_bgcolor=colors['background'],
            # font={'color': colors['text']},
            title="Slippage Suffered",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Dollar"),
            height=figure_height
        )
    )
    costs_commissions_figure = go.Figure(
        data=[go.Bar(
            x=backtest["portfolio"]["costs"].index,
            y=backtest["portfolio"]["costs"]["commission"],
        )],
        layout=go.Layout(
            # plot_bgcolor=colors['background'],
            # paper_bgcolor=colors['background'],
            # font={'color': colors['text']},
            title="Commissions Charged",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Dollar"),
            height=figure_height
        )
    )
    portfolio_value_figure = go.Figure(
        data=[go.Scatter(
            x=backtest["portfolio"]["portfolio_value"].index,
            y=backtest["portfolio"]["portfolio_value"]["balance"],
            name="Balance",
        ),go.Scatter(
            x=backtest["portfolio"]["portfolio_value"].index,
            y=backtest["portfolio"]["portfolio_value"]["margin_account"],
            name="Margin Account"
        ),go.Scatter(
            x=backtest["portfolio"]["portfolio_value"].index,
            y=backtest["portfolio"]["portfolio_value"]["market_value"],
            name="Market Value"
        ),go.Scatter(
            x=backtest["portfolio"]["portfolio_value"].index,
            y=backtest["portfolio"]["portfolio_value"]["total"],
            name="Total Portfolio Value"
        ),go.Scatter(
            x=backtest["portfolio"]["portfolio_value"].index,
            y=backtest["portfolio"]["portfolio_value"]["market_value"]+backtest["portfolio"]["portfolio_value"]["balance"]+backtest["portfolio"]["portfolio_value"]["margin_account"],
            name="Total Portfolio Value - MANUAL TEST"
        )],
        layout=go.Layout(
            # plot_bgcolor=colors['background'],
            # paper_bgcolor=colors['background'],
            # font={'color': colors['text']},
            title="Porfolio Value over Time",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Dollar"),
            height=figure_height
        )
    )


    @app.callback(Output("active-trades-value-graph", "figure"), [Input("date-picker", "date")])
    def update_portfolio_allocation_value_graph(date):
        blotter_history = backtest["broker"]["blotter_history"]
        try:
            active_trades = blotter_history.loc[blotter_history.date == date]
        except:
            active_trades = pd.DataFrame(columns=blotter_history.columns)
        
        short_trace = go.Bar(
            x=active_trades["ticker"],
            y=active_trades["short_value"], # NOTE: Need to rethink this...
            name="Short Value",
            marker=go.bar.Marker(color="rgb(237, 79, 35)")
        )
        long_trace = go.Bar(
            x=active_trades["ticker"],
            y=active_trades["long_value"],
            name="Long Value",
            marker=go.bar.Marker(color="rgb(90, 198, 17)")
        )
        net_trace = go.Bar(
            x=active_trades["ticker"],
            y=active_trades["net_value"],
            name="Net Value",
            marker=go.bar.Marker(color="rgb(46, 108, 232)")
        )

        data = [short_trace, long_trace, net_trace]
        
        layout=go.Layout(
            title='Portfolio Value Allocation ' + str(date),
            xaxis=dict(title="Ticker"),
            yaxis=dict(title="Dollar Value"),
            height=figure_height,
        )

        return dict(data=data, layout=layout)

    @app.callback(Output("active-trades-amount-graph", "figure"), [Input("date-picker", "date")])
    def update_portfolio_allocation_amount_graph(date):
        blotter_history = backtest["broker"]["blotter_history"]
        try:
            active_trades = blotter_history.loc[blotter_history.date == date]
            active_trades = active_trades.loc[active_trades.ticker != "Cash"]
            active_trades = active_trades.loc[active_trades.ticker != "margin Account"]
        except:
            active_trades = pd.DataFrame(columns=blotter_history.columns)

        short_trace = go.Bar(
            x=active_trades["ticker"],
            y=active_trades["short_position"], # NOTE:  Need to rethink
            name="Short Position",
            marker=go.bar.Marker(color="rgb(237, 79, 35)")
        )
        long_trace = go.Bar(
            x=active_trades["ticker"],
            y=active_trades["long_position"],
            name="Long Position",
            marker=go.bar.Marker(color="rgb(90, 198, 17)")
        )
        net_trace = go.Bar(
            x=active_trades["ticker"],
            y=active_trades["net_position"],
            name="Net Position",
            marker=go.bar.Marker(color="rgb(46, 108, 232)")
        )

        data = [short_trace, long_trace, net_trace]
        
        layout=go.Layout(
            title='Portfolio Share Allocation ' + str(date),
            xaxis=dict(title="Ticker"),
            yaxis=dict(title="Nr. Shares"),
            height=figure_height,
        )

        return dict(data=data, layout=layout)
        



    @app.callback(Output("active-trades-table", "data"), [Input("date-picker", "date")])
    def update_broker_blotter_history(date):
        date = pd.to_datetime(date)
        blotter_history = backtest["broker"]["blotter_history"]
        try:
            active_trades = blotter_history.loc[blotter_history.date == date]
        except:
            active_trades = pd.DataFrame(columns=active_trades.columns)
        return active_trades.to_dict("rows")


    # INSPECT PORTFOLIO WEEK BY WEEK


    # TRADE INSPECTION


    # STRATEGY AND ML STATISTICS

    # RAW DATA INSPECTION
    
    @app.callback(Output("candlestick-chart", "figure"), [Input("ticker-search-field", "value")])
    def update_candlestick_chart(ticker):
        ticker_sep = market_data.get_ticker_data(ticker)
        # ticker_sep = sep.loc[sep.ticker==ticker][start:stop]

        trace = go.Ohlc(
            x=ticker_sep.index,
            open=ticker_sep["open"],
            high=ticker_sep["high"],
            low=ticker_sep["low"],
            close=ticker_sep["close"]
        )
        
        data = [trace]
        
        layout = {
            'title': ticker + ' Chart',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Price'},
            'height': figure_height
        }

        return dict(data=data, layout=layout)


    # RAW DATA INSPECTION
    """
    dcc.DatePickerSingle(
        id='date-picker-single',
        date=dt(1997, 5, 10)
    )

    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=dt(1997, 5, 3),
        end_date_placeholder_text='Select a date!'
    )
    """



    # BACKTEST FINAL STATE TABLES



    """ APP LAYOUT """

    app.layout = html.Div(
        [
            html.H1(
                children='ML Driven Automated Trading System - Backtest Results',
                style={
                    'textAlign': 'center',
                }
            ),
            # CONFIG
            html.H2("Backtest Config"),
            html.Div(
                children=[
                    html.Div(children=[
                        html.Span("Start: {}".format(backtest.settings.start)),
                        html.Span("End: {}".format(backtest.settings.end)),
                    ])
                ]
            ),

            # SUMMARY STATISTICS
            html.H2("Backtest Summary Statistics"),
            html.Div(children=[
                html.Div(children=[
                    html.Span("Total Return: {}".format(backtest.stats.total_return)),
                    html.Span("Total Slippage: {}".format(backtest.stats.total_slippage))    
                ])
            ]),

            # - Summary Graphs
            dcc.Graph(id="portfolio-value", figure=portfolio_value_figure),

            dcc.Graph(id="costs-slippage", figure=costs_slippage_figure),
            dcc.Graph(id="costs-commissions", figure=costs_commissions_figure),
            # dcc.Graph(id="costs-total-charged", figure=costs_total_charged_figure),
            # etc..


            # INSPECT WEEK
            html.H2("Inspect State For Week"),
            html.H3(children="Portfolio Composition"),
            dcc.Slider( # NOTE: Calculate number of weeks to set this up above
                id="week-slider",
                min=0,
                max=10,
                marks={i: 'Label {}'.format(i) for i in range(10)},
                value=5,
            ),
            html.Span("Choose date: "),
            dcc.DatePickerSingle(
                id='date-picker',
                date=start_date
            ),
            # NOTE: UPDATE FILTER/NAMING
            dcc.Graph(id='active-trades-value-graph'), # Dollar value of each Trade (+ balance and margin account)
            dcc.Graph(id='active-trades-amount-graph'), # Amount of Stock For Each Trade

            # - Active Trades
            html.H3(children='Active Trades'),
            dash_table.DataTable( # NOTE: FILTER CONTENTS
                id='active-trades-table',
                columns=[{"name": i, "id": i} for i in backtest["broker"]["active_positions_history"].columns],
                # data=backtest["portfolio"]["blotter"].to_dict("rows"),
                filtering=True,
                sorting=True,
                sorting_type="multi",
                pagination_mode="fe",
                pagination_settings={
                    "displayed_pages": 1,
                    "current_page": 0,
                    "page_size": 50,
                },
                navigation="page",
            ),
            # TRADE INSPECTION



            # RAW DATA INSPECTION

            html.Span("Search for ticker: "),
            dcc.Input(id="ticker-search-field", type="text", value="AAPL"),
            dcc.Graph(id="candlestick-chart"),

            
            # BACKTEST'S FINAL STATE TABLES 

            
            html.H2(children='Trades'),
            dash_table.DataTable(
                id='broker-blotter-table',
                columns=[{"name": i, "id": i} for i in backtest["broker"]["blotter"].columns],
                data=backtest["broker"]["blotter"].to_dict("records"), # Why no signal id?
                filtering=True,
                sorting=True,
                sorting_type="multi",
                pagination_mode="fe",
                pagination_settings={
                    "displayed_pages": 1,
                    "current_page": 0,
                    "page_size": 50,
                },
                navigation="page",
            ),

            html.H2(
                children='Portfolio Signals',
                style={
                    'textAlign': 'center',
                }
            ),
            dash_table.DataTable(
                id='signals-table',
                columns=[{"name": i, "id": i} for i in backtest["portfolio"]["signals"].columns],
                data=backtest["portfolio"]["signals"].to_dict("records"), # Why no signal id?
                filtering=True,
                sorting=True,
                sorting_type="multi",
                pagination_mode="fe",
                pagination_settings={
                    "displayed_pages": 1,
                    "current_page": 0,
                    "page_size": 50,
                },
                navigation="page",
            ),

            html.H2(
                children='Portfolio Order History',
                style={
                    'textAlign': 'center',
                }
            ),
            dash_table.DataTable(
                id='order-history-table',
                columns=[{"name": i, "id": i} for i in backtest["portfolio"]["order_history"].columns],
                data=backtest["portfolio"]["order_history"].to_dict("rows"),
                filtering=True,
                sorting=True,
                sorting_type="multi",
                pagination_mode="fe",
                pagination_settings={
                    "displayed_pages": 1,
                    "current_page": 0,
                    "page_size": 50,
                },
                navigation="page",
            ),

            html.H2(
                children='Broker Cancelled Orders',
                style={
                    'textAlign': 'center',
                }
            ),
            dash_table.DataTable(
                id='broker-cancelled-orders-table',
                columns=[{"name": i, "id": i} for i in backtest["broker"]["cancelled_orders"].columns],
                data=backtest["broker"]["cancelled_orders"].to_dict("records"), # Why no signal id?
                filtering=True,
                sorting=True,
                sorting_type="multi",
                pagination_mode="fe",
                pagination_settings={
                    "displayed_pages": 1,
                    "current_page": 0,
                    "page_size": 50,
                },
                navigation="page",
            ),

            # BACKTEST LOG
            # NOTE: How to do this


        ]
    )


    """ CALLBACKS """


    app.run_server(debug=True)












# Showing trades as two scatter plots:
# Make two dfs, one with long and one with short trades (orders that was filled)
"""
Just a thought:
        Can trades be presented in a graph using a scatter plot and assign
        marker config to display direction and size of trade, and have text config
        to give information about the trade in the graph?
short_trace = go.Scatter(x=[], y=[], mode="markers", marker=dict(
    size=12,
    color="rgb(100, 50, 50)",
    symbol="arrow_down", # this is the key, have one Scatter for long trades and one scatter for short ones
))

long_trace = go.Scatter(x=[], y=[], mode="markers", marker=dict(
    size=12,
    color="rgb(100, 50, 50)",
    symbol="arrow_up", # this is the key, have one Scatter for long trades and one scatter for short ones
))

data = [short_trace, long_trace]

"""







"""
1. Visualization and performance reporting
    - This work will prepare many visualizations for the report
    - This work will also substitute testing work
    - It will be awesome to look at.

    - I want to illustrate all aspects of the system, LIVE.
    - This require some clever linking do Dash, but once this is done it will be easy to extend
    - I can also implement a loop slowdown parameter, so make it more visible to
    the user of the system what is going on.

    Whether I want it to be live depends on how long the backtest takes to run.
    As long as I can browse through the history of the simulation, I should be good.


    ######   I need to make the data available (and saved) at least partly before I code out the dash!!!!

    MASTER SLIDER, where all data is selected relative to the master slider!!!!
    
    Wanted on the screen: 
        # - Portfolio Composition (Pie chart / bar chart with slider) - stocks (bar chart prob better)
        - Portfolio Composition (Pie chart / bar chart with slider) - industry
            - Combine both in nested bar chart?
        
        # - Balance (as just a number) If not running simulation live, these must be charts
        - Total commission # working i think
        - Total slippage # not working (-2 each day in slippage...)
        - Sharpe ratio (make it simple to begin with?) NEED RISK FREE RATE IN SEP
        

        - Trades per month chart
            - Max trades per month
            - Lowest number of trades per month


        - Line chart of portfolio return (on different periods)
            - Max loss on single day / month
            - Max gain on single day / month


        # - Line chart of portfolio value
        - Line chart of portfolio variance

        - Line chart of sharpe ratio
        # - Candlestick chart, where you can select stock
        - Table of executed orders -> with associated signal, fill, commission, 
          slippage, open, execution price, and stock information (one mega master table),
          portfolio balance

        - Table of top 10 signals for each day
        - Table of non executed orders

        - Histogram of trade return with distribution layered on top (Distplots IS AWESOME!)
        - Other distributions?

        Strategy related:
        - Heatmap of feature importance
        - Make it possible to access the ML features and acociate them with signals?

    



Table columns for ATS Master table

    Direction of Time -->
    General Info:
    Date, 
    ticker, 
    order_id, 

    Signal:
    signal direction,
    signal certainty,
    barrier high,
    barrier low, 
    barrier vertical,

    Order:
    order amount (implies direction), 
    stop loss, 
    take profit, 
    timeout, 

    Fill:
    Fill price
    commission
    slippage

    After trade status:
    Balance,
    Portfolio Value,
    (return? need to find out if this is practical)


    I need sorting functionality for tables 


"""
