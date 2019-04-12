import glob
import os
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

from automated_trading_system.backtester.data_handler import DailyBarsDataHander, MLFeaturesDataHandler

DATA_SET = "../../datasets/testing/sep.csv"
SAVE_LOCATION = "./test_bundles"

if __name__ == "__main__":

    list_of_files = glob.glob('./backtests/*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    # Load data:
    pickle_path = latest_file  # "./backtests/backtest_state_20190408-001506.pickle"
    pickle_in = open(pickle_path, "rb")

    backtest = pickle.load(pickle_in)

    start_date = backtest["settings"]["start"]
    end_date = backtest["settings"]["end"]

    market_data = DailyBarsDataHander(
        source_path=DATA_SET,
        store_path=SAVE_LOCATION,
        file_name_time_data="time_data",
        file_name_ticker_data="ticker_data",
        start=start_date,
        end=end_date
    )

    feature_data = MLFeaturesDataHandler(
        source_path=DATA_SET,
        store_path=SAVE_LOCATION,
        file_name="feature_data",
    )

    app = dash.Dash(__name__)

    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    """ FIGURE OBJECTS AND SUCH """
    """
    portfolio_total_value_figure = go.Figure(
        data=[go.Scatter(
            x=backtest["perf"].index,
            y=backtest["perf"]["portfolio_value"],
        )],
        layout=go.Layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
        )
    )
    """

    commissions_charged_figure = go.Figure(
        data=[go.Scatter(
            x=backtest["portfolio"]["commissions_charged"].index,
            y=backtest["portfolio"]["commissions_charged"]["amount"],
        )],
        layout=go.Layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
        )
    )

    slippage_suffered_figure = go.Figure(
        data=[go.Scatter(
            x=backtest["portfolio"]["slippage_suffered"].index,
            y=backtest["portfolio"]["slippage_suffered"]["amount"],
        )],
        layout=go.Layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
        )
    )

    portfolio_value_figure = go.Figure(
        data=[go.Scatter(
            x=backtest["portfolio"]["portfolio_value"].index,
            y=backtest["portfolio"]["portfolio_value"]["balance"],
            name="Balance",
        ), go.Scatter(
            x=backtest["portfolio"]["portfolio_value"].index,
            y=backtest["portfolio"]["portfolio_value"]["margin_account"],
            name="Margin Account"
        ), go.Scatter(
            x=backtest["portfolio"]["portfolio_value"].index,
            y=backtest["portfolio"]["portfolio_value"]["market_value"],
            name="Market Value"
        ), go.Scatter(
            x=backtest["portfolio"]["portfolio_value"].index,
            y=backtest["portfolio"]["portfolio_value"]["market_value"] +
              backtest["portfolio"]["portfolio_value"]["balance"] +
              backtest["portfolio"]["portfolio_value"]["margin_account"],
            name="Total Portfolio Value"
        )],
        layout=go.Layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
        )
    )

    """ APP LAYOUT """

    app.layout = html.Div(
        [
            html.H1(
                children='ML Driven Automated Trading System',
                style={'textAlign': 'center'}
            ),

            dcc.DatePickerSingle(
                id='portfolio-allocation-date-picker',
                date=start_date
            ),
            dcc.Graph(id='portfolio-allocation-value-graph'),

            dcc.Graph(id='portfolio-allocation-amount-graph'),

            dcc.Input(id="ticker-search-field", type="text", value="AAPL"),
            dcc.Graph(id="candlestick-chart"),

            dcc.Graph(id="portfolio-value", figure=portfolio_value_figure),

            # Redundant, need to clean up
            # dcc.Graph(id="portfolio-total-value", figure=portfolio_total_value_figure)

            dcc.Graph(id="commissions-charged", figure=commissions_charged_figure),

            dcc.Graph(id="slippage-suffered", figure=slippage_suffered_figure),

            html.H2(
                children='Portfolio Blotter',
                style={'textAlign': 'center'}
            ),
            dash_table.DataTable(
                id='blotter-table',
                columns=[{"name": i, "id": i} for i in backtest["portfolio"]["blotter"].columns],
                data=backtest["portfolio"]["blotter"].to_dict("rows"),
            ),

            html.H2(
                children='Portfolio Order History',
                style={'textAlign': 'center'}
            ),
            dash_table.DataTable(
                id='order-history-table',
                columns=[{"name": i, "id": i} for i in backtest["portfolio"]["order_history"].columns],
                data=backtest["portfolio"]["order_history"].to_dict("rows"),
            ),

            html.H2(
                children='Portfolio Cancelled Orders',
                style={'textAlign': 'center'}
            ),
            dash_table.DataTable(
                id='cancelled-orders-table',
                columns=[{"name": i, "id": i} for i in backtest["portfolio"]["cancelled_orders"].columns],
                data=backtest["portfolio"]["cancelled_orders"].to_dict("rows"),
            ),

            html.H2(
                children='Portfolio Signals',
                style={'textAlign': 'center'}
            ),
            dash_table.DataTable(
                id='signals-table',
                columns=[{"name": i, "id": i} for i in backtest["portfolio"]["signals"].columns],
                data=backtest["portfolio"]["signals"].to_dict("rows"),  # Why no signal id?
            ),

        ]
    )

    """ CALLBACKS """


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
        }

        return dict(data=data, layout=layout)


    @app.callback(Output("portfolio-allocation-amount-graph", "figure"),
                  [Input("portfolio-allocation-date-picker", "date")])
    def update_portfolio_allocation_amount_graph(date):
        # ticker_sep = market_data.get_ticker_data(ticker)
        # ticker_sep = sep.loc[sep.ticker==ticker][start:stop]

        try:
            portfolio = backtest["portfolio"]["portfolio_history"].loc[date, :]
            portfolio = portfolio.reset_index()
            portfolio = portfolio.loc[portfolio.ticker != "Cash"]
        except:
            portfolio = pd.DataFrame(columns=["date", "ticker", "amount", "close", "value"])

        """
        close
        short_position
        long_position
        net_position
        short_value
        long_value
        net_value
        """

        short_trace = go.Bar(
            x=portfolio["ticker"],
            y=portfolio["short_position"],
            name="Short Position",
            marker=go.bar.Marker(color="rgb(237, 79, 35)")
        )

        long_trace = go.Bar(
            x=portfolio["ticker"],
            y=portfolio["long_position"],
            name="Long Position",
            marker=go.bar.Marker(color="rgb(90, 198, 17)")
        )

        net_trace = go.Bar(
            x=portfolio["ticker"],
            y=portfolio["net_position"],
            name="Net Position",
            marker=go.bar.Marker(color="rgb(46, 108, 232)")
        )

        data = [short_trace, long_trace, net_trace]

        layout = go.Layout(
            title='Portfolio Share Allocation ' + str(date),
            xaxis=dict(title="Ticker"),
            yaxis=dict(title="Nr. Shares")
        )

        return dict(data=data, layout=layout)


    @app.callback(Output("portfolio-allocation-value-graph", "figure"),
                  [Input("portfolio-allocation-date-picker", "date")])
    def update_portfolio_allocation_value_graph(date):
        # ticker_sep = market_data.get_ticker_data(ticker)
        # ticker_sep = sep.loc[sep.ticker==ticker][start:stop]

        try:
            portfolio = backtest["portfolio"]["portfolio_history"].loc[date, :]
            portfolio = portfolio.reset_index()

        except:
            portfolio = pd.DataFrame(columns=["date", "ticker", "amount", "close", "value"])

        short_trace = go.Bar(
            x=portfolio["ticker"],
            y=portfolio["short_value"],
            name="Short Value",
            marker=go.bar.Marker(color="rgb(237, 79, 35)")
        )

        long_trace = go.Bar(
            x=portfolio["ticker"],
            y=portfolio["long_value"],
            name="Long Value",
            marker=go.bar.Marker(color="rgb(90, 198, 17)")
        )

        net_trace = go.Bar(
            x=portfolio["ticker"],
            y=portfolio["net_value"],
            name="Net Value",
            marker=go.bar.Marker(color="rgb(46, 108, 232)")
        )

        data = [short_trace, long_trace, net_trace]

        layout = go.Layout(
            title='Portfolio Value Allocation ' + str(date),
            xaxis=dict(title="Ticker"),
            yaxis=dict(title="Dollar Value")
        )

        return dict(data=data, layout=layout)


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
