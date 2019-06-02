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


if __name__ == "__main__":


    list_of_files = glob.glob('./backtests/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    # Load data:
    pickle_path = latest_file # "./backtests/backtest_state_20190408-001506.pickle"
    pickle_in = open(pickle_path,"rb") 
    backtest = pickle.load(pickle_in) 

    backtest["broker"]["blotter_history"]["date"] = pd.to_datetime(backtest["broker"]["blotter_history"]["date"])
    backtest["broker"]["blotter_history"]["trade_date"] = pd.to_datetime(backtest["broker"]["blotter_history"]["trade_date"])
    backtest["broker"]["blotter_history"]["close_date"] = pd.to_datetime(backtest["broker"]["blotter_history"]["close_date"])
    backtest["broker"]["blotter_history"]["timeout"] = pd.to_datetime(backtest["broker"]["blotter_history"]["timeout"])

    backtest["broker"]["all_trades"]["trade_date"] = pd.to_datetime(backtest["broker"]["all_trades"]["trade_date"])
    backtest["broker"]["all_trades"]["close_date"] = pd.to_datetime(backtest["broker"]["all_trades"]["close_date"])
    backtest["broker"]["all_trades"]["timeout"] = pd.to_datetime(backtest["broker"]["all_trades"]["timeout"])

    # ML model results
    ml_model_results = pickle.load(open("../models/ml_strategy_models_results.pickle", "rb"))
    print(ml_model_results)

    # list_of_files = glob.glob('./logs/*') # * means all if need specific format then *.csv
    # latest_log_file = max(list_of_files, key=os.path.getctime)
    # with open(latest_log_file, "r") as f:
    #     log_lines = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    # log_lines = [html.P(line.strip()) for line in log_lines]
    all_log_dirs = ["./logs/" + d for d in os.listdir('./logs') if os.path.isdir("./logs/" + d)]
    latest_log_dir = max(all_log_dirs, key=os.path.getmtime)
    print("Loading log dir: ", latest_log_dir)
    
    log_lines = []

    files = [f for f in glob.glob(latest_log_dir + "/*.log", recursive=True)]
    print("Loading log files: ", files)
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
        formatted_log_lines = [html.P(line.strip()) for line in lines]
        
        log_lines.extend(formatted_log_lines)



    start_date = pd.to_datetime(backtest["settings"]["start"])
    end_date = pd.to_datetime(backtest["settings"]["end"])

    # NOTE: TEST DATA, NEED TO UPDATE LATER
    market_data = DailyBarsDataHander( 
        path_prices="../dataset_development/datasets/testing/sep.csv",
        path_snp500="../dataset_development/datasets/macro/snp500.csv",
        path_interest="../dataset_development/datasets/macro/t_bill_rate_3m.csv",
        path_corp_actions="../dataset_development/datasets/sharadar/SHARADAR_EVENTS.csv",
        store_path="./live_bundles", # NOTE: The important thing to set correctly
        start=start_date,
        end=end_date,
        rebuild=False
    )

    # NOTE: TEST DATA, NEED TO UPDATE LATER
    feature_data = MLFeaturesDataHandler(
        path_features="", # "../dataset_development/datasets/testing/ml_dataset.csv",
        store_path="./live_bundles",
        start=start_date,
        end=end_date
    )
    
    # NOTE: Need to create the strategy model result pickle
    # ml_model_results = pickle.load(open("./models/ml_strategy_models_results.pickle", "rb"))


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
    """
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
    """
    snp500 = market_data.get_ticker_data("snp500")
    snp500["return"] = (snp500["close"] / snp500.iloc[0]["close"]) - 1
    portfolio_value = backtest["portfolio"]["portfolio_value"]
    portfolio_value["return"] = (portfolio_value["total"] / portfolio_value.iloc[0]["total"]) - 1

    snp500_portfolio_return = go.Figure(
        data=[go.Scatter(
            x=snp500.index,
            y=snp500["return"],
            name="S&P500 Return"
        ),go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value["return"],
            name="Portfolio Return"
        )],
        layout=go.Layout(
            # plot_bgcolor=colors['background'],
            # paper_bgcolor=colors['background'],
            # font={'color': colors['text']},
            title="S&P500 and Portfolio Return",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Return"),
            height=figure_height
        )
    )


    costs_figure = go.Figure(
        data=[go.Bar(
            x=backtest["portfolio"]["costs"].index,
            y=backtest["portfolio"]["costs"]["commission"],
            name="Commissions Charged"
        ),go.Bar(
            x=backtest["portfolio"]["costs"].index,
            y=backtest["portfolio"]["costs"]["slippage"],
            name="Slippage Suffered"
        ),go.Bar(
            x=backtest["portfolio"]["costs"].index,
            y=backtest["portfolio"]["costs"]["margin_interest"],
            name="Margin Interest Charged"
        ),go.Bar(
            x=backtest["portfolio"]["costs"].index,
            y=backtest["portfolio"]["costs"]["account_interest"],
            name="Account Interest Charged"
        ),go.Bar(
            x=backtest["portfolio"]["costs"].index,
            y=backtest["portfolio"]["costs"]["short_dividends"],
            name="Payments to Cover Short Dividends"
        )],
        layout=go.Layout(
            # plot_bgcolor=colors['background'],
            # paper_bgcolor=colors['background'],
            # font={'color': colors['text']},
            title="Costs Charged",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Dollar"),
            height=figure_height
        )
    )
    received_figure = go.Figure(
        data=[go.Bar(
            x=backtest["portfolio"]["received"].index,
            y=backtest["portfolio"]["received"]["dividends"],
            name="Dividends Received"
        ),go.Bar(
            x=backtest["portfolio"]["received"].index,
            y=backtest["portfolio"]["received"]["interest"],
            name="Interest Received"
        )],
        layout=go.Layout(
            # plot_bgcolor=colors['background'],
            # paper_bgcolor=colors['background'],
            # font={'color': colors['text']},
            title="Money Received",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Dollar"),
            height=figure_height
        )
    )

    charged_proceeds_figure = go.Figure(
        data=[go.Bar(
                x=backtest["portfolio"]["costs"].index,
                y=backtest["portfolio"]["costs"]["charged"],
                name="Charged for Opening Trade"
            ),go.Bar(
                x=backtest["portfolio"]["received"].index,
                y=backtest["portfolio"]["received"]["proceeds"],
                name="Received Proceeds of Trade Close"
            ),go.Bar(
                x=backtest["portfolio"]["costs"].index,
                y=backtest["portfolio"]["costs"]["short_losses"],
                name="Losses from Short Trades"
            )
        ],
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
            name="Total Portfolio Value / Equity Curve"
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

    # Return of portfolio and return of benchmark graph




    # INSPECT PORTFOLIO WEEK BY WEEK
    date_range = pd.date_range(start_date-relativedelta(days=7), end_date+relativedelta(days=7), freq="W-MON")


    # TRADE INSPECTION

    # -  Trade ID in Input field?

    # STRATEGY AND ML STATISTICS

    # RAW DATA INSPECTION
    rf_rate_figure = go.Figure(
        data=[go.Scatter(
            x=market_data.rf_rate.index,
            y=market_data.rf_rate["daily"],
            name="Daily Risk Free Rate (based on 3 month T-bill secondary market rate)"
        )],
        layout=go.Layout(
            # plot_bgcolor=colors['background'],
            # paper_bgcolor=colors['background'],
            # font={'color': colors['text']},
            title="Daily Risk Free Rate",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Rate"),
            height=figure_height
        )
    )

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


    """ APP LAYOUT """


    app_layout = [
        html.H1(
            children='ML Driven Automated Trading System - Backtest Results',
            style={
                'textAlign': 'center',
            }
        ),
        # CONFIG
        html.H2(children="Backtest Config", style={'textAlign': 'center'}),
        html.Div(children=[
            html.Div(children=[
                html.P("Start: {}  ".format(backtest["settings"]["start"])),
                html.P("End: {}  ".format(backtest["settings"]["end"])),
            ])
        ], style={
            'textAlign': 'center',
        }),

        # SUMMARY STATISTICS
        html.H2(children="Summary Statistics", style={'textAlign': 'center'}),
        html.Div(children=[
            html.P("Total Return: {}%".format(round(backtest["stats"]["total_return"]*100, 4))), 
            html.P("Annualized Rate of Return: {}  ".format(backtest["stats"]["annualized_rate_of_return"])),
            html.P("Portfolio Standard Deviation of Returns: {} ".format(backtest["stats"]["std_portfolio_returns"])),
            html.P("S&P500 Return: {}%  ".format(round(snp500["return"].iloc[-1]*100, 4))),
            html.P("Annualized Rate of S&P500 Return: {}  ".format(backtest["stats"]["annualized_rate_of_index_return"])),
            html.P("S&P500 Standard Deviation of Returns: {} ".format(backtest["stats"]["std_snp500_returns"])),

            html.P("Normality of Returns Test Results: {}  ".format(backtest["stats"]["normality_test_result"])),
            html.P("Sharpe Ratio: {}  ".format(backtest["stats"]["sharpe_ratio"])),
            html.P("T-test on Excess Return: {}  ".format(backtest["stats"]["t_test_on_excess_return"])),
            html.P("Correlation to Underlying (S&P500): {}  ".format(backtest["stats"]["correlation_to_underlying"])),   
            

            html.H4(children="Received Money Related Stats"),
            html.Div(children=[
                html.P("Total Dividends Received: {}  ".format(backtest["stats"]["total_dividends"][0])),
                html.P("Total Interest Received: {}  ".format(backtest["stats"]["total_interest"][0])),
                html.P("Total Proceeds Received: {}  ".format(backtest["stats"]["total_proceeds"][0])),
            ]),
            html.H4(children="Cost Related Stats"),
            html.Div(children=[
                html.P("Total Commission: {}  ".format(backtest["stats"]["total_commission"][0])),
                html.P("Broker Fees Per Dollar: {}  ".format(backtest["stats"]["broker_fees_per_dollar"])),
                html.P("Broker Fees Per Stock: {}  ".format(backtest["stats"]["broker_fees_per_stock"])),
                html.P("Total Charged to Enter Trades: {}  ".format(backtest["stats"]["total_charged"][0])),
                html.P("Total Margin Interest Payed: {}  ".format(backtest["stats"]["total_margin_interest"][0])),
                html.P("Total Interest on Accounts: {}  ".format(backtest["stats"]["total_account_interest"][0])),
                html.P("Total Dividends covered on Short Trades: {}  ".format(backtest["stats"]["total_short_dividends"][0])),
                html.P("Total Losses on Closed Short Trades: {}  ".format(backtest["stats"]["total_short_losses"][0])),  
                html.P("Total Slippage: {}  ".format(backtest["stats"]["total_slippage"][0])),

            ]),
            html.H4(children="Backtest Stats"),
            html.Div(children=[
                html.P("Time Range: {}  ".format(backtest["stats"]["time_range"])),
                html.P("End Value: {}  ".format(backtest["stats"]["end_value"])),
                html.P("Total Return: {}  ".format(backtest["stats"]["total_return"])),
                html.P("Annualized Turover: {}  ".format(backtest["stats"]["annualized_turnover"])),

                html.P("Number of Unique Stocks: {}  ".format(backtest["stats"]["number_of_unique_stocks"])),
                html.P("Number of Trades: {}  ".format(backtest["stats"]["number_of_trades"])),

                html.P("Ratio of Longs: {}  ".format(backtest["stats"]["ratio_of_longs"])),
                html.P("PnL: {}  ".format(backtest["stats"]["pnl"])),
                html.P("PnL from Short Positions: {}  ".format(backtest["stats"]["pnl_short_positions"])),
                html.P("PnL form Long Positions: {}  ".format(backtest["stats"]["pnl_long_positions"])),
                html.P("Hit Ratio: {}  ".format(backtest["stats"]["hit_ratio"])),
                html.P("Average AUM: {}  ".format(backtest["stats"]["average_aum"])),

                html.P("Capacity: {}  ".format(backtest["stats"]["capacity"])),
                html.P("Maximum Dollar Position Size: {}  ".format(backtest["stats"]["maximum_dollar_position_size"])),
                html.P("Frequency of Bets: {}  ".format(backtest["stats"]["frequency_of_bets"])),
                html.P("Average Holding Period: {}  ".format(backtest["stats"]["average_holding_period"])),
                html.P("Average Return from Hits: {}  ".format(backtest["stats"]["average_return_from_hits"])),
                html.P("Average Return from Misses: {}  ".format(backtest["stats"]["average_return_from_misses"])),
                html.P("Highest Return from Hit: {}  ".format(backtest["stats"]["highest_return_from_hit"])),
                html.P("Lowest Return from Miss: {}  ".format(backtest["stats"]["lowest_return_from_miss"])),
                html.P("Closed trade counts by cause: {}  ".format(backtest["stats"]["closed_trades_by_cause"])),

            ]),
            html.H4(children="ML Model Stats (2012-03-01 - 2019-02-01)"),
            html.Div(children=[
                html.P("Side Model Accuracy: {}  ".format(ml_model_results["side_model"]["accuracy"])),
                html.P("Side Model Precision: {}  ".format(ml_model_results["side_model"]["precision"])),
                html.P("Side Model Recall: {}   ".format(ml_model_results["side_model"]["recall"])),
                html.P("Side Model F1: {}   ".format(ml_model_results["side_model"]["f1"])),
                html.P("Certainty Model Accuracy: {}  ".format(ml_model_results["certainty_model"]["accuracy"])),
                html.P("Certainty Model Precision: {}  ".format(ml_model_results["certainty_model"]["precision"])),
                html.P("Certainty Model Recall: {}  ".format(ml_model_results["certainty_model"]["recall"])),
                html.P("Certainty Model F1: {}  ".format(ml_model_results["certainty_model"]["f1"])),
                html.P("Statistical Significance of Models: {}  ".format(backtest["stats"]["statistical_significance_of_classification_models"])),

                
            ])
        ], style={
            'textAlign': 'center',
        }),

        # - Summary Graphs
        dcc.Graph(id="portfolio-value", figure=portfolio_value_figure),

        dcc.Graph(id="snp500-portfolio-return-graph", figure=snp500_portfolio_return),
        dcc.Graph(id="costs-graph", figure=costs_figure),
        dcc.Graph(id="received-graph", figure=received_figure),
        dcc.Graph(id="charged-proceeds-graph", figure=charged_proceeds_figure),

        

        # dcc.Graph(id="costs-total-charged", figure=costs_total_charged_figure),
        # etc..


        # INSPECT WEEK
        html.H2("Inspect State For Date", style={'textAlign': 'center'}, id="inspect-state-for-week-heading"),
        html.H3(children="Portfolio Composition"),
        html.Span("Choose Week: "),
        dcc.Slider( # NOTE: Calculate number of weeks to set this up above
            id="week-slider",
            min=0,
            max=len(date_range),
            marks={i: 'W{}'.format(i) for i in range(len(date_range))},
            value=0,
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
            columns=[{"name": i, "id": i} for i in backtest["broker"]["blotter_history"].columns],
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
        html.H2(children="Trade Inspection", style={'textAlign': 'center'}),
        # Input (order id maybe or add in trade id also?)
        html.P(children="Provide an order/ticker ID and various information and graphs associated with the trade is shown."),
        html.Span("Order/Trade ID: "),
        dcc.Input(id='order-id-input', type='text', value='1'),
        #info table
        html.H3(children="Trade Info"),
        dash_table.DataTable(
            id='trade-table',
            columns=[{"name": i, "id": i} for i in backtest["broker"]["all_trades"].columns],
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
        #order
        html.H3(children="Order for Trade"),
        dash_table.DataTable(
            id='order-table',
            columns=[{"name": i, "id": i} for i in backtest["portfolio"]["order_history"].columns],
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
        #signal
        html.H3(children="Signal behind Order"),
        dash_table.DataTable(
            id='signal-table',
            columns=[{"name": i, "id": i} for i in backtest["portfolio"]["signals"].columns],
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
        # graph (prices and triple barrier)
        dcc.Graph(id="trade-prices-graph"),
        dcc.Graph(id="trade-triple-barrier-graph"),




        # RAW DATA INSPECTION
        html.H2(children="Raw Data Inspection", style={'textAlign': 'center'}),

        html.H3(children="Candle Stick Chart"),
        html.Span("Ticker: "),
        dcc.Input(id="ticker-search-field", type="text", value="AAPL"),
        dcc.Graph(id="candlestick-chart"),

        html.H3(children="Ticker Sharadar Equity Prices"),
        dash_table.DataTable(
            id='sep-table',
            columns=[{"name": i, "id": i} for i in market_data.ticker_data.columns],
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
        html.H3(children="Ticker Sharadar Equity Prices"),
        dash_table.DataTable(
            id='corp-actions-table',
            columns=[{"name": i, "id": i} for i in market_data.corp_actions.columns],
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
        dcc.Graph(id="rf-rate-graph", figure=rf_rate_figure),
        html.H3(children="Feature/ML Algorithm Data"),
        dash_table.DataTable(
            id='features-table',
            columns=[{"name": i, "id": i} for i in feature_data.feature_data.columns],
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


        # BACKTEST'S FINAL STATE TABLES 

        html.H2(children="Final State Tables", style={'textAlign': 'center'}),
        html.H3(children='Trades'),
        dash_table.DataTable(
            id='broker-blotter-table',
            columns=[{"name": i, "id": i} for i in backtest["broker"]["all_trades"].columns],
            data=backtest["broker"]["all_trades"].to_dict("records"), # Why no signal id?
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

        html.H3(children='Portfolio Signals'),
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

        html.H3(children='Portfolio Order History',),
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

        html.H3(children='Broker Cancelled Orders'),
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
        html.H2(children="Backtest Log", style={'textAlign': 'center'}),
    ]
    app_layout.extend(log_lines)
    app.layout = html.Div(app_layout)

    # CONFIG
    # All in markdown

    # SUMMARY STATISTICS



    @app.callback(Output("date-picker", "date"), [Input("week-slider", "value")])
    def update_date_picker(week_nr):
        date = date_range[week_nr]
        return date

    # INSPECT PORTFOLIO WEEK BY WEEK
    @app.callback(Output("inspect-state-for-week-heading", "children"), [Input("date-picker", "date")])
    def update_inspect_state_for_week_heading(date):

        date = pd.to_datetime(date) 

        return "Inspect State For Date ({})".format(date.strftime("%Y-%m-%d"))

    # @app.callback(Output("active-trades-value-graph", "figure"), [Input("date-picker", "date")])
    # def update_active_trades_value_graph(date):
    @app.callback(Output("active-trades-value-graph", "figure"), [Input("date-picker", "date")])
    def update_active_trades_value_graph(date):    
        date = pd.to_datetime(date) 
        blotter_history = backtest["broker"]["blotter_history"]
        try:
            # print("DATE: ", type(date), date, type(blotter_history.date))
            active_trades = blotter_history.loc[blotter_history.date == date]
        except Exception as e:
            # print(e)
            active_trades = pd.DataFrame(columns=blotter_history.columns)
        # print(active_trades)
        # Active trades are not closed, to no close_price is available
        long_trades = active_trades.loc[active_trades.direction == 1]
        short_trades = active_trades.loc[active_trades.direction == -1]
        
        portfolio_value = backtest["portfolio"]["portfolio_value"]
        balance = portfolio_value.loc[portfolio_value.index == date]["balance"]
        margin_account = portfolio_value.loc[portfolio_value.index == date]["margin_account"]

        long_trace = go.Bar(
            x=long_trades["ticker"],
            y=long_trades["cur_value"],
            name="Value of Long Trade",
            marker=go.bar.Marker(color="rgb(90, 198, 17)")
        )
        short_trace = go.Bar(
            x=short_trades["ticker"],
            y=short_trades["cur_value"],
            name="Value of Short Trade",
            marker=go.bar.Marker(color="rgb(237, 79, 35)")
        )
        """
        balance_margin_account_trace = go.Bar(
            x=["Balance", "Margin Account"],
            y=[balance, margin_account],
            name="Accounts",
            marker=go.bar.Marker(color="rgb(46, 108, 232)")
        )
        """

        data = [short_trace, long_trace] # balance_margin_account_trace
        
        layout=go.Layout(
            title='Portfolio Value Allocation ' + str(date),
            xaxis=dict(title="Ticker"),
            yaxis=dict(title="Dollar Value"),
            height=figure_height,
        )

        return dict(data=data, layout=layout)

    # @app.callback(Output("active-trades-amount-graph", "figure"), [Input("date-picker", "date")])
    # def update_portfolio_allocation_amount_graph(date):
    @app.callback(Output("active-trades-amount-graph", "figure"), [Input("week-slider", "value"), Input("date-picker", "date")])
    def update_portfolio_allocation_amount_graph(week_nr, date):
        date = pd.to_datetime(date) 

        blotter_history = backtest["broker"]["blotter_history"]
        try:
            active_trades = blotter_history.loc[blotter_history.date == date]
        except:
            active_trades = pd.DataFrame(columns=blotter_history.columns)

        long_trades = active_trades.loc[active_trades.direction == 1]
        short_trades = active_trades.loc[active_trades.direction == -1]

        long_trace = go.Bar(
            x=long_trades["ticker"],
            y=long_trades["amount"],
            name="Long Position",
            marker=go.bar.Marker(color="rgb(90, 198, 17)")
        )
        short_trace = go.Bar(
            x=short_trades["ticker"],
            y=[abs(row["amount"]) for _, row in short_trades.iterrows()],
            name="Short Position",
            marker=go.bar.Marker(color="rgb(237, 79, 35)")
        )
        """ Is this interesting? To have the net position in each Stock? 
        net_trace = go.Bar(
            x=active_trades["ticker"],
            y=active_trades["net_position"],
            name="Net Position",
            marker=go.bar.Marker(color="rgb(46, 108, 232)")
        )
        """

        data = [short_trace, long_trace]
        
        layout=go.Layout(
            title='Portfolio Share Allocation ' + str(date),
            xaxis=dict(title="Ticker"),
            yaxis=dict(title="Nr. Shares"),
            height=figure_height,
        )

        return dict(data=data, layout=layout)
        

    # @app.callback(Output("active-trades-table", "data"), [Input("date-picker", "date")])
    # def update_broker_blotter_history(date):
    @app.callback(Output("active-trades-table", "data"), [Input("week-slider", "value"), Input("date-picker", "date")])
    def update_broker_blotter_history(week_nr, date):

        date = pd.to_datetime(date) 

        blotter_history = backtest["broker"]["blotter_history"]
        try:
            active_trades = blotter_history.loc[blotter_history.date == date]
        except:
            active_trades = pd.DataFrame(columns=active_trades.columns)
        return active_trades.to_dict("rows")

    # TRADE INSPECTION

    @app.callback(Output("trade-table", "data"), [Input("order-id-input", "value")])
    def update_trade_table(order_id):
        try: 
            order_id = int(order_id)
        except:
            return {}
        all_trades = backtest["broker"]["all_trades"]
        trade = all_trades.loc[all_trades.order_id == order_id]
        return trade.to_dict("rows")
    
    @app.callback(Output("order-table", "data"), [Input("order-id-input", "value")])
    def update_order_table(order_id):
        try: 
            order_id = int(order_id)
        except:
            return {}
        all_orders = backtest["portfolio"]["order_history"]
        order = all_orders.loc[all_orders.order_id == order_id]
        return order.to_dict("rows")

    @app.callback(Output("signal-table", "data"), [Input("order-id-input", "value")])
    def update_signal_table(order_id):
        try: 
            order_id = int(order_id)
        except:
            return {}
        all_orders = backtest["portfolio"]["order_history"]
        order = all_orders.loc[all_orders.order_id == order_id].iloc[-1]
        signal_id = order["signal_id"]
        all_signals = backtest["portfolio"]["signals"]
        signal = all_signals.loc[all_signals.signal_id == signal_id]
        return signal.to_dict("rows")


    @app.callback(Output("trade-prices-graph", "figure"), [Input("order-id-input", "value")])
    def update_trade_prices_graph(order_id):
        try: 
            order_id = int(order_id)
        except:
            return {}
        all_trades = backtest["broker"]["all_trades"]
        trade = all_trades.loc[all_trades.order_id == order_id]

        if len(trade) == 0:
            return {}
        trade = trade.iloc[-1]
        date0 = trade["trade_date"].strftime("%Y-%m-%d")
        date1 = trade["timeout"].strftime("%Y-%m-%d")
        ticker_sep = market_data.get_ticker_data(trade["ticker"])
        path_of_prices = ticker_sep.loc[date0:date1]
        
        stop_loss_barrier = (1 + trade["stop_loss"]) * trade["fill_price"] # Allways negative
        take_profit_barrier = (1 + trade["take_profit"]) * trade["fill_price"] # Allways positive (but meaning changes base on trade.direction)

        ohlc_trace = go.Ohlc(
            x=path_of_prices.index,
            open=path_of_prices["open"],
            high=path_of_prices["high"],
            low=path_of_prices["low"],
            close=path_of_prices["close"]
        )

        shapes = [
            # Line Vertical
            {
                'type': 'line',
                'x0': trade["timeout"],
                'y0': stop_loss_barrier,
                'x1': trade["timeout"],
                'y1': take_profit_barrier,
                'line': {
                    'color': 'rgb(55, 128, 191)',
                    'width': 4,
                    'dash': 'dashdot',
                },
            },
            # Take profit (Green), might be on top or bottom
            {
                'type': 'line',
                'x0': path_of_prices.index.min(),
                'y0': take_profit_barrier,
                'x1': path_of_prices.index.max(),
                'y1': take_profit_barrier,
                'line': {
                    'color': 'rgb(96, 250, 96)',
                    'width': 4,
                    'dash': 'dashdot',
                },
            },
            # Stop loss (Red), might be on top or bottom
            {
                'type': 'line',
                'x0': path_of_prices.index.min(),
                'y0': stop_loss_barrier,
                'x1': path_of_prices.index.max(),
                'y1': stop_loss_barrier,
                'line': {
                    'color': 'rgb(250, 96, 96)',
                    'width': 4,
                    'dash': 'dashdot',
                },
            }
        ]

        figure = {
            "data": [ohlc_trace], 
            "layout": {
                'title': trade.ticker + ' Price Chart (not adjusted for dividends)',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Price ($)'},
                'height': figure_height
            }
        }
        figure['layout'].update(shapes=shapes)
        # figure['layout'].update(annotations=annotations)

        return figure 

    @app.callback(Output("trade-triple-barrier-graph", "figure"), [Input("order-id-input", "value")])
    def update_trade_triple_barrier_chart(order_id):
        try: 
            order_id = int(order_id)
        except:
            return {}
        all_trades = backtest["broker"]["all_trades"]
        trade = all_trades.loc[all_trades.order_id == order_id]
        if len(trade) == 0:
            return {}
        trade = trade.iloc[-1]

        ticker_sep = market_data.get_ticker_data(trade["ticker"])
        date0 = trade["trade_date"].strftime("%Y-%m-%d")
        date1 = trade["timeout"].strftime("%Y-%m-%d")
        path_of_prices = ticker_sep.loc[date0:date1]
        
        
        path_of_returns = pd.DataFrame(index=path_of_prices.index, columns=["low_return", "high_return"])
        path_of_returns["low_return"] = (((path_of_prices["low"] + trade.dividends_per_share) / trade.fill_price) - 1) * trade.direction
        path_of_returns["high_return"] = (((path_of_prices["high"] + trade.dividends_per_share) / trade.fill_price) - 1) * trade.direction
        path_of_returns["close_return"] = (((path_of_prices["close"] + trade.dividends_per_share) / trade.fill_price) - 1) * trade.direction

        low_return_trace = go.Scatter(
            x=path_of_returns.index,
            y=path_of_returns["low_return"],
            name="Return Series of Daily Low Prices"
        )

        high_return_trace = go.Scatter(
            x=path_of_returns.index,
            y=path_of_returns["high_return"],
            name="Return Series of Daily High Prices"
        )

        close_return_trace = go.Scatter(
            x=path_of_returns.index,
            y=path_of_returns["close_return"],
            name="Return Series of Daily Close Prices"
        )
        
        figure = {
            "data": [low_return_trace, high_return_trace, close_return_trace], 
            "layout": {
                'title': trade["ticker"] +  ' High/Low Return Chart with Exit Limits',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Return'},
                'height': figure_height
            }
        }
        shapes = [
            # Line Vertical
            {
                'type': 'line',
                'x0': trade["timeout"],
                'y0': trade["stop_loss"],
                'x1': trade["timeout"],
                'y1': trade["take_profit"],
                'line': {
                    'color': 'rgb(55, 128, 191)',
                    'width': 4,
                    'dash': 'dashdot',
                },
            },
            # Take profit (Green), might be on top or bottom
            {
                'type': 'line',
                'x0': path_of_returns.index.min(),
                'y0': trade["take_profit"],
                'x1': path_of_returns.index.max(),
                'y1': trade["take_profit"],
                'line': {
                    'color': 'rgb(96, 250, 96)',
                    'width': 4,
                    'dash': 'dashdot',
                },
            },
            # Stop loss (Red), might be on top or bottom
            {
                'type': 'line',
                'x0': path_of_returns.index.min(),
                'y0': trade["stop_loss"],
                'x1': path_of_returns.index.max(),
                'y1': trade["stop_loss"],
                'line': {
                    'color': 'rgb(250, 96, 96)',
                    'width': 4,
                    'dash': 'dashdot',
                },
            }
        ]
        
        figure['layout'].update(shapes=shapes)
        
        if trade["CLOSED"]:
            annotations = [
                dict(
                    x=trade["close_date"],
                    y=trade["total_ret"],
                    text='Return: {}, Date: {}, Cause: {}'.format(round(trade["total_ret"],3), trade["close_date"].strftime("%Y-%m-%d"), trade["close_cause"]),
                    showarrow=True,
                    arrowhead=7,
                    font = dict(
                        color = "black",
                        size = 12
                    ),
                    ax=0,
                    ay=-40
                )
            ]
            figure['layout'].update(annotations=annotations)

        return figure 


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

    @app.callback(Output("sep-table", "data"), [Input("ticker-search-field", "value")])
    def update_sep_table(ticker):

        ticker_sep = market_data.get_ticker_data(ticker)
        
        return ticker_sep.to_dict("rows")


    @app.callback(Output("corp-actions-table", "data"), [Input("ticker-search-field", "value")])
    def update_corp_actions_table(ticker):

        ticker_sep = market_data.corp_actions.loc[market_data.corp_actions.ticker == ticker]
        
        return ticker_sep.to_dict("rows")



    # BACKTEST FINAL STATE TABLES




    """ CALLBACKS """


    app.run_server(debug=True)


def visualize_triple_barrier_method(rows, cols, dates, ticker, sep_triple_barrier, ticker_sep):
    if (rows*cols) != len(dates):
        raise ValueError("Number of rows and columns must match the number of dates") 

    figure = tls.make_subplots(rows=rows, cols=cols)
    
    all_traces = []
    all_shapes = []
    all_annotations = []

    for i, date in enumerate(dates, 1):
        row = sep_triple_barrier.loc[(sep_triple_barrier.ticker == ticker) & (sep_triple_barrier.index == date)].iloc[-1]

        """
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(row)
        """

        path_of_prices = ticker_sep[date:row["timeout"]]

        path_of_returns = pd.DataFrame(index=path_of_prices.index, columns=["ret"])
        path_of_returns["ret"] = ((path_of_prices["close"] / path_of_prices.iloc[0]["close"]) - 1)

        upper_barrier = row["stop_loss_barrier"]
        lower_barrier = row["take_profit_barrier"]


        trace = go.Scatter(x=path_of_returns.index, y=path_of_returns["ret"])

        shapes = [
            # Line Vertical
            {
                "xref": "x{}".format(i),
                "yref": "y{}".format(i),
                'type': 'line',
                'x0': row["timeout"],
                'y0': lower_barrier,
                'x1': row["timeout"],
                'y1': upper_barrier,
                'line': {
                    'color': 'rgb(55, 128, 191)',
                    'width': 3,
                },
            },
            # Line Horizontal
            {
                "xref": "x{}".format(i),
                "yref": "y{}".format(i),
                'type': 'line',
                'x0': path_of_returns.index.min(),
                'y0': upper_barrier,
                'x1': path_of_returns.index.max(),
                'y1': upper_barrier,
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 4,
                    'dash': 'dashdot',
                },
            },
            {
                "xref": "x{}".format(i),
                "yref": "y{}".format(i),
                'type': 'line',
                'x0': path_of_returns.index.min(),
                'y0': lower_barrier,
                'x1': path_of_returns.index.max(),
                'y1': lower_barrier,
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 4,
                    'dash': 'dashdot',
                },
            }
        ]
            
        annotations = [
            dict(
                xref='x{}'.format(i),
                yref='y{}'.format(i),
                x=row["date_of_touch"],
                y=row["return_tbm"],
                text='Return: {}, Label: {}, Date: {}'.format(row["return_tbm"], row["primary_label_tbm"], row["date_of_touch"]),
                showarrow=True,
                arrowhead=7,
                font = dict(
                    color = "black",
                    size = 12
                ),
                ax=0,
                ay=-40
            )
        ]

        all_shapes.extend(shapes)
        all_annotations.extend(annotations)
        all_traces.append(trace)


    trace_nr = 0
    for row in range(1, rows+1):
        for col in range(1, cols+1):
            figure.append_trace(all_traces[trace_nr], row, col)
            trace_nr += 1


    figure['layout'].update(shapes=all_shapes)
    figure['layout'].update(annotations=all_annotations)

    py.offline.plot(figure, auto_open=True)













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
