import pandas as pd

import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls


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





def candlestick_chart(ohlc: pd.DataFrame):
    cs_chart = go.Candlestick(x=ohlc.index,
                open=ohlc["open"],
                high=ohlc['high'],
                low=ohlc['low'],
                close=ohlc['close'])
    
    figure = dict(
        data=[cs_chart]
    )

    py.offline.plot(figure, auto_open=True)