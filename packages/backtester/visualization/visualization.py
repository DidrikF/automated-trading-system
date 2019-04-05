import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# from  matplotlib.finance import candlestick_ohlc
import datetime


def print_series(data, start, end, title, xlabel, ylabel):
    for label, series in data.items():
        series.plot(label=label, figsize=(16, 8))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_candle_stick_chart():
    pass



def plot_data(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel=xlabel, ylabel=ylabel,
        title=title)
    ax.grid()

    # fig.savefig("test.png")
    plt.show()