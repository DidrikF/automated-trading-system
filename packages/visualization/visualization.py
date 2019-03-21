import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def print_series(data, start, end, title, xlabel, ylabel):
    for label, series in data.items():
        series.plot(label=label, figsize=(16, 8))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def print_candle_stick_chart():
    pass

