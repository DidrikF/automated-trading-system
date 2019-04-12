import matplotlib.pyplot as plt
import pandas as pd

from automated_trading_system.multiprocessing.engine import pandas_mp_engine

"""
This file is intended to calculate values in preparation of running sep_features.py. This is because of
complications 
"""

"""
Need to calculate: 
- Market return for each date
- Equal weighted weekly market returns
- equal_weight_industry_avg(SEP[close]m-1 / SEP[close]m-12 - 1) 
- count(SEP[volume]d == 0) for 1 month of data / [monthly_turnover]m-1
"""

"""
Equal weighted weekly market returns
1. Calculate weekly returns for each company for all dates (use reindex, fill, shift and downsample)
2. calculate equal weighted market returns by calculate the average of the weekly returns for each date

Equal weight industry avg return
1. Same as above
2. Take the average of the weekly returns for each date for each industry
"""


def dividend_adjusting_prices_backwards(sep):
    """
    - sep contains all tickers for a date range.
    Adds dividend adjusted close prices to dataframe.
    """
    sep = sep.sort_values(by="date", ascending=False)
    adjustment_factor = 1

    # Looping backwards in time...
    for date, row in sep.iterrows():
        # At each date we want to adjust the price according to the accumulated 
        # adjustment factor from future dates, not the current date.
        sep.at[date, "adj_close"] = row["close"] / adjustment_factor

        # All the earlier dates than the current need to be adjusted according 
        # to the current accumulated adjustment factor, taking into account
        # any new dividend on the current date.
        adjustment_factor_update = (row["close"] + row["dividends"]) / row["close"]
        adjustment_factor = adjustment_factor * adjustment_factor_update

    return sep


def dividend_adjusting_prices_forwards(sep):
    adjustment_factor = 1
    # Looping forwards in time...
    for date, row in sep.iterrows():
        # Calculate adjustment factor
        adjustment_factor_update = (row["close"] + row["dividends"]) / row["close"]
        adjustment_factor = adjustment_factor * adjustment_factor_update
        # Update close
        sep.at[date, "adj_close_forward"] = row["close"] * adjustment_factor

    return sep


def add_weekly_and_12m_stock_returns(sep):
    """ sep only contains one ticker """
    sep_empty = True if (len(sep) == 0) else False

    if sep_empty == True:
        print("Got empty sep in add_weekly_and_12m_stock_returns, don't know why")
        return sep

    pd.options.mode.chained_assignment = None  # default='warn'
    # Reindex, forward fill and shift
    date_index = pd.date_range(sep.index.min(), sep.index.max())  # [0], [1]

    sep_filled = sep.reindex(date_index)
    sep_filled["adj_close"] = sep_filled["adj_close"].fillna(method="ffill")
    sep_filled_1w_behind = sep_filled.shift(periods=7)

    # Calculate weekly momentum/return
    sep_filled["mom1w"] = (sep_filled["adj_close"] / sep_filled_1w_behind["adj_close"]) - 1

    # Calculate 12 month momentum (mom12m_actual) for indmom in sep_industry_features.py
    sep_filled_12m_behind = sep_filled.shift(periods=365)
    sep_filled["mom12m_actual"] = (sep_filled["adj_close"] / sep_filled_12m_behind["adj_close"]) - 1

    # Downsample and set values
    sep_filled = sep_filled.loc[sep.index]

    sep[["mom1w", "mom12m_actual"]] = sep_filled[["mom1w", "mom12m_actual"]]

    return sep


def add_equally_weighted_weekly_market_returns(sep):
    # sep contains all tickers for a date range
    pd.options.mode.chained_assignment = None  # default='warn'

    dates = list(sep.index.unique())
    # print("add_equally_weighted_weekly_market_returns")
    # print("Dates: ", sep.index.min(), sep.index.max(), "Tickers: ", sep.ticker.unique())

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        pass

    for date in dates:
        sep_for_date = sep.loc[sep.index == date]
        avg_weekly_market_momentum = sep_for_date["mom1w"].mean()
        sep.loc[sep.index == date, "mom1w_ewa_market"] = avg_weekly_market_momentum

    return sep


if __name__ == "__main__":
    sep = pd.read_csv("./datasets/testing/sep_extended.csv", parse_dates=["date"], index_col="date")

    sep_adjusted = pandas_mp_engine(callback=dividend_adjusting_prices_backwards, atoms=sep, data=None, \
                                    molecule_key='sep', split_strategy='ticker', \
                                    num_processes=1, molecules_per_process=1)

    sep_adjusted.loc[sep_adjusted.ticker == "FCX"]["close"].plot(legend="close")
    sep_adjusted.loc[sep_adjusted.ticker == "FCX"]["adj_close"].plot(legend="adj_close")
    plt.show()

    # sep.sort_values(by=["ticker", "date"], ascending=True, inplace=True)
    # sep.to_csv("./datasets/testing/sep_prepared.csv")
