"""
This scripts applies the Triple Barrier Method to label samples.

Code is from Advances in Financial Machine Learning by Marco Lopez de Prado - Chapter 3
"""


import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import sys
import numpy as np
import math

from processing.engine import pandas_mp_engine
"""
Step by step guide to labeling via the triple barrier method and meta-labeling.

1.  Get return volatility of the price series for each stock by comptuting an exponentially weighted standard 
    deivation on monthly/weekly returns the past year.
"""

"""
def get_daily_volatility(close, span0=100):
    # Daily volatility, reindexed to close
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=30))
    df0 = df0[df0>0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values-1 # Daily returns
    df0 = df0.ewm(span=span0).std()
    return df0
"""




def getEvents(close: pd.Series, tEvents: pd.DatetimeIndex, ptSl: tuple, trgt: pd.Series, minRet: float, numThreads: int, t1=False):
    # sep_featured, sep, ptSl, minRet
    
    """
    Finds the time of the first barrier touch.

    close:      A pandas series of prices.
    tEvents:    The pandas timeindex containing the timestamps that will seed every triple barrier. 
                These are the timestamps selected by the sampling procedures discussed in Chapter 2, 
                Section 2.5.
    ptSl:       A non-negative float that sets the width of the two barriers. A 0 value means that the 
                respective horizontal barrier (profit taking and/or stop loss) will be disabled.
    t1:         A pandas series with the timestamps of the vertical barriers. We pass a False 
                when we want to disable vertical barriers.
    trgt:       A pandas series of targets, expressed in terms of absolute returns.
    minRet:     The minimum target return required for running a triple barrier search.
    """

    # Some of this preparation migth be done before giving it to this function, don't know...

    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt>minRet] # minRet # what is the result of this (sets those lower than minRet to NAN i think)
    
    # 2) Get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
        
    # 3) form events objects, apply stop loss on t1

    side_ = pd.Series(1.0, index=trgt.index) # ALlways assume long position when first calculating label for side.
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=["trgt"])

    df0 = pandas_mp_engine(func=apply_ptsl_on_t1, pdObj=('molecule', events.index), numThreads=numThreads, \
        close=close, events=events, ptSl=[ptSl, ptSl]) # notice that barriers are symmetric when first labeling for side.

    # drop those where none of the barrieres where touched (should be extremly rare, if not at all I think)
    events["t1"] = df0.dropna(how='all').min(axis=1) # pd.min ignores nan, here events["t1"] becomes the timesamp of the earliest barrier touch
    events = events.drop("side", axis=1)

    return events


def apply_ptsl_on_t1(close, events, ptSl, molecule): # Need to rewrite, molecule is not needed at the splitting will be done prior to this function being called
    """
    The output from this function is a pandas dataframe containing 
    the timestamps (if any) at which each barrier was touched
    
    close: A pandas series of prices.
    events: A pandas dataframe, with columns: ["t1", "trgt", "side"] (made in getEvents)
    t1: The timestamp of vertical barrier. When the value is np.nan, there will not be a vertical barrier.
    trgt: The unit width of the horizontal barriers.
    ptSl: A list of two non-negative float values:
    ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier. 
             If 0, there will not be an upper barrier.
    ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier. 
             If 0, there will not be a lower barrier.

    # Some thought and experimentation must go into how I set ptSl, and how trgt is calculated must then be
    # taken into account.

    molecule: A list with the subset of event indices that will be processed by a 
              single thread. Its use will become clear later on in the chapter.

    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    """

    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)

    # When first labeling for side prediction, ptSl should be two POSITIVE floats.
    if ptSl[0] > 0:
        pt = ptSl[0]*events_['trgt'] # trgt is the result of get_return_volatility.
    else: 
        pt = pd.Series(index=events.index) # NaNs
    if ptSl[1] > 0:
        sl = ptSl[1]*events_['trgt']
    else: 
        sl = pd.Series(index=events.index) # NaNs

    # If the events df does not have a value for each sample in events use the last available price as t1
    for loc, t1 in events_["t1"].fillna(close.index[-1]).iterrows(): # path prices (iteritems())
        df0 = close[loc:t1] # path prices
        # OBS where did "side" come from?
        df0 = (df0/close[loc] - 1)*events_.at[loc, "side"] # path returns (this takes into account both long and short positions)
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min() # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min() # earliest profit taking

    return out # this is a df with tree columns ["t1", "sl", "pt"] and the index is the date of the event 


barrier_configuration = [1, 1, 1] # [pt, sl, t1], where is this used exactly?



# Adding vertical barriers:
# Don't know where it is best to add this code
# Should be used to calculate the t1 argument to getEvents.
"""
t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays)) # Find the date in close.index where the date is numDays or later after the sample in tEvents
t1 = t1[t1<close.shape[0]]
t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]) # NaNs at end
"""

def getBins(events, close):
    # 1) Prices aligned with events
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindexed(px, method="bfill")

    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
    out["bin"] = np.sign(out["ret"])

    return out





# With the above you can label the dataset for side prediction, meta-labeling can be deferred to later.


# Metalabaling functions:

def get_events_metalabaling(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt>minRet] # minRet

    # 2) Get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)

    # 3) form events object, apply stop loss on t1
    if side is None: 
        side_, ptSl = pd.Series(1.0, index=trgt.index), [ ptSl[0], ptSl[0] ]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    
    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(subset=["trgt"])
    df0 = pandas_mp_engine(callback=apply_ptsl_on_t1, pdObj=("molecule", events.index), numThreads=numThreads, \
        close=inst["close"], events=events, ptSl=ptSl_)


    events["t1"] = df0.dropna(how="all").min(axis=1) # pd.min ignores nan
    if side is None: 
        events = events.drop("side", axis=1)

    return events


def getBins(events, close):
    """
    Compute event's outcome (include side information, if provided).
    events in a DataFrame where: 
    events.index is event's start time
    events["t1"] is event's endtime
    events["trgt"] is event's target
    events["side"] (optional) implies the algo's position side

    Case 1: ("side" not in events): bin (-1, 1) <- label by price action
    Case 2: ("side" in events): bin(0, 1) <- label by pnl (meta-labeling)
    """

    # 1) prices aligned with events
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")

    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] -1

    if "side" in events_:
        out["ret"] *= events_["side"] # meta-labeling
        out["bin"] = np.sign(out["ret"])
        
    if "side" in events_:
        out.loc[out["ret"] <= 0, "bin"] = 0 # Meta-labeling

    return out


"""

def drop_labels(events, min_pct=0.05):
    # Apply weights, drop labels with insufficient examples
    while True:
        df0 = events["bin"].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print("droppe label", df0.argmin(), df0.min())
        events = events[events["bin"] != df0.argmin()]

    return events

"""



def triple_barrier_search(events, sep, ptSl):
    barrier_touch_dates = events[["timeout"]].copy(deep=True)

    take_profit_barriers = ptSl[0] * events["ewmstd"] # I only consider the case where there is both horizontal barriers
    stop_loss_barriers = ptSl[1] * events["ewmstd"]

    for date, timeout in events["timeout"].fillna(sep.index[-1]).iterrows(): # If timeout is missing for an event (this should not be the case), we use the last date in sep (for the ticker) as timeout.
        path_of_prices = sep[date:timeout]["close"]
        path_of_returns = (path_of_prices / path_of_prices[0] - 1) * events.at[date, "side"] # remember we are still allways long
        barrier_touch_dates.loc[date, "earliest_take_profit_touch"] = path_of_returns[path_of_returns > take_profit_barriers[date]].index.min()
        barrier_touch_dates.loc[date, "earliest_stop_loss_touch"] = path_of_returns[path_of_returns < stop_loss_barriers[date]].index.min()

    return barrier_touch_dates



def get_first_barrier_touches(sep_featured, sep, ptSl: tuple, trgt: pd.Series, minRet: float):
    ewmstd = sep_featured[["ewmstd"]]

    # I only consider the case there there is a timeout
    timeout = sep_featured[["timeout"]] # NEED TO ADD

    side_long = pd.Series(1.0, index=ewmstd.index)

    # 3) form events objects, apply stop loss on t1
    events = pd.concat({'timeout': timeout, 'ewmstd': ewmstd, 'side': side_long}, axis=1) # .dropna(subset=["trgt"])

    
    sep_featured = pandas_mp_engine(callback=triple_barrier_search, atoms=sep_featured, \
        data={'sep': sep}, molecule_key='sep_sampled', split_strategy= 'ticker', \
            num_processes=1, molecules_per_process=1, ptSl=[0.9,0.9], minRet=None)

    # drop those where none of the barrieres where touched (should be extremly rare, if not at all I think)
    events["earliest_touch"] = df0.dropna(how='all').min(axis=1) # pd.min ignores nan, here events["t1"] becomes the timesamp of the earliest barrier touch
    events = events.drop("side", axis=1)

    return events


def get_primary_labels(events, sep):
    close = sep["close"]

    events_ = events.dropna(subset=["earliest_touch"])
    close_index = events_.index.union(events_["earliest_touch"].values).drop_duplicates()


    close_reindexed = close.reindexed(close_index, method="bfill")

    out = pd.DataFrame(index=events_.index)
    out["return"] = close_reindexed.loc[events_["timeout"]].values / close_reindexed.loc[events_.index] - 1
    out["primary_label"] = np.sign(out["return"])

    return out




# Think this will be the last callback in the sep pipeline
# How will this change when side is not allways long?
def add_labels_via_triple_barrier_method(sep_featured: pd.DataFrame, sep: pd.DataFrame, ptSl: tuple, min_ret):
    
    if ("ewmstd_2y_monthly" not in sep_featured.columns) or ("timeout" not in sep_featured.columns):
        return sep_featured

    ewmstd = sep_featured[["ewmstd_2y_monthly"]]

    # I only consider the case there there is a timeout
    timeout = sep_featured[["timeout"]] # NEED TO ADD

    side_long = pd.DataFrame(index=ewmstd.index)
    side_long["side"] = 1.0


    # 3) form events objects, apply stop loss on t1
    events = pd.concat({'timeout': timeout["timeout"], 'ewmstd': ewmstd["ewmstd_2y_monthly"], 'side': side_long["side"]}, axis=1) # .dropna(subset=["trgt"])
    # print(events)
    barrier_touch_dates = events[["timeout"]].copy(deep=True)

    take_profit_barriers = ptSl[0] * events["ewmstd"] # I only consider the case where there is both horizontal barriers
    stop_loss_barriers = ptSl[1] * events["ewmstd"]
    # print(stop_loss_barriers)

    for date, timeout in events["timeout"].fillna(sep.index[-1]).iteritems(): # If timeout is missing for an event (this should not be the case), we use the last date in sep (for the ticker) as timeout.
        path_of_prices = sep.loc[(sep.index >= date) & (sep.index <= timeout)][["close"]]
        path_of_returns = ((path_of_prices["close"] / path_of_prices.iloc[0]["close"]) - 1) * events.loc[events.index == date].iloc[0]["side"] # events.loc[date]["side"] # remember we are still allways long
        barrier_touch_dates.loc[date, "earliest_take_profit_touch"] = path_of_returns[path_of_returns > take_profit_barriers[date]].index.min()
        barrier_touch_dates.loc[date, "earliest_stop_loss_touch"] = path_of_returns[path_of_returns < stop_loss_barriers[date]].index.min()
        

    # print(barrier_touch_dates)
    # print(events)
    # drop those where none of the barrieres where touched (should be extremly rare, if not at all I think)
    events["earliest_touch"] = barrier_touch_dates.dropna(how='all').min(axis=1) # pd.min ignores nan, here events["t1"] becomes the timesamp of the earliest barrier touch

    # events = events.drop("side", axis=1)
    # events_ = events.dropna(subset=["earliest_touch"]) # Does not do anything.
    # sample_date__earliest_touch_date = events_.index.union(events_["earliest_touch"].values).drop_duplicates()
    # close_reindexed = close.reindex(close_index, method="bfill")

    # out = pd.DataFrame(index=events.index)
    events["return"] = sep.loc[events["earliest_touch"]]["close"].values / sep.loc[events.index]["close"].values - 1

    events["primary_label"] = np.sign(events["return"])

    sep_featured[["return_tbm", "primary_label_tbm"]] = events[["return", "primary_label"]]
    sep_featured["date_of_touch"] = events["earliest_touch"]
    sep_featured["take_profit_barrier"] = take_profit_barriers
    sep_featured["stop_loss_barrier"] = stop_loss_barriers

    return sep_featured




def equity_risk_premium_labeling(sep_featured, tb_rate):
    # 1.    To get the risk free rate over the month, take the most recent 3-month t-bill rate and make 
    #       it into a decimal number and divide it by 3 to get the monthly rate
    # 2.    Subtract the risk free rate from the return
    
    for date, row in sep_featured.iterrows():
        tb_rate_3m = tb_rate.loc[date]["rate"]

        rf_rate_1m = tb_rate_3m / 3
        rf_rate_2m = (tb_rate_3m / 3) * 2
        rf_rate_3m = tb_rate_3m

        sep_featured.at[date, "erp_1m"] = row["return_1m"] - rf_rate_1m
        sep_featured.at[date, "erp_2m"] = row["return_2m"] - rf_rate_2m
        sep_featured.at[date, "erp_3m"] = row["return_3m"] - rf_rate_3m

    return sep_featured


def process_tree_month_t_bill_rates(tb_rate):
    tb_rate = tb_rate.rename(columns={"DATE": "date", "DTB3": "rate"})
    tb_rate = tb_rate.set_index("date")
    tb_rate.index.name = "date"
    tb_rate.loc[tb_rate.rate == "."] = math.nan
    tb_rate["rate"] = pd.to_numeric(tb_rate["rate"])
    
    date_index = pd.date_range(tb_rate.index.min(), tb_rate.index.max())
    tb_rate = tb_rate.reindex(date_index)
    tb_rate["rate"] = tb_rate["rate"].fillna(method="ffill")
    tb_rate["rate"] = tb_rate["rate"] / 100
    tb_rate = tb_rate.round(4)

    return tb_rate

if __name__ == "__main__":
    tb_rate = pd.read_csv("./datasets/excel/three_month_treasury_bill_rate.csv", parse_dates=["DATE"], low_memory=False)

    tb_rate = process_tree_month_t_bill_rates(tb_rate)

    tb_rate.to_csv("./datasets/macro/t_bill_rate_3m.csv", index=True)