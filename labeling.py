"""
This scripts applies the Triple Barrier Method to label samples.

Code is from Advances in Financial Machine Learning by Marco Lopez de Prado - Chapter 3
"""


import pandas as pd
from packages.multiprocessing.engine import pandas_mp_engine




def get_daily_volatility(close, span0=100):
    # Daily volatility, reindexed to close
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0 = df0[df0>0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values-1 # Daily returns
    df0 = df0.ewm(span=span0).std()
    return df0




def apply_ptsl_on_t1(close, events, ptSl, molecule):
    """
    The output from this function is a pandas dataframe containing 
    the timestamps (if any) at which each barrier was touched
    
    close: A pandas series of prices.
    events: A pandas dataframe, with columns,
    t1: The timestamp of vertical barrier. When the value is np.nan, there will not be a vertical barrier.
    trgt: The unit width of the horizontal barriers.
    ptSl: A list of two non-negative float values:
    ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier. 
             If 0, there will not be an upper barrier.
    ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier. 
             If 0, there will not be a lower barrier.
    molecule: A list with the subset of event indices that will be processed by a 
              single thread. Its use will become clear later on in the chapter.


    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    """

    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0]*events_['trgt']
    else: 
        pt = pd.Series(index=events.index) # NaNs
    if ptSl[1] > 0:
        sl = ptSl[1]*events_['trgt']
    else: 
        sl = pd.Series(index=events.index) # NaNs

    for loc, t1 in events_["t1"]: # path prices
        df0 = close[loc:t1] # path prices
        df0 = (df0/close[loc] - 1)*events_.at[loc, "side"] # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min() # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min() # earliest profit taking

    return out


barrier_configuration = [1, 1, 1] # [pt, sl, t1]



def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False):
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

    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt>minRet] # minRet
    
    # 2) Get t1 (max holding period)

    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
        
    # 3) form events objects, apply stop loss on t1

    side_ = pd.Series(1.0, index=trgt.index)
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=["trgt"])

    df0 = pandas_mp_engine(func=apply_ptsl_on_t1, pdObj=('molecule', events.index), numThreads=numThreads, \
        close=close, events=events, ptSl=[ptSl, ptSl])

    events["t1"] = df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    events = events.drop("side", axis=1)

    return events


# Adding vertical barriers:
# Don't know where it is best to add this code

t1 = close.index. searchsorted(tEvents + pd.Timedelta(days=numDays))
t1 = t1[t1<close.shape[0]]
t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]) # NaNs at end


def getBins(events, close):
    # 1) Prices aligned with events
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindexed(px, method="bfill")

    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_index] - 1
    out["bin"] = np.sign(out["ret"])

    return out






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
    
    events = pd.concat({"t1": t1: "trgt": trgt, "side": side_}, axis=1).dropna(subset=["trgt"])
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




def drop_labels(events, min_pct=0.05):
    # Apply weights, drop labels with insufficient examples
    while True:
        df0 = events["bin"].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print("droppe label", df0.argmin(), df0.min())
        events = events[events["bin"] != df0.argmin()]

    return events