"""
This scripts applies the Triple Barrier Method to label samples.

"""


import pandas as pd





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