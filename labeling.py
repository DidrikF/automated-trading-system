import pandas as pd







def get_daily_volatility(close, span0=100):
    # Daily volatility, reindexed to close
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0 = df0[df0>0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values-1 # Daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


def apply_ptsl_on_t1(close, events, ptsl, molecule):
    events_ = events.loc[molecule]
    # Etc