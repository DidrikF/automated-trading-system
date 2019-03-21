import pandas as pd

from np import norm # ???


def getSignal(events, stepSize, prob: pd.Series, pred, numClasses, numThreads, **kargs):
    """
    Translates probabilities to bet size (Get signals from predictions).

    It handles the possibility that the prediction comes from a 
    meta-labeling estimator, as well from a standard labeling estimator.

    events -> a df of all bars (before sampling)
    stepSize -> 
    prob -> Series of predictions from primary ML model
    pred -> the prediction (probability) from the model which outputted the highest probability
    """

    if prob.shape[0] == 0: # Number of rows
        return pd.Series()
    
    # 1) generae signals from multinomial classification (one-vs-rest, OvR)
    z_value = (prob - 1/numClasses) / (prob * (1 - prob))**0.5 # t-value (z?) of OvR
    size = pred*(2*norm.cdf(z_value) - 1) # signal = side (pred or x) * size (m)
    
    if "side" in events:
        signal_continuous = size * events.loc[size.index, 'side'] # meta-labeling (only determines size (0 or 1 prediction), we need to add in the side for the signal to be complete)
    else:
        signal_continuous = size # If no meta-labeling, then both size and side can be derived from the primary model's prediction.
    # 2) compute average signal among those concurrently open
    """
    time_index  signal  t1
    01-04-2012  0.74    01-05-2012  
    """
    signals = signal_continuous.to_frame('signal').join(events[['t1']], how='left') # events["t1"] is the time of the first barrier being touched (need to investigate if this is the case)
    """
    df1.join(df2) -> is a convenient method for combining the columns of two 
    potentially differently-indexed DataFrames into a single result DataFrame
    """

    signals_averaged = avgActiveSignal(signals, numThreads)

    signals_discretized = discreteSignal(signals_averaged=signals_averaged, stepSize=stepSize)
    return signals_discretized




def avgActiveSignal(signals, numThreads):
    """
    Every bet is associated with a holding period, spanning form the 
    time it originated to the time the first barrier is touched, t1.
    --> I dont have the time of the first barrier being touched for new predictions...
    --> t1 was the timestamp of the vertical barrier originally...

    # A more sensible approach is to average all sizes 
    # across all bets still active at a given point in time. (WHAT DOES THIS MEAN???)



    """
    # compute the averate signal among those active
    # 1) time points where signals change (either one starts or one ends)
    # times where signals end
    time_points = set(signals['t1'].dropna().values) # Drop missing values and only get the unique signals
    # times where signals start
    time_points = time_points.union(signals.index.values) # add inn all signal dates not allready in time_points
    time_points = list(time_points)
    time_points.sort() # Earliest dates first

    out = mpPandasObj(mpAvgActiveSignals, ('molecule', time_points), numThreads, signals=signals)
    return out

def mpAvgActiveSignals(signals, molecule): # Molecule = time_points
    """
    Motivation: Averaging reduces some of the excess turnover that will happen if bets are replaced
    in their entirety when a new signal (data that triggers a trade) comes in, but still it is likely 
    that small trades will be triggered with every prediction.

    At time loc, average signal among those still active
    Signal is active if:
        a) issued before or at loc AND
        b) loc before signal's endtime, or endtime is still unknown (NaT)
    
    ->  molecule: A list with the subset of event indices that will be processed
        by a single thread. Its use will become clear later on in the chapter.
    ->
    """
    out = pd.Series()
    for loc in molecule: # loc is the location where the signal (side*size [-1, 1]) changes
        df0 = (signals.index.values <= loc) & ((loc < signals["t1"]) | pd.isnull(signals["t1"])) # this is a conditional statement to select rows form the signlas dataframe
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0 # no signal active at this time
        
    return out

def discreteSignal(signals_averaged, stepSize):
    """
    Size discretization to prevent overtrading
    """
    signals_discretized = (signals_averaged/stepSize).round() * stepSize # discretize
    signals_discretized[signals_discretized > 1] = 1 # cap
    signals_discretized[signals_discretized < -1] = -1 # floor
    return signals_discretized



# _________________________Dynamic Position Size and Limit Price_________________________


"""
Limit order:{
    ticker: 'AAPL',
    type: 'buy',
    size: '56 stocks',
    limit_price: '$100',
}


w controls the curvature of the function that translates price divergence to bet sizes


"""


def betSize(w, x):
    return x * (w + x**2)**-0.5


def getTPos(w, f, mP, maxPos):
    # calculate the desired position associated with price difference (current less forecasted price), and the calibrated w.
    return int(betSize(w, f-mP)*maxPos)

def invPrice(f, w, m):
    return f - m * (w / (1 - m**2))**0.5


def limitPrice(tPos, pos, f, w, maxPos):
    sgn = (1 if tPos >= pos else -1)
    lP = 0
    for j in range(abs(pos + sgn), abs(tPos+ 1)): # xrange returns a generator and is more memory efficient
        lP += invPrice(f, w, j/float(maxPos))
    lP = lP / (tPos - pos)
    return lP



def getW(x, m):
    # 0 < alpha < 1
    return x**2 * (m**-2 - 1)

# An example of usage
def main():
    pos = 0 # current position ???
    maxPos = 100 # maximum size of position (dollars? stocks? I think dollars are more accurat and also works well when dealing with different stocks)
    mP = 100 # current price
    f = 115 # forecasted price
    # This is used to calibrate w. When the price divergence is $10, we want the desired position to be 95% of the maximum possible position.
    wParams = {
        'divergence': 10,   # probably a function of the stocks volatility
        'm': 0.95           
    }

    # W is calibrated such that price difference gets translated as desired into 
    w = getW(wParams['divergence'], wParams['m']) # calibrate w
    """
    get the desired position (a percent of maxPos (dollar amount i think is best, 
    but you can denominate all in shares as well, its just important to be consistent 
    as I understand it))
    """
    tPos = getTPos(w, f, mP, maxPos) 


    """
    The limit price is calculated based on the 
    """
    lP = limitPrice(tPos, pos, f, w, maxPos) # limit price for order
    
    return

if __name__ == '__main__':
    main()