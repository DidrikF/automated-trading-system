from numpy import square
import numpy as np
import math
from scipy.stats import t
import pandas as pd

def zero_benchmarked_r_squared(predictions, labels):
    """
    The denominator in the r-quared calculation is generally teh sum of squared excess returns without demeaning. 
    (NOTE: what does this mean)
    In many out of sample forecasting problems, predictions are compared against the historical mean reaturn.
    However, the historical mean return is so noisy compared to individual stock returns that it artificially 
    lowers the bar for "good" forecasting performance. This problem is avoided by setting the mean excess retur
    to zero when calculating R-squared.
    """

    return 1 - ((square(predictions - labels).sum()) / (square(labels)).sum())


def sample_binary_predictor(y_pred: pd.Series, y_true: pd.Series, n_samples: int, sample_size: int, replace: bool=True):
    """
    The test set has over 300 000 samples
    """
    correct = y_true.eq(y_pred) # a series of bool, but you can treat the bools as numbers
    observations = []
    for _ in range(n_samples):
        sample = np.random.choice(correct, size=sample_size, replace=replace)
        accuracy = sample.sum() / sample.shape[0]
        observations.append(accuracy)
    
    return observations

def single_sample_t_test(observations: np.array, mean0, alpha):
    """
    This is an implementation of single-sided and single sample test of the mean of a normal distribution
    with unknown variance.
    In the context of this project, observations are some form of monthly returns or monthly return difference.
    NOTE: Requires that observations are pre-computed.
    """

    mean_obs = observations.mean()
    sample_std = math.sqrt(square(observations - mean_obs).sum() / (len(observations) - 1))

    t_statistic = (mean_obs - mean0) / (sample_std / math.sqrt(len(observations)))

    #  test if t_statistic > t(alpha, n-1)
    # ppf(q, df, loc=0, scale=1)	Percent point function (inverse of cdf — percentiles).
    t_val = t.ppf(1 - alpha, df=len(observations)-1)

    if t_statistic > t_val:
        return "Reject H0 with t_statistic={}, t_val={} and alpha={}".format(t_statistic, t_val, alpha)

    else:
        return "Filed to reject H0 with t_statistic={}, t_val={} and alpha={}".format(t_statistic, t_val, alpha)



def two_sample_t_test_with_unknown_variance(observations, benchmark):
    pass