from numpy import square

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



def single_sample_t_test(observations, benchmark):
    pass


def two_sample_t_test_with_unknown_variance(observations, benchmark):
    pass