import pandas as pd

"""
General approach for producing signals for automated trading system with ML:


I need to have a model for every year from 2008 (2010) - 20019 (might only end up backtesting from 2010-2018).

Each time a new model is made, I run validation with the last 30% of the data that is available at the time. This leaves 70% of data for training.

Once the optimal hyperparameters are found, I train each model on all the data that is available at the given time.

I might later run more advanced validation schemes (custom cross validation).

The model is tested on data from the end of the validation set and to the end of the dataset.
This makes it so that more and more data is available for training and validation and less and less is available for testing

"""

n_estimators = 100 # Nr of trees
criterion = "gini" # function to measure quality of split (impurity measure)
max_depth = 5 # depth of each tree
min_samples_split = 2 # minimum number of samples required to split an internal node
min_samples_leaf = 5 # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
min_weight_fraction_leaf = 0 # Not relevant unless samples are weighted unequally 