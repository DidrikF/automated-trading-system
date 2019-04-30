import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

"""
General approach for producing signals for automated trading system with ML:


I need to have a model for every year from 2008 (2010) - 20019 (might only end up backtesting from 2010-2018).

Each time a new model is made, I run validation with the last 30% of the data that is available at the time. This leaves 70% of data for training.

Once the optimal hyperparameters are found, I train each model on all the data that is available at the given time.

I might later run more advanced validation schemes (custom cross validation).

The model is tested on data from the end of the validation set and to the end of the dataset.
This makes it so that more and more data is available for training and validation and less and less is available for testing

"""

# Random Forest Classifier Configuration
# A lot of the options has to do with how trees are grown. These settings are probably also available for decision trees.
n_estimators = 100 # Nr of trees
criterion = "gini" # function to measure quality of split (impurity measure)
max_depth = 5 # depth of each tree
min_samples_split = 2 # minimum number of samples required to split an internal node
min_samples_leaf = 5 # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
max_features = "auto" # the number of features to consider when looking for the best split (auto = sqrt(n_features))
bootstrap = True # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
n_jobs = -1 # Number of jobs to run in parallell during fit and predict (-1 means to use all available)
random_state = 999 # Think this is important for reproducability of the results
verbose = 0 # controls verbosity during fit and predicting (may want to set this to other value, but what?)


# These are basically turned off.
min_weight_fraction_leaf = 0 # Not relevant unless samples are weighted unequally 
max_leaf_nodes = None # allows unlimited number of leaf nodes
min_impurity_decrease = 0 # A node will not be split if the split does not decrease impurity by more than min_impurity_decrease
# min_impurity_split = 1e-7 # threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf. (depreciated)
oob_score = False # Whether to use out-of-bag samples to estimate the generalization accuracy.
warm_start = False # whether to use trees allready built and add more, or start from scratch
class_weight = None # use this attribute to set weight of different feature columns

# Attributes listed in the docs are attributes the classifier you get when instantiating an object from the RandomForestClassifier Class


# Random Forest Regressor Congfiguration (basically all the same except the function to measure quality of split)

criterion = "mse"

# The classifier is able to return features ranked by importance (regressor.feature_importances_)
# But I might be more interested in gathering feature importance data from each 
# model based feature importance being measured as increase in MSE or R-squared.




# You can generate a plot for precition and recall, see chapter 3 in hands-on machine learning





