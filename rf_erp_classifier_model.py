import pandas as pd
import pickle

# Algorithms
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


# Performance Metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

# Other
import scipy
from scipy.stats import randint

from cross_validation import PurgedKFold, cv_score
from dataset_columns import features, labels, base_cols

from model_and_performance_visualization import plot_feature_importances

"""
General approach for producing signals for automated trading system with ML:


I need to have a model for every year from 2008 (2010) - 20019 (might only end up backtesting from 2010-2018).

Each time a new model is made, I run validation with the last 30% of the data that is available at the time. This leaves 70% of data for training.

Once the optimal hyperparameters are found, I train each model on all the data that is available at the given time.

I might later run more advanced validation schemes (custom cross validation).

The model is tested on data from the end of the validation set and to the end of the dataset.
This makes it so that more and more data is available for training and validation and less and less is available for testing

"""


# You can generate a plot for precition and recall, see chapter 3 in hands-on machine learning


# DATASET PREPARATION
dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date", "timeout"], index_col=["date"])
dataset = dataset.loc[dataset.primary_label_tbm != 0]

dataset = dataset.sort_values(by=["date"]) # important for cross validation

# Feature scaling not needed

# Encoding Categorical Features:
# Not using industry now


train_start = dataset.index.min() # Does Date index included in features when training a model?
train_end = pd.to_datetime("2012-01-01")

test_start = pd.to_datetime("2012-03-01")
test_end = dataset.index.max()

train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets


train_x = train_set[features]
train_y = train_set["primary_label_tbm"]

test_x = test_set[features]
test_y = test_set["primary_label_tbm"]

training_model = True

if training_model:
    
    rf_classifier = RandomForestClassifier(random_state=100)

    # Define parameter space:
    num_samples = len(train_set)
    parameter_space = {
        "n_estimators": [20, 50, 100, 200, 500, 1000],
        "max_depth": [1, 2, 4, 8, 10, 15, 20, 25, 30], # max depth should be set lower I think
        "min_samples_split": [int(num_samples*0.02),int(num_samples*0.04),int(num_samples*0.06),int(num_samples*0.08)], # I have 550,000 samples for training -> 5500
        "min_samples_leaf": [int(num_samples*0.02),int(num_samples*0.04),int(num_samples*0.06),int(num_samples*0.08)], # 2-8% of samples 
        "max_features": [1, 5, 8, 10, 15, 20, 30, 40, 50],
        "class_weight": [None, "balanced_subsample"],
        "bootstrap": [True, False],
        "criterion": ["entropy", "gini"]
    }

    
    t1 = pd.Series(index=train_set.index, data=train_set["timeout"])

    random_search = RandomizedSearchCV(
        estimator=rf_classifier,
        param_distributions=parameter_space,
        n_iter=2,
        #  NOTE: need to update to use the date and timout columns
        cv=PurgedKFold(n_splits=5, t1=t1), # a CV splitter object implementing a split method yielding arrays of train and test indices
        # Need to figure out if just using built in scorers will work with the custom PurgedKFold splitter
        scoring="accuracy", # a string or a callable to evaluate the predictions on the test set (use custom scoring function that works with PurgedKFold)
        n_jobs=6,
        verbose=1
    )

    random_search.fit(train_x, train_y) # Validation is part of the test set in this case....
    
    print("Best Score (Accuracy): \n", random_search.best_score_)
    print("Best Params: \n", random_search.best_params_)
    print("Best Index: \n", random_search.best_index_)
    print("CV Results: \n", random_search.cv_results_)

    # I need a better way to visualize the performance... I think

    best_rf_classifier: RandomForestClassifier = random_search.best_estimator_
    test_x_pred = best_rf_classifier.predict(test_x)
    accuracy = accuracy_score(test_y, test_x_pred)
    precision = precision_score(test_y, test_x_pred)
    recall = recall_score(test_y, test_x_pred)

    print("OOS Accuracy: ", accuracy)
    print("OOS Precision: ", precision)
    print("OOS Recall: ", recall)

    plot_feature_importances(best_rf_classifier, train_x.columns)

    # Store model:
    pickle.dump(best_rf_classifier, open("./models/rf_erp_classifier_model.pickle", 'wb'))
    




    """
    RandomForestClassifier(
        n_estimators = 1000, # Nr of trees
        criterion = "entropy", # function to measure quality of split (impurity measure)
        max_depth = 5, # depth of each tree
        min_samples_split = 2, # minimum number of samples required to split an internal node
        min_samples_leaf = 5, # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
        # NOTE: Set to a lower value to force discrepancy between trees
        max_features = "auto", # the number of features to consider when looking for the best split (auto = sqrt(n_features))
        # NOTE: Need to read up on
        class_weight = "balanced_subsample", # use this attribute to set weight of different feature columns
        bootstrap = True, # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
        n_jobs = 6, # Number of jobs to run in parallell during fit and predict (-1 means to use all available)
        # NOTE: Set to a sufficiently large value (e.g. 5%) such that out-of bag accuracy converges to out of sample (k-fold) accuracy.
        # min_weight_fraction_leaf = 0, # Not relevant unless samples are weighted unequally 
    )
    """


    # NOTE: Use BaggingClassifier on RandomForestClassifier where mex-samples is set to average uniqueness between samples
    # average_uniqueness = 666 # NOTE: not implemented
    # bagging_classifier = BaggingClassifier(vase_estimator=clf0, n_estimators=1000, max_samples=average_uniqueness,max_features=1)


    # Hyper parameter tuning
    # - n_estimators (at some point more does not help, performance may even drop)
    # - max_depth (too high and the classifier overfits)
    # - min_sample_split (contoll how many branches you get, as it will stop if too few samples are available to calculate the next split)
    # - min_samples_leaf ()
    # - max_features (constrain to generate discrepancy)


    """
    Using sequential bootstrapping instead of normal bootstraping:

    """

    """
    How to address RF overfitting:
    1. Set a parameter max_features to a lower value, as a way of forcing discrepancy between trees.
    2. Early stopping: Set the regularization parameter min_weight_fraction_leaf to a sufficiently 
    large value (e.g., 5%) such that out-of-bag accuracy converges to out-of-sample (k-fold) accuracy.
    3. Use BaggingClassifier on DecisionTreeClassifier where max_samples is set to the average uniqueness
    (avgU) between samples.
    
    1. clf=DecisionTreeClassifier(criterion=‘entropy’,max_features=‘auto’,class_weight=‘balanced’)
    2. bc=BaggingClassifier(base_estimator=clf,n_estimators=1000,max_samples=avgU,max_features=1.)
    3, Use BaggingClassifier on RandomForestClassifier where max_samples is set to the average uniqueness 
    (avgU) between samples.
    4, clf=RandomForestClassifier(n_estimators=1,criterion=‘entropy’,bootstrap=False,class_weight=‘balanced_subsample’)
    5. bc=BaggingClassifier(base_estimator=clf,n_estimators=1000,max_samples=avgU,max_features=1.)
    6. Modify the RF class to replace standard bootstrapping with sequential bootstrapping.
    """


# Model Testing:
# Calculate Performance Metrics

# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# print(roc_auc)


"""

# Random Forest Classifier Configuration
# A lot of the options has to do with how trees are grown. These settings are probably also available for decision trees.
n_estimators = 1000, # Nr of trees
criterion = "entropy", # function to measure quality of split (impurity measure)
max_depth = 5, # depth of each tree
min_samples_split = 2, # minimum number of samples required to split an internal node
min_samples_leaf = 5, # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
# NOTE: Set to a lower value to force discrepancy between trees
max_features = "auto", # the number of features to consider when looking for the best split (auto = sqrt(n_features))

# NOTE: Need to read up on
class_weight = "balanced_subsample", # use this attribute to set weight of different feature columns

bootstrap = True, # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
n_jobs = 6, # Number of jobs to run in parallell during fit and predict (-1 means to use all available)

# NOTE: Set to a sufficiently large value (e.g. 5%) such that out-of bag accuracy converges to out of sample (k-fold) accuracy.
# min_weight_fraction_leaf = 0, # Not relevant unless samples are weighted unequally 
    
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



"""

"""
First output, need to do a better job of generalizing


Best Score:
 0.5065522970590343
Best Params:
 {'class_weight': None, 'criterion': 'entropy', 'max_depth': 39, 'max_features': 57, 'min_samples_leaf': 22390, 'min_samples_split': 13637, 'n_estimators': 971}
Best Index:
 1
CV Results:
 {
    'mean_fit_time': array([2222.89805403, 3656.15981908]), 
    'std_fit_time': array([1055.76469708, 1736.67012387]), 
    'mean_score_time': array([5.66908627, 4.9568913 ]), 
    'std_score_time': array([0.20852488, 0.65835138]), 
    'param_class_weight': masked_array(data=['balanced_subsample', None],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_criterion': masked_array(data=['entropy', 'entropy'],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_max_depth': masked_array(data=[54, 39],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_max_features': masked_array(data=[29, 57],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_min_samples_leaf': masked_array(data=[18179, 22390],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_min_samples_split': masked_array(data=[26168, 13637],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_n_estimators': masked_array(data=[861, 971],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'params': [
                {'class_weight': 'balanced_subsample', 
                'criterion': 'entropy', 
                'max_depth': 54, 
                'max_features': 29, 
                'min_samples_leaf': 18179, 
                'min_samples_split': 26168, 
                'n_estimators': 861
                }, 
                {
                    'class_weight': None, 
                    'criterion': 'entropy', 
                    'max_depth': 39, 
                    'max_features': 57, 
                    'min_samples_leaf': 22390, 
                    'min_samples_split': 13637, 
                    'n_estimators': 971}
                ], 
                    'split0_test_score': array([0.34422877, 0.51496245]), 
                    'split1_test_score': array([0.41051415, 0.5061814 ]), 
                    'split2_test_score': array([0.40437897, 0.51146158]), 
                    'split3_test_score': array([0.42644714, 0.51551704]), 
                    'split4_test_score': array([0.37215912, 0.48463876]), 
                    'mean_test_score': array([0.39154568, 0.5065523 ]), 
                    'std_test_score': array([0.02952065, 0.0114502 ]), 
                    'rank_test_score': array([2, 1]), 
                    
                    'split0_train_score': array([0.38461718, 0.53389197]), 
                    'split1_train_score': array([0.44641139, 0.58192798]), 
                    'split2_train_score': array([0.48507628, 0.60720768]), 
                    'split3_train_score': array([0.56298404, 0.6538934 ]), 
                    'split4_train_score': array([0.6378981 , 0.68455441]), 
                    'mean_train_score': array([0.5033974 , 0.61229509]), 
                    'std_train_score': array([0.08869365, 0.05300362])}
"""