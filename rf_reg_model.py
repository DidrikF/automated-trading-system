import pandas as pd
import pickle

# Algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


# Performance Metrics 
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Other
import scipy
from scipy.stats import randint

from cross_validation import PurgedKFold, cv_score
from dataset_columns import features, labels, base_cols
from model_performance_reporting import zero_benchmarked_r_squared

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
train_y = train_set["erp_1m"]

test_x = test_set[features]
test_y = test_set["erp_1m"]

training_model = True

if training_model:
    
    rf_regressor = RandomForestRegressor(random_state=100)

    # Define parameter space:
    num_samples = len(train_set)
    parameter_space = {
        "n_estimators": [20, 50], # 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "max_depth": [1, 2, 4, 8, 10, 15, 20, 25, 30], # max depth should be set lower I think
        "min_samples_split": [int(num_samples*0.02),int(num_samples*0.04),int(num_samples*0.06),int(num_samples*0.08)], # I have 550,000 samples for training -> 5500
        "min_samples_leaf": [int(num_samples*0.02),int(num_samples*0.04),int(num_samples*0.06),int(num_samples*0.08)], # 2-8% of samples 
        "max_features": [1, 5, 10, 15, 20, 30, 40, 50, 60],
        "class_weight": [None, "balanced_subsample"],
        "bootstrap": [True, False],
        "criterion": ["mse"]
    }

    t1 = pd.Series(index=train_set.index, data=train_set["timeout"])

    random_search = RandomizedSearchCV(
        estimator=rf_regressor,
        param_distributions=parameter_space,
        n_iter=2,
        #  NOTE: need to update to use the date and timout columns
        cv=PurgedKFold(n_splits=5, t1=t1), # a CV splitter object implementing a split method yielding arrays of train and test indices
        # Need to figure out if just using built in scorers will work with the custom PurgedKFold splitter
        scoring="mse", # a string or a callable to evaluate the predictions on the test set (use custom scoring function that works with PurgedKFold)
        n_jobs=6,
        verbose=1
    )

    random_search.fit(train_x, train_y, ) # Validation is part of the test set in this case....
    
    print("Best Score: \n", random_search.best_score_)
    print("Best Params: \n", random_search.best_params_)
    print("Best Index: \n", random_search.best_index_)
    print("CV Results: \n", random_search.cv_results_)
    
    # Measure performance
    best_rf_regressor = random_search.best_estimator_
    test_x_pred = best_rf_regressor.predict(test_x)
    r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
    
    print("OOS Zero Benchmarked R Squared: ", r_squared)
    print("OOS MSE: ", mean_squared_error(test_x_pred, test_y))
    print("OOS MAE: ", mean_absolute_error(test_x_pred, test_y))

    # Save Best Model
    pickle.dump(best_rf_regressor, open("./models/rf_erp_regression_model.pickle", 'wb'))
