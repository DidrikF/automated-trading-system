import pandas as pd
import pickle

# Algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Performance Metrics 
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Other
import scipy
from scipy.stats import randint

from cross_validation import PurgedKFold, cv_score
from dataset_columns import features, labels, base_cols
from model_performance_reporting import zero_benchmarked_r_squared

n_jobs = 64

# DATASET PREPARATION
dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date", "timeout"], index_col=["date"])
dataset = dataset.loc[dataset.erp_1m != 0]

dataset = dataset.sort_values(by=["date"]) # important for cross validation


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
    
    rf_regressor = RandomForestRegressor(
        n_estimators=1000,
        min_weight_fraction_leaf=0.2,
        max_features=5,
        class_weight="balanced_subsample",
        bootstrap=True,
        criterion="mse",
        n_jobs=n_jobs,
    )


    # Measure performance
    test_x_pred = rf_regressor.predict(test_x)
    r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
    
    print("OOS Zero Benchmarked R Squared: ", r_squared)
    print("OOS MSE: ", mean_squared_error(test_x_pred, test_y))
    print("OOS MAE: ", mean_absolute_error(test_x_pred, test_y))

    # Save Best Model
    pickle.dump(rf_regressor, open("./models/rf_erp_regression_model.pickle", 'wb'))





"""
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
"""