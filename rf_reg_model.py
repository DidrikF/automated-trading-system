import pandas as pd
import pickle
import numpy as np

# Algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Performance Metrics 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Other
import scipy
from scipy.stats import randint

from cross_validation import PurgedKFold, cv_score
from dataset_columns import features, labels, base_cols
from performance_measurement import zero_benchmarked_r_squared

from model_and_performance_visualization import plot_feature_importances

n_jobs = 6

# DATASET PREPARATION
print("Reading dataset")
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
    rf_regressor = RandomForestRegressor(verbose=True, n_jobs=6)

    parameter_space = {
        "n_estimators": [200, 500, 1000], 
        "min_weight_fraction_leaf": [0.05, 0.10, 0.20], 
        "max_features": [5, 10],
        "bootstrap": [True], 
        "criterion": ["mse"]
    }

    t1 = pd.Series(index=train_set.index, data=train_set["timeout"])

    grid_search = GridSearchCV(
        estimator=rf_regressor,
        param_grid=parameter_space,
        cv=PurgedKFold(n_splits=3, t1=t1), # a CV splitter object implementing a split method yielding arrays of train and test indices
        scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"], # a string or a callable to evaluate the predictions on the test set
        refit="neg_mean_squared_error",
        n_jobs=n_jobs,
        verbose=2,
        error_score=np.nan
    )

    grid_search.fit(train_x, train_y)
    best_estimator = grid_search.best_estimator_
    # Measure performance
    test_x_pred = grid_search.predict(test_x)
    r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
    r2 = r2_score(test_y, test_x_pred)
    mse = mean_squared_error(test_y, test_x_pred)
    mae = mean_absolute_error(test_y, test_x_pred)

    print("OOS Zero Benchmarked R Squared: ", r_squared)
    print("OOS R Squared: ", r2)
    print("OOS MSE: ", mse)
    print("OOS MAE: ", mae)

    plot_feature_importances(best_estimator, train_x.columns)

    # Save Best Model
    pickle.dump(best_estimator, open("./models/rf_erp_regression_model.pickle", 'wb'))

    results = {
        "r_squared": r_squared,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "best_params": grid_search.best_params_,
        "cv_results": grid_search.cv_results_,
    }
    
    pickle.dump(results, open("./models/rf_erp_regression_results.pickle", "wb"))
    