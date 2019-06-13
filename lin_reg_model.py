
# Algorithms
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict # Don't know if I will use
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Performance Metrics 
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Other
import scipy
from scipy.stats import randint
import pandas as pd
import pickle

from cross_validation import PurgedKFold, cv_score
from dataset_columns import features, labels, base_cols

from performance_measurement import zero_benchmarked_r_squared

# DATASET PREPARATION
print("Reading ML Dataset")
dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date", "timeout"], index_col=["date"])
dataset = dataset.loc[dataset.primary_label_tbm != 0]

dataset = dataset.sort_values(by=["date"]) # important for cross validation


# Feature scaling
std_scaler = StandardScaler()
# Maybe not standardize labels..., also many of the columns are not numeric at this point, do this process below...
dataset[features] = std_scaler.fit_transform(dataset[features]) 
scaled_dataset = dataset
# scaled_dataset["erp_1m"] = dataset["erp_1m"]

train_start = scaled_dataset.index.min() # Does Date index included in features when training a model?
train_end = pd.to_datetime("2012-01-01")

test_start = pd.to_datetime("2012-03-01")
test_end = scaled_dataset.index.max()

train_set = scaled_dataset.loc[(scaled_dataset.index >= train_start) & (scaled_dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
test_set = scaled_dataset.loc[(scaled_dataset.index >= test_start) & (scaled_dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

print("train tickers: ", len(train_set["ticker"].unique()))
print("test tickers: ", len(test_set["ticker"].unique()))

# Need to select the top principal components...
print("train set shape ", train_set.shape)
print("test set shape ", test_set.shape)
# print("Start and end dates: ", min(dataset.index), max(dataset.index))
# print("train set num tickers", len(train_set["ticker"].unique()))
# print("test set num tickers", len(test_set["ticker"].unique()))
print(scaled_dataset.columns)

train_x = train_set[features]
train_y = train_set["erp_1m"]

test_x = test_set[features]
test_y = test_set["erp_1m"]


regressor = linear_model.LinearRegression()
regressor.fit(train_x, train_y)

# IN SAMPLE
train_x_pred = regressor.predict(train_x)
r_squared = zero_benchmarked_r_squared(train_x_pred, train_y)
print("In sample results")
print("OOS Zero Benchmarked R-Squared: ", r_squared)
print("R-squared: ", r2_score(train_y, train_x_pred))
print("OOS MSE: ", mean_squared_error(train_y, train_x_pred))
print("OOS MAE: ", mean_absolute_error(train_y, train_x_pred))


# OUT OF SAMPLE
test_x_pred = regressor.predict(test_x)
r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
print("Out of sample results")
print("OOS Zero Benchmarked R-Squared: ", r_squared)
print("R-squared: ", r2_score(test_y, test_x_pred))
print("OOS MSE: ", mean_squared_error(test_y, test_x_pred))
print("OOS MAE: ", mean_absolute_error(test_y, test_x_pred))
