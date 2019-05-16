import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import load_model
from tensorflow.keras import Model

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from dataset_columns import features, labels, base_cols
from performance_measurement import zero_benchmarked_r_squared


# Load Dataset
dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date", "timeout"], index_col=["date"])
dataset = dataset.loc[dataset.primary_label_tbm != 0]

# Base for all models
dataset = dataset.sort_values(by=["date"]) # important for cross validation


# Standardized Datasets: (used for pca/lin-reg and dnn)
std_scaler = StandardScaler()
std_dataset = dataset.copy(deep=True)
std_dataset[features] = std_scaler.fit_transform(std_dataset[features]) 

# I guess this data is not needed here...
std_train_start = std_dataset.index.min() # Does Date index included in features when training a model?
std_train_end = pd.to_datetime("2009-09-01")
std_validation_start = pd.to_datetime("2010-01-01")
std_validation_end = pd.to_datetime("2012-01-01") 
std_test_start = pd.to_datetime("2012-03-01") # NOTE: The test set start for all models and backtests
std_test_end = std_dataset.index.max()


std_train_set = std_dataset.loc[(std_dataset.index >= std_train_start) & (std_dataset.index < std_train_end)] # NOTE: Use for labeling and constructing new train/test sets
std_validation_set = std_dataset.loc[(std_dataset.index >= std_validation_start) & (std_dataset.index < std_validation_end)]
std_test_set = std_dataset.loc[(std_dataset.index >= std_test_start) & (std_dataset.index <= std_test_end)] # NOTE: Use for labeling and constructing new train/test sets


std_train_x = std_train_set[features]
std_train_y = std_train_set["return_1m"] # maybe I don't need to update to erp_1m, this is also not adjuseted for dividends...

std_validation_x = std_validation_set[features]
std_validation_y = std_validation_set["return_1m"]

std_test_x = std_test_set[features]
std_test_y = std_test_set["return_1m"]


# Non Normalized Datasets: Used for all random forests (regression and classification)
train_start = dataset.index.min() # Does Date index included in features when training a model?
train_end = pd.to_datetime("2012-01-01")
test_start = pd.to_datetime("2012-03-01")
test_end = dataset.index.max()

train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

train_x = train_set[features]
train_y = train_set["primary_label_tbm"] # maybe I don't need to update to erp_1m, this is also not adjuseted for dividends...
test_x = test_set[features]
test_y = test_set["primary_label_tbm"]

# TODO:
# Calculate statistics on the test set
# Random Forest Regression Model:
# Equity Risk Premium Classification Model:
# Automated Trading System Models:
# Produce relevant graphs 
# Produce relevant tables
# Store information in a formant that can be used in the report

# Load each model: (3 regresson models, 3 classification models)
try:
    lin_reg_model: BaseEstimator = pickle.load(open("./models/lin_reg_model.pickle", 'rb'))
    
    test_x_pred = regressor.predict(test_x)
    r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
    print("Lin Reg ERP REGRESSOR:")
    print("OOS Zero Benchmarked R-Squared: ", r_squared)
    print("OOS MSE: ", mean_squared_error(test_x_pred, test_y))
    print("OOS MAE: ", mean_absolute_error(test_x_pred, test_y))
except:
    print("LIN-REG MODEL NOT AVAILABLE!")


try:
    rf_reg_model: BaseEstimator = pickle.load(open("./models/rf_erp_regression_model.pickle"))
    
    test_x_pred = rf_reg_model.predict(test_x)
    r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
    print("RF ERP REGRESSOR:")
    print("OOS Zero Benchmarked R Squared: ", r_squared)
    print("OOS MSE: ", mean_squared_error(test_x_pred, test_y))
    print("OOS MAE: ", mean_absolute_error(test_x_pred, test_y))
except:
    print("RF ERP REGRESSION MODEL NOT AVAILABLE!")

try:
    dnn_reg_model: Model = load_model("./models/dnn_regression_model.h5")
    
    test_x_pred = model.predict(test_x).flatten()
    r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
    print("DNN ERP REGRESSOR:")
    print("OOS Zero Benchmarked R-Squared: ", r_squared)
    print("OOS MSE: ", mean_squared_error(test_x_pred, test_y))
    print("OOS MAE: ", mean_absolute_error(test_x_pred, test_y))

except:
    print("DNN REGRESSION MODEL NOT AVAILABLE!")


try:
    rf_classifier: BaseEstimator = pickle.load(open("./models/rf_erp_classifier_model.pickle"))
    
    test_x_pred = best_rf_classifier.predict(test_x)
    accuracy = accuracy_score(test_y, test_x_pred)
    precision = precision_score(test_y, test_x_pred)
    recall = recall_score(test_y, test_x_pred)

    print("OOS Accuracy: ", accuracy)
    print("OOS Precision: ", precision)
    print("OOS Recall: ", recall)
except:
    print("RF ERP CLASSIFIER NOT AVAILABLE!")



# NOTE: Don't know how to best evaluate these models
rf_side_classifier: BaseEstimator = pickle.load(open("./models/rf_side_classifier.pickle"))

rf_certainty_classifier: BaseEstimator = pickle.load(open("./models/rf_certainty_classifier.pickle"))




