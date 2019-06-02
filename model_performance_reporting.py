import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import load_model
from tensorflow.keras import Model

from sklearn.metrics import mean_squared_error, r2_score, f1_score, mean_absolute_error, \
    accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from dataset_columns import features, labels, base_cols
from performance_measurement import zero_benchmarked_r_squared, sample_binary_predictor, single_sample_t_test


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
train_start = std_dataset.index.min()
train_end = pd.to_datetime("2012-01-01")
test_start = pd.to_datetime("2012-03-01")
test_end = std_dataset.index.max()

# Lin reg dataset
lr_dataset = pd.read_csv("./dataset_development/datasets/completed/reduced_dataset.csv")
lr_test_set = lr_dataset.loc[(std_dataset.index >= test_start) & (std_dataset.index <= test_end)].drop()

# Normalized dataset

# Normal dataset

train_set = std_dataset.loc[(std_dataset.index >= train_start) & (std_dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
test_set = std_dataset.loc[(std_dataset.index >= test_start) & (std_dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

test_x = std_test_set[features]


std_train_x = std_train_set[features]


train_erp_1m_y = test_set["erp_1m"]
test_erp_1m_direction_y = test_set["erp_1m_direction"]


# Non Normalized Datasets: Used for all random forests (regression and classification)

# train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
# train_x = train_set[features]

test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

test_x = test_set[features]


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
    r_squared = zero_benchmarked_r_squared(test_x_pred, train_erp_1m_y)
    print("Lin Reg ERP REGRESSOR:")
    print("OOS Zero Benchmarked R-Squared: ", r_squared)
    print("OOS MSE: ", mean_squared_error(train_erp_1m_y, test_x_pred))
    print("OOS MAE: ", mean_absolute_error(train_erp_1m_y, test_x_pred))
except:
    print("LIN-REG MODEL NOT AVAILABLE!")


try:
    rf_reg_model: BaseEstimator = pickle.load(open("./models/rf_erp_regression_model.pickle"))
    
    test_x_pred = rf_reg_model.predict(test_x)
    r_squared = zero_benchmarked_r_squared(test_x_pred, train_erp_1m_y)
    print("RF ERP REGRESSOR:")
    print("OOS Zero Benchmarked R Squared: ", r_squared)
    print("OOS MSE: ", mean_squared_error(train_erp_1m_y, test_x_pred))
    print("OOS MAE: ", mean_absolute_error(train_erp_1m_y, test_x_pred))
except:
    print("RF ERP REGRESSION MODEL NOT AVAILABLE!")

try:
    dnn_reg_model: Model = load_model("./models/dnn_regression_model.h5")
    
    test_x_pred = model.predict(test_x).flatten()
    r_squared = zero_benchmarked_r_squared(test_x_pred, train_erp_1m_y)
    print("DNN ERP REGRESSOR:")
    print("OOS Zero Benchmarked R-Squared: ", r_squared)
    print("OOS MSE: ", mean_squared_error(train_erp_1m_y, test_x_pred))
    print("OOS MAE: ", mean_absolute_error(train_erp_1m_y, test_x_pred))

except:
    print("DNN REGRESSION MODEL NOT AVAILABLE!")


try:
    rf_classifier: BaseEstimator = pickle.load(open("./models/rf_erp_classifier_model.pickle"))
    
    test_x_pred = best_rf_classifier.predict(test_x)
    
    observations = sample_binary_predictor(test_x_pred, test_y, 300, 1000)
    t_test_res = single_sample_t_test(observations, 0.5, 0.05)

    accuracy = accuracy_score(test_erp_1m_direction_y, test_x_pred)
    precision = precision_score(test_erp_1m_direction_y, test_x_pred)
    recall = recall_score(test_erp_1m_direction_y, test_x_pred)
    f1 = f1_score(test_erp_1m_direction_y, test_x_pred)

    print("T-test of accuracy compared to random model result: ", t_test_res)
    print("OOS Accuracy: ", accuracy)
    print("OOS Precision: ", precision)
    print("OOS Recall: ", recall)
    print("OOS F1 Score: ", f1)
except:
    print("RF ERP CLASSIFIER NOT AVAILABLE!")



# NOTE: Don't know how to best evaluate these models
# rf_side_classifier: BaseEstimator = pickle.load(open("./models/rf_side_classifier.pickle"))
# rf_certainty_classifier: BaseEstimator = pickle.load(open("./models/rf_certainty_classifier.pickle"))

try:
    ml_strategy_models_results = pickle.load(open("./models/ml_strategy_models_results.pickle", "rb"))

    print("Side Model Accuracy: {}  ".format(ml_strategy_models_results["side_model"]["accuracy"]))
    print("Side Model Precision: {}  ".format(ml_strategy_models_results["side_model"]["precision"]))
    print("Side Model Recall: {}   ".format(ml_strategy_models_results["side_model"]["recall"]))
    print("Side Model F1: {}   ".format(ml_strategy_models_results["side_model"]["f1"]))


    print("Side Model Accuracy: {}  ".format(ml_strategy_models_results["certainty_model"]["accuracy"]))
    print("Side Model Accuracy: {}  ".format(ml_strategy_models_results["certainty_model"]["precision"]))
    print("Side Model Accuracy: {}  ".format(ml_strategy_models_results["certainty_model"]["recall"]))
    print("Side Model F1: {}  ".format(ml_strategy_models_results["certainty_model"]["f1"]))
except:
    print("ML STRATEGY MODEL RESULTS NOT AVAILABLE!")