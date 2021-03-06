
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

dataset_x = dataset[features]
# dataset_y = dataset["erp_1m"] # No need to scale dependent variable


# Feature scaling
std_scaler = StandardScaler()
# Maybe not standardize labels..., also many of the columns are not numeric at this point, do this process below...
dataset_x = std_scaler.fit_transform(dataset_x) 

# Encoding Categorical Features: NOTE: Not using industry now
pca = PCA(n_components=5) # NOTE: How many principal complents?
reduced_dataset = pca.fit_transform(dataset_x)


ex_variance=np.var(reduced_dataset,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)

# correlation_matrix =np.corrcoef(reduced_dataset)
# print("Correlation Matrix")
# print(correlation_matrix)


reduced_dataset = pd.DataFrame(index=dataset.index, data=reduced_dataset)

reduced_dataset["erp_1m"] = dataset["erp_1m"]

reduced_dataset.to_csv("./dataset_development/datasets/completed/reduced_dataset.csv")

train_start = reduced_dataset.index.min() # Does Date index included in features when training a model?
train_end = pd.to_datetime("2012-01-01")

test_start = pd.to_datetime("2012-03-01")
test_end = reduced_dataset.index.max()

train_set = reduced_dataset.loc[(reduced_dataset.index >= train_start) & (reduced_dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
test_set = reduced_dataset.loc[(reduced_dataset.index >= test_start) & (reduced_dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

# Need to select the top principal components...
print("train set shape ", train_set.shape)
print("test set shape ", test_set.shape)
# print("Start and end dates: ", min(dataset.index), max(dataset.index))
# print("train set num tickers", len(train_set["ticker"].unique()))
# print("test set num tickers", len(test_set["ticker"].unique()))
print(reduced_dataset.columns)

train_x = train_set[[0,1,2,3,4]]
train_y = train_set["erp_1m"]

test_x = test_set[[0,1,2,3,4]]
test_y = test_set["erp_1m"]


regressor = linear_model.LinearRegression()
regressor.fit(train_x, train_y)

# r_squared = regressor.score(test_x, test_y) # Returns the coefficient of determination R^2 of the prediction.

#______ Calculate Performance measures ________
print("Variance Ratios: ", ex_variance_ratio) 

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


# HISTORICAL MEAN FORECAST - OUT OF SAMPLE
print("Historical Mean Forecast Restult")
historical_mean = train_y.mean()
historical_pred = pd.Series(index=test_y.index, data=historical_mean)
# print("historical pred:")
# print(historical_pred)
print("Historic Mean: ", historical_mean)
r_squared = zero_benchmarked_r_squared(historical_pred, test_y)
print("OOS Zero Benchmarked R-Squared: ", r_squared)
print("R-squared: ", r2_score(test_y, historical_pred))
print("OOS MSE: ", mean_squared_error(test_y, historical_pred))
print("OOS MAE: ", mean_absolute_error(test_y, historical_pred))

# Zero forecast - OUT OF SAMPLE
print("Zero Forecast Restult")
zero_pred = pd.Series(index=test_y.index, data=0)
# print("historical pred:")
# print(zero_pred)

r_squared = zero_benchmarked_r_squared(zero_pred, test_y)
print("OOS Zero Benchmarked R-Squared: ", r_squared)
print("R-squared: ", r2_score(test_y, zero_pred))
print("OOS MSE: ", mean_squared_error(test_y, zero_pred))
print("OOS MAE: ", mean_absolute_error(test_y, zero_pred))

# save the model to disk
pickle.dump(regressor, open("./models/lin_reg_model.pickle", 'wb'))

