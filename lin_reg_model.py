
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
print("Start and end dates: ", min(dataset.index), max(dataset.index))
print("train set num tickers", len(train_set["ticker"].unique()))
print("test set num tickers", len(test_set["ticker"].unique()))
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

train_x = train_set[features]
train_y = train_set["erp_1m"] # Need to add erp_1m to the dataset, can be done in finalize_dataset, but maybe just rebuild the dataset first, only need to run the last task to add dividend adjusted labels_tbm

test_x = test_set[features]
test_y = test_set["erp_1m"]


regressor = linear_model.LinearRegression()
regressor.fit(train_x, train_y)

# r_squared = regressor.score(test_x, test_y) # Returns the coefficient of determination R^2 of the prediction.

# Calculate Performance measures
test_x_pred = regressor.predict(test_x)
r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
print("Variance Ratios: ", ex_variance_ratio) 
print("OOS Zero Benchmarked R-Squared: ", r_squared)
print("OOS MSE: ", mean_squared_error(test_x_pred, test_y))
print("OOS MAE: ", mean_absolute_error(test_x_pred, test_y))


# save the model to disk
pickle.dump(regressor, open("./models/lin_reg_model.pickle", 'wb'))

