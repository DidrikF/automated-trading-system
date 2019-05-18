
# Algorithms
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict # Don't know if I will use
from sklearn.metrics import mean_squared_error, r2_score

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
pca = PCA()
reduced_dataset_x = pca.fit_transform(dataset_x)

train_start = dataset.index.min() # Does Date index included in features when training a model?
train_end = pd.to_datetime("2012-01-01")

test_start = pd.to_datetime("2012-03-01")
test_end = dataset.index.max()

train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

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

print("OOS Zero Benchmarked R-Squared: ", r_squared)
print("OOS MSE: ", mean_squared_error(test_x_pred, test_y))
print("OOS MAE: ", mean_absolute_error(test_x_pred, test_y))


# save the model to disk
pickle.dump(regressor, open("./models/lin_reg_model.pickle", 'wb'))

