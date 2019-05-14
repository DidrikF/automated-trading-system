import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from dataset_columns import features, labels, base_cols

print("TF version: ", tf.version)
print("TF Keras version: ", tf.keras.__version__)

"""
One cannot expect deep intuition about what type of network is best suited for this dataset.
"""


model = tf.keras.Sequential()

# Add a densly-connected layer with 94 units to the model

model.add(layers.Dense(94, activation="relu")) # Relu is almost allwasy right
model.add(layers.Dense(94, activation="relu"))
model.add(layers.Dense(94, activation="relu"))

# What is the appropriate output layer type? 

model.compile(
    optimizer=tf.train.AdamOptimizer(0.001), # tf.train.GradientDecentOptimizer()
    loss="mse", # 
    metrics=["mse", "mae"],
)


# DATASET PREPARATION
dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date", "timeout"], index_col=["date"])
dataset = dataset.loc[dataset.primary_label_tbm != 0]

dataset = dataset.sort_values(by=["date"]) # important for cross validation
# Feature scaling
std_scaler = StandardScaler()
# Maybe not standardize labels..., also many of the columns are not numeric at this point, do this process below...
dataset_x = std_scaler.fit_transform(dataset[features]) 


# Encoding Categorical Features:
# Not using industry now


train_start = dataset.index.min() # Does Date index included in features when training a model?
train_end = pd.to_datetime("2009-09-01")

test_start = pd.to_datetime("2010-01-01")
test_end = dataset.index.max()

train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets


train_x = train_set[features]
train_y = train_set["primary_label_tbm"] # maybe I don't need to update to erp_1m, this is also not adjuseted for dividends...

test_x = test_set[features]
test_y = test_set["primary_label_tbm"]

validate_x = []
validate_y = []

model.fit(
    train_x, 
    train_y, 
    epochs=10, # must be tuned for this particular problem
    batch_size=32, # I think 32 is a decent number regardless
    validation_data=(validate_x, validate_y),
)

# Parameters are tuned to make the accuracy on the testing set similar to that of the training set, both should be high!

# How to do hyperparameter tuning?


predictions = model.predict(test_x)
model.predict_proba(test_x)

model.save("./models/dnn_regression_model.tf") # What type?