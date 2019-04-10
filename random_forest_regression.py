import numpy as np 
import matplotlib as plt
import pandas as pd

from compile_dataset import selected_sf1_features, selected_industry_sf1_features, selected_sep_features

dataset = pd.read_csv(
    "./datasets/ml_ready_live/dataset.csv", 
    parse_dates=["date", "datekey", "calendardate", "timeout"],
    index_col="date"
)
dataset = dataset["1998":"2008"]


feature_columns = selected_sf1_features + selected_industry_sf1_features + selected_sep_features
feature_columns.remove("return_1m")
feature_columns.remove("return_2m")
feature_columns.remove("return_3m")

X = dataset[feature_columns]
Y = dataset["return_1m"]

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(criterion="mse", random_state=42)

regressor.fit(X, Y, )


"""
General approach for producing regression models: 

This is somewhat asside to the more important part of the project, which is to produce an automated trading system.
The methods used in runing regressions are therefore somewhat simpler than what will be used in the training of the
automated trading system.

Training set from beginning to 2007
Validation set from 2008 to 20011
Testing set from 2011-2019

Train and tune hyperparameters with training and validation sets.
No advanced cross validation.

Regress on monthly returns.

Produce statistics for performance and feature importance.

Run regressions with random forest, linear regression and deep neural network (use Keras and TensorFlow).

"""