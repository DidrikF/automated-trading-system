import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import math
from copy import deepcopy
import sys
import numpy as np
import pickle
import copy

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

from dataset_columns import features, labels, base_cols
from model_and_performance_visualization import plot_history

from performance_measurement import zero_benchmarked_r_squared
from cross_validation import PurgedKFold

print("TF version: ", tf.version)
print("TF Keras version: ", tf.keras.__version__)

"""
One cannot expect deep intuition about what type of network is best suited for this dataset.
"""

# DATASET PREPARATION
dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date", "timeout"], index_col=["date"])
dataset = dataset.loc[dataset.primary_label_tbm != 0]

dataset = dataset.sort_values(by=["date"]) # important for cross validation
# Feature scaling
std_scaler = StandardScaler()
dataset[features] = std_scaler.fit_transform(dataset[features]) 

train_end = pd.to_datetime("2012-01-01")
test_start = pd.to_datetime("2012-03-01")

train_set = dataset.loc[dataset.index < train_end]
test_set = dataset.loc[dataset.index >= test_start]

test_x = test_set[features]
test_y = test_set["erp_1m"]


class KerasGridSearchCV(object):
    def __init__(self, parameter_space: dict, train_set: pd.DataFrame, n_splits: int=3):

        self.parameter_space = parameter_space
        self.results = {}
       
        self.train_set = train_set
       
        t1 = pd.Series(index=train_set.index, data=train_set["timeout"])
       
        self.purged_k_fold = PurgedKFold(n_splits=n_splits, t1=t1)
        

        self.tb_callback = keras.callbacks.TensorBoard(
            log_dir='./tensor_board_logs', 
            histogram_freq=0, 
            batch_size=32, 
            write_graph=True,
            write_grads=False, 
            write_images=False, 
            embeddings_freq=0, 
            embeddings_layer_names=None, 
            embeddings_metadata=None, 
            embeddings_data=None, 
            update_freq='epoch'
        )

        self.best_score = {}
        self.best_model = None
        self.best_params = None
        self.best_model_history = None

        self.param_grid = self._get_param_grid(self.parameter_space)
        print("Parameter grid (len {})", len(self.param_grid))
        for grid in self.param_grid:
            print(grid)

        self.ensamble_scores = []
        self.ensemble = []
        self.ensamble_histories = []


    def simple_search(self, model_trainer: callable):

        for index, params in enumerate(self.param_grid):
            # generate test, validation split
            cross_val_iterator = self.purged_k_fold.split(self.train_set)
            for train_index, test_index in cross_val_iterator:
                train_set = self.train_set.iloc[train_index] # self.train_set.index < train_end
                validation_set = self.train_set.iloc[test_index] # self.train_set.index >= validation_start

                train_x = train_set[features]
                train_y = train_set["erp_1m"]
                validation_x = validation_set[features]
                validation_y = validation_set["erp_1m"]            

                scores, model, history = model_trainer(train_x, train_y, validation_x, validation_y, params)

                if index not in self.results:
                    self.results[index] = {
                        "score": None, 
                        "res_list": [],
                        "params": params,
                        "params_index": index
                    }

                res = {
                    "scores": scores,
                    "history": history,
                }
                self.results[index]["res_list"].append(res)

        self._compile_results()
        self._find_best()
        
    
    def _compile_results(self):
        for grid_index, res_obj in self.results.items():
            mse_sum = 0
            for res in res_obj["res_list"]:
                mse_sum += res["scores"]["mse"]

            self.results[grid_index]["score"] = mse_sum / len(res_obj["res_list"])

    def _get_param_grid(self, param_space: dict):
        return list(ParameterGrid(self.parameter_space))
    
    def _find_best(self): # TODO: update if I want to use cross validation, but this is just too time consuming I think
        """
        Select the best model based on mse and r-squared. How to evaluate based on both scores?
        """
        cur_best_key = -1
        cur_best_mse = 999999999999
        for key, res_obj in self.results.items():
            if res_obj["score"] < cur_best_mse: # should also optimize r-squared
                cur_best_mse = res_obj["score"]
                cur_best_key = key

        self.best_score = self.results[cur_best_key]["score"]
        self.best_params = self.results[cur_best_key]["params"]
        self.best_model_history = self.results[cur_best_key]["res_list"][0]["history"]
        # self.best_model = self.results[cur_best_key][1] # Need to train ensable

    # https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/
    
    def train_ensemble(self, model_trainer, train_end: pd.datetime, validation_start: pd.datetime, n_estimators: int=10, params: dict=None):

        if params is None:
            params = self.best_params
        
        train_set = self.train_set.loc[self.train_set.index < train_end]
        validation_set = self.train_set.loc[self.train_set.index >= validation_start] 
        train_x = train_set[features]
        train_y = train_set["erp_1m"]
        validation_x = validation_set[features]
        validation_y = validation_set["erp_1m"]      

        for _ in range(n_estimators):
            score, model, history = model_trainer(train_x, train_y, validation_x, validation_y, params)
            self.ensamble_scores.append(score)
            self.ensemble.append(model)
            self.ensamble_histories.append(history)


    def ensemble_predict(self, X: pd.DataFrame):
        print("X Head: ", X.head())
        print("Ensemble : ", self.ensemble)
        predictions = None
        for net in self.ensemble:
            net_preds = net.predict(X).flatten()
            print("Net preds: ", net_preds)
            if predictions is None:
                predictions = np.array(net_preds)
            else:
                print("shapes of preds: ", predictions.shape, net_preds.shape)
                predictions = np.vstack((predictions, net_preds))


        print("Predictions: ", predictions)
        return np.mean(predictions, axis=0)



def model_trainer(train_x, train_y, validation_x, validation_y, params):
    print("Training model with params: ", params)
    model = tf.keras.Sequential()

    # NOTE: What is input dim in relation to the first layer in the model? Do I need to output 94 units
    model.add(layers.Dense(train_x.shape[1], kernel_initializer='normal',input_dim=train_x.shape[1], activation=params["activation"]))

    # The Hidden Layer:
    model.add(layers.Dropout(rate=params["dropout"])) # Low dropout from inputs
    model.add(layers.Dense(10, kernel_initializer='normal',activation=params["activation"], kernel_regularizer=l2(params["lambd"])))

    # The Output Layer :
    model.add(layers.Dense(1, kernel_initializer='normal',activation='linear')) # Is linear activation function for the output layer good?


    def coeff_determination(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )


    model.compile(
        optimizer=tf.train.AdamOptimizer(params["initial_learning_rate"]), # adam is a good default # tf.train.GradientDecentOptimizer()
        loss="mse", 
        metrics=["mae", "mse", coeff_determination],
    )

    # Regularization:
    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    # https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', # loss on the validation set!
        mode='auto', 
        verbose=0, 
        patience=params["patience"], # was 0 # the number of epochs over which we wish to see improvement
        min_delta=0, # was 0 and has been 0.1 for all testing so far 
        baseline=None, 
        restore_best_weights=False # whether to restore model weights from the epoch with the best value of the monitored quantity
    )

    history = model.fit(
        train_x, 
        train_y, 
        epochs=params["epochs"], # must be tuned for this particular problem
        batch_size=params["batch_size"], # I think 32 is a decent number regardless
        validation_data=(validation_x, validation_y), # NOTE: 3 years currently, this is used to evaluate the model at the end of each epoch
        verbose=1,
        callbacks=[early_stopping_callback], # learning_rate_callback, early_stop, PrintDot()
        # validation_split=0.2
    )

    validation_x_pred = model.predict(validation_x).flatten()
    r_squared = zero_benchmarked_r_squared(validation_x_pred, validation_y)
    r2 = r2_score(validation_y, validation_x_pred)
    mse = mean_squared_error(validation_y, validation_x_pred)
    mae = mean_absolute_error(validation_y, validation_x_pred)

    scores = {
        "r_squared": r_squared, 
        "r2_score": r2,
        "mse": mse, 
        "mae": mae
    }

    return (scores, model, history)


"""
parameter_space = {
    "initial_learning_rate": [0.1], #, 0.001, 0.0001, 0.00001
    "activation": ["relu"],
    "dropout": [0.5], # A common value is a probability of 0.5 for retaining the output of each node in a hidden layer and a value close to 1.0, such as 0.8, for retaining inputs from the visible layer
    "lambd": [0.7], # [0.01, 0.1, 0.3, 1] # https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/playground-exercise-examining-l2-regularization
    "epochs": [10], # 5,10 ,20,40     10, 100, 500, 1000 and larger....
    "patience": [1], # 10
    # "lr_factor": [0.01, 0.1], # decreasing the learning rate by a factor of two or an order of magnitude
    "batch_size": [128] # 32, 64, 128 # https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
}
"""

parameter_space = {
    "initial_learning_rate": [0.01],
    "activation": ["relu"],
    "dropout": [0.2],
    "lambd": [0.3], 
    "epochs": [500],
    "patience": [10], 
    "batch_size": [100000] 
}

grid_search_cv = KerasGridSearchCV(parameter_space=parameter_space, train_set=train_set, n_splits=3)

if True:
    print("Grid Search...")
    grid_search_cv.simple_search(model_trainer=model_trainer)
    print(grid_search_cv.best_params)
    print(grid_search_cv.best_score)
    print(grid_search_cv.best_model_history)
    print(grid_search_cv.best_model_history.__dict__.keys())

    results = {
        "best_history": grid_search_cv.best_model_history.history,
        "best_params": grid_search_cv.best_params,
        "best_score": grid_search_cv.best_score,
    }
    plot_history(grid_search_cv.best_model_history)

else:
    params = {
        "initial_learning_rate": 0.001,
        "activation": "relu",
        "dropout": 0,
        "lambd": 0.3, 
        "epochs": 500,
        "patience": 10, 
        "batch_size": 100000 
    }
    grid_search_cv.best_params = params
    results = {
        "params": params
    }


train_end = pd.to_datetime("2008-09-01")
validation_start = pd.to_datetime("2009-01-01")

print("Training Ensemble...")
grid_search_cv.train_ensemble(model_trainer=model_trainer, train_end=train_end, validation_start=validation_start, n_estimators=10)



test_x_pred = grid_search_cv.ensemble_predict(test_x)
print("Final Predictions: ", test_x_pred)
print(test_x_pred) # Empty...
print("print test set shapes: ", test_x.shape, test_y.shape)
z_r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
r2 = r2_score(test_y, test_x_pred)
mse = mean_squared_error(test_y, test_x_pred)
mae = mean_absolute_error(test_y, test_x_pred)
print("OOS Zero Benchmarked R-Squared: ", z_r_squared)
print("R-squared: ", r2)
print("OOS MSE: ", mse)
print("OOS MAE: ", mae)

results["r_squared"] = z_r_squared 
results["r2"] = r2
results["mse"] = mse 
results["mae"] = mae


print("Results: ", results)
print("Saving Results...")
pickle.dump(results, open("./models/dnn_reg_model_results.pickle", "wb"))

