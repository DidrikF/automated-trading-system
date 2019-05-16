import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import math
from copy import deepcopy
import sys

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

# Interesting to plot feature distributions.
# sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# Encoding Categorical Features:
# Not using industry now

validation_end = pd.to_datetime("2012-01-01")
test_start = pd.to_datetime("2012-03-01")
train_set = dataset.loc[dataset.index < validation_end]
test_set = dataset.loc[dataset.index >= test_start]
test_x = test_set[features]
test_y = test_set["return_1m"]
"""
TODO:
- Add dropout X
- Add Early Stopping X
- Add L2 regularization X
- Learning rate X
- Ensemble # NOTE: drop for now
"""



# You can customize logging output during model training.
"""
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
"""


class KerasGridSearchCV(object):
    def __init__(self, parameter_space: dict, train_set: pd.DataFrame, n_splits: int=3):

        self.parameter_space = parameter_space
        self.results = []
       
        self.train_set = train_set
       
        t1 = pd.Series(index=train_set.index, data=train_set["timeout"])
       
        self.purged_k_fold = PurgedKFold(n_splits=n_splits, t1=t1)
        
        self.fb_callback = keras.callbacks.TensorBoard(
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

    def simple_search(self, model_trainer: callable, train_end: pd.datetime, validation_start: pd.datetime, n_iter: int):
        iter = 0
        param_grid = self._get_param_grid(self.parameter_space)
        print("Parameter grid (len {})", len(param_grid))
        print(param_grid)

        for params in param_grid:
            if iter >= n_iter:
                break
            # generate test, validation split
            train_set = self.train_set.loc[self.train_set.index < train_end]
            validation_set = self.train_set.loc[self.train_set.index >= validation_start]

            train_x = train_set[features]
            train_y = train_set["return_1m"]

            validation_x = validation_set[features]
            validation_y = validation_set["return_1m"]            

            scores, model, history = model_trainer(train_x, train_y, validation_x, validation_y, params)
            
            self.results.append((scores, model, history, deepcopy(params)))

            iter += 1

        return self._find_best()
    

    def _get_param_grid(self, param_space: dict):
        return list(ParameterGrid(self.parameter_space))
    
    def _find_best(self): # TODO: update if I want to use cross validation, but this is just too time consuming I think
        """
        Select the best model based on mse and r-squared. How to evaluate based on both scores?
        """
        cur_best_index = -1
        cur_best_mse = 999999999999
        for index, result in enumerate(self.results):
            if result[0]["mse"] < cur_best_mse: # should also optimize r-squared
                cur_best_index = index

        self.best_score = self.results[cur_best_index][0]
        self.best_model = self.results[cur_best_index][1]
        self.best_model_history = self.results[cur_best_index][2]
        self.best_params = self.results[cur_best_index][3]

    # https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/
    def train_ensemble(self, params=None):
        # samples = bootstrap(self.train_set)
        # for sample in samples:
        #   # make train and validation sets
        #   dnn = train_model(sample_test_x, sample_test_y, sample_val_x, sample_val_y, self.best_params)
        #   self.ensamble.push(dnn)
        pass





def model_trainer(train_x, train_y, validation_x, validation_y, params):
    print("Training model with params: ", params)
    model = tf.keras.Sequential()

    # NOTE: What is input dim in relation to the first layer in the model? Do I need to output 94 units
    model.add(layers.Dense(train_x.shape[1], kernel_initializer='normal',input_dim=train_x.shape[1], activation=params["activation"]))

    # The Hidden Layers :
    model.add(layers.Dropout(rate=params["dropout"][0])) # Low dropout from inputs
    model.add(layers.Dense(32, kernel_initializer='normal',activation=params["activation"], kernel_regularizer=l2(params["lambd"])))
    model.add(layers.Dropout(rate=params["dropout"][1])) # higher dropout for hidden layers
    model.add(layers.Dense(16, kernel_initializer='normal',activation=params["activation"],kernel_regularizer=l2(params["lambd"])))
    model.add(layers.Dropout(rate=params["dropout"][1])) # higher dropout for hidden layers
    model.add(layers.Dense(8, kernel_initializer='normal',activation=params["activation"], kernel_regularizer=l2(params["lambd"])))

    # The Output Layer :
    model.add(layers.Dense(1, kernel_initializer='normal',activation='linear')) # Is linear activation function for the output layer good?

    model.compile(
        optimizer=tf.train.AdamOptimizer(params["initial_learning_rate"]), # adam is a good default # tf.train.GradientDecentOptimizer()
        loss="mse", 
        metrics=["mae"],
    )

    # Regularization:
    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    # https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', # loss on the validation set!
        mode='auto', 
        verbose=0, 
        patience=params["patience"], # was 0 # the number of epochs over which we wish to see improvement
        min_delta=0.1, # was 0 
        baseline=None, 
        restore_best_weights=False # whether to restore model weights from the epoch with the best value of the monitored quantity
    )

    # learning_rate_callback = keras.callbacks.LearningRateScheduler(schedule, verbose=0) # NOTE: need schedule function

    """
    learning_rate_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', # loss on the validation set
        mode='auto', 
        verbose=0, 
        patience=params["patience"], # the number of epochs over which we wish to see improvement
        factor=params["lr_factor"], #  a small positive value, often in the range between 0.0 and 1.0, factor by which to reduce the learning rate
        min_delta=0.0001, # threshold for measuring the new optimum, to only focus on significant changes
        cooldown=0, # number of epochs to wait before resuming normal operation after lr has been reduced
        min_lr=0 # lower bound on the learning rate, no limit here
    )
    """


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
    mse = mean_squared_error(validation_x_pred, validation_y)
    mae = mean_absolute_error(validation_x_pred, validation_y)

    scores = {"r_squared": r_squared, "mse": mse, "mae": mae} # might want to average these scores over all folds to decide which params are best

    return (scores, model, history)




# Parameters are tuned to make the accuracy on the testing set similar to that of the training set, both should be high!
parameter_space = {
    "initial_learning_rate": [0.1, 0.01], #, 0.001, 0.0001, 0.00001
    "activation": ["relu"],
    "dropout": [[0.8, 0.5]],
    "lambd": [0.1, 0.5], # [0.01, 0.1, 0.3, 1] # https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/playground-exercise-examining-l2-regularization
    "epochs": [1], # 5,10 ,20,40     10, 100, 500, 1000 and larger....
    "patience": [2, 5], # 10
     "lr_factor": [0.01, 0.1], # decreasing the learning rate by a factor of two or an order of magnitude
    "batch_size": [32] # , 64, 128 # https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
}



# NOTE:Rather test on validation, test first on the test set when you are happy with the model...
grid_search_cv = KerasGridSearchCV(parameter_space=parameter_space, train_set=train_set, n_splits=3)

train_end = pd.to_datetime("2008-09-01")
validation_start = pd.to_datetime("2009-01-01")

grid_search_cv.simple_search(
    model_trainer=model_trainer, 
    train_end=train_end, 
    validation_start=validation_start, 
    n_iter=2
)


test_x_pred = grid_search_cv.best_model.predict(test_x).flatten()

# Plot how the loss function developed over the epochs, see the error on both training and validation sets
plot_history(grid_search_cv.best_model_history)

"""
loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
print("Testing set Mean Sqrd Error: {:5.2f} MPG".format(mse))
"""


r_squared = zero_benchmarked_r_squared(test_x_pred, test_y)
print("OOS Zero Benchmarked R-Squared: ", r_squared)
print("OOS MSE: ", mean_squared_error(test_x_pred, test_y))
print("OOS MAE: ", mean_absolute_error(test_x_pred, test_y))



# Finally save the model for use elsewhere!
# pickle.dump(grid_search_cv.best_model, open("./models/dnn_regression_model.h5", 'wb'))

grid_search_cv.best_model.save("./models/dnn_regression_model.h5") # TODO: Need to do this differently and figure out how to load models
# NOTE: This should save the model





"""
def cv_search(self, model_trainer: callable, n_iter: int):
    iter = 0

    param_grid = self._grid_search(self.parameter_space)
    print(param_grid)
    sys.exit()
    for params in param_grid:
        if iter >= n_iter:
            break
        # generate test, validation split
        train_indices, validation_indices = self.purged_k_fold.split(self.train_set)

        # Made test and validation sets, X and y

        result, model, history = model_trainer(train_x, train_y, validation_x, validation_y, params)
        
        self.results.append((result, model, history, params))
        iter += 1


    return self._find_best()
    """





"""
train_start = dataset.index.min() # Does Date index included in features when training a model?
train_end = pd.to_datetime("2009-09-01")
test_start = pd.to_datetime("2010-01-01")
test_end = dataset.index.max()
"""
"""
train_start = dataset.index.min() # Does Date index included in features when training a model?
train_end = pd.to_datetime("2008-09-01") # NOTE: Pushed back further
validation_start = pd.to_datetime("2009-01-01")
test_start = pd.to_datetime("2012-03-01")
test_end = dataset.index.max()


train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
validation_set = dataset.loc[(dataset.index >= validation_start) & (dataset.index < validation_end)]
test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets


train_x = train_set[features]
train_y = train_set["return_1m"] # maybe I don't need to update to erp_1m, this is also not adjuseted for dividends...

validation_x = validation_set[features]
validation_y = validation_set["return_1m"]

test_x = test_set[features]
test_y = test_set["return_1m"]
"""




















"""
c:\pycode/automated-trading-system (master -> origin)
(master -> origin-tf) λ python dnn_reg_model.py
TF version:  <module 'tensorflow._api.v1.version' from 'C://Anaconda3//envs//master-tf//lib//site-packages//tensorflow//_api//v1//version//__init__.py'>
TF Keras version:  2.2.4-tf
C:/Anaconda3/envs/master-tf/lib/site-packages/sklearn/preprocessing/data.py:645: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
  return self.partial_fit(X, y)
C:/Anaconda3/envs/master-tf/lib/site-packages/sklearn/base.py:464: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
  return self.fit(X, **fit_params).transform(X)
WARNING:tensorflow:From C:/Anaconda3/envs/master-tf/lib/site-packages/tensorflow/python/ops/resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From C:/Anaconda3/envs/master-tf/lib/site-packages/tensorflow/python/keras/utils/losses_utils.py:170: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 94)                8836
_________________________________________________________________
dense_1 (Dense)              (None, 32)                3040
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9
=================================================================
Total params: 12,549
Trainable params: 12,549
Non-trainable params: 0
_________________________________________________________________
Model Summary:
 None
Train on 426433 samples, validate on 105640 samples
WARNING:tensorflow:From C:/Anaconda3/envs/master-tf/lib/site-packages/tensorflow/python/ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
2019-05-14 21:15:30.434997: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2019-05-14 21:15:30.582302: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties:
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.835
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.63GiB
2019-05-14 21:15:30.595955: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-05-14 21:15:30.939925: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-05-14 21:15:30.949114: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0
2019-05-14 21:15:30.954197: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N
2019-05-14 21:15:30.960449: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6385 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
Epoch 1/10
2019-05-14 21:15:31.449155: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
426433/426433 [==============================] - 31s 73us/sample - loss: 84.8138 - mean_squared_error: 84.8047 - mean_absolute_error: 0.1320 - val_loss: 0.1624 - val_mean_squared_error: 0.1624 - val_mean_absolute_error: 0.1096
Epoch 2/10
426433/426433 [==============================] - 31s 72us/sample - loss: 84.8120 - mean_squared_error: 84.7971 - mean_absolute_error: 0.1318 - val_loss: 0.1631 - val_mean_squared_error: 0.1631 - val_mean_absolute_error: 0.1116
Epoch 3/10
426433/426433 [==============================] - 31s 73us/sample - loss: 84.8081 - mean_squared_error: 84.8056 - mean_absolute_error: 0.1317 - val_loss: 0.1626 - val_mean_squared_error: 0.1626 - val_mean_absolute_error: 0.1113
Epoch 4/10
426433/426433 [==============================] - 30s 70us/sample - loss: 84.8019 - mean_squared_error: 84.7885 - mean_absolute_error: 0.1319 - val_loss: 0.1624 - val_mean_squared_error: 0.1624 - val_mean_absolute_error: 0.1112
Epoch 5/10
426433/426433 [==============================] - 30s 70us/sample - loss: 84.7862 - mean_squared_error: 84.7701 - mean_absolute_error: 0.1322 - val_loss: 0.1636 - val_mean_squared_error: 0.1636 - val_mean_absolute_error: 0.1105
Epoch 6/10
426433/426433 [==============================] - 30s 70us/sample - loss: 84.7844 - mean_squared_error: 84.7744 - mean_absolute_error: 0.1320 - val_loss: 0.1634 - val_mean_squared_error: 0.1634 - val_mean_absolute_error: 0.1130
Epoch 7/10
426433/426433 [==============================] - 30s 70us/sample - loss: 84.7087 - mean_squared_error: 84.6923 - mean_absolute_error: 0.1322 - val_loss: 0.1670 - val_mean_squared_error: 0.1670 - val_mean_absolute_error: 0.1130
Epoch 8/10
426433/426433 [==============================] - 30s 70us/sample - loss: 84.6145 - mean_squared_error: 84.6069 - mean_absolute_error: 0.1325 - val_loss: 0.1673 - val_mean_squared_error: 0.1673 - val_mean_absolute_error: 0.1114
Epoch 9/10
426433/426433 [==============================] - 30s 70us/sample - loss: 84.4745 - mean_squared_error: 84.4546 - mean_absolute_error: 0.1370 - val_loss: 0.2452 - val_mean_squared_error: 0.2452 - val_mean_absolute_error: 0.1152
Epoch 10/10
426433/426433 [==============================] - 30s 70us/sample - loss: 84.6711 - mean_squared_error: 84.6692 - mean_absolute_error: 0.1320 - val_loss: 0.2610 - val_mean_squared_error: 0.2610 - val_mean_absolute_error: 0.1148
343127/343127 [==============================] - 11s 32us/sample - loss: 6.0491 - mean_squared_error: 6.0491 - mean_absolute_error: 0.1074
Testing set Mean Abs Error:  6.05 MPG

NOTE: WARNING:tensorflow:TensorFlow optimizers do not make it possible to access optimizer attributes or optimizer state after instantiation. As a result, we cannot save the optimizer as part of the model save file.You will have to compile your model again after loading it. Prefer using a Keras optimizer instead (see keras.io/optimizers).

"""


