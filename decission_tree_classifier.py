import sys
import matplotlib as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.metrics import confusion_matrix

from compile_dataset import selected_sf1_features, selected_industry_sf1_features, selected_sep_features

from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

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



dataset = pd.read_csv(
    "./datasets/ml_ready_live/dataset_with_nans.csv", 
    parse_dates=["date", "datekey", "calendardate", "timeout"],
    index_col="date"
)



train_start_date = pd.to_datetime("1998-01-01")
train_end_date = pd.to_datetime("2006-12-31")
validation_start_date = pd.to_datetime("2007-01-01")
validation_end_date = pd.to_datetime("2010-12-31")
test_start_date = pd.to_datetime("2011-01-01")
test_end_date = pd.to_datetime("2019-06-01")

feature_columns = selected_sf1_features + selected_industry_sf1_features + selected_sep_features
drop_cols = ['return_1m', 'return_2m', 'return_3m', 'ewmstd_2y_monthly', 'return_tbm', \
    'primary_label_tbm', 'take_profit_barrier', 'stop_loss_barrier', 'age', "industry", "timeout"]

feature_columns = [col for col in feature_columns if col not in drop_cols]



training_set = dataset.loc[(dataset.index>=train_start_date) & (dataset.index<=train_end_date)]

validation_set = dataset.loc[(dataset.index >= validation_start_date) & (dataset.index <= validation_end_date)]

test_set = dataset.loc[(dataset.index >= test_start_date) & (dataset.index <= test_end_date)]



X_train = training_set[feature_columns]
Y_train = training_set["return_1m"]

X_validation = validation_set[feature_columns]
Y_validation = validation_set["return_1m"]

X_test = test_set[feature_columns]
Y_test = test_set["return_1m"]



regressor = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=5)

regressor.fit(X_train, Y_train)


Y_validation_pred = regressor.predict(X_validation)
Y_validation_true = Y_validation


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

class_names = ["Long", "Short"]

# conf_matrix = confusion_matrix(Y_validation_true, Y_validation_pred)

# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_validation_true, Y_validation_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_validation_true, Y_validation_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()