import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc


def plot_feature_importances(estimator, cols):
    y = estimator.feature_importances_
    fig, ax = plt.subplots() 
    width = 0.4 # the width of the bars 
    ind = np.arange(len(y)) # the x locations for the groups
    ax.barh(ind, y, width, color="green")
    ax.set_yticks(ind+width/10)
    ax.set_yticklabels(cols, minor=False)
    plt.title("Feature importance in RandomForest Classifier")
    plt.xlabel("Relative importance")
    plt.ylabel("feature") 
    plt.figure(figsize=(5,5))
    fig.set_size_inches(6.5, 4.5, forward=True)

    plt.show()



def plot_history(history):
    """
    Plot how the loss function developed over the epochs, see the error on both training and validation sets
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(
        hist['epoch'], 
        hist['mean_squared_error'],
        label='Training Error'
    )
    plt.plot(
        hist['epoch'], 
        hist['val_mean_squared_error'],
        label = 'Validation Error'
    )
    plt.ylim([0,20])
    plt.legend()
    plt.show()



def plot_error(y_true, y_pred):
    """
    Plot prediction errors as a histogram. Prediction error size at the x-axis and
    number of occurrences at the y-axis.
    """
    error = y_pred - y_true
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()

