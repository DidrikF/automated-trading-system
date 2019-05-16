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
    #plot
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


def plot_trees(estimator, features):
    from sklearn.tree import export_graphviz
    export_graphviz(
        estimator, 
        out_file="tree.dot", # Need to make into jpeg...
        feature_names = features,
        class_names = ["Long", "Short"],
        rounded = True, 
        proportion = False, 
        precision = 2, 
        filled = True
    )


def plot_history(history):
    """
    Plot how the loss function developed over the epochs, see the error on both training and validation sets
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
            label='Train Error')
    plt.plot(
        hist['epoch'], 
        hist['val_mean_absolute_error'],
        label = 'Val Error'
    )
    plt.ylim([0,5])
    plt.legend()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(
        hist['epoch'], 
        hist['mean_squared_error'],
        label='Train Error'
    )
    plt.plot(
        hist['epoch'], 
        hist['val_mean_squared_error'],
        label = 'Val Error'
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
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()





# NOTE: not my code and I need to adapt it
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

"""
plot_confusion_matrix(Y_validation_true, Y_validation_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
"""