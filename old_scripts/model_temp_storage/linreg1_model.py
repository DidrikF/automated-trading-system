"""
Assumptions of linear regression:
    - Linearity
    - Homoscedasticity
    - Multivariate normality
    - Independence of errors
    - Lack of multicollinearity


y = b0 + b1*x1 + b2*x2

5 methods of building a model
    - All-in
    - Backwards Elimination
    - Forward Selection
    - Bidirectional Elimination
    - Score Comparison

"""

from dataset_builder.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys


if __name__ == "__main__":

    dataset = Dataset('./processed_datasets/fundamentals_final.csv')

    X = dataset.data.iloc[:, [1,3]].values
    y = dataset.data.iloc[:, 2].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # print(len(X_train), len(X_test), len(y_train), len(y_test))

    regressor = LinearRegression()

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test[:, 0], X_test[:, 1], y_pred)
    ax.set_xlabel('EPS')
    ax.set_ylabel('BVPS')
    ax.set_zlabel('Price')
    plt.show()

    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y)
    ax.set_xlabel('EPS')
    ax.set_ylabel('BVPS')
    ax.set_zlabel('Price')
    plt.show()
    """