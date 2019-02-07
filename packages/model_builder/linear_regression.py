from sklearn.linear_model import LinearRegression


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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split

