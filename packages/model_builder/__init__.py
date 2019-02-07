"""
Package containing procedures to train, tune and test multiple ML
models. Users should in a simple way be able to specify the kind of
procedure to execute.

Users should be able to specify: 
    - What type of model to use
    - What hyperparemeters to tune and in what ranges
    - What cost/objective function(s) to use
    - What penalty to add to the objective function (A type of regularization?)
        - Elastic Net
        - LASSO
        - Ridge Regression
        - Huber Loss
    - What kind of input regularization to apply (can be multiple kinds)
    - What kind of output regularization to apply (can be multiple kinds)
    - What type to test statistics and measurements to gather
    - Tracker to keep track of the number of models and configurations
      that has been explored.

"""