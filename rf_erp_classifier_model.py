import pandas as pd
import pickle
import numpy as np
import sys

# Algorithms
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Performance Metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

# Other
import scipy
from scipy.stats import randint

from cross_validation import PurgedKFold, cv_score
from dataset_columns import features, labels, base_cols
from performance_measurement import sample_binary_predictor, single_sample_t_test
from model_and_performance_visualization import plot_feature_importances

# NOTE: Train on own computer to print feature importances
n_jobs = 5

# DATASET PREPARATION
dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date", "timeout"], index_col=["date"])
dataset = dataset.loc[dataset.erp_1m != 0]

dataset = dataset.sort_values(by=["date"]) # important for cross validation


train_start = dataset.index.min() # Does Date index included in features when training a model?
train_end = pd.to_datetime("2012-01-01")

test_start = pd.to_datetime("2012-03-01")
test_end = dataset.index.max()

train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

train_x = train_set[features]
train_y = train_set["erp_1m_direction"]

test_x = test_set[features]
test_y = test_set["erp_1m_direction"]

training_model = True

if training_model:

    rf_classifier = RandomForestClassifier(random_state=100, verbose=True, n_jobs=2)

    parameter_space = {
        "n_estimators": [200], #, 500], # [100, 200, 500] # looks like 1000 is too much, it overfits
        "min_weight_fraction_leaf": [0.05], #  0.10, 0.20], # [0.05, 0.10, 0.20] early stopping
        "max_features": [3], # 5, 7], # 1, 5, 10, 15, 20, 30 may even push it??
        "class_weight": ["balanced_subsample"],
        "bootstrap": [True], # , False
        "criterion": ["entropy"] # , "gini"
        # "max_depth": [1, 2, 4, 8, 10, 15], # max depth should be set lower I think
        # "min_samples_split": [int(num_samples*0.04),int(num_samples*0.06),int(num_samples*0.08)], # I have 550,000 samples for training -> 5500
        # "min_samples_leaf": [int(num_samples*0.04),int(num_samples*0.06),int(num_samples*0.08)], # 2-8% of samples 
    }

    t1 = pd.Series(index=train_set.index, data=train_set["timeout"])
        
    grid_search = GridSearchCV(
        estimator=rf_classifier,
        param_grid=parameter_space,
        #  NOTE: need to update to use the date and timout columns
        cv=PurgedKFold(n_splits=3, t1=t1), # a CV splitter object implementing a split method yielding arrays of train and test indices
        # Need to figure out if just using built in scorers will work with the custom PurgedKFold splitter
        scoring=["accuracy", "balanced_accuracy", "precision", "recall", "f1"], # a string or a callable to evaluate the predictions on the test set (use custom scoring function that works with PurgedKFold)
        refit="f1",
        n_jobs=n_jobs,
        verbose=2,
        error_score=np.nan
    )

    grid_search.fit(train_x, train_y)

    print("Best Score (F1): \n", grid_search.best_score_)
    print("Best Params: \n", grid_search.best_params_)
    print("Best Index: \n", grid_search.best_index_)
    print("CV Results: \n")
    for key, val in grid_search.cv_results_.items():
        print("{}: {}".format(key, val))

    best_params = grid_search.best_params_
    best_classifier: RandomForestClassifier = grid_search.best_estimator_


    test_x_pred = best_classifier.predict(test_x)
    accuracy = accuracy_score(test_y, test_x_pred)
    precision = precision_score(test_y, test_x_pred)
    recall = recall_score(test_y, test_x_pred)
    f1 = f1_score(test_y, test_x_pred)

    print("OOS Accuracy: ", accuracy)
    print("OOS Precision: ", precision)
    print("OOS Recall: ", recall)
    print("OOS F1 score: ", f1)
    print("Prediction distribution: \n", pd.Series(test_x_pred).value_counts())

    plot_feature_importances(best_classifier, train_x.columns)
    sys.exit()

    # T-test
    observations = sample_binary_predictor(y_pred=test_x_pred, y_true=test_y, n_samples=200, sample_size=3000)
    t_test_results_50 = single_sample_t_test(observations, mean0=0.5, alpha=0.05)
    t_test_results_51 = single_sample_t_test(observations, mean0=0.51, alpha=0.05)
    t_test_results_51 = single_sample_t_test(observations, mean0=0.52, alpha=0.05)
    t_test_results_52 = single_sample_t_test(observations, mean0=0.53, alpha=0.05)
    t_test_results_53 = single_sample_t_test(observations, mean0=0.54, alpha=0.05)
    t_test_results_54 = single_sample_t_test(observations, mean0=0.55, alpha=0.05)
    t_test_results_55 = single_sample_t_test(observations, mean0=0.56, alpha=0.05)

    
    print("T-test results: {}".format(t_test_results_50))
    print("Mean of random samples: ", observations.mean())
    print("Standard deviations of random samples: ", observations.std())


    print("Saving Model...")
    pickle.dump(best_classifier, open("./models/rf_erp_classifier_model.pickle", 'wb'))
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "t_test_results_50_mean": t_test_results_50, 
        "t_test_results_51_mean": t_test_results_51, 
        "t_test_results_52_mean": t_test_results_52, 
        "t_test_results_53_mean": t_test_results_53, 
        "t_test_results_54_mean": t_test_results_54, 
        "t_test_results_55_mean": t_test_results_55, 
        "sample_mean": observations.mean(),
        "sample_std": observations.std(),
        "prediction_distribution": pd.Series(test_x_pred).value_counts(),
        "best_params": grid_search.best_params_,
        "cv_results": grid_search.cv_results_,
    }
    
    pickle.dump(results, open("./models/rf_erp_classifier_results.pickle", "wb"))
    




    # ________________________Experiment with extra trained classifier______________________________

    train_start = dataset.index.min() # Does Date index included in features when training a model?
    train_end = pd.to_datetime("2016-01-01")

    test_start = pd.to_datetime("2016-03-01")
    test_end = dataset.index.max()

    train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
    test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

    train_x = train_set[features]
    train_y = train_set["erp_1m_direction"]

    test_x = test_set[features]
    test_y = test_set["erp_1m_direction"]


    et_classifier = RandomForestClassifier(
        n_estimators=best_params["n_estimators"], # Nr of trees
        # NOTE: Set to a lower value to force discrepancy between trees
        min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
        max_features=best_params["max_features"], # the number of features to consider when looking for the best split (auto = sqrt(n_features))
        class_weight=best_params["class_weight"],
        criterion=best_params["criterion"], # function to measure quality of split (impurity measure)
        bootstrap=best_params["bootstrap"], # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
        n_jobs=n_jobs, # Number of jobs to run in parallell during fit and predict (-1 means to use all available)
    )
    print("Training new 'extra trained' classifier...")
    et_classifier.fit(train_x, train_y)

    old_test_x_pred = best_classifier.predict(test_x)
    new_test_x_pred = et_classifier.predict(test_x)

    old_accuracy = accuracy_score(test_y, old_test_x_pred)
    old_precision = precision_score(test_y, old_test_x_pred)
    old_recall = recall_score(test_y, old_test_x_pred)
    old_f1 = f1_score(test_y, old_test_x_pred)
    
    new_accuracy = accuracy_score(test_y, new_test_x_pred)
    new_precision = precision_score(test_y, new_test_x_pred)
    new_recall = recall_score(test_y, new_test_x_pred)
    new_f1 = f1_score(test_y, new_test_x_pred)

    print("Model trained on old data")
    print("OOS Accuracy: ", old_accuracy)
    print("OOS Precision: ", old_precision)
    print("OOS Recall: ", old_recall)
    print("OOS F-1: ", old_f1)

    print("Model trained on new data")
    print("OOS Accuracy: ", new_accuracy)
    print("OOS Precision: ", new_precision)
    print("OOS Recall: ", new_recall)
    print("OOS F-1: ", new_f1)

