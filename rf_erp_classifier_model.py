import pandas as pd
import pickle
import numpy as np

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
        "n_estimators": [200, 500, 1000], # [100, 200, 500]
        "min_weight_fraction_leaf": [0.05, 0.10, 0.20], # [0.05, 0.10, 0.20] early stopping
        "max_features": [5, 10], # 1, 5, 10, 15, 20, 30 may even push it??
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

    print("Best Score (Accuracy): \n", grid_search.best_score_)
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


    # T-test
    observations = sample_binary_predictor(y_pred=test_x_pred, y_true=test_y, n_samples=200, sample_size=3000)
    t_test_results = single_sample_t_test(observations, mean0=0.5, alpha=0.05)
    
    print("T-test results: {}".format(t_test_results))
    print("Mean of random samples: ", observations.mean())
    print("Standard deviations of random samples: ", observations.std())


    print("Saving Model...")
    pickle.dump(best_classifier, open("./models/rf_erp_classifier_model.pickle", 'wb'))
    pickle.dump(best_params, open("./models/rf_erp_classifier_best_params.pickle", "wb"))
    




    # ____________________________________________________________________________________

    # Experiment with extra trained classifier:
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
         # max_depth=best_params["max_depth"], # depth of each tree
        # min_samples_split=best_params["min_samples_split"], # minimum number of samples required to split an internal node
        # min_samples_leaf=best_params["min_samples_leaf"], # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
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


    """ RESULT:
    OOS Accuracy:  0.5404167140271572
    OOS Precision:  0.5455320336761093
    OOS Recall:  0.6219836551904432
    OOS F1 score:  0.5812547301334395
    Prediction distribution:
    1.0    200617
    -1.0    142496
    dtype: int64
    """



    """
    rf_classifier = RandomForestClassifier(
        n_estimators=1000,
        min_weight_fraction_leaf=0.2,
        max_features=5,
        class_weight="balanced_subsample",
        bootstrap=True,
        criterion="entropy",
        n_jobs=n_jobs,
    )

    RandomForestClassifier(
        n_estimators = 1000, # Nr of trees
        criterion = "entropy", # function to measure quality of split (impurity measure)
        max_depth = 5, # depth of each tree
        min_samples_split = 2, # minimum number of samples required to split an internal node
        min_samples_leaf = 5, # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
        # NOTE: Set to a lower value to force discrepancy between trees
        max_features = "auto", # the number of features to consider when looking for the best split (auto = sqrt(n_features))
        # NOTE: Need to read up on
        class_weight = "balanced_subsample", # use this attribute to set weight of different feature columns
        bootstrap = True, # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
        n_jobs = 6, # Number of jobs to run in parallell during fit and predict (-1 means to use all available)
        # NOTE: Set to a sufficiently large value (e.g. 5%) such that out-of bag accuracy converges to out of sample (k-fold) accuracy.
        # min_weight_fraction_leaf = 0, # Not relevant unless samples are weighted unequally 
    )
    """



    """
    How to address RF overfitting:
    1. Set a parameter max_features to a lower value, as a way of forcing discrepancy between trees.
    2. Early stopping: Set the regularization parameter min_weight_fraction_leaf to a sufficiently 
    large value (e.g., 5%) such that out-of-bag accuracy converges to out-of-sample (k-fold) accuracy.
    3. Use BaggingClassifier on DecisionTreeClassifier where max_samples is set to the average uniqueness
    (avgU) between samples.
    
    1. clf=DecisionTreeClassifier(criterion=‘entropy’,max_features=‘auto’,class_weight=‘balanced’)
    2. bc=BaggingClassifier(base_estimator=clf,n_estimators=1000,max_samples=avgU,max_features=1.)
    3, Use BaggingClassifier on RandomForestClassifier where max_samples is set to the average uniqueness 
    (avgU) between samples.
    4, clf=RandomForestClassifier(n_estimators=1,criterion=‘entropy’,bootstrap=False,class_weight=‘balanced_subsample’)
    5. bc=BaggingClassifier(base_estimator=clf,n_estimators=1000,max_samples=avgU,max_features=1.)
    6. Modify the RF class to replace standard bootstrapping with sequential bootstrapping.
    """



"""
First output, need to do a better job of generalizing


Best Score:
 0.5065522970590343
Best Params:
 {'class_weight': None, 'criterion': 'entropy', 'max_depth': 39, 'max_features': 57, 'min_samples_leaf': 22390, 'min_samples_split': 13637, 'n_estimators': 971}
Best Index:
 1
CV Results:
 {
    'mean_fit_time': array([2222.89805403, 3656.15981908]), 
    'std_fit_time': array([1055.76469708, 1736.67012387]), 
    'mean_score_time': array([5.66908627, 4.9568913 ]), 
    'std_score_time': array([0.20852488, 0.65835138]), 
    'param_class_weight': masked_array(data=['balanced_subsample', None],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_criterion': masked_array(data=['entropy', 'entropy'],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_max_depth': masked_array(data=[54, 39],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_max_features': masked_array(data=[29, 57],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_min_samples_leaf': masked_array(data=[18179, 22390],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_min_samples_split': masked_array(data=[26168, 13637],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'param_n_estimators': masked_array(data=[861, 971],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'params': [
                {'class_weight': 'balanced_subsample', 
                'criterion': 'entropy', 
                'max_depth': 54, 
                'max_features': 29, 
                'min_samples_leaf': 18179, 
                'min_samples_split': 26168, 
                'n_estimators': 861
                }, 
                {
                    'class_weight': None, 
                    'criterion': 'entropy', 
                    'max_depth': 39, 
                    'max_features': 57, 
                    'min_samples_leaf': 22390, 
                    'min_samples_split': 13637, 
                    'n_estimators': 971}
                ], 
                    'split0_test_score': array([0.34422877, 0.51496245]), 
                    'split1_test_score': array([0.41051415, 0.5061814 ]), 
                    'split2_test_score': array([0.40437897, 0.51146158]), 
                    'split3_test_score': array([0.42644714, 0.51551704]), 
                    'split4_test_score': array([0.37215912, 0.48463876]), 
                    'mean_test_score': array([0.39154568, 0.5065523 ]), 
                    'std_test_score': array([0.02952065, 0.0114502 ]), 
                    'rank_test_score': array([2, 1]), 
                    
                    'split0_train_score': array([0.38461718, 0.53389197]), 
                    'split1_train_score': array([0.44641139, 0.58192798]), 
                    'split2_train_score': array([0.48507628, 0.60720768]), 
                    'split3_train_score': array([0.56298404, 0.6538934 ]), 
                    'split4_train_score': array([0.6378981 , 0.68455441]), 
                    'mean_train_score': array([0.5033974 , 0.61229509]), 
                    'std_train_score': array([0.08869365, 0.05300362])}
"""


"""



"""