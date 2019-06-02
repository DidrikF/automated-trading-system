import pandas as pd

# Algorithms
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Performance Metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Other
import scipy
from scipy.stats import randint
import numpy as np
import pickle


from dataset_development.labeling import meta_labeling_via_triple_barrier_method
from dataset_development.processing.engine import pandas_mp_engine
from dataset_development.sep_features import dividend_adjusting_prices_backwards

from dataset_columns import features, labels, base_cols
from cross_validation import PurgedKFold

"""
C:\Anaconda3\envs\master\lib\site-packages\sklearn\model_selection\_validation.py:559: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details:
ValueError: Found array with 0 sample(s) (shape=(0, 93)) while a minimum of 1 is required.
"""

if __name__ == "__main__":
    # CONFIG
    num_processes = 32 # 32
    n_jobs = 64 # 64

    # DATASET PREPARATION
    print("Reading inn Dataset")
    dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date", "timeout"], index_col="date")
    dataset = dataset.loc[dataset.primary_label_tbm != 0]
    dataset = dataset.sort_values(by="date")

    print("Labels After dropping zero labels")
    print(dataset["primary_label_tbm"].value_counts())

    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
        print(dataset.isnull().sum())
        print(dataset.columns)
    """


    train_start = dataset.index.min()
    # train_start = pd.to_datetime("2010-01-01")
    train_end = pd.to_datetime("2012-01-01")

    test_start = pd.to_datetime("2012-03-01")
    test_end = dataset.index.max()
    # test_end = pd.to_datetime("2013-01-01")

    train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
    test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

    print("Shapes: ", train_set.shape, test_set.shape)

    train_x = train_set[features]
    train_y = train_set["primary_label_tbm"]

    test_x = test_set[features]
    test_y = test_set["primary_label_tbm"]

    print("Train set label distribution:")
    print(train_set["primary_label_tbm"].value_counts())
    print("Test set label distribution:")
    print(test_set["primary_label_tbm"].value_counts())


    # You can generate a plot for precition and recall, see chapter 3 in hands-on machine learning
    training_model = True
    if training_model:
        rf_classifier = RandomForestClassifier(random_state=100, verbose=True, n_jobs=6)

        # Define parameter space:
        # num_samples = len(train_set)
        # many estimators with few features, early stopping and limited depth
        parameter_space = {
            "n_estimators": [50, 100, 200], # [50, 100, 200, 500, 1000] 
            "min_weight_fraction_leaf": [0.20, 0.25, 0.30], # [0.05, 0.10, 0.20] early stopping
            "max_features": [3, 5], # [3,5,10]
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

        print("Training Side Classifier...")
        grid_search.fit(train_x, train_y) # Validation is part of the test set in this case....
        print("DONE TRAINING SIDE CLASSIFIER!")

        print("Best Score (F-1): \n", grid_search.best_score_)
        print("Best Params: \n", grid_search.best_params_)
        print("Best Index: \n", grid_search.best_index_)
        print("CV Results: \n")
        for key, val in grid_search.cv_results_.items():
            print("{}: {}".format(key, val))

        best_params = grid_search.best_params_
        side_classifier: RandomForestClassifier = grid_search.best_estimator_

        # Save
        print("Saving Side Model...")
        pickle.dump(side_classifier, open("./models/side_classifier.pickle", "wb"))
        pickle.dump(best_params, open("./models/side_classifier_best_params.pickle", "wb"))

    else:
        print("Reading inn side model")
        side_classifier = pickle.load(open("./models/side_classifier.pickle", "rb"))
        best_params = pickle.load(open("./models/side_classifier_best_params.pickle", "rb"))


    print("Reading SEP")
    adjust_sep = False
    if adjust_sep:
        sep = pd.read_csv("./dataset_development/datasets/sharadar/SEP_PURGED.csv", parse_dates=["date"], index_col="date")
        print("Adjusting prices for dividends")
        sep_adjusted = pandas_mp_engine(
            callback=dividend_adjusting_prices_backwards, 
            atoms=sep, 
            data=None,
            molecule_key='sep', 
            split_strategy= 'ticker_new',
            num_processes=num_processes, 
            molecules_per_process=1
        )
        print("Writing dividend adjusted sep to disk")
        sep_adjusted.to_csv("./dataset_development/datasets/sharadar/SEP_PURGED_ADJUSTED.csv")
    else:
        sep_adjusted = pd.read_csv("./dataset_development/datasets/sharadar/SEP_PURGED_ADJUSTED.csv", parse_dates=["date"], index_col="date")

    # What data to train on and make predictions for when training the model
    # I think i do K-fold cross validation on the test set and then make predictions on all of the test set
    # Then use the predictions on the test set to set side when labeling the for the second ml model. 

    side_predictions = side_classifier.predict(train_x)
    train_set["side_prediction"] = pd.Series(side_predictions)
    train_set_with_predictions = train_set

    # NOTE: must allways relabel and retrain the certainty model every time the side model changes... (this is not every time though...)
    # NOTE: maybe better to have a sepereate script for model testing and performance measurement.

    print("Meta Labeling of train set")
    train_set_with_meta_labels = pandas_mp_engine(
        callback=meta_labeling_via_triple_barrier_method, 
        atoms=train_set_with_predictions,
        data={'sep': sep_adjusted}, 
        molecule_key='dataset', 
        split_strategy= 'ticker_new', 
        num_processes=num_processes, 
        molecules_per_process=1, 
        ptSl=[1, -0.7], # What is best here? 
        min_ret=None
    )

    print("len train_set_with_meta_labels: ", len(train_set_with_meta_labels))
    train_set_with_meta_labels = train_set_with_meta_labels.replace([np.inf, -np.inf], np.nan)
    train_set_with_meta_labels = train_set_with_meta_labels.dropna(axis=0, subset=features+["m_primary_label_tbm"])
    print("len train_set_with_meta_labels: ", len(train_set_with_meta_labels))


    # Set up training of second model

    certainty_train_x = train_set_with_meta_labels[features] 
    certainty_train_y = train_set_with_meta_labels["m_primary_label_tbm"]

    print("certainty train y value counts: ", certainty_train_y.value_counts())
    print("nans: ", np.any(np.isnan(certainty_train_x)))
    print("infinities: ", np.all(np.isfinite(certainty_train_x)))
    # print("nans: ", certainty_train_x.isinf().sum())


    # NOTE: Use the same params as the best side classifier? Ideally not... but for now...
    certainty_classifier = RandomForestClassifier(
        n_estimators=best_params["n_estimators"], # Nr of trees
        # NOTE: Set to a lower value to force discrepancy between trees
        min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
        max_features=best_params["max_features"], # the number of features to consider when looking for the best split (auto = sqrt(n_features))
        criterion=best_params["criterion"], # function to measure quality of split (impurity measure)
        class_weight=best_params["class_weight"], # use this attribute to set weight of different feature columns
        bootstrap=best_params["bootstrap"], # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
        n_jobs=n_jobs, # Number of jobs to run in parallell during fit and predict (-1 means to use all available)
         # max_depth=best_params["max_depth"], # depth of each tree
        # min_samples_split=best_params["min_samples_split"], # minimum number of samples required to split an internal node
        # min_samples_leaf=best_params["min_samples_leaf"], # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
    )


    print("Training Certainty Classifier...")
    certainty_classifier.fit(certainty_train_x, certainty_train_y)

    # Save
    print("Saving Certainty Model...")
    pickle.dump(certainty_classifier, open("./models/certainty_classifier.pickle", "wb"))


    #____________________ MODEL TESTING ____________________________

    # Testing Side Classifier
    side_score = side_classifier.score(test_x, test_y)
    test_x_pred = side_classifier.predict(test_x)
    side_accuracy = accuracy_score(test_y, test_x_pred)
    side_precision = precision_score(test_y, test_x_pred)
    side_recall = recall_score(test_y, test_x_pred)
    side_f1 = f1_score(test_y, test_x_pred)

    print("Side Classifier Metrics: ")
    print("OOS Accuracy: ", side_accuracy)
    print("OOS Precision: ", side_precision)
    print("OOS Recall: ", side_recall)
    print("OOS f1: ", side_f1)

    # Testing Certainty Classifier
    # Generate side predictions for TEST SET: # NOTE: First time the test set is used...
    test_side_predictions = side_classifier.predict(test_x)
    test_set["side_prediction"] = test_side_predictions
    test_set_with_predictions = test_set

    # Run triple barrier search using the side predictions on the test set -> This will be the correct labels for the certainty-model
    print("Running triple barrier search on test set with side set by side classifier... (Meta labeling on test set)")
    test_set_meta_labeled = pandas_mp_engine(
        callback=meta_labeling_via_triple_barrier_method, 
        atoms=test_set_with_predictions,
        data={'sep': sep_adjusted}, 
        molecule_key='dataset', 
        split_strategy= 'ticker_new', 
        num_processes=num_processes, 
        molecules_per_process=1, 
        ptSl=[1, -0.7], # NOTE: less tolerant for movement downwards... 
        min_ret=None
    )

    print("len test_set_meta_labeled: ", len(test_set_meta_labeled))
    test_set_meta_labeled = test_set_meta_labeled.replace([np.inf, -np.inf], np.nan)
    test_set_meta_labeled = test_set_meta_labeled.dropna(axis=0, subset=features+["m_primary_label_tbm"])
    print("len test_set_meta_labeled: ", len(test_set_meta_labeled))

    # Score the certainty model 
    certainty_test_x = test_set_meta_labeled[features] 
    certainty_test_y = test_set_meta_labeled["m_primary_label_tbm"]

    certainty_test_x_pred = certainty_classifier.predict(certainty_test_x)
    certainty_accuracy = accuracy_score(certainty_test_y, certainty_test_x_pred)
    certainty_precision = precision_score(certainty_test_y, certainty_test_x_pred)
    certainty_recall = recall_score(certainty_test_y, certainty_test_x_pred)
    certainty_f1 = f1_score(certainty_test_y, certainty_test_x_pred)

    print("Certainty Classifier Metrics: ")    
    print("OOS Accuracy: ", certainty_accuracy)
    print("OOS Precision: ", certainty_precision)
    print("OOS Recall: ", certainty_recall)
    print("OOS F1: ", certainty_f1)


    results = {
        "side_model": {
            "accuracy": side_accuracy,
            "precision": side_precision,
            "recall": side_recall,
            "f1": side_f1,
            "cv_results": grid_search.cv_results_,
            "best_params": grid_search.best_params_,
            "test_labels": test_y.value_counts(),
            "pred_labels": pd.Series(data=test_x_pred).value_counts(),
        },
        "certainty_model": {
            "accuracy": certainty_accuracy,
            "precision": certainty_precision,
            "recall": certainty_recall,
            "f1": certainty_f1,
            "test_labels": certainty_test_y.value_counts(),
            "pred_labels": pd.Series(data=certainty_test_x_pred).value_counts(),
        }
    }
    print("Results: ", results)
    print("Saving Results...")

    pickle.dump(results, open("./models/ml_strategy_models_results.pickle", "wb"))

    print("DONE!")

"""
Output of basic model training: Training with -1 lower barrier and -1, 1 labeling

(master -> origin) λ python ml_strategy_models.py
Reading inn Dataset
Labels After dropping zero labels
 1.0    465261
-1.0    436908
Name: primary_label_tbm, dtype: int64
Training Side Classifier...
Saving Side Model...
Reading SEP
ml_strategy_models.py:290: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  train_set["side_prediction"] = pd.Series(side_predictions)
Meta Labeling of train set
Number of jobs:  7579
2019-05-11 13:02:44.430865 100.0% 7579/7579 - meta_labeling_via_triple_barrier_method done after 7.26 minutes. Remaining 0.0 minutes..
Training Certainty Classifier...
Saving Certainty Model...
Side Classifier Accuracy:  0.5419696661705016 # NOTE: More data seemed to help somewhat...
ml_strategy_models.py:351: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  test_set["side_prediction"] = test_side_predictions
Running triple barrier search on test set with side set by side classifier... (Meta labeling on test set)
Number of jobs:  7261
2019-05-11 13:12:06.662944 100.0% 7261/7261 - meta_labeling_via_triple_barrier_method done after 7.64 minutes. Remaining 0.0 minutes..
Certainty Classifier Accuracy:  0.5495498449210835 # NOTE: Higher accuracy on second model!

"""


"""
Training with -0.5 lower barrier and 0,1 labeling

(master -> origin) λ python ml_strategy_models.py
Reading inn Dataset
Labels After dropping zero labels
 1.0    465261
-1.0    436908
Name: primary_label_tbm, dtype: int64
Train set label distribution:
 1.0    217895
-1.0    208538
Name: primary_label_tbm, dtype: int64
Test set label distribution:
 1.0    237373
-1.0    220136
Name: primary_label_tbm, dtype: int64
Training Side Classifier...
Saving Side Model...
Reading SEP
ml_strategy_models.py:294: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  train_set["side_prediction"] = pd.Series(side_predictions)
Meta Labeling of train set
Number of jobs:  7579
2019-05-12 17:40:04.698720 100.0% 7579/7579 - meta_labeling_via_triple_barrier_method done after 6.3 minutes. Remaining 0.0 minutes...
Training Certainty Classifier...
Saving Certainty Model...
Side Classifier Accuracy:  0.5422516278368295
ml_strategy_models.py:356: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  test_set["side_prediction"] = test_side_predictions
Running triple barrier search on test set with side set by side classifier... (Meta labeling on test set)
Number of jobs:  7261
2019-05-12 17:48:19.853732 100.0% 7261/7261 - meta_labeling_via_triple_barrier_method done after 6.78 minutes. Remaining 0.0 minutes..
Certainty Classifier Accuracy:  0.489347750536055

"""

"""
{'side_model': {
    'accuracy': 0.5405986704631229, 
    'precision': 0.548164014385144, 
    'recall': 0.625687441419263, 
    'cv_results': {'mean_fit_time': array([2.5686957]), 
    'std_fit_time': array([2.39578189]),
     'mean_score_time': array([0.25544715]),
      'std_score_time': array([0.00885381]), 
      'param_n_estimators': masked_array(data=[20],
             mask=[False],
       fill_value='?',
            dtype=object), 'param_min_weight_fraction_leaf': masked_array(data=[0.1],
             mask=[False],
       fill_value='?',
            dtype=object), 'param_max_features': masked_array(data=[20],
             mask=[False],
       fill_value='?',
            dtype=object), 'param_criterion': masked_array(data=['entropy'],
             mask=[False],
       fill_value='?',
            dtype=object), 'param_class_weight': masked_array(data=['balanced_subsample'],
             mask=[False],
       fill_value='?',
            dtype=object), 'param_bootstrap': masked_array(data=[True],
             mask=[False],
       fill_value='?',
            dtype=object), 'params':
             [{'n_estimators': 20, 
             'min_weight_fraction_leaf': 0.1, 
             'max_features': 20, 'criterion': 
             'entropy', 'class_weight': 
             'balanced_subsample', 'bootstrap': True}],
              'split0_test_score': array([0.49030169]), 
              'split1_test_score': array([0.48618842]),
               'split2_test_score': array([0.49347173]),
                'mean_test_score': array([0.48998728]),
                 'std_test_score': array([0.0029817]), 
                 'rank_test_score': array([1], dtype=int32),
                  'split0_train_score': array([0.95175938]),
                   'split1_train_score': array([0.92796616]),
                   'split2_train_score': array([0.74132879]), 
                   'mean_train_score': array([0.87368478]),
                    'std_train_score': array([0.09409254])},
                     'best_params': {'n_estimators': 20,
                      'min_weight_fraction_leaf': 0.1, 
                      'max_features': 20, 'criterion': 
                      'entropy', 'class_weight': 
                      'balanced_subsample', 
                      'bootstrap': True}}, 


                'certainty_model': {
                        'accuracy': 0.5352420532339338,
                       'precision': 0.5452637819170709,
                        'recall': 0.6484847128340571}}

"""

"""

split0_test_score: [0.4968272]
split1_test_score: [0.4878566]
split2_test_score: [0.48547426]
mean_test_score: [0.4900527]
std_test_score: [0.00488804]
rank_test_score: [1]
split0_train_score: [0.90030569]
split1_train_score: [0.90175316]
split2_train_score: [0.71619287]
mean_train_score: [0.83941724]
std_train_score: [0.08713479]
Saving Side Model...
Reading SEP
ml_strategy_models.py:155: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  train_set["side_prediction"] = pd.Series(side_predictions)
Meta Labeling of train set
Number of jobs:  8061
2019-05-16 16:38:38.616784 100.0% 8061/8061 - meta_labeling_via_triple_barrier_method done after 1.41 minutes. Remaining 0.0 minutes..
Training Certainty Classifier...
Saving Certainty Model...
Side Classifier Metrics:
OOS Accuracy:  0.5404296368400039
OOS Precision:  0.5473141915767823
OOS Recall:  0.6340440188361772
ml_strategy_models.py:222: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  test_set["side_prediction"] = test_side_predictions
Running triple barrier search on test set with side set by side classifier... (Meta labeling on test set)
Number of jobs:  6602
2019-05-16 16:40:41.148306 100.0% 6602/6602 - meta_labeling_via_triple_barrier_method done after 0.85 minutes. Remaining 0.0 minutes..
OOS Accuracy:  0.5327298638696459
OOS Precision:  0.5445449447789941
OOS Recall:  0.625282748426687


"""


"""
params: [{'n_estimators': 300, 'min_weight_fraction_leaf': 0.2, 'max_features': 5, 'criterion': 'entropy', 'class_weight': 'balanced_subsample', 'bootstrap': True}]
split0_test_score: [0.4968381]
split1_test_score: [0.48691348]
split2_test_score: [0.48558329]
mean_test_score: [0.4897783]
std_test_score: [0.00502149]
rank_test_score: [1]
split0_train_score: [0.90035031]
split1_train_score: [0.90281725]
split2_train_score: [0.70116126]
mean_train_score: [0.83477627]
std_train_score: [0.09448545]
Saving Side Model...
Reading SEP
ml_strategy_models.py:155: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  train_set["side_prediction"] = pd.Series(side_predictions)
Meta Labeling of train set
Number of jobs:  8061
2019-05-16 16:51:22.422277 100.0% 8061/8061 - meta_labeling_via_triple_barrier_method done after 1.37 minutes. Remaining 0.0 minutes..
Training Certainty Classifier...
Saving Certainty Model...
Side Classifier Metrics:
OOS Accuracy:  0.5403392912828194
OOS Precision:  0.5482842453469371
OOS Recall:  0.6214131649972333
ml_strategy_models.py:222: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

"""


"""

DONE TRAINING SIDE CLASSIFIER!
Best Score (Accuracy):
 0.5588232383244957
Best Params:
 {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.2, 'n_estimators': 200}
Best Index:
 6
CV Results:

mean_fit_time: [ 22.99247424 116.2861836  591.19373178  18.23488649  90.27693462
 446.21771661  13.19054008  62.3763574  316.62517373  40.23785345
 191.96455606 944.71733991  27.68358819 137.6297187  692.58647251
  18.1145273   91.10986773 449.05331039]
std_fit_time: [ 10.07189116  54.03907939 269.15865748   7.26571574  38.17827894
 190.58302098   4.34038784  20.38847132 108.36304626  20.28947802
  98.31725508 478.73037574  13.49162604  67.10749522 335.33419349
   7.90094433  39.15953476 190.46912359]
mean_score_time: [ 2.81446934 13.83632008 68.2646327   2.56014967 11.77074289 57.826636
  2.36030944 10.32171353 50.97692593  2.89375965 13.09595696 63.67228905
  2.48535013 11.24191864 56.47243524  2.10835854  9.84764942 44.00029071]
std_score_time: [0.05865366 0.75570194 2.01806382 0.04352607 0.06767862 1.07136747
 0.15146329 0.2182148  2.00411352 0.06304402 0.30051712 2.48963867
 0.06474195 0.12523753 1.04763428 0.04236708 0.10699206 6.08658855]
param_bootstrap: [True True True True True True True True True True True True True True
 True True True True]
param_class_weight: ['balanced_subsample' 'balanced_subsample' 'balanced_subsample'
 'balanced_subsample' 'balanced_subsample' 'balanced_subsample'
 'balanced_subsample' 'balanced_subsample' 'balanced_subsample'
 'balanced_subsample' 'balanced_subsample' 'balanced_subsample'
 'balanced_subsample' 'balanced_subsample' 'balanced_subsample'
 'balanced_subsample' 'balanced_subsample' 'balanced_subsample']
param_criterion: ['entropy' 'entropy' 'entropy' 'entropy' 'entropy' 'entropy' 'entropy'
 'entropy' 'entropy' 'entropy' 'entropy' 'entropy' 'entropy' 'entropy'
 'entropy' 'entropy' 'entropy' 'entropy']
param_max_features: [5 5 5 5 5 5 5 5 5 10 10 10 10 10 10 10 10 10]
param_min_weight_fraction_leaf: [0.05 0.05 0.05 0.1 0.1 0.1 0.2 0.2 0.2 0.05 0.05 0.05 0.1 0.1 0.1 0.2 0.2
 0.2]
param_n_estimators: [200 1000 5000 200 1000 5000 200 1000 5000 200 1000 5000 200 1000 5000 200
 1000 5000]
params: [{'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.05, 'n_estimators': 200}, 
{'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.05, 'n_estimators': 1000}, 
{'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.05, 'n_estimators': 5000},
 {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.1, 'n_estimators': 200},
  {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.1, 'n_estimators': 1000},
   {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.1, 'n_estimators': 5000}, 
   {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.2, 'n_estimators': 200}, <----
    {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.2, 'n_estimators': 1000},
     {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 5, 'min_weight_fraction_leaf': 0.2, 'n_estimators': 5000},
      {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 10, 'min_weight_fraction_leaf': 0.05, 'n_estimators': 200}, 
      {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 10, 'min_weight_fraction_leaf': 0.05, 'n_estimators': 1000},
       {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 10, 'min_weight_fraction_leaf': 0.05, 'n_estimators': 5000}, 
       {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 10, 'min_weight_fraction_leaf': 0.1, 'n_estimators': 200}, 
       {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 10, 'min_weight_fraction_leaf': 0.1, 'n_estimators': 1000},
        {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 10, 'min_weight_fraction_leaf': 0.1, 'n_estimators': 5000}, 
        {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 10, 'min_weight_fraction_leaf': 0.2, 'n_estimators': 200},
         {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 10, 'min_weight_fraction_leaf': 0.2, 'n_estimators': 1000},
          {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_features': 10, 'min_weight_fraction_leaf': 0.2, 'n_estimators': 5000}]
split0_test_accuracy: [0.54168441 0.54020835 0.53961225 0.54023674 0.53912969 0.53893099
 0.53620597 0.53728462 0.53651821 0.54855375 0.54736155 0.54687899
 0.54685061 0.54548809 0.54409719 0.54517585 0.54279145 0.54151409]
split1_test_accuracy: [0.5067416  0.50336371 0.50492492 0.52235374 0.5182662  0.51894746
 0.52947855 0.52791734 0.52791734 0.49958841 0.50131994 0.50248375
 0.51077238 0.51301485 0.51522893 0.51559794 0.5168753  0.52133186]
split2_test_accuracy: [0.4978994  0.49699103 0.49639491 0.50141933 0.50048257 0.50136255
 0.50579085 0.50451346 0.50383218 0.49849551 0.49812649 0.49835358
 0.50482571 0.50366186 0.5035767  0.50721017 0.50391734 0.50306574]
mean_test_accuracy: [0.51544197 0.51352119 0.51364419 0.52133679 0.519293   0.51974718
 0.52382529 0.52323865 0.52275609 0.51554605 0.51560282 0.51590561
 0.52081638 0.52072176 0.52096777 0.52266147 0.52119486 0.52197074]
std_test_accuracy: [0.01890419 0.0190493  0.01868961 0.01586341 0.01579428 0.01534764
 0.0130445  0.01378173 0.01383407 0.0233444  0.02249478 0.02196645
 0.0185685  0.01792399 0.01703284 0.01628429 0.01616151 0.01570294]
rank_test_accuracy: [16 18 17  6 12 11  1  2  3 15 14 13  9 10  8  4  7  5]
split0_train_accuracy: [0.57342244 0.57193732 0.57234649 0.56384494 0.56404195 0.56325392
 0.55479784 0.55702552 0.5558738  0.58116627 0.58087834 0.58077226
 0.57342244 0.5727405  0.5720434  0.56551191 0.56257198 0.56145057]
split1_train_accuracy: [0.74676647 0.75185928 0.75187544 0.73211861 0.72967729 0.73108388
 0.66874151 0.68348639 0.67459419 0.75666106 0.75507663 0.75363772
 0.72864257 0.73140723 0.73265214 0.67483671 0.68971092 0.68963008]
split2_train_accuracy: [0.95988747 0.9587177  0.9591323  0.85146961 0.8535278  0.8439772
 0.7782187  0.77502036 0.76530688 0.95240986 0.95935441 0.96142741
 0.866795   0.86692826 0.87083734 0.71482935 0.7381654  0.74121567]
mean_train_accuracy: [0.76002546 0.7608381  0.76111808 0.71581105 0.71574901 0.71277167
 0.66725269 0.67184409 0.66525829 0.7634124  0.76510313 0.76527913
 0.72295333 0.723692   0.72517763 0.65172599 0.66348277 0.66409877]
std_train_accuracy: [0.15805201 0.15803002 0.15803984 0.11798712 0.11859177 0.11533398
 0.09121726 0.08937597 0.08575517 0.15163473 0.15467478 0.15561968
 0.11983639 0.12022549 0.12209656 0.06311103 0.07404593 0.07557671]
split0_test_balanced_accuracy: [0.54168301 0.54017797 0.53959711 0.54023775 0.53917437 0.53896452
 0.53626071 0.53737398 0.53660562 0.54866806 0.54744816 0.54697213
 0.54702714 0.54565666 0.54424126 0.54533506 0.54294728 0.54163595]
split1_test_balanced_accuracy: [0.52507716 0.5238801  0.52491843 0.53033605 0.52756273 0.52804043
 0.52995214 0.52951121 0.53005551 0.52121621 0.52244698 0.52402309
 0.52340981 0.52334605 0.52538316 0.52168911 0.52095523 0.52420304]
split2_test_balanced_accuracy: [0.52362656 0.52433717 0.52385105 0.52620036 0.52543283 0.52633318
 0.52768893 0.52644335 0.52590591 0.52193699 0.52111622 0.5220058
 0.52288698 0.52344363 0.52396319 0.52274458 0.52180249 0.52101651]
mean_test_balanced_accuracy: [0.53012897 0.52946513 0.52945558 0.53225811 0.53072336 0.53111276
 0.53130062 0.53110956 0.53085573 0.53060717 0.53033721 0.53100042
 0.53110806 0.53081552 0.53119594 0.52992298 0.5285684  0.52895191]
std_test_balanced_accuracy: [0.00819143 0.00757748 0.00718442 0.00588969 0.00603874 0.00559565
 0.00362699 0.0046033  0.00440463 0.01277446 0.01211155 0.01132377
 0.0112586  0.01049442 0.0092427  0.01090658 0.01017336 0.00906288]
rank_test_balanced_accuracy: [13 15 16  1 10  4  2  5  8 11 12  7  6  9  3 14 18 17]
split0_train_balanced_accuracy: [0.57290665 0.57134066 0.57177776 0.5633092  0.56339367 0.56252103
 0.55397504 0.55627815 0.55508338 0.58145041 0.5810538  0.58094733
 0.57374336 0.57284847 0.57203462 0.56548987 0.56226771 0.56091497]
split1_train_balanced_accuracy: [0.74787031 0.75272007 0.75272508 0.73178549 0.72942723 0.73075922
 0.66937341 0.68341039 0.6749812  0.75778519 0.75621608 0.75477614
 0.72866389 0.73139056 0.73256816 0.67492965 0.68959855 0.68952348]
split2_train_balanced_accuracy: [0.95681948 0.95646276 0.95702937 0.8490436  0.85249531 0.84389473
 0.77464135 0.77264876 0.76412352 0.95043559 0.9561995  0.95944825
 0.86998904 0.86892923 0.87193001 0.71309182 0.73183152 0.73578926]
mean_train_balanced_accuracy: [0.75919881 0.7601745  0.76051074 0.71471276 0.7151054  0.71239166
 0.6659966  0.6707791  0.66472937 0.76322373 0.7644898  0.76505724
 0.7241321  0.72438942 0.72551093 0.65117045 0.66123259 0.6620759 ]
std_train_balanced_accuracy: [0.15693633 0.15731377 0.15737463 0.1172736  0.11845893 0.11560224
 0.09011828 0.08878334 0.08564761 0.15068665 0.15326429 0.15469327
 0.12098424 0.1209758  0.12253344 0.06255642 0.07207146 0.07398325]
split0_test_precision: [0.54389746 0.54188599 0.54157506 0.54249702 0.54220704 0.54179388
 0.53940557 0.54115701 0.54032353 0.55349671 0.55157559 0.55122817
 0.55325043 0.55157611 0.54949385 0.55101044 0.5483694  0.54625075]
split1_test_precision: [0.63931567 0.63862543 0.6396421  0.64176765 0.6393593  0.63978527
 0.63936086 0.63923151 0.6398835  0.63589493 0.63716937 0.63909931
 0.63596075 0.63533416 0.63737349 0.63273679 0.63162503 0.63450343]
split2_test_precision: [0.46079117 0.46094941 0.46060491 0.4627488  0.46218454 0.46279727
 0.464484   0.46358778 0.46317032 0.46007757 0.45958987 0.46008454
 0.46191498 0.46192893 0.46216914 0.46241772 0.46114123 0.4605365 ]
mean_test_precision: [0.54800226 0.54715442 0.54727484 0.54900531 0.54791777 0.54812628
 0.54775093 0.5479929  0.54779325 0.54982392 0.54944579 0.55013819
 0.55037622 0.5496139  0.54967965 0.54872247 0.54704603 0.54709771]
std_test_precision: [0.07293992 0.07263136 0.07320249 0.07322872 0.07244375 0.07239349
 0.07163648 0.07186882 0.07233578 0.07182394 0.072512   0.07308637
 0.07108278 0.07080581 0.07152683 0.06955112 0.06960584 0.07102406]
rank_test_precision: [10 16 15  7 12  9 14 11 13  3  6  2  1  5  4  8 18 17]
split0_train_precision: [0.58075701 0.5788084  0.57938038 0.57153575 0.5710488  0.56981464
 0.56145889 0.5639215  0.56262364 0.5944866  0.59327649 0.59316627
 0.58689299 0.58456095 0.58299876 0.57642659 0.5717271  0.56926407]
split1_train_precision: [0.77730463 0.77824352 0.77806447 0.74003872 0.73888209 0.7391739
 0.69037963 0.6966478  0.69336257 0.78790564 0.78654791 0.78503907
 0.74176262 0.7438908  0.74411383 0.6901662  0.70224915 0.70224042]
split2_train_precision: [0.96473829 0.96653012 0.96727281 0.89460225 0.89992411 0.89590204
 0.83950965 0.84010546 0.83581113 0.96336269 0.96412921 0.96913059
 0.92164644 0.91813016 0.9181482  0.79605631 0.8037376  0.80798114]
mean_train_precision: [0.77426664 0.77452735 0.77490589 0.73539224 0.73661833 0.73496353
 0.69711606 0.70022492 0.69726578 0.78191831 0.78131787 0.78244531
 0.75010068 0.74886064 0.74842026 0.6875497  0.69257129 0.69316187]
std_train_precision: [0.15677442 0.15830854 0.15837217 0.13193226 0.13427233 0.13315791
 0.11361364 0.11278    0.11156247 0.15065253 0.15144515 0.15349775
 0.13678964 0.13622439 0.13685806 0.08968254 0.09496478 0.09766703]
split0_test_recall: [0.54199164 0.54690856 0.54295241 0.54001356 0.52927546 0.53153611
 0.52413247 0.51757658 0.51723748 0.52334125 0.52825817 0.52633661
 0.50791229 0.5083079  0.51232056 0.51005991 0.50842093 0.51463773]
split1_test_recall: [0.44259173 0.43158384 0.43497445 0.49442638 0.48574083 0.48713423
 0.52782164 0.52234092 0.5204366  0.42392011 0.42740362 0.42712494
 0.46655829 0.47686948 0.47970274 0.49428704 0.50260102 0.51128658]
split2_test_recall: [0.75673261 0.77211232 0.77262285 0.75073389 0.75149968 0.75258456
 0.72610083 0.72514359 0.72590938 0.73433312 0.72941927 0.73631142
 0.68653478 0.70268028 0.708679   0.66349713 0.6838545  0.68366305]
mean_test_recall: [0.58043699 0.58353312 0.58351478 0.59505647 0.58883712 0.59041677
 0.59268372 0.5883524  0.58785985 0.56052985 0.5616921  0.56325602
 0.5536672  0.5626179  0.56689943 0.55594701 0.56495769 0.56986138]
std_test_recall: [0.1310968  0.14141147 0.14079648 0.11164118 0.11638324 0.11609226
 0.09435083 0.09674416 0.09762311 0.12942489 0.12554319 0.12889596
 0.09545509 0.09986585 0.10113239 0.07632047 0.08410511 0.08048042]
rank_test_recall: [ 9  7  8  1  4  3  2  5  6 16 15 13 18 14 11 17 12 10]
split0_train_recall: [0.59574973 0.5977652  0.59696494 0.58703577 0.59210409 0.5949791
 0.59041465 0.58937728 0.59008862 0.56886689 0.57328314 0.57319423
 0.55953051 0.56806663 0.5724236  0.56646611 0.57574321 0.58463499]
split1_train_recall: [0.7143036  0.72654416 0.72688852 0.74191529 0.73703159 0.74063175
 0.65015809 0.68572144 0.6632126  0.72360142 0.72156654 0.72015778
 0.72801553 0.73189744 0.73512194 0.67210343 0.69301568 0.69276524]
split2_train_recall: [0.97019995 0.96629728 0.96620092 0.85962419 0.85699831 0.8442544
 0.79024331 0.78299205 0.76928451 0.95904601 0.96995905 0.96807998
 0.85605878 0.86020236 0.86716454 0.72066972 0.75945555 0.75945555]
mean_train_recall: [0.76008443 0.76353554 0.76335146 0.72952508 0.72871133 0.72662175
 0.67693869 0.68603026 0.67419524 0.75050477 0.75493624 0.75381066
 0.71453494 0.72005548 0.72490336 0.65307975 0.67607148 0.67895193]
std_train_recall: [0.15625867 0.15270941 0.15292909 0.1116281  0.10830253 0.10224726
 0.08374873 0.0790432  0.07356745 0.1604219  0.16365226 0.16295823
 0.12143187 0.11955751 0.12054424 0.0643745  0.07595124 0.07203547]
split0_test_f1: [0.54294287 0.54438569 0.54226286 0.54125244 0.53566322 0.53661598
 0.53165936 0.5291042  0.52852853 0.53799675 0.53966513 0.5384949
 0.52961282 0.52905882 0.5302565  0.52974496 0.52764011 0.52997323]
split1_test_f1: [0.52306848 0.51507761 0.51781814 0.55854343 0.55206271 0.5531208
 0.57826175 0.57490479 0.57401194 0.50870886 0.51162015 0.51204098
 0.53824514 0.54481295 0.54741082 0.55500795 0.55977446 0.56626971]
split2_test_f1: [0.5727949  0.57726991 0.57714218 0.57256887 0.57235899 0.57314347
 0.56654882 0.5655907  0.56551244 0.56571864 0.56388752 0.56631
 0.55225873 0.5574202  0.55947403 0.54500183 0.55083787 0.55034419]
mean_test_f1: [0.5462685  0.54557743 0.54574076 0.55745477 0.55336146 0.55429324
 ###0.55882324### 0.55653314 0.55601755 0.53747448 0.53839069 0.53894837 <---- BEST MEAN F1 GIVES THE CHOSEN PARAMS
 0.54003878 0.54376386 0.54571365 0.54325157 0.5460841  0.54886236]
std_test_f1: [0.02043643 0.02540382 0.02434342 0.012808   0.01500907 0.01493528
 0.01979416 0.01976455 0.01974509 0.02327702 0.02135703 0.0221575
 0.00933172 0.0116022  0.0119882  0.01038761 0.01354264 0.01485505]
rank_test_f1: [ 8 12 10  2  6  5  1  3  4 18 17 16 15 13 11 14  9  7]
split0_train_f1: [0.58815784 0.58813409 0.58804123 0.57918208 0.58138587 0.58212504
 0.57557283 0.57636846 0.57602893 0.58139464 0.58310848 0.58300926
 0.57288521 0.57619577 0.57766279 0.57140294 0.57372813 0.57684715]
split1_train_f1: [0.74447363 0.75150573 0.75160638 0.74097582 0.73795568 0.73990211
 0.66966546 0.69114144 0.67795254 0.75438568 0.75265727 0.75120008
 0.73482479 0.73784539 0.73959055 0.68101507 0.69760187 0.69747065]
split2_train_f1: [0.96746141 0.96641369 0.96673657 0.8767645  0.87793682 0.86931177
 0.81413184 0.8105439  0.80116915 0.96119951 0.96703534 0.968605
 0.88764269 0.88822278 0.89192839 0.75648952 0.78096937 0.7829672 ]
mean_train_f1: [0.76669763 0.7686845  0.76879472 0.73230747 0.73242612 0.73044631
 0.68645671 0.6926846  0.68505021 0.76565994 0.76760037 0.76760478
 0.73178423 0.73408798 0.73639391 0.66963585 0.68409979 0.68576167]
std_train_f1: [0.15564539 0.154909   0.15507873 0.12164204 0.12112954 0.11743399
 0.09811238 0.09560795 0.09205003 0.15525949 0.15709324 0.15784561
 0.12851719 0.1274122  0.1283183  0.07598849 0.08514287 0.08455449]


"""