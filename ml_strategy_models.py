import pandas as pd

# Algorithms
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


# Performance Metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

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

if __name__ == "__main__":
    # CONFIG
    num_processes = 32
    n_jobs = 64

    # DATASET PREPARATION
    print("Reading inn Dataset")
    dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date", "timeout"], index_col="date")
    dataset = dataset.loc[dataset.primary_label_tbm != 0]

    print("Labels After dropping zero labels")
    print(dataset["primary_label_tbm"].value_counts())

    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
        print(dataset.isnull().sum())
        print(dataset.columns)
    """


    train_start = dataset.index.min()
    train_end = pd.to_datetime("2012-01-01")

    test_start = pd.to_datetime("2012-03-01")
    test_end = dataset.index.max()

    train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
    test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

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
        side_classifier = RandomForestClassifier(
            n_estimators=1000,
            min_weight_fraction_leaf=0.2,
            max_features=5,
            class_weight=None,
            bootstrap=True,
            criterion="entropy",
            n_jobs=n_jobs,
        )
        
        print("Training Side Classifier...")
        side_classifier.fit(train_x, train_y) # Validation is part of the test set in this case....
        print("DONE TRAINING SIDE CLASSIFIER!")

        # Save
        print("Saving Side Model...")
        pickle.dump(side_classifier, open("./models/side_classifier.pickle", "wb"))

    else:
        print("Reading inn side model")
        side_classifier = pickle.load(open("./models/side_classifier.pickle", "rb"))


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
        ptSl=[1, -0.8], # 
        min_ret=None
    )


    # Set up training of second model

    certainty_train_x = train_set_with_meta_labels[features] 
    certainty_train_y = train_set_with_meta_labels["m_primary_label_tbm"]

    certainty_classifier = RandomForestClassifier(
        n_estimators=1000,
        min_weight_fraction_leaf=0.2,
        max_features=5,
        class_weight=None,
        bootstrap=True,
        criterion="entropy",
        n_jobs=n_jobs,
    )


    print("Training Certainty Classifier...")
    certainty_classifier.fit(certainty_train_x, certainty_train_y)

    # Save
    print("Saving Certainty Model...")
    pickle.dump(certainty_classifier, open("./models/certainty_classifier.pickle", "wb"))

    # Testing Side Classifier
    side_score = side_classifier.score(test_x, test_y)
    print("Side Classifier Metrics: ")
    test_x_pred = side_classifier.predict(test_x)
    side_accuracy = accuracy_score(test_y, test_x_pred)
    side_precision = precision_score(test_y, test_x_pred)
    side_recall = recall_score(test_y, test_x_pred)
    side_f1_score = f1_score(test_y, test_x_pred)

    print("OOS Accuracy: ", side_accuracy)
    print("OOS Precision: ", side_precision)
    print("OOS Recall: ", side_recall)
    print("OOS F1 score: ", side_f1_score)


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
        ptSl=[1, -0.8], # NOTE: less tolerant for movement downwards... 
        min_ret=None
    )

    # Score the certainty model 
    certainty_test_x = test_set_meta_labeled[features] 
    certainty_test_y = test_set_meta_labeled["m_primary_label_tbm"]

    certainty_test_x_pred = certainty_classifier.predict(certainty_test_x)
    certainty_accuracy = accuracy_score(certainty_test_y, certainty_test_x_pred)
    certainty_precision = precision_score(certainty_test_y, certainty_test_x_pred)
    certainty_recall = recall_score(certainty_test_y, certainty_test_x_pred)
    certainty_f1_score = f1_score(certainty_test_y, certainty_test_x_pred)

    print("OOS Accuracy: ", certainty_accuracy)
    print("OOS Precision: ", certainty_precision)
    print("OOS Recall: ", certainty_recall)
    print("OOS F1 score: ", certainty_f1_score)



    results = {
        "side_model": {
            "accuracy": side_accuracy,
            "precision": side_precision,
            "recall": side_recall,
            "f1": side_f1_score,
        },
        "certainty_model": {
            "accuracy": certainty_accuracy,
            "precision": certainty_precision,
            "recall": certainty_recall,
            "f1": certainty_f1_score,
        }
    }
    pickle.dump(results, open("./models/ml_strategy_models_results.pickle", "wb"))

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