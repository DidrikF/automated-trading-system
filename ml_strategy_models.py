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
        rf_classifier = RandomForestClassifier(random_state=100)

        # Define parameter space:
        num_samples = len(train_set)
        parameter_space = {
            "n_estimators": [20], # 50, 100, 200, 500, 1000
            "max_depth": [1, 2, 4, 8, 10, 15], # max depth should be set lower I think
            "min_samples_split": [int(num_samples*0.02),int(num_samples*0.04),int(num_samples*0.06),int(num_samples*0.08)], # I have 550,000 samples for training -> 5500
            "min_samples_leaf": [int(num_samples*0.02),int(num_samples*0.04),int(num_samples*0.06),int(num_samples*0.08)], # 2-8% of samples 
            "max_features": [5, 10, 15, 20, 30], # 30 may even push it??
            "class_weight": [None, "balanced_subsample"],
            "bootstrap": [True], # , False
            "criterion": ["entropy"] # , "gini"
        }

        
        t1 = pd.Series(index=train_set.index, data=train_set["timeout"])

        random_search = RandomizedSearchCV(
            estimator=rf_classifier,
            param_distributions=parameter_space,
            n_iter=2, # NOTE: Need to update 
            #  NOTE: need to update to use the date and timout columns
            cv=PurgedKFold(n_splits=5, t1=t1), # a CV splitter object implementing a split method yielding arrays of train and test indices
            # Need to figure out if just using built in scorers will work with the custom PurgedKFold splitter
            scoring="accuracy", # a string or a callable to evaluate the predictions on the test set (use custom scoring function that works with PurgedKFold)
            n_jobs=n_jobs,
            verbose=1
        )

        print("Training Side Classifier...")
        random_search.fit(train_x, train_y) # Validation is part of the test set in this case....
    
        print("Best Score (Accuracy): \n", random_search.best_score_)
        print("Best Params: \n", random_search.best_params_)
        print("Best Index: \n", random_search.best_index_)
        print("CV Results: \n", random_search.cv_results_)

        best_params = random_search.best_params_
        side_classifier: RandomForestClassifier = random_search.best_estimator_

        # Save
        print("Saving Side Model...")
        pickle.dump(side_classifier, open("./models/side_classifier.pickle", "wb"))
        pickle.dump(best_params, open("./models/side_classifier_best_params.pickle", "wb"))

    else:
        print("Reading inn side model")
        side_classifier = pickle.load(open("./models/side_classifier.pickle", "rb"))
        best_params = pickle.load(open("./models/side_classifier_best_params.pickle", "rb"))


    print("Reading SEP")
    adjust_sep = True # NOTE: set to false
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

    # NOTE: Use the same params as the best side classifier? Ideally not... but for now...
    certainty_classifier = RandomForestClassifier(
        n_estimators=best_params["n_estimators"], # Nr of trees
        criterion=best_params["criterion"], # function to measure quality of split (impurity measure)
        max_depth=best_params["max_depth"], # depth of each tree
        min_samples_split=best_params["min_samples_split"], # minimum number of samples required to split an internal node
        min_samples_leaf=best_params["min_samples_leaf"], # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
        # NOTE: Set to a lower value to force discrepancy between trees
        max_features=best_params["max_features"], # the number of features to consider when looking for the best split (auto = sqrt(n_features))
        # NOTE: Need to read up on
        class_weight=best_params["class_weight"], # use this attribute to set weight of different feature columns
        bootstrap=best_params["bootstrap"], # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
        n_jobs=n_jobs, # Number of jobs to run in parallell during fit and predict (-1 means to use all available)
        # NOTE: Set to a sufficiently large value (e.g. 5%) such that out-of bag accuracy converges to out of sample (k-fold) accuracy.
        # min_weight_fraction_leaf = 0, # Not relevant unless samples are weighted unequally 
    )

    print("Training Certainty Classifier...")
    certainty_classifier.fit(certainty_train_x, certainty_train_y)

    # Save
    print("Saving Certainty Model...")
    pickle.dump(certainty_classifier, open("./models/certainty_classifier.pickle", "wb"))

    # Testing Side Classifier
    side_score = side_classifier.score(test_x, test_y)
    print("Side Classifier Metrics: ")
    test_x_pred = side_classifier.predict(certainty_test_x)
    side_accuracy = accuracy_score(test_y, test_x_pred)
    side_precision = precision_score(test_y, test_x_pred)
    side_recall = recall_score(test_y, test_x_pred)

    print("OOS Accuracy: ", side_accuracy)
    print("OOS Precision: ", side_precision)
    print("OOS Recall: ", side_recall)


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

    print("OOS Accuracy: ", certainty_accuracy)
    print("OOS Precision: ", certainty_precision)
    print("OOS Recall: ", certainty_recall)


    results = {
        "side_model": {
            "accuracy": side_accuracy,
            "precision": side_precision,
            "recall": side_recall
        },
        "certainty_model": {
            "accuracy": certainty_accuracy,
            "precision": certainty_precision,
            "recall": certainty_recall,
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