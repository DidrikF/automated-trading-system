import pandas as pd

# Algorithms
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


# Performance Metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

# Other
import scipy
from scipy.stats import randint
import numpy as np
import pickle


from dataset_development.labeling import meta_labeling_via_triple_barrier_method
from dataset_development.processing.engine import pandas_mp_engine
from dataset_development.sep_features import dividend_adjusting_prices_backwards

from dataset_columns import features, labels, base_cols



if __name__ == "__main__":
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

    # Feature scaling not needed

    # Encoding Categorical Features:
    # Not using industry now


    # Train, validate, test split
    # NOTE: combine and run cross validation (1995-2009) and then run predictions over the test period with no retraining...
    # NOTE: Maybe move the boundaries to include more data in the training set...
    train_start = dataset.index.min() # Does Date index included in features when training a model?
    train_end = pd.to_datetime("2009-09-01")
    
    test_start = pd.to_datetime("2010-01-01")
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
        # NOTE: USE THE SAME PARAMETERS THAT WAS FOUND TO BE OPTIMAL FOR rf_erp_classifier_model
        side_classifier = RandomForestClassifier(
            n_estimators = 200, # Nr of trees
            criterion = "entropy", # function to measure quality of split (impurity measure)
            max_depth = 10, # depth of each tree
            min_samples_split = int(0.05*len(train_x)), # minimum number of samples required to split an internal node
            min_samples_leaf = int(0.05*len(train_x)), # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
            # NOTE: Set to a lower value to force discrepancy between trees
            max_features = "auto", # the number of features to consider when looking for the best split (auto = sqrt(n_features))
            # NOTE: Need to read up on
            # class_weight = "balanced_subsample", # use this attribute to set weight of different feature columns
            bootstrap = True, # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
            n_jobs = 6, # Number of jobs to run in parallell during fit and predict (-1 means to use all available)
            # NOTE: Set to a sufficiently large value (e.g. 5%) such that out-of bag accuracy converges to out of sample (k-fold) accuracy.
            # min_weight_fraction_leaf = 0, # Not relevant unless samples are weighted unequally 
        )
        
        # NOTE: See AFML for tips on how to train side and certainty classifiers
        print("Training Side Classifier...")
        side_classifier.fit(train_x, train_y)

        # Save
        print("Saving Side Model...")
        pickle.dump(side_classifier, open("./models/simple_side_classifier.pickle", "wb"))

    else:
        print("Reading inn side model")
        side_classifier = pickle.load(open("./models/simple_side_classifier.pickle", "rb"))


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
            num_processes=6, 
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
        num_processes=6, 
        molecules_per_process=1, 
        ptSl=[1, -0.5],
        min_ret=None
    )


    # Set up training of second model


    # How to test second model? On train_set? I don't have correct labels to compare to...
    # Correct label for test set can be derived from primary_label_tbm and side prediction on test set.
    # I just need to test the two models sequentially as I did during training...

    certainty_train_x = train_set_with_meta_labels[features] 
    certainty_train_y = train_set_with_meta_labels["m_primary_label_tbm"]


    certainty_classifier = RandomForestClassifier(
        n_estimators = 200, # Nr of trees
        criterion = "entropy", # function to measure quality of split (impurity measure)
        max_depth = 10, # depth of each tree
        min_samples_split = int(0.05*len(train_x)), # minimum number of samples required to split an internal node
        min_samples_leaf = int(0.05*len(train_x)), # The minimum number of samples required to be at a leaf node, may cause smoothing in regression models
        # NOTE: Set to a lower value to force discrepancy between trees
        max_features = "auto", # the number of features to consider when looking for the best split (auto = sqrt(n_features))
        # NOTE: Need to read up on
        # class_weight = "balanced_subsample", # use this attribute to set weight of different feature columns
        bootstrap = True, # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
        n_jobs = 6, # Number of jobs to run in parallell during fit and predict (-1 means to use all available)
        # NOTE: Set to a sufficiently large value (e.g. 5%) such that out-of bag accuracy converges to out of sample (k-fold) accuracy.
        # min_weight_fraction_leaf = 0, # Not relevant unless samples are weighted unequally 
    )

    print("Training Certainty Classifier...")
    certainty_classifier.fit(certainty_train_x, certainty_train_y)

    # Save
    print("Saving Certainty Model...")
    pickle.dump(certainty_classifier, open("./models/simple_certainty_classifier.pickle", "wb"))


    # Testing Side Classifier
    side_score = side_classifier.score(test_x, test_y)
    print("Side Classifier Accuracy: ", side_score)

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
        num_processes=6, 
        molecules_per_process=1, 
        ptSl=[1, -0.8], # NOTE: less tolerant for movement downwards... 
        min_ret=None
    )

    # Score the certainty model 


    # Derive correct labels for test_set
    certainty_test_x = test_set_meta_labeled[features] 
    certainty_test_y = test_set_meta_labeled["m_primary_label_tbm"]

    certainty_score = certainty_classifier.score(certainty_test_x, certainty_test_y)
    print("Certainty Classifier Accuracy: ", certainty_score)




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