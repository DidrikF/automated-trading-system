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


    training_model = True
    if training_model:
        side_classifier = RandomForestClassifier(
            n_estimators=1000,
            min_weight_fraction_leaf=0.2, # 0,1
            max_features=5,
            class_weight="balanced_subsample",
            bootstrap=True,
            criterion="entropy",
            n_jobs=n_jobs,
        )
        
        print("Training Side Classifier...")
        side_classifier.fit(train_x, train_y)
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
        class_weight="balanced_subsample",
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
