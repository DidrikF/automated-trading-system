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
            "n_estimators": [200], # [50, 100, 200, 500, 1000] 
            "min_weight_fraction_leaf": [0.20], # [0.05, 0.10, 0.20] early stopping
            "max_features": [5], # [3,5,10]
            "class_weight": ["balanced_subsample"],
            "bootstrap": [True], # , False
            "criterion": ["entropy"] # , "gini"
        }

        
        t1 = pd.Series(index=train_set.index, data=train_set["timeout"])

        grid_search = GridSearchCV(
            estimator=rf_classifier,
            param_grid=parameter_space,
            cv=PurgedKFold(n_splits=3, t1=t1),
            scoring=["accuracy", "balanced_accuracy", "precision", "recall", "f1"],
            refit="f1",
            n_jobs=n_jobs,
            verbose=2,
            error_score=np.nan
        )

        print("Training Side Classifier...")
        grid_search.fit(train_x, train_y) 
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


    side_predictions = side_classifier.predict(train_x)
    train_set["side_prediction"] = pd.Series(side_predictions)
    train_set_with_predictions = train_set


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
        min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
        max_features=best_params["max_features"], # the number of features to consider when looking for the best split (auto = sqrt(n_features))
        criterion=best_params["criterion"], # function to measure quality of split (impurity measure)
        class_weight=best_params["class_weight"], # use this attribute to set weight of different feature columns
        bootstrap=best_params["bootstrap"], # Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
        n_jobs=n_jobs, # Number of jobs to run in parallell during fit and predict (-1 means to use all available)
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
