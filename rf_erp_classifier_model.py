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
        "n_estimators": [200, 500], # [100, 200, 500] # looks like 1000 is too much, it overfits
        "min_weight_fraction_leaf": [0.05, 0.10, 0.20], # [0.05, 0.10, 0.20] early stopping
        "max_features": [3, 5, 7], # 1, 5, 10, 15, 20, 30 may even push it??
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



""" Grid Search Cross Validation Results - Seems to overfit to the same degree, but cv-test F1 is lower than I'd like
'accuracy': 0.5253845705628715,
  'best_params': { 'bootstrap': True,
                   'class_weight': 'balanced_subsample',
                   'criterion': 'entropy',
                   'max_features': 10,
                   'min_weight_fraction_leaf': 0.05,
                   'n_estimators': 1000},
  'cv_results': { 'mean_fit_time': array([ 128.36591752,  342.38639235,  650.72928333,  101.31353943,
        267.00083407,  496.76866341,   75.04061373,  182.66999857,
        379.25802135,  232.80405021,  542.38835645, 1080.88415718,
        170.13113006,  401.03491767,  795.61717804,  117.43159389,
        295.09865546,  445.58668152]),
                  'mean_score_time': array([15.38859312, 36.80461113, 58.51183208, 12.98027444, 28.99910347,
       55.01701514, 12.1797328 , 26.3114531 , 51.2691648 , 15.71830479,
       34.26610184, 63.58767446, 13.47443024, 31.20621991, 52.09860492,
       13.49953008, 27.69702832, 30.95979222]),
                  'mean_test_accuracy': array([0.50546774, 0.50522977, 0.50546047, 0.50430879, 0.50436329,
       0.50415257, 0.50508627, 0.50485012, 0.50469208, 0.50453768,
       0.50451769, 0.50471933, 0.50353496, 0.50338055, 0.50368209,
       0.50437782, 0.5040236 , 0.50441234]),
                  'mean_test_balanced_accuracy': array([0.51559685, 0.51514432, 0.51530101, 0.51389185, 0.5139101 ,
       0.51371126, 0.51338861, 0.51299711, 0.51313589, 0.51393266,
       0.51399284, 0.51422593, 0.51222775, 0.51220722, 0.51254951,
       0.51155842, 0.51191944, 0.51259351]),
                  'mean_test_f1': array([0.50607279, 0.50613302, 0.50742949, 0.50274174, 0.50191041,
       0.50047885, 0.49408189, 0.49251331, 0.49331304, 0.50722498,
       0.50832118, 0.50932892, 0.49768133, 0.49739563, 0.49958112,
       0.49298715, 0.49474963, 0.49576017]),
                  'mean_test_precision': array([0.49323426, 0.49285598, 0.49308948, 0.49205103, 0.49214224,
       0.4919802 , 0.49176471, 0.49157235, 0.49157726, 0.49181905,
       0.49174346, 0.49205123, 0.49043148, 0.4904362 , 0.49075768,
       0.49014043, 0.490334  , 0.49108148]),
                  'mean_test_recall': array([0.54838159, 0.54837345, 0.54991631, 0.53999457, 0.53876844,
       0.53674198, 0.52110914, 0.51863015, 0.52071179, 0.54861106,
       0.55117764, 0.55301146, 0.5308013 , 0.53086878, 0.53407488,
       0.51674857, 0.52235261, 0.52403829]),
                  'mean_train_accuracy': array([0.68305837, 0.67962806, 0.68121944, 0.6316646 , 0.63106534,
       0.63236653, 0.5934942 , 0.59779007, 0.60065997, 0.68508118,
       0.68475519, 0.68585771, 0.64168238, 0.639953  , 0.63913121,
       0.59769795, 0.60237538, 0.60165905]),
                  'mean_train_balanced_accuracy': array([0.68408426, 0.68084089, 0.682521  , 0.63274118, 0.632137  ,
       0.63341661, 0.59443428, 0.59895797, 0.60184687, 0.68604136,
       0.68585259, 0.68697096, 0.64329482, 0.64142141, 0.64061032,
       0.59931571, 0.60401547, 0.60317276]),
                  'mean_train_f1': array([0.68635559, 0.68250272, 0.68415627, 0.63603465, 0.63568858,
       0.63681475, 0.59501589, 0.60034605, 0.6031724 , 0.68445235,
       0.68455981, 0.68595237, 0.64613487, 0.64375259, 0.64298087,
       0.59789346, 0.60375615, 0.60317454]),
                  'mean_train_precision': array([0.66486771, 0.66151488, 0.66375483, 0.61342774, 0.61268873,
       0.61434234, 0.57727448, 0.58113536, 0.58369076, 0.67075719,
       0.66975994, 0.67054436, 0.62365989, 0.62265558, 0.62127652,
       0.58323492, 0.58750827, 0.58616922]),
                  'mean_train_recall': array([0.71019756, 0.70600553, 0.7071998 , 0.66126498, 0.6613515 ,
       0.66191277, 0.61487572, 0.62202639, 0.62513335, 0.69996025,
       0.70140094, 0.70343718, 0.67196936, 0.66788666, 0.66774618,
       0.61581959, 0.62331963, 0.62319873]),
                  'param_bootstrap': masked_array(data=[True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object),
                  'param_class_weight': masked_array(data=['balanced_subsample', 'balanced_subsample',
                   'balanced_subsample', 'balanced_subsample',
                   'balanced_subsample', 'balanced_subsample',
                   'balanced_subsample', 'balanced_subsample',
                   'balanced_subsample', 'balanced_subsample',
                   'balanced_subsample', 'balanced_subsample',
                   'balanced_subsample', 'balanced_subsample',
                   'balanced_subsample', 'balanced_subsample',
                   'balanced_subsample', 'balanced_subsample'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object),
                  'param_criterion': masked_array(data=['entropy', 'entropy', 'entropy', 'entropy', 'entropy',
                   'entropy', 'entropy', 'entropy', 'entropy', 'entropy',
                   'entropy', 'entropy', 'entropy', 'entropy', 'entropy',
                   'entropy', 'entropy', 'entropy'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object),
                  'param_max_features': masked_array(data=[5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10,
                   10, 10],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object),
                  'param_min_weight_fraction_leaf': masked_array(data=[0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.05,
                   0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object),
                  'param_n_estimators': masked_array(data=[200, 500, 1000, 200, 500, 1000, 200, 500, 1000, 200,
                   500, 1000, 200, 500, 1000, 200, 500, 1000],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object),
                  'params': [ { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 5,
                                'min_weight_fraction_leaf': 0.05,
                                'n_estimators': 200},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 5,
                                'min_weight_fraction_leaf': 0.05,
                                'n_estimators': 500},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 5,
                                'min_weight_fraction_leaf': 0.05,
                                'n_estimators': 1000},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 5,
                                'min_weight_fraction_leaf': 0.1,
                                'n_estimators': 200},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 5,
                                'min_weight_fraction_leaf': 0.1,
                                'n_estimators': 500},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 5,
                                'min_weight_fraction_leaf': 0.1,
                                'n_estimators': 1000},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 5,
                                'min_weight_fraction_leaf': 0.2,
                                'n_estimators': 200},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 5,
                                'min_weight_fraction_leaf': 0.2,
                                'n_estimators': 500},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 5,
                                'min_weight_fraction_leaf': 0.2,
                                'n_estimators': 1000},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 10,
                                'min_weight_fraction_leaf': 0.05,
                                'n_estimators': 200},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 10,
                                'min_weight_fraction_leaf': 0.05,
                                'n_estimators': 500},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 10,
                                'min_weight_fraction_leaf': 0.05,
                                'n_estimators': 1000},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 10,
                                'min_weight_fraction_leaf': 0.1,
                                'n_estimators': 200},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 10,
                                'min_weight_fraction_leaf': 0.1,
                                'n_estimators': 500},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 10,
                                'min_weight_fraction_leaf': 0.1,
                                'n_estimators': 1000},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 10,
                                'min_weight_fraction_leaf': 0.2,
                                'n_estimators': 200},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 10,
                                'min_weight_fraction_leaf': 0.2,
                                'n_estimators': 500},
                              { 'bootstrap': True,
                                'class_weight': 'balanced_subsample',
                                'criterion': 'entropy',
                                'max_features': 10,
                                'min_weight_fraction_leaf': 0.2,
                                'n_estimators': 1000}],
                  'rank_test_accuracy': array([ 1,  3,  2, 13, 12, 14,  4,  5,  7,  8,  9,  6, 17, 18, 16, 11, 15,
       10]),
                  'rank_test_balanced_accuracy': array([ 1,  3,  2,  8,  7,  9, 10, 12, 11,  6,  5,  4, 15, 16, 14, 18, 17,
       13]),
                  'rank_test_f1': array([ 6,  5,  3,  7,  8,  9, 15, 18, 16,  4,  2,  1, 11, 12, 10, 17, 14,
       13]),
                  'rank_test_precision': array([ 1,  3,  2,  6,  4,  7,  9, 12, 11,  8, 10,  5, 16, 15, 14, 18, 17,
       13]),
                  'rank_test_recall': array([ 5,  6,  3,  7,  8,  9, 15, 17, 16,  4,  2,  1, 12, 11, 10, 18, 14,
       13]),
                  'split0_test_accuracy': array([0.52110343, 0.52037864, 0.52074921, 0.51889635, 0.51884731,
       0.51879826, 0.51801353, 0.51865657, 0.51884731, 0.51974649,
       0.51996447, 0.52019335, 0.51814976, 0.51814432, 0.51848764,
       0.51813887, 0.51787184, 0.51831325]),
                  'split0_test_balanced_accuracy': array([0.5208962 , 0.52009014, 0.52046229, 0.51895373, 0.51886128,
       0.51880651, 0.51824133, 0.51882087, 0.51904943, 0.51945322,
       0.51959768, 0.51977767, 0.51803759, 0.51803255, 0.5183881 ,
       0.51826394, 0.5179265 , 0.51844961]),
                  'split0_test_f1': array([0.53121232, 0.532947  , 0.53326328, 0.5208547 , 0.52216738,
       0.52229706, 0.51453726, 0.51722536, 0.51620822, 0.53247036,
       0.53487869, 0.53654916, 0.52538916, 0.52537105, 0.52533468,
       0.51795235, 0.51991252, 0.51776927]),
                  'split0_test_precision': array([0.52790619, 0.5269211 , 0.52728303, 0.52666586, 0.52646398,
       0.52639499, 0.52637431, 0.5268051 , 0.52713803, 0.52629702,
       0.52626165, 0.52631796, 0.52534404, 0.52534003, 0.52571877,
       0.52613858, 0.52562569, 0.52635549]),
                  'split0_test_recall': array([0.53456013, 0.53911232, 0.53938073, 0.51517039, 0.51794035,
       0.51826244, 0.50322089, 0.5079878 , 0.50572245, 0.53879023,
       0.54378261, 0.54718602, 0.52543428, 0.52540207, 0.52495115,
       0.51001696, 0.51432222, 0.50945868]),
                  'split0_train_accuracy': array([0.54206128, 0.54231218, 0.54196754, 0.53686962, 0.53705159,
       0.53674831, 0.53421727, 0.53419797, 0.53469425, 0.5447219 ,
       0.5445868 , 0.5446061 , 0.53720047, 0.53730249, 0.53789803,
       0.53244444, 0.53233691, 0.53269809]),
                  'split0_train_balanced_accuracy': array([0.54500964, 0.5453727 , 0.54524617, 0.54001032, 0.54021355,
       0.53995918, 0.53692864, 0.53709019, 0.53742428, 0.54709302,
       0.54712244, 0.54718625, 0.54093616, 0.54088158, 0.54134847,
       0.53588579, 0.53601022, 0.53603108]),
                  'split0_train_f1': array([0.54418266, 0.54515265, 0.54623127, 0.54044146, 0.54075521,
       0.54078342, 0.53505436, 0.53623614, 0.53563911, 0.54298177,
       0.54393042, 0.54424044, 0.54461452, 0.54370352, 0.54344172,
       0.53816679, 0.53957655, 0.5376997 ]),
                  'split0_train_precision': array([0.50682179, 0.50703364, 0.50667619, 0.50199742, 0.50216397,
       0.50188208, 0.49954776, 0.49953204, 0.49999229, 0.50943631,
       0.50926773, 0.50927475, 0.50226681, 0.50236907, 0.50292177,
       0.49793629, 0.49785111, 0.49816283]),
                  'split0_train_recall': array([0.58749007, 0.58946919, 0.59248528, 0.58526209, 0.58577168,
       0.58622202, 0.5759946 , 0.57876181, 0.57675899, 0.58125644,
       0.58365627, 0.58436141, 0.59476067, 0.59244972, 0.59106315,
       0.58546948, 0.5889359 , 0.58405328]),
                  'split1_test_accuracy': array([0.48159411, 0.48132163, 0.48077667, 0.47990474, 0.47998649,
       0.47993199, 0.4848148 , 0.48360499, 0.48339246, 0.47954507,
       0.47957232, 0.47900556, 0.48080937, 0.4803843 , 0.48018812,
       0.48320173, 0.48246604, 0.48221536]),
                  'split1_test_balanced_accuracy': array([0.512359  , 0.5115244 , 0.5107478 , 0.50875255, 0.50877756,
       0.50877299, 0.50967352, 0.50807198, 0.5087085 , 0.50817364,
       0.50851415, 0.50808978, 0.50717953, 0.50715906, 0.50706077,
       0.50478723, 0.50627312, 0.50679226]),
                  'split1_test_f1': array([0.54201531, 0.54028285, 0.53920782, 0.53542778, 0.53534021,
       0.53543177, 0.52840674, 0.52625474, 0.52849952, 0.53452646,
       0.53540901, 0.53532805, 0.52929784, 0.53007797, 0.53018766,
       0.51748508, 0.52337743, 0.52538563]),
                  'split1_test_precision': array([0.4345319 , 0.4339955 , 0.43347019, 0.43214213, 0.43216321,
       0.43215697, 0.43306868, 0.43191498, 0.43232973, 0.43174553,
       0.43196822, 0.43166088, 0.43115633, 0.43112073, 0.43104498,
       0.42959966, 0.43061956, 0.43096553]),
                  'split1_test_recall': array([0.72014687, 0.7155157 , 0.71317453, 0.70359235, 0.70323414,
       0.70356677, 0.67757081, 0.67332344, 0.6796945 , 0.70153264,
       0.70398895, 0.70452626, 0.68528516, 0.68799734, 0.68856024,
       0.65057698, 0.66706752, 0.67278612]),
                  'split1_train_accuracy': array([0.68389382, 0.68613149, 0.68724893, 0.63623234, 0.63477967,
       0.63601723, 0.60631855, 0.60454462, 0.6041172 , 0.68840268,
       0.69011515, 0.69070181, 0.64319676, 0.63526296, 0.63702851,
       0.60098558, 0.60550282, 0.60309474]),
                  'split1_train_balanced_accuracy': array([0.68332791, 0.68580729, 0.68702115, 0.63557526, 0.63407911,
       0.63532371, 0.60597782, 0.60429887, 0.60390106, 0.68846052,
       0.69022099, 0.69077262, 0.64305184, 0.6350913 , 0.63684997,
       0.6015072 , 0.60587708, 0.60333375]),
                  'split1_train_f1': array([0.69587654, 0.69543531, 0.69547785, 0.6507174 , 0.64984854,
       0.65095932, 0.61741158, 0.61435693, 0.61352424, 0.69346701,
       0.69463386, 0.69560412, 0.65114145, 0.64365294, 0.64548238,
       0.59980499, 0.60652837, 0.60606061]),
                  'split1_train_precision': array([0.68773607, 0.69311489, 0.69552893, 0.64185928, 0.64006838,
       0.64129741, 0.61627554, 0.61534935, 0.61518015, 0.70075814,
       0.7031567 , 0.70321186, 0.65390629, 0.6458953 , 0.64753896,
       0.61844287, 0.62171549, 0.61806522]),
                  'split1_train_recall': array([0.70421202, 0.69777132, 0.69542678, 0.65982342, 0.65993222,
       0.66091682, 0.61855182, 0.61336771, 0.61187721, 0.68632603,
       0.68631515, 0.68815923, 0.64839989, 0.64142609, 0.64343881,
       0.58225762, 0.59206554, 0.59451344]),
                  'split2_test_accuracy': array([0.51370572, 0.5139891 , 0.51485559, 0.51412534, 0.51425613,
       0.51372752, 0.51243052, 0.51228883, 0.51183651, 0.51432153,
       0.51401635, 0.51495913, 0.51164578, 0.51161308, 0.51237057,
       0.51179292, 0.51173297, 0.51270845]),
                  'split2_test_balanced_accuracy': array([0.51353533, 0.51381839, 0.51469294, 0.51396927, 0.51409146,
       0.51355428, 0.51225097, 0.51209847, 0.51164974, 0.51417112,
       0.51386671, 0.51481034, 0.51146613, 0.51143006, 0.51219964,
       0.51162408, 0.51155871, 0.51253865]),
                  'split2_test_f1': array([0.44499039, 0.44516888, 0.44981707, 0.45194246, 0.44822335,
       0.44370741, 0.43930136, 0.43405951, 0.43523107, 0.45467784,
       0.45467554, 0.45610926, 0.43835668, 0.43673754, 0.44322071,
       0.44352374, 0.44095864, 0.44412533]),
                  'split2_test_precision': array([0.51726481, 0.51765148, 0.51851535, 0.51734523, 0.51779968,
       0.51738878, 0.51585129, 0.51599711, 0.51526416, 0.51741474,
       0.51700065, 0.518175  , 0.51479421, 0.51484796, 0.51550943,
       0.5146832 , 0.51475687, 0.51592357]),
                  'split2_test_recall': array([0.3904369 , 0.39049147, 0.39719284, 0.40122022, 0.39513004,
       0.38839593, 0.38253495, 0.37457844, 0.37671764, 0.40550953,
       0.40576056, 0.40732131, 0.38168364, 0.37920609, 0.38871244,
       0.38965107, 0.38566735, 0.38986936]),
                  'split2_train_accuracy': array([0.82322002, 0.81044051, 0.81444187, 0.72189184, 0.72136477,
       0.72433405, 0.6399468 , 0.65462764, 0.66316847, 0.82211895,
       0.8195636 , 0.82226521, 0.74464991, 0.74729356, 0.7424671 ,
       0.65966383, 0.66928641, 0.6691843 ]),
                  'split2_train_balanced_accuracy': array([0.82391523, 0.81134268, 0.81529568, 0.72263795, 0.72211833,
       0.72496693, 0.64039638, 0.65548485, 0.66421526, 0.82257055,
       0.82021433, 0.822954  , 0.74589644, 0.74829135, 0.74363253,
       0.66055413, 0.6701591 , 0.67015345]),
                  'split2_train_f1': array([0.81900758, 0.8069202 , 0.8107597 , 0.7169451 , 0.716462  ,
       0.71870151, 0.63258174, 0.65044506, 0.66035384, 0.81690829,
       0.81511517, 0.81801254, 0.74264863, 0.74390131, 0.7400185 ,
       0.6557086 , 0.66516352, 0.66576332]),
                  'split2_train_precision': array([0.80004526, 0.78439612, 0.78905938, 0.69642653, 0.69583383,
       0.69984754, 0.61600013, 0.6285247 , 0.63589983, 0.80207712,
       0.79685539, 0.79914648, 0.71480657, 0.71970238, 0.71336882,
       0.6333256 , 0.64295823, 0.64227962]),
                  'split2_train_recall': array([0.83889059, 0.83077609, 0.83368735, 0.73870943, 0.73835059,
       0.73859947, 0.65008074, 0.67394966, 0.68676386, 0.83229828,
       0.83423141, 0.83779091, 0.77274753, 0.76978417, 0.76873658,
       0.67973168, 0.68895744, 0.69102948]),
                  'std_fit_time': array([ 37.54574212, 126.68627951, 254.29181948,  41.34428853,
        92.363999  , 182.92850744,  21.97044948,  48.74439506,
       119.21987115,  91.51428083, 250.52620823, 454.26080312,
        82.32122342, 176.10962616, 361.97004055,  47.88773624,
       120.10230432, 119.33090309]),
                  'std_score_time': array([0.81491357, 0.63435126, 1.61175819, 0.25030665, 0.38164354,
       2.09752545, 1.45551373, 0.43990837, 1.30667488, 1.10948708,
       1.86392663, 1.12210949, 0.89290386, 0.60019571, 4.64376523,
       0.44707818, 0.54037198, 5.38613382]),
                  'std_test_accuracy': array([0.01714925, 0.0171057 , 0.01761916, 0.01736587, 0.01733864,
       0.01725122, 0.01451419, 0.01524586, 0.01533067, 0.0178107 ,
       0.01780543, 0.01830754, 0.01628733, 0.01647798, 0.01679942,
       0.01519625, 0.01544817, 0.01586157]),
                  'std_test_balanced_accuracy': array([0.00377786, 0.00362046, 0.00398917, 0.00416498, 0.00411866,
       0.00409768, 0.00358911, 0.00443399, 0.0043505 , 0.00460796,
       0.00452571, 0.00478943, 0.00446538, 0.00447297, 0.00463098,
       0.00550204, 0.00476431, 0.00475925]),
                  'std_test_f1': array([0.04341624, 0.04321195, 0.04081025, 0.03640978, 0.0383414 ,
       0.04049991, 0.03914723, 0.04149702, 0.04137545, 0.03716582,
       0.03793371, 0.03763518, 0.04197909, 0.04293465, 0.03990194,
       0.03497634, 0.03806217, 0.03664341]),
                  'std_test_precision': array([0.04173561, 0.04179239, 0.04230894, 0.0425326 , 0.04255888,
       0.04246096, 0.04172616, 0.04241432, 0.0421739 , 0.04263294,
       0.04243629, 0.04283169, 0.04213462, 0.04216058, 0.04242853,
       0.04306354, 0.04245704, 0.04272126]),
                  'std_test_recall': array([0.13495781, 0.13285202, 0.12921386, 0.12468464, 0.1266422 ,
       0.12932972, 0.12111018, 0.12219402, 0.12414302, 0.12105023,
       0.12186341, 0.12140326, 0.12400283, 0.12612269, 0.12258218,
       0.1066288 , 0.11502134, 0.11595942]),
                  'std_train_accuracy': array([0.11478409, 0.10955949, 0.11131885, 0.07560403, 0.07529136,
       0.07662506, 0.04410617, 0.04939666, 0.05250632, 0.11327122,
       0.11232277, 0.1134056 , 0.08469765, 0.08579262, 0.08352821,
       0.0519891 , 0.05595312, 0.05572951]),
                  'std_train_balanced_accuracy': array([0.11386399, 0.10863857, 0.11029316, 0.07458434, 0.07427501,
       0.07554113, 0.04302196, 0.04848174, 0.05178258, 0.11247624,
       0.11153208, 0.1126138 , 0.08367486, 0.08479291, 0.08262492,
       0.05091922, 0.05478187, 0.05475535]),
                  'std_train_f1': array([0.11239861, 0.10725671, 0.1082896 , 0.07280142, 0.07242742,
       0.07332014, 0.04284914, 0.04766654, 0.05143806, 0.11201156,
       0.11093964, 0.11197517, 0.08092456, 0.08173044, 0.08027162,
       0.04800528, 0.05130813, 0.05232157]),
                  'std_train_precision': array([0.1207952 , 0.11541637, 0.11745144, 0.08188176, 0.08140122,
       0.08303619, 0.0549612 , 0.05795242, 0.05978518, 0.12133893,
       0.11975857, 0.12057302, 0.08936599, 0.0902349 , 0.08789872,
       0.06062049, 0.0639876 , 0.0630102 ]),
                  'std_train_recall': array([0.10272106, 0.09868504, 0.0988216 , 0.06265291, 0.06229816,
       0.06221182, 0.03035704, 0.03933964, 0.0458771 , 0.10293985,
       0.10285154, 0.10402464, 0.07454963, 0.07477521, 0.07454349,
       0.04521169, 0.04643053, 0.0481533 ])},
  'f1': 0.529696553277115,
  'precision': 0.529823507669562,
  'prediction_distribution':  1.0    173152
-1.0    170088
dtype: int64,
  'recall': 0.5295696597107975,
  'sample_mean': 0.5244033333333333,
  'sample_std': 0.008659137113798088,
  't_test_results': 'Reject H0 (mean of observations are greater) with '
                    't_statistic=39.75585278682773, p-value=0.0, '
                    'critical_value=1.652546746165939 and alpha=0.05'}

"""