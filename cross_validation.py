import pandas as pd
import numpy as np
from sklearn.model_selection._split import _BaseKFold
from dataset_columns import features, labels, base_cols
"""
How to address cross validation in finance:
1. Avoid overlapping labels in the test and training sets
2. Employ purging and embargoing to elimitate label overlap
3. The test set contains known information and you can re-arrange observations (disregard chronological order)
    in the pursuit of higher out of sample performance.
"""


class PurgedKFold(_BaseKFold):
    """
    Code from: "Advances in Financial Machine Learning" by López de Prado (2018)   
    Class implementing purged k-fold cross-validation
    """
    def __init__(self, n_splits=3, t1=None, pct_embargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label through dates must be a pd.Series")
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    # X is the dataset, y is the desired labels 
    def split(self, X, y=None, groups=None):
        """
        Splits observations into train and test sets and applies purging and embargoing.
        Assumes observations sorted by start date
        Assumes continouos test set
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and thru_date_values must have the same index")

        indices = np.arange(X.shape[0]) # or integer arguments the function is equivalent to the Python built-in range function, but returns an ndarray rather than a list.
        
        embargo = int(X.shape[0]*self.pct_embargo) # embargo is an integer number of bars NOTE: I think 0 is fine for me
        test_starts = [(i[0], i[-1]+1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)] # Devides the index into sub sets that are as equal in size as possible
        # For each iteration, a different couple of indexes delimits the test set (validation set...)
        for i, j in test_starts: # a list of tuples of [start_index, stop_index) # up to but not including the stop index is the indended use of these numbers
            t0 = self.t1.index[i] # start date/index of test set
            test_indices = indices[i:j]
            # searchsorted: Find indices where elements should be inserted to maintain order. I think this only returns numeric indices
            latest_end_date_of_observations_in_test_set = self.t1[test_indices].max() # observations starting after this date will not overlap with the test set
            # Find the first observation with a start date after the test sets last end date
            last_index_of_test_set_in_dataset = self.t1.index.searchsorted(latest_end_date_of_observations_in_test_set) # get the highest date from t1 values (observation end dates)
            print(latest_end_date_of_observations_in_test_set, last_index_of_test_set_in_dataset)
            # ensures the trining data does not overlap with the start of the test set
            # get the indexes/start dates of observations with end dates less than the start date of the train set
            train_indices = self.t1.index.searchsorted(self.t1[self.t1<=t0].index) 
            
            # right train (with embargo), this conditions captures the case when there are training batch(s) to the right of the test set
            if last_index_of_test_set_in_dataset < X.shape[0]: 
                train_indices = np.concatenate((train_indices, indices[last_index_of_test_set_in_dataset+embargo:])) # adding together the indexes before and after the test set
            yield train_indices, test_indices # here it returns the two list of numeric indexes that belong to the train and test sets
    


def cv_score(clf, X, y, sample_weight, scoring="neg_log_loss", t1=None, cv=None, cv_gen=None, pct_embargo=None):
    """
    Cross validation scoring function
    """
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise Exception("wrong scoring method.")

    from sklearn.metrics import log_loss, accuracy_score
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
    score = []

    # calculates the score for each train/test split
    for train, test in cv_gen.split(X=X):
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])
            # need to negate because scores are evaluated such that higher numbers are better
            score_ = -log_loss(y_true=y.iloc[test], y_pred=prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_) 
        else: # accuracy_score
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y_true=y.iloc[test], y_pred=pred, sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    
    return np.array(score_)


def get_train_times(t1, test_times):
    """
    Code from: "Advances in Financial Machine Learning" by López de Prado (2018)
    
    This function implements purging of observations from the training set. If the testing set is contigonous,
    in the sense that no training observations occure between the first and last testing observation, then purging can be accelerated: 
    The object test_times can ve a pandas serries with a single item, spanning the entire testing set.
    Given test_times, find the time of the training observations.
    - t1.index: Time when the observation started.
    - t1.value: Time when the observation ended.
    - testTimes: Times of testing observations
    """
    train_set = t1.copy(deep=True)
    for test_start, test_end in test_times.iterrows(): 
        drop_df0 = train_set[(test_start<=train_set.index) & (train_set.index <= test_end)].index # Training observations starts within test
        drop_df1 = train_set[(test_start<=train_set) & (test_end<=train_set)].index # Training observation ends within test_test
        drop_df2 = train_set[(train_set.index<=test_start) & (test_end<=train_set)].index # training observations envelopes test
        train_set = train_set.drop(drop_df0.union(drop_df1).union(drop_df2))
    return train_set


def get_embargo_times(times, pct_embargo):
    # Code from: "Advances in Financial Machine Learning" by López de Prado (2018)
    step = int(times.shape[0]*pct_embargo)
    if step == 0:
        embargo = pd.Series(times, index=times)
    else:
        embargo = pd.Series(times[step:], index=times[:-step])
        embargo = embargo.append(pd.Series(times[-1], index=times[-step:]))
    return embargo

"""
# Exmaple useage:

test_times = pd.Series(embargo[dt1], index=dt0) # Include embargo before purge
train_times = get_train_times(t1, test_times)
test_times = t1.loc[dt0:dt1].index

"""


if __name__ == "__main__":
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

    # train_start = dataset.index.min()
    train_start = pd.to_datetime("2010-01-01")
    train_end = pd.to_datetime("2012-01-01")

    test_start = pd.to_datetime("2012-03-01")
    # test_end = dataset.index.max()
    test_end = pd.to_datetime("2013-01-01")

    train_set = dataset.loc[(dataset.index >= train_start) & (dataset.index < train_end)] # NOTE: Use for labeling and constructing new train/test sets
    test_set = dataset.loc[(dataset.index >= test_start) & (dataset.index <= test_end)] # NOTE: Use for labeling and constructing new train/test sets

    print("Shapes: ", train_set.shape, test_set.shape)

    train_x = train_set[features]
    train_y = train_set["primary_label_tbm"]

    test_x = test_set[features]
    test_y = test_set["primary_label_tbm"]

    t1 = pd.Series(index=train_set.index, data=train_set["timeout"])

    kfold = PurgedKFold(n_splits=3, t1=t1)

    iterator = kfold.split(X=train_x)
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
