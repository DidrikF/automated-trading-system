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

features = [
    "age",
    "agr",
    "beta",
    "betasq",
    "bm",
    "bm_ia",
    "cash",
    "cashdebt",
    "cashpr",
    "cfp_ia",
    "chatoia",
    "chcsho",
    "chdebtc_sale",
    "chdebtnc_lagat",
    "chdebtnc_ppne",
    "chint_lagat",
    "chint_laginvcap",
    "chint_laglt",
    "chint_lagsgna",
    "chinv",
    "chinvt_lagact",
    "chinvt_lagcor",
    "chinvt_lagsale",
    "chlct_lagat",
    "chlt_laginvcap",
    "chltc_laginvcap",
    "chmom",
    "chpay_lagact",
    "chpmia",
    "chppne_laglt",
    "chtl_lagat",
    "chtx",
    "cinvest",
    "currat",
    "debtc_sale",
    "dep_ppne",
    "depr",
    "dolvol",
    "dy",
    "egr",
    "ep",
    "eqt_marketcap",
    "gma",
    "grcapx",
    "idiovol",
    "ill",
    "indmom",
    "invest",
    "ipo",
    "lev",
    "lgr",
    "maxret",
    "mom12m",
    "mom1m",
    "mom24m",
    "mom6m",
    "ms",
    "mve",
    "mve_ia",
    "nincr",
    "operprof",
    "pchcapex_ia",
    "pchcurrat",
    "pchdepr",
    "pchgm_pchsale",
    "pchint",
    "pchlt",
    "pchppne",
    "pchquick",
    "pchsale_pchrect",
    "pchsale_pchxsga",
    "ps",
    "quick",
    "rd_sale",
    "retvol",
    "roaq",
    "roavol",
    "roeq",
    "roic",
    "rsup",
    "salecash",
    "salerec",
    "sgr",
    "sin",
    "sp",
    "std_dolvol",
    "std_turn",
    "sue",
    "tang",
    "tangibles_marketcap",
    "tb",
    "turn",
    "zerotrade",
    # "industry", # Why not just use whatever industry classification I want even after generating all other features?
    # "calendardate",
    # "date",
    # "datekey",
    # "ewmstd_2y_monthly",
    # "primary_label_tbm",
    # "return_1m",
    # "return_2m",
    # "return_3m",
    # "return_tbm",
    # "size",
    # "stop_loss_barrier",
    # "take_profit_barrier",
    # "ticker",
    # "timeout",
]


labels = [
    "return_1m",
    "return_2m",
    "return_3m",
    "timeout",
    "ewmstd_2y_monthly",
    "return_tbm",
    "primary_label_tbm",
    "take_profit_barrier",
    "stop_loss_barrier",
]

base_cols = ["ticker", "date", "calendardate", "datekey"]

def plot_feature_importances(estimator, cols):
    import matplotlib.pyplot as plt
    y = estimator.feature_importances_
    #plot
    fig, ax = plt.subplots() 
    width = 0.4 # the width of the bars 
    ind = np.arange(len(y)) # the x locations for the groups
    ax.barh(ind, y, width, color="green")
    ax.set_yticks(ind+width/10)
    ax.set_yticklabels(cols, minor=False)
    plt.title("Feature importance in RandomForest Classifier")
    plt.xlabel("Relative importance")
    plt.ylabel("feature") 
    plt.figure(figsize=(5,5))
    fig.set_size_inches(6.5, 4.5, forward=True)

    plt.show()


def plot_trees(estimator, features):
    from sklearn.tree import export_graphviz
    export_graphviz(
        estimator, 
        out_file="tree.dot", 
        feature_names = features,
        class_names = ["Long", "Short"],
        rounded = True, 
        proportion = False, 
        precision = 2, 
        filled = True
    )

# DATASET PREPARATION
print("Reading inn Dataset")
dataset = pd.read_csv("./dataset_development/datasets/completed/ml_dataset.csv", parse_dates=["date"])

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
train_start = dataset["date"].min()
train_end = pd.to_datetime("2006-01-01")
validation_start = pd.to_datetime("2006-01-01")
validation_end = pd.to_datetime("2010-01-01")

test_start = pd.to_datetime("2010-01-01")
test_end = dataset["date"].max()

train_set = dataset.loc[(dataset.date >= train_start) & (dataset.date < train_end)]
validation_set = dataset.loc[(dataset.date >= validation_start) & (dataset.date < validation_end)]
test_set = dataset.loc[(dataset.date >= test_start) & (dataset.date <= test_end)]


train_x = train_set[features]
train_y = train_set["primary_label_tbm"]

test_x = test_set[features]
test_y = test_set["primary_label_tbm"]


# You can generate a plot for precition and recall, see chapter 3 in hands-on machine learning
training_model = False
if training_model:
    

    rf_classifier = RandomForestClassifier(
        n_estimators = 1000, # Nr of trees
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
    print("Training Random Forest Classifier...")
    rf_classifier.fit(train_x, train_y)

    # Save
    print("Saving Model...")
    pickle.dump(rf_classifier, open("./models/simple_rf_classifier.pickle", "wb"))

else:
    print("Reading inn model")
    rf_classifier = pickle.load(open("./models/simple_rf_classifier.pickle", "rb"))


score = rf_classifier.score(test_x, test_y)

print("Score: ", score)
print("Feature importance:")
plot_feature_importances(rf_classifier, train_x.columns)



def purge_dataset():
    # Drop from the training set any observation i where Yi is a 
    # function of information used to determine Yj, and j belongs to the testing set.

    # For example, Yi and Yj should not span overlapping periods (see Chapter 4 for a 
    # discussion of sample uniqueness).

    # Avoid overfitting the classifier. In this way, even if some leakage occurs, 
    # the classifier will not be able to profit from it. Use:

    # Bagging of classifiers, while controlling for oversampling on redundant examples, 
    # so that the individual classifiers are as diverse as possible.

    # Set max_samples to the average uniqueness.

    # Apply sequential bootstrap

    pass

def getTrainTimes():
    """
    See SNIPPET 7.1 Purging Observations in the training set
    This removes from the training set any observations with overlapping labels in the test set.
    NOTE: Don't seem to difficult
    """
    pass

def getEmbargoTimes():
    """
    See SNIPPET 7.2 Embargo on training observations
    Futher removes overlap in labels between testing fold and SUBSEQUENT testing fold.
    """
    pass

from sklearn.model_selection._split import _BaseKFold

class PurgedKFold(_BaseKFold):
    """
    Ectend KFold class to work with labels that span intervals. The train is purged of observations
    overlapping test-label intervals. Test set is assumed continous (shuffle=False), w/o training samples 
    in between.  
    """
    def __init__(self, n_splits: int=3):
        self.n_splits = n_splits
    
    def split(self, x, y, groups=None):

        yield
    
    