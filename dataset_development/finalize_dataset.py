import pandas as pd
import sys
from dateutil.relativedelta import *
import numpy as np
import math

from processing.engine import pandas_mp_engine

# The order sets the order of columns in the final dataset
selected_sf1_features = [
    "roaq", 
    "chtx",
    "rsup",
    "sue",
    "cinvest",
    "nincr",
    "roavol",
    "cashpr", 
    "cash", 
    "bm",
    "currat",
    "depr",
    "ep",
    "lev",
    "quick",
    "rd_sale", 
    "roic",
    "salecash",
    "saleinv",
    "salerec",
    "sp",
    "tb",
    "sin",
    "tang",
    "debtc_sale",
    "eqt_marketcap",
    "dep_ppne",
    "tangibles_marketcap",
    "agr",
    "cashdebt",
    "chcsho",
    "chinv",
    "egr",
    "gma",
    "invest",
    "lgr",
    "operprof",
    "pchcurrat",
    "pchdepr",
    "pchgm_pchsale",
    "pchquick",
    "pchsale_pchinvt",
    "pchsale_pchrect",
    "pchsale_pchxsga",
    "pchsaleinv",
    "rd",
    "roeq",
    "sgr",
    "grcapx",
    "chtl_lagat",
    "chlt_laginvcap",
    "chlct_lagat",
    "chint_lagat",
    "chinvt_lagsale",
    "chint_lagsgna",
    "chltc_laginvcap",
    "chint_laglt",
    "chdebtnc_lagat",
    "chinvt_lagcor",
    "chppne_laglt", 
    "chpay_lagact",
    "chint_laginvcap",
    "chinvt_lagact",
    "pchppne", 
    "pchlt",
    "pchint",
    "chdebtnc_ppne",
    "chdebtc_sale",
    "age",
    "ipo",
    # "profitmargin",
    # "chprofitmargin",
    # "industry",
    # "change_sales",
    "ps"
]


selected_industry_sf1_features = [
    "bm_ia",
    "cfp_ia",
    "chatoia",
    "mve_ia",
    "pchcapex_ia",
    "chpmia",
    "herf",
    "ms"
]

selected_sep_features = [
    "industry",
    # "sector",
    # "siccode",
    # Need for industry calculation
    # "mom12m_actual",
    "indmom",
    # Needed for beta calculation
    # "mom1w",
    # "mom1w_ewa_market", # This is used for idiovol
    # Calculated using forward filling and matrix multiplication
    "mom1m",
    "mom6m",
    "mom12m",
    "mom24m",
    # "mom12m_to_7m",
    "chmom",
    # Calculated only for samples
    "mve",
    "beta",
    "betasq",
    "idiovol",
    "ill",
    "dy",
    "turn",
    "dolvol",
    "maxret",
    "retvol",
    "std_dolvol",
    "std_turn",
    "zerotrade",

    # Labels
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

sizes = ["nano", "micro", "small", "mid", "large", "mega"]

def merge_datasets(sep_featured, sf1_featured, selected_features) -> pd.DataFrame:

    sf1_featured = sf1_featured.drop_duplicates(subset=["ticker", "datekey"], keep="last")

    dataset = sep_featured.merge(sf1_featured, on=["datekey", "ticker"], suffixes=("", "_sf1")) # Only the rows with matching datekey and ticker will be kept
    
    dataset = dataset[selected_features]
    
    # Drop first two (one of calendardate) years
    dataset.sort_values(by=["ticker", "calendardate"])

    return dataset

"""
Notes:
1. You need to filter out samples that rely on very old financial statements (see the age column)
"""
def fix_nans_and_drop_rows(dataset: pd.DataFrame, metadata: pd.DataFrame, features: list, size_rvs: dict):
    """
    dataset and sep is given per industry. (might update to industry in the future. Depends on how nans should be amended)
    Note that dataset must have a size column with 
    """
    # 1. Drop all with less than two years of sep history (this will fix a lot, as a lot of missing values is due to too little history)
    # 2. Drop all without a label
    dataset = dataset.dropna(axis=0, subset=["mom24m", "primary_label_tbm", "return_1m"]) # NOTE: illiquidity does not seem to work... no time to look into it

    # 3. Drop rows with outdated labels
    dataset = dataset.loc[dataset.age <= 180]

    """
    drop_indexes = []
    for index, row in dataset.iterrows(): 
        if row["age"] > 180: # pd.Timedelta('180 days')
            drop_indexes.append(index)

    dataset = dataset.drop(drop_indexes, axis=0)
    """
    
    # Calculate random variables
    # Size classifications: Nano <$50m; 2 - Micro < $300m; 3 - Small < $2bn; 4 - Mid <$10bn; 5 - Large < $200bn; 6 - Mega >= $200b
    nano_dataset = dataset.loc[dataset["size"] == "nano"]
    micro_dataset = dataset.loc[dataset["size"] == "micro"]
    small_dataset = dataset.loc[dataset["size"] == "small"]
    mid_dataset = dataset.loc[dataset["size"] == "mid"]
    large_dataset = dataset.loc[dataset["size"] == "large"]
    mega_dataset = dataset.loc[dataset["size"] == "mega"]

    size_ind_rvs = {}

    for feature in features:
        size_ind_rvs[feature] = {
        "nano": (nano_dataset[feature].mean(), nano_dataset[feature].std()),
        "micro": (micro_dataset[feature].mean(), micro_dataset[feature].std()),
        "small": (small_dataset[feature].mean(), small_dataset[feature].std()),
        "mid": (mid_dataset[feature].mean(), mid_dataset[feature].std()),
        "large": (large_dataset[feature].mean(), large_dataset[feature].std()),
        "mega": (mega_dataset[feature].mean(), mega_dataset[feature].std()),
    }


    # Fill nans from normal distributions
    for index, row in dataset.iterrows():
        size = row["size"]
        for feature in features:            
            if pd.isnull(row[feature]):
                mean = size_ind_rvs[feature][size][0]
                std = size_ind_rvs[feature][size][1]
                # market = False
                if pd.isnull(mean) or pd.isnull(std):
                    # market = True
                    mean = size_rvs[feature][size][0] # I guess this could also be null
                    std = size_rvs[feature][size][1]
                
                if pd.isnull(mean) or pd.isnull(std):
                    dataset.at[index, feature] = np.random.normal(0, 1) # Not optimal, but only a very small part of the data gets this treatment (ill)
                else:
                    dataset.at[index, feature] = np.random.normal(mean, std)
                
                """
                if (row["ticker"] == "AAPL") and (feature == "bm") and (row["datekey"] == pd.to_datetime("2003-12-19")):
                    print("AAPL mean and std: ", mean, std)
                    print
                    if market == True: 
                        print("Used market mean and std")
                    else:
                        print("Used industry mean and std")
                """


    # 4. Drop remaining rows with nans so no nans get through
    # This does not work well... Need to fix way more for this approach to be viable
    # NOTE: Check how much drop first...
    # dataset = dataset.dropna()

    return dataset


# Maybe this belongs in the training scripts?
def feature_scaling(dataset: pd.DataFrame()):
    """
    To make the dataset suitable for machine learning, all features must be expressed as numbers (floats, ints) and be scaled
    to the same size (Size scaling, is probably not needed for decision trees...)
    """    
    
    scaled_dataset = dataset

    return scaled_dataset




def finalize_dataset(sep, metadata, sep_featured=None, sf1_featured=None, num_processes=6):

    sf1_featured = sf1_featured.drop_duplicates(subset=["ticker", "datekey"], keep="last")

    # 2. Select features from SEP, SF1 etc.
    selected_features = base_cols + selected_sf1_features + selected_industry_sf1_features + selected_sep_features

    dataset = merge_datasets(sep_featured, sf1_featured, selected_features)
    
    # 3. Make all values numeric:
    dataset["age"] = pd.to_timedelta(dataset["age"])
    dataset["age"] = dataset["age"].dt.days # pd.to_numeric(dataset["age"].apply())

    # dataset.to_csv("./datasets/ml_ready_live/dataset_with_nans.csv", index=False)

    """
    merged_length = len(dataset)
    merged_cols = set(dataset.columns)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Nan status after merge")
        print("Dataset length: ", merged_length)
        print(dataset.isnull().sum())
    """ 

    # 2. Drop columns with too many missing values
    columns_to_drop = ["saleinv", "pchsale_pchinvt", "pchsaleinv", "rd", "herf"]
    dataset = dataset.drop(columns_to_drop, axis=1)

    
    features = list(set(dataset.columns) - set(labels) - set(base_cols) - set(["industry"]))

    # 4. Calculate mean and var for each feature for each size category for the whole market
    # Size classifications: Nano <$50m; 2 - Micro < $300m; 3 - Small < $2bn; 4 - Mid <$10bn; 5 - Large < $200bn; 6 - Mega >= $200bn
    
    dataset = dataset.dropna(axis=0, subset=["mve"])

    dataset["size"] = pd.NaT
    dataset["size"].loc[dataset.mve < math.log(50e6)] = "nano"
    dataset["size"].loc[(dataset.mve >= math.log(50e6)) & (dataset.mve < math.log(300e6))] = "micro"
    dataset["size"].loc[(dataset.mve >= math.log(300e6)) & (dataset.mve < math.log(2e9))] = "small"
    dataset["size"].loc[(dataset.mve >= math.log(2e9)) & (dataset.mve < math.log(10e9))] = "mid"
    dataset["size"].loc[(dataset.mve >= math.log(10e9)) & (dataset.mve < math.log(200e9))] = "large"
    dataset["size"].loc[dataset.mve >= math.log(200e9)] = "mega"

    nano_dataset = dataset.loc[dataset["size"] == "nano"]
    micro_dataset = dataset.loc[dataset["size"] == "micro"]
    small_dataset = dataset.loc[dataset["size"] == "small"]
    mid_dataset = dataset.loc[dataset["size"] == "mid"]
    large_dataset = dataset.loc[dataset["size"] == "large"]
    mega_dataset = dataset.loc[dataset["size"] == "mega"]

    print(features)
    size_rvs = {}
    for feature in features:
        size_rvs[feature] = {
            "nano": (nano_dataset[feature].mean(), nano_dataset[feature].std()),
            "micro": (micro_dataset[feature].mean(), micro_dataset[feature].std()),
            "small": (small_dataset[feature].mean(), small_dataset[feature].std()),
            "mid": (mid_dataset[feature].mean(), mid_dataset[feature].std()),
            "large": (large_dataset[feature].mean(), large_dataset[feature].std()),
            "mega": (mega_dataset[feature].mean(), mega_dataset[feature].std()),
        }


    # 5. Fix Nans and drop rows    
    dataset = pandas_mp_engine(callback=fix_nans_and_drop_rows, atoms=dataset, data={"metadata": metadata}, molecule_key="dataset", \
        split_strategy="industry_new", num_processes=num_processes, molecules_per_process=1, features=features, size_rvs=size_rvs)
    
    # dataset.to_csv("./datasets/ml_ready_live/dataset_without_nans.csv", index=False)

    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("\n\nNan Status After fixing Nans:")
        print("New dataset length: ", len(dataset))
        print("Percentage dropped: ", ((merged_length - len(dataset))/merged_length) * 100)
        print("Dropped columns: ", merged_cols.difference(set(dataset.columns)))
        print(dataset.isnull().sum())
        print(dataset.describe())
    """

    return dataset




if __name__ == "__main__":

    sep = pd.read_csv("./datasets/sharadar/SEP_PURGED.csv", parse_dates=["date"], index_col="date", low_memory=False)

    metadata = pd.read_csv("./datasets/sharadar/METADATA_PURGED.csv")
    
    sep_featured = pd.read_csv("./datasets/ml_ready_live/sep_featured_labeled.csv", parse_dates=["date", "datekey", "age"])
        
    sf1_featured = pd.read_csv("./datasets/ml_ready_live/sf1_featured.csv", parse_dates=["calendardate", "datekey"])

    dataset = finalize_dataset(sep=sep, metadata=metadata, sep_featured=sep_featured, sf1_featured=sf1_featured)

    dataset.to_csv("./datasets/ml_ready_live/dataset_without_nans.csv", index=False)

    # Report on final dataset
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("\n\nFinal Dataset Report:")
        print("New dataset length: ", len(dataset))
        print("Columns lenght: ", len(dataset.columns))
        print("Columns: ", list(dataset.columns))
        print(dataset.isnull().sum())
        print(dataset.describe())
        



    # NEED TO TEST THE ABOVE CODE, NEED TO USE TESTSETS, BECAUSE IT TAKE TOO MUCH TIME!!! 
    # NEED TO STEP BACK...


    """
    # 4. Scale Features
    scaled_dataset = pandas_mp_engine(callback=feature_type_fixing_and_scaling, atoms=dataset, data=None, molecule_key="dataset", \
        split_strategy="ticker_new", num_processes=6, molecules_per_process=1)



    """
