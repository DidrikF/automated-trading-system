import pandas as pd
import pytest
import math
from dateutil.relativedelta import *
import numpy as np

import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from datetime import datetime

from ..finalize_dataset import finalize_dataset, fix_nans_and_drop_rows, merge_datasets, base_cols, selected_industry_sf1_features, \
    selected_sep_features, selected_sf1_features, labels



dataset = pd.DataFrame()
ce_dataset = pd.DataFrame()

@pytest.fixture(scope='module', autouse=True)
def setup():
    global metadata, ce_dataset
    metadata = pd.read_csv("../datasets/sharadar/METADATA_PURGED.csv")

    if False:
        dataset = pd.read_csv("../datasets/ml_ready_live/dataset_with_nans.csv", parse_dates=["date", "datekey", "calendardate"], low_memory=False)
        ce_dataset = dataset.loc[dataset.industry == "Consumer Electronics"]
        ce_dataset.to_csv("./testing_datasets/ce_dataset.csv", index=False)
    else:
        ce_dataset = pd.read_csv("./testing_datasets/ce_dataset.csv", parse_dates=["date", "datekey", "calendardate"], low_memory=False)

    yield
    

@pytest.mark.skip()
def test_merge_datasets():
    global dataset

    selected_features = base_cols + selected_sf1_features + selected_industry_sf1_features + selected_sep_features
    sep_featured = pd.read_csv("./testing_datasets/sep_featured.csv", parse_dates=["date", "datekey"])
    sf1_featured = pd.read_csv("./testing_datasets/sf1_featured.csv", parse_dates=["datekey", "calendardate"])

    dataset = merge_datasets(sep_featured, sf1_featured, selected_features)

    assert set(dataset.columns) == set(selected_features)
    assert len(sep_featured) == len(dataset)

    dataset_row = dataset.iloc[50]
    sf1_row = sf1_featured.loc[(sf1_featured.datekey == dataset_row["datekey"]) & (sf1_featured.ticker == dataset_row["ticker"])].iloc[-1]
    
    sep_row = sep_featured.loc[(sep_featured.date == dataset_row["date"]) & (sep_featured.ticker == dataset_row["ticker"])].iloc[-1]


    assert dataset_row["bm"] == sf1_row["bm"]
    assert dataset_row["return_1m"] == sep_row["return_1m"]





# @pytest.mark.skip()
def test_finalize_dataset():
    global metadata, dataset

    sep = pd.read_csv("../datasets/testing/sep.csv")
    sep_featured = pd.read_csv("./testing_datasets/sep_featured.csv", parse_dates=["date", "datekey"])
    sf1_featured = pd.read_csv("./testing_datasets/sf1_featured.csv", parse_dates=["datekey", "calendardate"])

    cols = ["bm", "chinv"]

    date0 = "2004-01-20"
    datekey0 = "2003-12-19"

    date1 = "2006-05-05"
    datekey1 = "2005-02-01"

    sf1_featured["bm"].loc[(sf1_featured.ticker == "AAPL") & (sf1_featured.datekey == datekey0)] = pd.NaT
    # sf1_featured["bm"].loc[(sf1_featured.ticker == "AAPL") & (sf1_featured.datekey == datekey1)] = pd.NaT

    
    # Get desired_value
    selected_features = base_cols + selected_sf1_features + selected_industry_sf1_features + selected_sep_features
    check_dataset = merge_datasets(sep_featured, sf1_featured, selected_features)
    check_dataset["age"] = pd.to_timedelta(check_dataset["age"])
    check_dataset["age"] = check_dataset["age"].dt.days
    check_dataset["bm"].loc[(check_dataset.ticker == "AAPL") & (check_dataset.datekey == datekey0)] = pd.NaT
    
    check_ind_dataset = check_dataset.loc[check_dataset.industry == "Consumer Electronics"]
    check_ind_dataset = check_ind_dataset.loc[(check_ind_dataset["mve"] >= math.log(2e9)) & (check_ind_dataset["mve"] < math.log(10e9))]
    check_ind_dataset = check_ind_dataset.dropna(axis=0, subset=["mom24m", "primary_label_tbm", "return_1m"])
    check_ind_dataset = check_ind_dataset.loc[check_ind_dataset.age <= 180]

    dataset = finalize_dataset(
        sep=sep,
        metadata=metadata,
        sep_featured=sep_featured,
        sf1_featured=sf1_featured,
    )
        
    ind_mean = check_ind_dataset["bm"].mean()
    ind_std = check_ind_dataset["bm"].std()
    ind_val = ind_mean + ind_std

    market_mean = check_dataset["bm"].mean() 
    market_std = check_dataset["bm"].std()
    market_val = market_mean + market_std

    print("desired: ", dataset.loc[(dataset.ticker == "AAPL") & (dataset.datekey == datekey0)].iloc[-1]["bm"])
    print("industry: ", ind_val)
    print("market: ", market_val)

    assert dataset.loc[(dataset.ticker == "AAPL") & (dataset.datekey == datekey0)].iloc[-1]["bm"] == ind_val
        

@pytest.mark.skip()
def test_feature_scaling():
    pass



"""
AAPL
date        datekey
2004-01-20  2003-12-19
2005-02-01  2005-02-01

"""





"""
# @pytest.mark.skip()
def test_fix_nans_and_drop_rows():
    # Produce lacking sep_featured and sf1_featured
    # Take note of what dates and values are missing and test against those
    global ce_dataset, metadata, dataset

    # ____ Setup from finalize_dataset function _____
    ce_dataset["age"] = pd.to_timedelta(ce_dataset["age"])

    columns_to_drop = ["saleinv", "pchsale_pchinvt", "pchsaleinv", "rd", "herf"]
    ce_dataset = ce_dataset.drop(columns_to_drop, axis=1)

    ce_dataset["age"] = ce_dataset["age"].dt.days
    
    exclude_cols = labels.copy()
    exclude_cols.extend(base_cols)
    features = list(set(ce_dataset.columns) - set(exclude_cols) - set(["industry"]))

    # 4. Calculate mean and var for each feature for each size category for the whole market
    # Size classifications: Nano <$50m; 2 - Micro < $300m; 3 - Small < $2bn; 4 - Mid <$10bn; 5 - Large < $200bn; 6 - Mega >= $200bn
    
    ce_dataset = ce_dataset.dropna(axis=0, subset=["mve"])

    ce_dataset["size"] = pd.NaT
    ce_dataset["size"].loc[ce_dataset.mve < math.log(50e6)] = "nano"
    ce_dataset["size"].loc[ce_dataset.mve < math.log(300e6)] = "micro"
    ce_dataset["size"].loc[ce_dataset.mve < math.log(2e9)] = "small"
    ce_dataset["size"].loc[ce_dataset.mve < math.log(10e9)] = "mid"
    ce_dataset["size"].loc[ce_dataset.mve < math.log(200e9)] = "large"
    ce_dataset["size"].loc[ce_dataset.mve >= math.log(200e9)] = "mega"

    nano_ce_dataset = ce_dataset.loc[ce_dataset["size"] == "nano"]
    micro_ce_dataset = ce_dataset.loc[ce_dataset["size"] == "micro"]
    small_ce_dataset = ce_dataset.loc[ce_dataset["size"] == "small"]
    mid_ce_dataset = ce_dataset.loc[ce_dataset["size"] == "mid"]
    large_ce_dataset = ce_dataset.loc[ce_dataset["size"] == "large"]
    mega_ce_dataset = ce_dataset.loc[ce_dataset["size"] == "mega"]

    size_rvs = {}
    for feature in features:
        size_rvs[feature] = size_rvs = {
        "nano": (nano_ce_dataset[feature].mean(), nano_ce_dataset[feature].std()),
        "micro": (micro_ce_dataset[feature].mean(), micro_ce_dataset[feature].std()),
        "small": (small_ce_dataset[feature].mean(), small_ce_dataset[feature].std()),
        "mid": (mid_ce_dataset[feature].mean(), mid_ce_dataset[feature].std()),
        "large": (large_ce_dataset[feature].mean(), large_ce_dataset[feature].std()),
        "mega": (mega_ce_dataset[feature].mean(), mega_ce_dataset[feature].std()),
    }

    print(size_rvs)

    # _____ Test Code _______
    # select one row with missing value
    check_row = ce_dataset.loc[(ce_dataset.ticker == "CLRC1") & (ce_dataset.date == "2003-02-14")].iloc[-1]

    # Testing CLRC1, zerotrade at 2003-02-14, 2003-03-14
    size_CLRC1 = check_row["size"]
    ce_size_dataset = ce_dataset.loc[ce_dataset["size"] == size_CLRC1]
    

    # calculate the wanted fabricated value 
    mean = ce_size_dataset["zerotrade"].mean()
    std = ce_size_dataset["zerotrade"].std()
    desired_value = mean + std # Cannot check against randomly generated value


    exclude_cols = labels.copy()
    exclude_cols.extend(base_cols)
    features = list(set(dataset.columns) - set(exclude_cols) - set(["industry"]))
    fixed_dataset = fix_nans_and_drop_rows(ce_dataset, metadata, features, size_rvs)

    print(fixed_dataset.head())

    # Compare the manually calculated values and the result from the above function
    # fixed_row = fixed_dataset.loc[(fixed_dataset.ticker == "CLRC1") & (fixed_dataset.date == "2003-02-14")]# .iloc[-1]

    # print(fixed_row)

    # assert fixed_row["zerotrade"] == check_row["zerotrade"]
"""