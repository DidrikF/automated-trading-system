import pandas as pd
import pytest
import math
from dateutil.relativedelta import *
import numpy as np
import sys, os

import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from datetime import datetime

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, ".."))

from finalize_dataset import finalize_dataset, fix_nans_and_drop_rows, merge_datasets, base_cols, selected_industry_sf1_features, \
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



def test_finalize_dataset():
    global metadata, dataset
    """
    Requires to make the filling of missing values deterministic in the "fix_nans_and_drop_rows" function.
    """

    # sep = pd.read_csv("../datasets/testing/sep.csv")
    sep_featured = pd.read_csv("./testing_datasets/sep_featured.csv", parse_dates=["date", "datekey"])
    sf1_featured = pd.read_csv("./testing_datasets/sf1_featured.csv", parse_dates=["datekey", "calendardate"])

    date0 = "2004-01-20"
    datekey0 = "2003-12-19"
    date1 = "2006-05-05"
    datekey1 = "2005-02-01"

    sf1_featured["bm"].loc[(sf1_featured.ticker == "AAPL") & (sf1_featured.datekey == datekey0)] = pd.NaT
    sf1_featured["bm"].loc[(sf1_featured.ticker == "AAPL") & (sf1_featured.datekey == datekey1)] = pd.NaT

    
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

    print("desired for date0: ", dataset.loc[(dataset.ticker == "AAPL") & (dataset.datekey == datekey0)].iloc[-1]["bm"])
    print("desired for date1: ", dataset.loc[(dataset.ticker == "AAPL") & (dataset.datekey == datekey1)].iloc[-1]["bm"])
    
    print("industry: ", ind_val)
    print("market: ", market_val)

    # assert dataset.loc[(dataset.ticker == "AAPL") & (dataset.datekey == datekey0)].iloc[-1]["bm"] == ind_val

