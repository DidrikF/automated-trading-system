import numpy as np
import pytest
import timeit
import shutil
import os
import math
import multiprocessing as mp
from packages.multiprocessing.engine import pandas_mp_engine, split_df_into_molecules, pandas_chaining_mp_engine
import time
import pandas as pd
from sampling import extend_sep_for_sampling, rebase_at_each_filing_sampling
from sep_preparation import dividend_adjusting_prices_backwards, add_weekly_and_12m_stock_returns, add_equally_weighted_weekly_market_returns
from sep_industry_features import add_indmom
from sep_features import add_sep_features
from sf1_features import add_sf1_features
from sf1_industry_features import add_industry_sf1_features
from packages.helpers.helpers import get_calendardate_x_quarters_later



atoms_configs = {
    "sep": { # atoms_info
        "disk_name": "sep",
        "csv_path": "./datasets/testing/sep.csv",
        "parse_dates": ["date"],
        "index_col": "date",
        "report_every": 500000,
        "length": 31971372,
        "sort_by": ["ticker", "date"],
        "cache": True,
    },
    "sf1_art": {
        "disk_name": "sf1",
        "csv_path": "./datasets/testing/sf1_art.csv",
        "parse_dates": ["calendardate", "datekey"],
        "index_col": "calendardate",
        "report_every": 20000,
        "length": 433417,
        "sort_by": ["ticker", "calendardate", "datekey"],
        "cache": True,
    },
    "metadata": {
        "disk_name": "metadata",
        "csv_path": "./datasets/sharadar/METADATA_PURGED.csv",
        "parse_dates": ["firstpricedate"],
        "index_col": None,
        "report_every": 7000,
        "length": 14135,
        "sort_by": None,
        "cache": True,
    },
}

# Output from each task is input to the next.
sep_tasks = [
    { # sorted values ???
        "name": "Extend sep for sampling",
        "callback": extend_sep_for_sampling, # Callback to modify primary molecules individually, requires splitting according to split_strategy
        "molecule_key": "sep", # What key is used to pass the primary molecule to the callback
        "data": { # Other data than the primary molecules the callback needs
            "sf1_art": "sf1_art", # kw name -> molecule_dict_name (also same as in cache in most cases)
            "metadata": "metadata", 
        },
        "kwargs": {}, # Key word arguments to the callback
        "split_strategy": "ticker", # How the molecules needs to be split for this task
        "save_result_to_disk": False, # Whether to combine and store the resulting molecules to disk (as a csv file)
        "sort_by": ["ticker", "date"], # Sorting parameters, used both for molecules individually and when combined
        "cache_result": True,  # Whether to cache the resulting molecules, because they are needed later in the chain
        "disk_name": "sep_extended", # Name of molecules saved as pickle in cache_dir, as csv file in save_dir or dict in molecules_dict
    },
    {
        "name": "Dividend adjusting close price",
        "callback": dividend_adjusting_prices_backwards,
        "molecule_key": "sep",
        "data": None,
        "kwargs": {},
        "split_strategy": "ticker",
        "save_result_to_disk": False,
        "cache_result": True,
        "disk_name": "sep_extended_divadj",
    },
    {
        "name": "Add weekly and 12 month stock returns",
        "callback": add_weekly_and_12m_stock_returns,
        "molecule_key": "sep",
        "data": None,
        "kwargs": {},
        "split_strategy": "ticker",
        "save_result_to_disk": False,
        "cache_result": True,
        "disk_name": "sep_extended_divadj_ret",
    },
    {
        "name": "Add equally weighted weekly market returns",
        "callback": add_equally_weighted_weekly_market_returns,
        "molecule_key": "sep",
        "data": None,
        "kwargs": {},
        "split_strategy": "date",
        "save_result_to_disk": False,
        "cache_result": True,
        "disk_name": "sep_extended_divadj_ret_market",
    },
    {
        "name": "Add industry momentum",
        "callback": add_indmom,
        "molecule_key": "sep",
        "data": None,
        "kwargs": {},
        "split_strategy": "industry",
        "save_result_to_disk": False,
        "cache_result": True,
        "add_to_molecules_dict": True, # Config used if some task later than the one immediately following needes this task's output 
        "split_strategy_for_molecule_dict": "ticker", # Config used if some task later than the one immediately following needes this task's output 
        "disk_name": "sep_extended_divadj_ret_market_ind",
    },
    { # sorted values, This is first needed when running add_sep_features
        "name": "Sample observations using rebase_at_each_filing_sampling",
        "callback": rebase_at_each_filing_sampling, # This returns samples...
        "molecule_key": "observations",
        "data": None,
        "kwargs": {
            "days_of_distance": 20
        },
        "split_strategy": "ticker",
        "save_result_to_disk": False,
        "cache_result": True,
        "disk_name": "sep_sampled",
        "atoms_key": "sep", # Indicates what atoms_config to use when splitting data for this task
    },
    { # Sorted values
        "name": "Add sep features, final step of SEP pipeline",
        "callback": add_sep_features, # adj_close keyerror
        "molecule_key": "sep_sampled",
        "data": {
            "sep": "sep_extended_divadj_ret_market_ind", # I need to get the updated sep df
            "sf1_art": "sf1_art"
        },
        "kwargs": {},
        "split_strategy": "ticker",
        "save_result_to_disk": False,
        "cache_result": False,
        "disk_name": "sep_featured",
    }
]



def test_cache_completed_tasks():
    global sep_tasks, atoms_configs
    base_test_dir = "./testing"
    cache_dir = "./testing/molecule_cache"
    save_dir = "./testing/molecule_save"
    num_processes = 4
    try:
        os.makedirs(cache_dir)
        os.makedirs(save_dir)
    except:
        shutil.rmtree(base_test_dir)
        os.makedirs(cache_dir)
        os.makedirs(save_dir)

    sep_featured = pandas_chaining_mp_engine(tasks=sep_tasks, primary_atoms="sep", atoms_configs=atoms_configs, \
            split_strategy="ticker", num_processes=num_processes, cache_dir=cache_dir, save_dir=save_dir, sort_by=["ticker", "date"], \
                molecules_per_process=2, resume=False)
    try:
        assert os.path.isfile(cache_dir + "/" + sep_tasks[0]["disk_name"] + ".pickle")
        assert os.path.isfile(cache_dir + "/" + sep_tasks[1]["disk_name"] + ".pickle")
        assert os.path.isfile(cache_dir + "/" + sep_tasks[2]["disk_name"] + ".pickle")
        assert os.path.isfile(cache_dir + "/" + sep_tasks[3]["disk_name"] + ".pickle")
        assert os.path.isfile(cache_dir + "/" + sep_tasks[4]["disk_name"] + ".pickle")
        assert os.path.isfile(cache_dir + "/" + sep_tasks[5]["disk_name"] + ".pickle")
    finally:
        # Delete cache dir and save dir
        shutil.rmtree(base_test_dir)


# Output from each task is input to the next.
sep_tasks_not_caching_all = [
    { # sorted values ???
        "name": "Extend sep for sampling",
        "callback": extend_sep_for_sampling, # Callback to modify primary molecules individually, requires splitting according to split_strategy
        "molecule_key": "sep", # What key is used to pass the primary molecule to the callback
        "data": { # Other data than the primary molecules the callback needs
            "sf1_art": "sf1_art", # kw name -> molecule_dict_name (also same as in cache in most cases)
            "metadata": "metadata", 
        },
        "kwargs": {}, # Key word arguments to the callback
        "split_strategy": "ticker", # How the molecules needs to be split for this task
        "save_result_to_disk": False, # Whether to combine and store the resulting molecules to disk (as a csv file)
        "sort_by": ["ticker", "date"], # Sorting parameters, used both for molecules individually and when combined
        "cache_result": True,  # Whether to cache the resulting molecules, because they are needed later in the chain
        "disk_name": "sep_extended", # Name of molecules saved as pickle in cache_dir, as csv file in save_dir or dict in molecules_dict
    },
    {
        "name": "Dividend adjusting close price",
        "callback": dividend_adjusting_prices_backwards,
        "molecule_key": "sep",
        "data": None,
        "kwargs": {},
        "split_strategy": "ticker",
        "save_result_to_disk": False,
        "cache_result": True,
        "disk_name": "sep_extended_divadj",
    },
    {
        "name": "Add weekly and 12 month stock returns",
        "callback": add_weekly_and_12m_stock_returns,
        "molecule_key": "sep",
        "data": None,
        "kwargs": {},
        "split_strategy": "ticker",
        "save_result_to_disk": False,
        "cache_result": True,
        "disk_name": "sep_extended_divadj_ret",
    },
    {
        "name": "Add equally weighted weekly market returns",
        "callback": add_equally_weighted_weekly_market_returns,
        "molecule_key": "sep",
        "data": None,
        "kwargs": {},
        "split_strategy": "date",
        "save_result_to_disk": False,
        "cache_result": True,
        "disk_name": "sep_extended_divadj_ret_market",
    },
    {
        "name": "Add industry momentum",
        "callback": add_indmom,
        "molecule_key": "sep",
        "data": None,
        "kwargs": {},
        "split_strategy": "industry",
        "save_result_to_disk": False,
        "cache_result": False,
        "add_to_molecules_dict": True, # Config used if some task later than the one immediately following needes this task's output 
        "split_strategy_for_molecule_dict": "ticker", # Config used if some task later than the one immediately following needes this task's output 
        "disk_name": "sep_extended_divadj_ret_market_ind",
    },
    { # sorted values, This is first needed when running add_sep_features
        "name": "Sample observations using rebase_at_each_filing_sampling",
        "callback": rebase_at_each_filing_sampling, # This returns samples...
        "molecule_key": "observations",
        "data": None,
        "kwargs": {
            "days_of_distance": 20
        },
        "split_strategy": "ticker",
        "save_result_to_disk": False,
        "cache_result": False,
        "disk_name": "sep_sampled",
    },
    { # Sorted values
        "name": "Add sep features, final step of SEP pipeline",
        "callback": add_sep_features, # adj_close keyerror
        "molecule_key": "sep_sampled",
        "data": {
            "sep": "sep_extended_divadj_ret_market_ind", # I need to get the updated sep df
            "sf1_art": "sf1_art"
        },
        "kwargs": {},
        "split_strategy": "ticker",
        "save_result_to_disk": False,
        "cache_result": False,
        "disk_name": "sep_featured",
    }
]




def test_resume_from_cache():
    global sep_tasks, atoms_configs
    base_test_dir = "./testing"
    cache_dir = "./testing/molecule_cache"
    save_dir = "./testing/molecule_save"
    num_processes = 4

    try:
        os.makedirs(cache_dir)
        os.makedirs(save_dir)
    except:
        shutil.rmtree(base_test_dir)
        os.makedirs(cache_dir)
        os.makedirs(save_dir)

    sep_featured = pandas_chaining_mp_engine(tasks=sep_tasks_not_caching_all, primary_atoms="sep", atoms_configs=atoms_configs, \
            split_strategy="ticker", num_processes=num_processes, cache_dir=cache_dir, save_dir=save_dir, sort_by=["ticker", "date"], \
                molecules_per_process=2)


    assert os.path.isfile(cache_dir + "/" + sep_tasks[0]["disk_name"] + ".pickle")
    assert os.path.isfile(cache_dir + "/" + sep_tasks[1]["disk_name"] + ".pickle")
    assert os.path.isfile(cache_dir + "/" + sep_tasks[2]["disk_name"] + ".pickle")
    assert os.path.isfile(cache_dir + "/" + sep_tasks[3]["disk_name"] + ".pickle")
    
    assert os.path.isfile(cache_dir + "/" + sep_tasks[4]["disk_name"] + ".pickle") == False

    sep_featured_2 = pandas_chaining_mp_engine(tasks=sep_tasks_not_caching_all, primary_atoms="sep", atoms_configs=atoms_configs, \
            split_strategy="ticker", num_processes=num_processes, cache_dir=cache_dir, save_dir=save_dir, sort_by=["ticker", "date"], \
                molecules_per_process=2, resume=True)

    # Using print output to see that it in fact resumed from the desired task.

    failed = False
    pos = None
    errors = []
    len_sep_featured = len(sep_featured)

    for index in range(0, len_sep_featured):
        for column in sep_featured.columns:
            correct_val = sep_featured.iloc[index][column]
            if isinstance(correct_val, str):
                if correct_val != sep_featured_2.iloc[index][column]:
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            elif isinstance(correct_val, pd.Timestamp) or isinstance(correct_val, pd.Timedelta):
                if str(correct_val) != str(sep_featured_2.iloc[index][column]):
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            elif math.isnan(correct_val):
                if not math.isnan(sep_featured_2.iloc[index][column]):
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            else:
                if correct_val != pytest.approx(sep_featured_2.iloc[index][column]):
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            


    if failed == True:
        print("Shape: ", sep_featured.shape, sep_featured_2.shape)
        for pos in errors:
            print("Failed at position: ", pos, " Corr: ", sep_featured.iloc[pos[0]][pos[1]], "Othr: ", sep_featured_2.iloc[pos[0]][pos[1]])

    assert len(errors) == 0

    shutil.rmtree(base_test_dir)