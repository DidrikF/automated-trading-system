from sampling import (extend_sep_for_sampling, 
    first_filing_based_sampling, 
    rebase_at_each_filing_sampling)
import pandas as pd
import pytest
import math
import sys

from packages.multiprocessing.engine import pandas_chaining_mp_engine

from packages.multiprocessing.engine import pandas_mp_engine
from sampling import extend_sep_for_sampling, rebase_at_each_filing_sampling
from sep_preparation import dividend_adjusting_prices_backwards, add_weekly_and_12m_stock_returns, add_equally_weighted_weekly_market_returns
from sep_industry_features import add_indmom
from sep_features import add_sep_features

from sf1_features import add_sf1_features
from sf1_industry_features import add_industry_sf1_features


@pytest.fixture(scope='module', autouse=True)
def setup():
    # Will be executed before the first test in the module   
    # sep = pd.read_csv("./datasets/testing/sep.csv", parse_dates=["date"], index_col="date", low_memory=False)
    # metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", low_memory=False)

    sf1_art = pd.read_csv("./datasets/testing/sf1_art.csv", parse_dates=["calendardate", "datekey"], low_memory=False)
    sf1_arq = pd.read_csv("./datasets/testing/sf1_arq.csv", parse_dates=["calendardate", "datekey"], low_memory=False)

    
    print("sf1_arq len before: ", len(sf1_arq))
    print("sf1_art len before: ", len(sf1_art))


    sf1_art.drop_duplicates(subset=["ticker", "calendardate", "datekey"], keep="last", inplace=True)
    sf1_arq.drop_duplicates(subset=["ticker", "calendardate", "datekey"], keep="last", inplace=True)

    print("sf1_arq len after: ", len(sf1_arq))
    print("sf1_art len after: ", len(sf1_art))

    sf1_art.to_csv("./datasets/testing/sf1_art_no_duplicates.csv", index=False)
    sf1_arq.to_csv("./datasets/testing/sf1_arq_no_duplicates.csv", index=False)

    yield
    # Will be executed after the last test in the module


def test_sep_featured():
    num_processes = 1
    save_path = "./testing_datasets"

    print("\n\nOLD METHOD\n\n")


    sep = pd.read_csv("./datasets/testing/sep.csv", parse_dates=["date"], index_col="date", low_memory=False)
    
    sf1_art = pd.read_csv("./datasets/testing/sf1_art.csv", parse_dates=["calendardate", "datekey", "reportperiod"],\
        index_col="calendardate", low_memory=False)
    
    metadata = pd.read_csv("./datasets/sharadar/METADATA_PURGED.csv", parse_dates=["firstpricedate"], low_memory=False)

    sep_extended = pandas_mp_engine(callback=extend_sep_for_sampling, atoms=sep, \
                data={"sf1_art": sf1_art, "metadata": metadata}, \
                    molecule_key='sep', split_strategy='ticker', \
                        num_processes=num_processes, molecules_per_process=1)

    sep_extended.sort_values(by=["ticker", "date"], ascending=True, inplace=True)


    sep_adjusted = pandas_mp_engine(callback=dividend_adjusting_prices_backwards, atoms=sep_extended, data=None, \
        molecule_key='sep', split_strategy= 'ticker', \
            num_processes=num_processes, molecules_per_process=1)
        
    sep_adjusted_plus_returns = pandas_mp_engine(callback=add_weekly_and_12m_stock_returns, atoms=sep_adjusted, data=None, \
        molecule_key='sep', split_strategy= 'ticker', \
            num_processes=num_processes, molecules_per_process=1)

    sep_prepared = pandas_mp_engine(callback=add_equally_weighted_weekly_market_returns, atoms=sep_adjusted_plus_returns, data=None, \
        molecule_key='sep', split_strategy= 'date', \
            num_processes=num_processes, molecules_per_process=1)


    sep_prepared_plus_indmom = pandas_mp_engine(callback=add_indmom, atoms=sep_prepared, data=None, \
        molecule_key='sep', split_strategy= 'industry', \
            num_processes=num_processes, molecules_per_process=1)

    sep_prepared_plus_indmom.sort_values(by=["ticker", "date"], inplace=True)
    
    sep_sampled = pandas_mp_engine(callback=rebase_at_each_filing_sampling, atoms=sep_prepared_plus_indmom, data=None, \
        molecule_key='observations', split_strategy='ticker', num_processes=num_processes, molecules_per_process=1, \
            days_of_distance=20)

    sep_sampled.sort_values(by=["ticker", "date"], ascending=True, inplace=True)
    

    sep_featured = pandas_mp_engine(callback=add_sep_features, atoms=sep_sampled, \
        data={'sep': sep_prepared_plus_indmom, "sf1_art": sf1_art}, molecule_key='sep_sampled', split_strategy= 'ticker', \
            num_processes=num_processes, molecules_per_process=1)


    sep_featured.sort_values(by=["ticker", "date"], inplace=True)


    sep_featured.to_csv(save_path + "/sep_featured.csv") # I really think this is the correct result

    #______________________CHAINING MP ENGINE____________________________

    print("\n\nNew METHOD\n\n")

    cache_dir = "./testing_datasets/molecules_cache"
    save_dir = "./testing_datasets/molecules_save"

    atoms_configs = {
        "sep": { # atoms_info
            "cache_name": "sep",
            # "csv_path": "./datasets/sharadar/SEP_PURGED.csv",
            "csv_path": "./datasets/testing/sep.csv",
            # "versions: 2, # use to identify different cached versions
            # "stage_of_development": "momentum_done", # used to cache the molecules at different stages of development
            "parse_dates": ["date"],
            "index_col": "date",
            "report_every": 500000,
            "length": 31971372,
            "sort_by": ["date"],
            "cache": True,
        },
        "sf1_art": {
            "cache_name": "sf1",
            # "csv_path": "./datasets/sharadar/SHARADAR_SF1_ART.csv",
            "csv_path": "./datasets/testing/sf1_art.csv",
            "parse_dates": ["calendardate", "datekey"],
            "index_col": "calendardate",
            "report_every": 20000,
            "length": 433417,
            "sort_by": ["calendardate", "datekey"],
            "cache": True,
        },
        "metadata": {
            "cache_name": "metadata",
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
            "cache_result": False,  # Whether to cache the resulting molecules, because they are needed later in the chain
            "disk_name": "sep_extended", # Name of molecules saved as pickle in cache_dir or as one csv file in save_dir
        },
        {
            "name": "Dividend adjusting close price",
            "callback": dividend_adjusting_prices_backwards,
            "molecule_key": "sep",
            "data": None,
            "kwargs": {},
            "split_strategy": "ticker",
            "save_result_to_disk": False,
            "cache_result": False,
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
            "cache_result": False,
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
            "cache_result": False,
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
            "add_to_molecules_dict": True, # But split the wrong way
            "split_strategy_for_molecule_dict": "ticker",
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
            "disk_name": "sep_extended",
        }
    ]


    sep_featured_2 = pandas_chaining_mp_engine(tasks=sep_tasks, primary_atoms="sep", atoms_configs=atoms_configs, \
            split_strategy="ticker", num_processes=num_processes, cache_dir=cache_dir, save_dir=save_dir, sort_by=["ticker", "date"], \
                molecules_per_process=5)

    sep_featured_2 = sep_featured_2.sort_values(by=["ticker", "date"]) # Should not need this

    sep_featured_2.to_csv("./testing_datasets/sep_featured_2.csv")


    """
    sep_featured = sep_featured.fillna("NA")
    sep_featured_2 = sep_featured_2.fillna("NA")
    eq_result = sep_featured.eq(sep_featured_2)
    eq_result.to_csv("./testing_datasets/eq_result_sep_featured.csv")
    """

    assert sep_featured.shape[0] == sep_featured_2.shape[0]
    assert sep_featured.shape[1] == sep_featured_2.shape[1]
    

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




@pytest.mark.skip()
def testing_sep_featured_fast():
    sep_featured = pd.read_csv("./testing_datasets/sep_featured.csv", parse_dates=["date"], index_col="date", low_memory=False)
    sep_featured_2 = pd.read_csv("./testing_datasets/sep_featured_2.csv", parse_dates=["date"], index_col="date", low_memory=False)


    len_sep_featured = len(sep_featured)

    failed = False
    pos = None
    errors = []

    for index in range(0, len_sep_featured):
        for column in sep_featured.columns:
            correct_val = sep_featured.iloc[index][column]
            if isinstance(correct_val, str):
                if correct_val != sep_featured_2.iloc[index][column]:
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
            # print("Row: ", sep_featured_2.iloc[pos[0]])
            # print("ticker: ", sep_featured_2.iloc[pos[0]]["ticker"])


    assert len(errors) == 0



@pytest.mark.skip()
def testing_sf1_featured():
    save_path = "./testing_datasets"

    sf1_art = pd.read_csv("./datasets/testing/sf1_art_no_duplicates.csv", parse_dates=["calendardate", "datekey"],\
        index_col="calendardate", low_memory=False)
    
    sf1_arq = pd.read_csv("./datasets/testing/sf1_arq_no_duplicates.csv", parse_dates=["calendardate", "datekey"],\
        index_col="calendardate", low_memory=False)
    
    metadata = pd.read_csv("./datasets/sharadar/METADATA_PURGED.csv", parse_dates=["firstpricedate"], low_memory=False)


    num_processes = 1

    sf1_art = sf1_art.sort_values(by=["ticker", "calendardate", "datekey"])

    sf1_arq = sf1_arq.sort_values(by=["ticker", "calendardate", "datekey"])

    sf1_featured = pandas_mp_engine(callback=add_sf1_features, atoms=sf1_art, \
        data={"sf1_arq": sf1_arq, 'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'ticker', \
            num_processes=num_processes, molecules_per_process=1)

    sf1_featured.sort_values(by=["ticker", "calendardate", "datekey"])


    sf1_featured = pandas_mp_engine(callback=add_industry_sf1_features, atoms=sf1_featured, \
        data={'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'industry', \
            num_processes=num_processes, molecules_per_process=1)

    sf1_featured = sf1_featured.sort_values(by=["ticker", "calendardate", "datekey"])

    sf1_featured.to_csv(save_path + "/sf1_featured.csv")

    sf1_atoms_configs = {
        "sf1_art": {
            "cache_name": "sf1_art",
            # "csv_path": "./datasets/sharadar/SHARADAR_SF1_ART.csv",
            "csv_path": "./datasets/testing/sf1_art_no_duplicates.csv",
            "parse_dates": ["calendardate", "datekey"],
            "index_col": "calendardate",
            "report_every": 20000,
            "length": 433417,
            "sort_by": ["ticker", "calendardate", "datekey"],
            "cache": True,
        },
        "sf1_arq": {
            "cache_name": "sf1_arq",
            # "csv_path": "./datasets/sharadar/SHARADAR_SF1_ART.csv",
            "csv_path": "./datasets/testing/sf1_arq_no_duplicates.csv",
            "parse_dates": ["calendardate", "datekey"],
            "index_col": "calendardate",
            "report_every": 20000,
            "length": 433417,
            "sort_by": ["ticker", "calendardate", "datekey"],
            "cache": True,
        },
        "metadata": {
            "cache_name": "metadata",
            "csv_path": "./datasets/sharadar/METADATA_PURGED.csv",
            "parse_dates": ["firstpricedate"],
            "index_col": None,
            "report_every": 7000,
            "length": 14135,
            "sort_by": None,
            "cache": True,
        },
    }


    sf1_tasks = [
        { # sorted values ???
            "name": "Add sf1 features",
            "callback": add_sf1_features, # Callback to modify primary molecules individually, requires splitting according to split_strategy
            "molecule_key": "sf1_art", # What key is used to pass the primary molecule to the callback
            "data": { # Other data than the primary molecules the callback needs
                "sf1_arq": "sf1_arq", # kw name -> molecule_dict_name (also same as in cache in most cases)
                "metadata": "metadata", 
            },
            "kwargs": {}, # Key word arguments to the callback
            "split_strategy": "ticker", # How the molecules needs to be split for this task
            "save_result_to_disk": False, # Whether to combine and store the resulting molecules to disk (as a csv file)
            "sort_by": ["ticker", "calendardate", "datekey"], # Sorting parameters, used both for molecules individually and when combined
            "cache_result": False,  # Whether to cache the resulting molecules, because they are needed later in the chain
            "disk_name": "sf1_art_featured", # Name of molecules saved as pickle in cache_dir or as one csv file in save_dir
        },
        {
            "name": "Add industry sf1 features",
            "callback": add_industry_sf1_features,
            "molecule_key": "sf1_art",
            "data": {
                "metadata": "metadata", 
            },
            "kwargs": {},
            "split_strategy": "industry",
            "save_result_to_disk": False,
            "sort_by": ["ticker", "calendardate", "datekey"], # Sorting parameters, used both for molecules individually and when combined
            "cache_result": False,
            "disk_name": "sf1_art_featured_plus_ind",
        }
    ]


    cache_dir = "./datasets/molecules_cache"
    save_dir = "./datasets/molecules_save"

    sf1_featured_2 = pandas_chaining_mp_engine(tasks=sf1_tasks, primary_atoms="sf1_art", atoms_configs=sf1_atoms_configs, \
        split_strategy="ticker", num_processes=num_processes, cache_dir=cache_dir, save_dir=save_dir, sort_by=["ticker", "calendardate", "datekey"], \
            molecules_per_process=5)

    sf1_featured_2 = sf1_featured_2.sort_values(by=["ticker", "calendardate", "datekey"])

    
    sf1_featured_2.to_csv(save_path + "/sf1_featured_2.csv")

    assert sf1_featured.shape[0] == sf1_featured_2.shape[0]
    assert sf1_featured.shape[1] == sf1_featured_2.shape[1]
    

    failed = False
    pos = None
    errors = []
    len_sf1_featured = len(sf1_featured)

    for index in range(0, len_sf1_featured):
        for column in sf1_featured.columns:
            correct_val = sf1_featured.iloc[index][column]
            if isinstance(correct_val, str):
                if correct_val != sf1_featured_2.iloc[index][column]:
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            elif isinstance(correct_val, pd.Timestamp) or isinstance(correct_val, pd.Timedelta):
                if str(correct_val) != str(sf1_featured_2.iloc[index][column]):
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            elif math.isnan(correct_val):
                if not math.isnan(sf1_featured_2.iloc[index][column]):
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            else:
                if correct_val != pytest.approx(sf1_featured_2.iloc[index][column]):
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            


    if failed == True:
        print("Shape: ", sf1_featured.shape, sf1_featured_2.shape)
        for pos in errors:
            print("Failed at position: ", pos, " Corr: ", sf1_featured.iloc[pos[0]][pos[1]], "Othr: ", \
                sf1_featured_2.iloc[pos[0]][pos[1]], sf1_featured.iloc[pos[0]]["datekey"], sf1_featured.iloc[pos[0]]["ticker"], \
                    sf1_featured_2.iloc[pos[0]]["datekey"], sf1_featured_2.iloc[pos[0]]["ticker"])

    assert len(errors) == 0


@pytest.mark.skip()
def testing_sf1_featured_fast():
    sf1_featured = pd.read_csv("./testing_datasets/sf1_featured.csv", parse_dates=["calendardate", "datekey"], index_col="calendardate", low_memory=False)
    sf1_featured_2 = pd.read_csv("./testing_datasets/sf1_featured_2.csv", parse_dates=["calendardate", "datekey"], index_col="calendardate", low_memory=False)


    assert sf1_featured.shape[0] == sf1_featured_2.shape[0]
    assert sf1_featured.shape[1] == sf1_featured_2.shape[1]
    

    failed = False
    pos = None
    errors = []
    len_sf1_featured = len(sf1_featured)

    for index in range(0, len_sf1_featured):
        for column in sf1_featured.columns:
            correct_val = sf1_featured.iloc[index][column]
            if isinstance(correct_val, str):
                if correct_val != sf1_featured_2.iloc[index][column]:
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            elif isinstance(correct_val, pd.Timestamp) or isinstance(correct_val, pd.Timedelta):
                if str(correct_val) != str(sf1_featured_2.iloc[index][column]):
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            elif math.isnan(correct_val):
                if not math.isnan(sf1_featured_2.iloc[index][column]):
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            else:
                if correct_val != pytest.approx(sf1_featured_2.iloc[index][column]):
                    failed = True
                    pos = (index, column)
                    errors.append(pos)
            


    if failed == True:
        print("Shape: ", sf1_featured.shape, sf1_featured_2.shape)
        for pos in errors:
            print("Failed at position: ", pos, " Corr: ", sf1_featured.iloc[pos[0]][pos[1]], "Othr: ", sf1_featured_2.iloc[pos[0]][pos[1]], sf1_featured.iloc[pos[0]]["datekey"], sf1_featured.iloc[pos[0]]["ticker"])

    assert len(errors) == 0
