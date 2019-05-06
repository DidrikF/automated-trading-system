import pandas as pd
import pytest
import math
import sys
import os


myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, ".."))
sys.path.insert(0, os.path.join(myPath))

from processing.engine import pandas_mp_engine, pandas_chaining_mp_engine
from sampling import extend_sep_for_sampling, rebase_at_each_filing_sampling
from sep_features import add_sep_features, add_indmom, dividend_adjusting_prices_backwards, add_weekly_and_12m_stock_returns, add_equally_weighted_weekly_market_returns
from sf1_features import add_sf1_features
from sf1_industry_features import add_industry_sf1_features
from generate_features import generate_sep_featured, generate_sf1_featured
from labeling import add_labels_via_triple_barrier_method, equity_risk_premium_labeling

save_path = ""
cache_dir = ""



@pytest.fixture(scope='module', autouse=True)
def setup():
    global save_path, cache_dir

    save_path = "./testing_datasets"
    cache_dir = "./testing_datasets/molecules_cache"

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Will be executed before the first test in the module   
    # sep = pd.read_csv("./datasets/testing/sep.csv", parse_dates=["date"], index_col="date", low_memory=False)
    # metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", low_memory=False)

    """
    sf1_art = pd.read_csv("../datasets/testing/sf1_art.csv", parse_dates=["calendardate", "datekey"], low_memory=False)
    sf1_arq = pd.read_csv("../datasets/testing/sf1_arq.csv", parse_dates=["calendardate", "datekey"], low_memory=False)

    
    print("sf1_arq len before: ", len(sf1_arq))
    print("sf1_art len before: ", len(sf1_art))


    sf1_art.drop_duplicates(subset=["ticker", "calendardate", "datekey"], keep="last", inplace=True)
    sf1_arq.drop_duplicates(subset=["ticker", "calendardate", "datekey"], keep="last", inplace=True)

    print("sf1_arq len after: ", len(sf1_arq))
    print("sf1_art len after: ", len(sf1_art))

    sf1_art.to_csv("../datasets/testing/sf1_art_no_duplicates.csv", index=False)
    sf1_arq.to_csv("../datasets/testing/sf1_arq_no_duplicates.csv", index=False)
    """

    yield
    # Will be executed after the last test in the module

def test_sep_featured():
    global save_path, cache_dir
    num_processes = 6
    

    print("\n\nSEP_FEATURED - OLD METHOD\n\n")

    sep = pd.read_csv("../datasets/testing/sep.csv", parse_dates=["date"], index_col="date", low_memory=False)
    
    sf1_art = pd.read_csv("../datasets/testing/sf1_art.csv", parse_dates=["calendardate", "datekey", "reportperiod"],\
        index_col="calendardate", low_memory=False)
    
    metadata = pd.read_csv("../datasets/sharadar/METADATA_PURGED.csv", parse_dates=["firstpricedate"], low_memory=False)

    tb_rate = tb_rate = pd.read_csv("../datasets/macro/t_bill_rate_3m.csv", parse_dates=["date"], index_col="date")

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

    sep_adjusted_plus_returns.sort_values(by=["ticker", "date"], ascending=True, inplace=True)

    sep_prepared = pandas_mp_engine(callback=add_equally_weighted_weekly_market_returns, atoms=sep_adjusted_plus_returns, data=None, \
        molecule_key='sep', split_strategy= 'date', \
            num_processes=num_processes, molecules_per_process=1)

    sep_prepared.sort_values(by=["ticker", "date"], ascending=True, inplace=True)

    sep_prepared_plus_indmom = pandas_mp_engine(callback=add_indmom, atoms=sep_prepared, data=None, \
        molecule_key='sep', split_strategy= 'industry', \
            num_processes=num_processes, molecules_per_process=1)

    sep_prepared_plus_indmom.sort_values(by=["ticker", "date"], inplace=True)

    # sep_prepared_plus_indmom.to_csv("../datasets/testing/sep_prepared.csv")
    
    sep_sampled = pandas_mp_engine(callback=rebase_at_each_filing_sampling, atoms=sep_prepared_plus_indmom, data=None, \
        molecule_key='observations', split_strategy='ticker', num_processes=num_processes, molecules_per_process=1, \
            days_of_distance=20)

    sep_sampled.sort_values(by=["ticker", "date"], ascending=True, inplace=True)
    

    sep_featured = pandas_mp_engine(callback=add_sep_features, atoms=sep_sampled, \
        data={'sep': sep_prepared_plus_indmom, "sf1_art": sf1_art}, molecule_key='sep_sampled', split_strategy= 'ticker', \
            num_processes=num_processes, molecules_per_process=1)

    sep_featured.sort_values(by=["ticker", "date"], ascending=True, inplace=True)


    tbm_labeled_sep = pandas_mp_engine(callback=add_labels_via_triple_barrier_method, atoms=sep_featured, \
        data={'sep': sep_prepared_plus_indmom}, molecule_key='sep_featured', split_strategy= 'ticker', \
            num_processes=num_processes, molecules_per_process=1, ptSl=[1, -1], min_ret=None)

    tbm_labeled_sep.sort_values(by=["ticker", "date"], ascending=True, inplace=True)

    erp_labeled_sep = pandas_mp_engine(callback=equity_risk_premium_labeling, atoms=tbm_labeled_sep, \
        data=None, molecule_key='sep_featured', split_strategy= 'ticker', \
            num_processes=num_processes, molecules_per_process=1, tb_rate=tb_rate)

    erp_labeled_sep.sort_values(by=["ticker", "date"], ascending=True, inplace=True)

    sep_featured = erp_labeled_sep


    sep_featured.sort_values(by=["ticker", "date"], inplace=True)


    sep_featured.to_csv(save_path + "/sep_featured.csv") # I really think this is the correct result


    #______________________CHAINING MP ENGINE____________________________

    print("\n\nSEP_FEATURED - NEW METHOD\n\n")

    sep_featured_2 = generate_sep_featured(
        num_processes=num_processes, 
        cache_dir=cache_dir, 
        tb_rate=tb_rate, 
        sep_path="../datasets/testing/sep.csv", # paths relative to the engine I think
        sf1_art_path="../datasets/testing/sf1_art.csv", 
        metadata_path="../datasets/sharadar/METADATA_PURGED.csv",
        resume=False
    )

    sep_featured_2 = sep_featured_2.sort_values(by=["ticker", "date"]) # Should not need this

    sep_featured_2.to_csv(save_path + "/sep_featured_2.csv")


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
def testing_sf1_featured():
    global save_path, cache_dir
    num_processes = 6

    print("\n\nSF1_FEATURED - OLD METHOD\n\n")

    sf1_art = pd.read_csv("../datasets/testing/sf1_art_no_duplicates.csv", parse_dates=["calendardate", "datekey"],\
        index_col="calendardate", low_memory=False)
    
    sf1_arq = pd.read_csv("../datasets/testing/sf1_arq_no_duplicates.csv", parse_dates=["calendardate", "datekey"],\
        index_col="calendardate", low_memory=False)
    
    metadata = pd.read_csv("../datasets/sharadar/METADATA_PURGED.csv", parse_dates=["firstpricedate"], low_memory=False)


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


    print("\n\nSF1_FEATURED - NEW METHOD\n\n")

    sf1_featured_2 = generate_sf1_featured(
            num_processes=num_processes,
            cache_dir=cache_dir,
            sf1_art_path="../datasets/testing/sf1_art_no_duplicates.csv",
            sf1_arq_path="../datasets/testing/sf1_arq_no_duplicates.csv",
            metadata_path="../datasets/sharadar/METADATA_PURGED.csv",
            resume=False
        )
        

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











# _______________ DON'T THINK I NEED THIS, WOULD LIKE TO REMOVE THE BELOW _______________________


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
