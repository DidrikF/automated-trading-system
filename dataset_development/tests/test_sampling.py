import pandas as pd
import pytest

from ..sampling import (extend_sep_for_sampling, 
    first_filing_based_sampling, 
    rebase_at_each_filing_sampling)
from ..dataset import Dataset
from ..multiprocessing.engine import pandas_mp_engine

"""
Test datasets:
/datasets/testing/...
Complete data for for AAPL, NTK (Consumer Electronics) and FCX (Copper)
sep.csv
sf1_art.csv
sf1_arq.csv
"""

sep = None
sf1_art = None
sf1_arq = None
metadata = None

sep_extended = None
sep_sampled = None


@pytest.fixture(scope='module', autouse=True)
def setup():
    # Will be executed before the first test in the module
    global sep, sf1_art, sf1_arq, metadata
    global sep_extended, sep_sampled
    sep = pd.read_csv("../datasets/testing/sep.csv", parse_dates=["date"], index_col="date", low_memory=False)
    sf1_art = pd.read_csv("../datasets/testing/sf1_art.csv", parse_dates=["calendardate", "datekey"], index_col="calendardate", low_memory=False)
    sf1_arq = pd.read_csv("../datasets/testing/sf1_arq.csv", parse_dates=["calendardate", "datekey"], index_col="calendardate", low_memory=False)
    metadata = pd.read_csv("../datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", low_memory=False)
    
    yield
    
    # Will be executed after the last test in the module
    if isinstance(sep_extended, pd.DataFrame):
        sep_extended.sort_values(by=["ticker", "date"], inplace=True)
        sep_extended.to_csv("../datasets/testing/sep_extended.csv")
    if isinstance(sep_sampled, pd.DataFrame):
        sep_sampled.sort_values(by=["ticker", "date"], inplace=True)
        sep_sampled.to_csv("../datasets/testing/sep_sampled.csv")


def test_extend_sep_for_sampling():
    global sep, sf1_art, sf1_arq, metadata
    global sep_extended

    sep_extended = pandas_mp_engine(callback=extend_sep_for_sampling, atoms=sep, \
        data={"sf1_art": sf1_art, "metadata": metadata}, \
            molecule_key='sep', split_strategy='ticker', \
                num_processes=1, molecules_per_process=1)

    """
    Test that a SEP file containing multiple tickers will get the correct sf1 
    datekey from sf1_art file with multiple tickers.
    """
    # Tests for AAPL
    # date_1998_12_23 = pd.to_datetime("1998-12-23") 
    #date_1999_01_04 = pd.to_datetime("1999-01-04") # datekey should be: 1998-12-23
    # date_1999_02_05 = pd.to_datetime("1999-02-05") # datekey should be: 1999-12-23
    # date_1999_02_08 = pd.to_datetime("1999-02-08") # datekey should be: 1999-02-08
    # date_1999_02_11 = pd.to_datetime("1999-02-11") # datekey should be: 1999-02-08

    sep_extended_aapl = sep_extended.loc[sep_extended["ticker"] == "AAPL"]

    assert sep_extended_aapl.loc["1999-01-04"]["datekey"] == pd.to_datetime("1998-12-23") 
    assert sep_extended_aapl.loc["1999-02-05"]["datekey"] == pd.to_datetime("1998-12-23")
    assert sep_extended_aapl.loc["1999-02-08"]["datekey"] == pd.to_datetime("1999-02-08")
    assert sep_extended_aapl.loc["1999-02-11"]["datekey"] == pd.to_datetime("1999-02-08")

    # Test metadata was set correctly
    metadata_aapl = metadata.loc[metadata["ticker"] == "AAPL"].iloc[-1]

    assert sep_extended_aapl.loc["1999-01-04"]["industry"] == metadata_aapl["industry"]
    assert sep_extended_aapl.loc["1999-01-04"]["sector"] == metadata_aapl["sector"]
    assert sep_extended_aapl.loc["1999-01-04"]["siccode"] == metadata_aapl["siccode"]
    assert sep_extended_aapl.loc["1999-01-04"]["sharesbas"] == \
        sf1_art.loc[sf1_art.datekey == sep_extended_aapl.loc["1999-01-04"]["datekey"]].iloc[-1]["sharesbas"]

    # Tests for NTK
    sep_extended_ntk = sep_extended.loc[sep_extended["ticker"] == "NTK"]

    assert sep_extended_ntk.loc["2011-03-31"]["datekey"] == pd.to_datetime("2011-03-31")
    assert sep_extended_ntk.loc["2011-05-04"]["datekey"] == pd.to_datetime("2011-03-31")


def test_rebase_at_each_filing_sampling():
    global sep_extended
    global sep_sampled


    sep_sampled = pandas_mp_engine(callback=rebase_at_each_filing_sampling, atoms=sep_extended, data=None, \
        molecule_key='observations', split_strategy='ticker', num_processes=1, molecules_per_process=1, \
            days_of_distance=20)

    sep_sampled = sep_sampled.sort_values(by=["ticker", "date"])

    sep_sampled.to_csv("../datasets/testing/sep_sampled_latest_implementation.csv")

    sep_sampled_aapl = sep_sampled.loc[sep_sampled.ticker == "AAPL"]

    assert sep_sampled_aapl.index[0] == pd.to_datetime("1997-12-31")
    assert sep_sampled_aapl.index[1] == pd.to_datetime("1998-02-09")
    assert sep_sampled_aapl.index[2] == pd.to_datetime("1998-03-09")
    assert sep_sampled_aapl.index[3] == pd.to_datetime("1998-04-09")
    assert sep_sampled_aapl.index[4] == pd.to_datetime("1998-05-11")
    assert sep_sampled_aapl.index[5] == pd.to_datetime("1998-06-11")
    assert sep_sampled_aapl.index[6] == pd.to_datetime("1998-07-10")
    assert sep_sampled_aapl.index[7] == pd.to_datetime("1998-08-10")
    assert sep_sampled_aapl.index[8] == pd.to_datetime("1998-09-10")

    
    """
    AAPL
    Date
    1997-12-31
    1998-02-09
    1998-03-09
    1998-04-09
    1998-05-11
    1998-06-11
    1998-07-10
    1998-08-10
    1998-09-10
    1998-10-09
    1998-11-10
    1998-12-23
    1999-02-08
    1999-03-08
    1999-04-08
    1999-05-11
    1999-06-11
    Datekey
    1997-12-05
    1998-02-09
    1998-02-09
    1998-02-09
    1998-05-11
    1998-05-11
    1998-05-11
    1998-08-10
    1998-08-10
    1998-08-10
    1998-08-10
    1998-12-23
    1999-02-08
    1999-02-08
    1999-02-08
    1999-05-11
    1999-05-11
    1999-05-11
    1999-08-06
    1999-08-06
    1999-08-06
    """




@pytest.mark.skip()
def test_rebase_at_each_filing_sampling_OLD():
    global sep_extended
    global sep_sampled


    sep_sampled = pandas_mp_engine(callback=rebase_at_each_filing_sampling, atoms=sep_extended, data=None, \
        molecule_key='observations', split_strategy='ticker', num_processes=1, molecules_per_process=1, \
            days_of_distance=20)
    
    # Tests for AAPL
    first_9_apple_samples = [
        pd.to_datetime("1999-02-08"), 
        pd.to_datetime("1999-03-08"), 
        pd.to_datetime("1999-04-08"), 
        pd.to_datetime("1999-05-11"), 
        pd.to_datetime("1999-06-11"), 
        pd.to_datetime("1999-07-12"), 
        pd.to_datetime("1999-08-06")
    ]

    first_6_ntk_samples = [
        pd.to_datetime("2011-03-31"),
        # pd.to_datetime("2011-04-29"), This sample is less than 20 day before a new filing, and should therefore not be among the samples
        pd.to_datetime("2011-05-12"),
        pd.to_datetime("2011-06-13"),
        pd.to_datetime("2011-07-12"),
        pd.to_datetime("2011-08-09"),
        pd.to_datetime("2011-09-09"),

    ]

    apple_samples = sep_sampled.loc[sep_sampled["ticker"] == "AAPL"].loc["1999-02-08":"1999-08-06"]
    ntk_samples = sep_sampled.loc[sep_sampled["ticker"] == "NTK"].loc["2011-03-31":"2011-09-09"]

    # print(ntk_samples[["datekey"]])

    index = 0
    for date, sample in apple_samples.iterrows():
        assert first_9_apple_samples[index] == date
        index += 1

    index = 0
    for date, sample in ntk_samples.iterrows(): 
        assert first_6_ntk_samples[index] == date
        index += 1



@pytest.mark.skip(reason="Not interested in this atm, this test is not completed")
def test_first_filing_based_sampling():
    global sep_extended
    global sep_sampled

    sep_extended = pandas_mp_engine(callback=first_filing_based_sampling, atoms=sep_extended, \
        data=None, molecule_key='sep', split_strategy='ticker', num_processes=1, molecules_per_process=1)


    print(sep_sampled["date"])
    print(sep_sampled["datekey"])
    assert False
