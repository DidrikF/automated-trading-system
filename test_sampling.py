from sampling import (extend_sep_for_sampling, 
    first_filing_based_sampling, 
    rebase_at_each_filing_sampling)
import pandas as pd
import pytest
from packages.dataset_builder.dataset import Dataset

"""
Test datasets:
/datasets/testing/...
Complete data for for AAPL, NTK (Consumer Electronics) and FCX (Copper)
sep.csv
sf1_art.csv
sf1_arq.csv
macro.csv

"""

testing_index_filename_tuple = (0, "filepath")
sep_extended = None
sep_sampled = None


@pytest.fixture(scope='module', autouse=True)
def setup():
    # Will be executed before the first test in the module
    
    yield
    # Will be executed after the last test in the module
    sep_extended.to_csv("./datasets/testing/sep_extended.csv")
    sep_sampled.to_csv("./datasets/testing/sep_sampled.csv")


def test_extend_sep_for_sampling():
    global sep_extended
    sep_extended = extend_sep_for_sampling(testing_index_filename_tuple, True)
    """
    Test that a SEP file containing multiple tickers will get the correct sf1 
    datekey from sf1_art file with multiple tickers.
    """

    sep_extended = sep_extended.fillna(value=pd.NaT)

    # Tests for AAPL
    date_1998_12_23 = pd.to_datetime("1998-12-23") 
    date_1999_01_04 = pd.to_datetime("1999-01-04") # datekey should be: 1998-12-23
    date_1999_02_05 = pd.to_datetime("1999-02-05") # datekey should be: 1999-12-23
    date_1999_02_08 = pd.to_datetime("1999-02-08") # datekey should be: 1999-02-08
    date_1999_02_11 = pd.to_datetime("1999-02-11") # datekey should be: 1999-02-08

    assert sep_extended.loc[sep_extended["date"] == date_1999_01_04].iloc[-1]["datekey"] == date_1998_12_23
    assert sep_extended.loc[sep_extended["date"] == date_1999_02_05].iloc[-1]["datekey"] == date_1998_12_23
    assert sep_extended.loc[sep_extended["date"] == date_1999_02_08].iloc[-1]["datekey"] == date_1999_02_08
    assert sep_extended.loc[sep_extended["date"] == date_1999_02_11].iloc[-1]["datekey"] == date_1999_02_08


    # Tests for NTK
    date_2011_03_31 = pd.to_datetime("2011-03-31")
    date_2011_05_04 = pd.to_datetime("2011-05-04")

    assert sep_extended.loc[sep_extended["date"] == date_2011_03_31].iloc[-1]["datekey"] == date_2011_03_31
    assert sep_extended.loc[sep_extended["date"] == date_2011_05_04].iloc[-1]["datekey"] == date_2011_03_31




@pytest.mark.skip(reason="Not interested in this atm")
def test_first_filing_based_sampling():
    global sep_extended
    sep_sampled = first_filing_based_sampling(testing_index_filename_tuple, sep_extended=sep_extended, testing=True)
    

    print(sep_sampled["date"])
    print(sep_sampled["datekey"])
    assert False





def test_rebase_at_each_filing_sampling():
    global sep_extended
    global sep_sampled
    sep_extended.reset_index(inplace=True)
    # sep_extended.to_csv("./sep_test.csv")
    #dataset = Dataset.from_df(sep_extended)
    #dataset.to_csv("./sep_test.csv")
    sep_sampled = rebase_at_each_filing_sampling(testing_index_filename_tuple, 20, sep_extended, True)

    # print(samples[["date", "datekey"]])

    # Tests for AAPL
    first_9_apple_samples = [
        pd.to_datetime("1999-01-04"), 
        # pd.to_datetime("1999-02-04"), 
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
        # pd.to_datetime("2011-04-29"),
        pd.to_datetime("2011-05-12"),
        pd.to_datetime("2011-06-13"),
        pd.to_datetime("2011-07-12"),
        pd.to_datetime("2011-08-09"),
        pd.to_datetime("2011-09-09"),

    ]

    apple_samples = sep_sampled.loc[sep_sampled["ticker"] == "AAPL"].reset_index().iloc[0:8]
    ntk_samples = sep_sampled.loc[sep_sampled["ticker"] == "NTK"].reset_index().iloc[0:6]

    index = 0
    for i, sample in apple_samples.iterrows():
        assert first_9_apple_samples[index] == sample["date"]
        index += 1

    index = 0
    for i, sample in ntk_samples.iterrows(): 
        assert first_6_ntk_samples[index] == sample["date"]
        index += 1

