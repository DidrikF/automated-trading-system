from sampling import (extend_sep_for_sampling, 
    first_filing_based_sampling, 
    rebase_at_each_filing_sampling)
import pandas as pd
import pytest

from packages.multiprocessing.engine import pandas_chaining_mp_engine



@pytest.fixture(scope='module', autouse=True)
def setup():
    # Will be executed before the first test in the module
    """    
    global sep, sf1_art, sf1_arq, metadata
    global sep_extended, sep_sampled
    sep = pd.read_csv("./datasets/testing/sep.csv", parse_dates=["date"], index_col="date", low_memory=False)
    sf1_art = pd.read_csv("./datasets/testing/sf1_art.csv", parse_dates=["calendardate", "datekey"], index_col="calendardate", low_memory=False)
    sf1_arq = pd.read_csv("./datasets/testing/sf1_arq.csv", parse_dates=["calendardate", "datekey"], index_col="calendardate", low_memory=False)
    metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", low_memory=False)
    """
    yield
    # Will be executed after the last test in the module


@pytest.mark.skip()
def test_sep_preparation():
    # want same output as when running test scripts
    sep_prepared = pd.read_csv("./datasets/testing/dataset_20190328-152047/sep_prepared_almost.csv")
    sep_prepared_new_mp_engine = pd.read_csv("./datasets/ml_ready/dataset_20190328-151006/sep_prepared.csv")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(sep_prepared.head())
        print(sep_prepared_new_mp_engine.head())

    assert len(sep_prepared) == len(sep_prepared_new_mp_engine)

    sep_prepared = sep_prepared.fillna(value="NA")
    sep_prepared_new_mp_engine = sep_prepared_new_mp_engine.fillna(value="NA")

    result = sep_prepared.eq(sep_prepared_new_mp_engine)

    result.to_csv("./sep_preparation_equal.csv")

    assert sep_prepared.equals(sep_prepared_new_mp_engine)



def test_sep_featured():
    sep_featured = pd.read_csv("./datasets/testing/dataset_20190328-152047/sep_featured.csv") # ./datasets/testing/sep_featured.csv
    sep_featured_ml = pd.read_csv("./datasets/ml_ready_2/dataset_20190328-212016/sep_featured.csv")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(sep_featured.head())
        print(sep_featured_ml.head())

    assert len(sep_featured) == len(sep_featured_ml)

    sep_featured = sep_featured.fillna(value="NA")
    sep_featured_ml = sep_featured_ml.fillna(value="NA")

    result = sep_featured.eq(sep_featured_ml)

    result.to_csv("./sep_featured_equal.csv")

    assert sep_featured.equals(sep_featured_ml)