import pytest
import pandas as pd
from sep_features import add_sep_features

"""
Each step is performed for each industry separately

Step-by-Step Dataset Construction:
1. Extend the SEP dataset with information usefull for sampling (most recent 10-K filing date, Industry classifications)
2. Use different sampling techniques to get monthly observations
    1. At first use timebars (sampling at a fixed time interval), but try to respect the different fiscal years
3. Calculate the various price and volume based features
4. Add inn SF1 and DAILY data
5. Compute features based on SF1
6. Select the features you want and combine into one ML ready dataset
"""


testing_index_filename_tuple = (0, "filepath")
sep_extended = None
sep_sampled = None
sep_featured = None


@pytest.fixture(scope='module', autouse=True)
def setup():
    # Will be executed before the first test in the module
    sep_extended = pd.read_csv("./datasets/testing/sep_extended.csv", index_col=0)
    sep_sampled = pd.read_csv("./datasets/testing/sep_sampled.csv", index_col=0)

    yield
    # Will be executed after the last test in the module
    sep_featured.to_csv("./datasets/testing/sep_featured.csv")



def test_add_sep_features():
    global sep_extended
    global sep_sampled
    global sep_featured

    sep_featured = add_sep_features(testing_index_filename_tuple, True)

    # Test return calculation
    # AAPL
    date_1999_01_04 = pd.to_datetime("1999-01-04") # Close: 1.473
    # date_1999_02_04 = pd.to_datetime("1999-02-04") # Close: 1.3530000000000002
    date_2002_12_19 = pd.to_datetime("2002-12-19") # Cannot calculate return

    assert sep_featured.loc[(sep_featured["ticker"] == "AAPL") & (sep_featured["date"] == date_1999_01_04)].iloc[-1]["return"] == pytest.approx((1.3530000000000002 / 1.473) -1)
    assert pd.isnull(sep_featured.loc[(sep_featured["ticker"] == "AAPL") & (sep_featured["date"] == date_2002_12_19)].iloc[-1]["return"])
    
    # NTK
    date_2011_06_13 = pd.to_datetime("2011-06-13") # Close: 38.88
    # date_2011_07_13 = pd.to_datetime("2011-07-13") # Close: 34.25
    assert sep_featured.loc[(sep_featured["ticker"] == "NTK") & (sep_featured["date"] == date_2011_06_13)].iloc[0]["return"] == pytest.approx((34.25 / 38.88) - 1)

    
    # Test momentum calculation (only for 6 and 12 months because the code is so similar)
    # AAPL
    date_1999_12_22 = pd.to_datetime("1999-12-22") # Close:  3.569
    # date_1999_06_22 = pd.to_datetime("1999-06-22") # Close: 1.621
    
    date_2002_06_14 = pd.to_datetime("2002-06-14") # Close: 1.436
    # date_2001_06_14 = pd.to_datetime("2001-06-14") # Close: 1.42

    assert sep_featured.loc[(sep_featured["ticker"] == "AAPL") & (sep_featured["date"] == date_1999_12_22)].iloc[-1]["mom6m"] == pytest.approx((3.569 / 1.621) -1)
    assert sep_featured.loc[(sep_featured["ticker"] == "AAPL") & (sep_featured["date"] == date_2002_06_14)].iloc[-1]["mom12m"] == pytest.approx((1.436 / 1.42) -1) 


    # Test beta calculation

    # assert False


# Determine if ANY Value in a Series is Missing
# s.isnull().values.any()