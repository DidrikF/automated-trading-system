from sf1_industry_features import add_industry_sf1_features
import pandas as pd
import pytest
from packages.dataset_builder.dataset import Dataset
import numpy as np

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
industry_sf1_art_featured = None


@pytest.fixture(scope='module', autouse=True)
def setup():
    global industry_sf1_art_featured
    # Will be executed before the first test in the module
    industry_sf1_art_featured = add_industry_sf1_features(testing_index_filename_tuple, True)
    yield
    
    # Will be executed after the last test in the module
    industry_sf1_art_featured.to_csv("./datasets/testing/industry_sf1_art_featured.csv", index=False)

def test_industry_adjusted_change_in_profit_margin():
    # Industry-adjusted change in profit margin	Soliman (chpmia), 
    # Formula: (SF1[netinc]t-1 / SF1[revenueusd]t-1) - (SF1[netinc]t-2 / SF1[revenueusd]t-2) - industry_mean((SF1[netinc]t-1 / SF1[revenueusd]t-1) - (SF1[netinc]t-2 / SF1[revenueusd]t-2))  --> [chprofitmargin]t-1 - industry_mean([chprofitmargin]t-1)
    # 2-digit SIC - fiscal-year mean adjusted change in income before extraordinary items (ib) divided by sales (sale).
    global industry_sf1_art_featured
    date_2010_03_31 = pd.to_datetime("2010-03-31")

    cnvr_row = industry_sf1_art_featured.loc[(industry_sf1_art_featured["ticker"] == "CNVR") & (industry_sf1_art_featured["calendardate"] == date_2010_03_31)].iloc[-1]
    ipg_row = industry_sf1_art_featured.loc[(industry_sf1_art_featured["ticker"] == "IPG") & (industry_sf1_art_featured["calendardate"] == date_2010_03_31)].iloc[-1]

    assert cnvr_row["chpmia"] == pytest.approx(0.5761777844 - 0.27694930664)
    assert ipg_row["chpmia"] == pytest.approx(-0.02227917111 - 0.27694930664)



# @pytest.mark.skip(reason="Not interested in this atm")
def test_industry_adjusted_percent_change_in_capital_expenditure():
    # Industry-adjusted % change in capital expenditure (pchcapx_ia), Formula: ((SF1[capex]t-1 / SF1[capex]2-1) - 1) - industry_mean((SF1[capex]t-1 / SF1[capex]2-1))
    # 2-digit SIC - fiscal-year mean adjusted percent change in capital expenditures (capx).
    global industry_sf1_art_featured
    date_2010_03_31 = pd.to_datetime("2010-03-31")


    cnvr_row = industry_sf1_art_featured.loc[(industry_sf1_art_featured["ticker"] == "CNVR") & (industry_sf1_art_featured["calendardate"] == date_2010_03_31)].iloc[-1]
    ipg_row = industry_sf1_art_featured.loc[(industry_sf1_art_featured["ticker"] == "IPG") & (industry_sf1_art_featured["calendardate"] == date_2010_03_31)].iloc[-1]


    print("cnvr pchcapex_ia: ", cnvr_row["pchcapex_ia"], type(cnvr_row["pchcapex_ia"]), np.isnan(cnvr_row["pchcapex_ia"]))

    assert cnvr_row["pchcapex_ia"] == pytest.approx(-0.01106953539 - (-0.23142309256))
    assert ipg_row["pchcapex_ia"] == pytest.approx(-0.45177664974 - (-0.23142309256))



"""
Ticker  date        capex        Netinc      Revenueusd  profit_margin   im_profit_margin    pch_capx   im_pch_capex    ch_profit_margin    im_ch_profit_margin
CNVR    2008-03-31  -9156000    71145000    664726000
CNVR    2008-06-30  -10558000   70006000    679881000
CNVR    2008-09-30  -11099000   55174000    675889000
CNVR    2008-12-31  -7227000    -214112000  625806000
CNVR    2009-03-31  -6414000    -220062000  584813000
CNVR    2009-06-30  -4446000    -221679000  551346000
CNVR    2009-09-30  -4244000    -198648000  528607000
CNVR    2009-12-31  -4625000    68616000    422723000
CNVR    2010-03-31  -6343000    76628000    383364000
CNVR    2010-06-30  -4056000    73798000    352601000
CNVR    2010-09-30  -5485000    84945000    329208000
CNVR    2010-12-31  -7416000    90510000    430798000
IPG     2008-03-31  -151500000  230700000   6680300000
IPG     2008-06-30  -139900000  188800000   6863300000
IPG     2008-09-30  -134000000  256400000   7043400000
IPG     2008-12-31  -138400000  295000000   6962700000
IPG     2009-03-31  -118200000  290800000   6802800000
IPG     2009-06-30  -107300000  223500000   6441500000
IPG     2009-09-30  -99300000   201900000   6128200000
IPG     2009-12-31  -67100000   121300000   6027600000
IPG     2010-03-31  -64800000   123700000   6043600000
IPG     2010-06-30  -67700000   178400000   6187000000
IPG     2010-09-30  -73100000   199600000   6321100000
IPG     2010-12-31  -96300000   261100000   6531900000
RLOC    2010-03-31  0           0           0
RLOC    2010-06-30  0           0           0
RLOC    2010-09-30  0           0           0
RLOC    2010-12-31  -9929000    -11147000   291689000
RLOC    2011-03-31  -12180000   -12340000   312121000
RLOC    2011-06-30  -12575000   -10895000   334511000


CNVR    2008-03-31  -9156000    71145000    664726000
CNVR    2009-03-31  -6414000    -220062000  584813000   -0.3762946446   -0.16677377023  -0.2994757536   
CNVR    2010-03-31  -6343000    76628000    383364000   0.1998831398    0.11017553641   -0.01106953539  -0.23142309256  0.5761777844    0.27694930664

IPG     2008-03-31  -151500000  230700000   6680300000
IPG     2009-03-31  -118200000  290800000   6802800000  0.04274710413   -0.16677377023  -0.21980198019
IPG     2010-03-31  -64800000   123700000   6043600000  0.02046793302   0.11017553641   -0.45177664974  -0.23142309256  -0.02227917111  0.27694930664

Missing
RLOC    2010-03-31  0           0           0

"""