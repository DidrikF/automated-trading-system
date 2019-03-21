import pandas as pd
import pytest

from packages.dataset_builder.dataset import Dataset
from sf1_features import add_sf1_features
from packages.helpers.helpers import 

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

sf1_art_featured = None


@pytest.fixture(scope='module', autouse=True)
def setup():
    global sf1_art_featured
    # Will be executed before the first test in the module
    sf1_art = pd.read_csv("./datasets/testing/sf1_art.csv", parse_dates=["datekey"], index_col="datekey")
    sf1_arq = pd.read_csv("./datasets/testing/sf1_arq.csv", parse_dates=["datekey"], index_col="datekey")
    metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", parse_dates=["firstpricedate"], index_col="datekey")
    yield
    
    # Will be executed after the last test in the module
    sf1_art_featured.to_csv("./datasets/testing/sf1_art_featured.csv", index=False)



@pytest.skip
def test_add_sf_features_asset_growth():
    global sf1_art_featured

    sf1_art_featured = pandas_mp_engine(callback=add_sf1_features, atoms=sf1_art, \
        data={"sf1_arq": sf1_arq, 'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'ticker', \
            num_processes=1, molecules_per_process=1)
    sf1_art_aapl = sf1_art_featured.loc[sf1_art_featured.ticker=="AAPL"]
    sf1_art_ntk = sf1_art_featured.loc[sf1_art_featured.ticker=="NTK"]

    # IMPLEMENT NEW WAY OF SELECTING FILINGS, TEST IT, REWRITE CODE, CONTINUE WRITING TESTS HERE.

    # AAPL Assets: 4289000000, 1y earlier: 4233000000
    assert sf1_art_featured.loc["1998-12-23"]["agr"] == pytest.approx((4289000000/4233000000) - 1)
    # NTK Assets: 1983000000, 1y earlier: 1942600000
    assert sf1_art_ntk.loc["2013-05-09"]["agr"] == pytest.approx((1983000000/1942600000) - 1)


def test_add_sf_features_book_to_market():
    global sf1_art_featured
    
    date_1998_12_23 = pd.to_datetime("1998-12-23") # AAPL Equityusd: 1642000000, Marketcap: 5387080402
    date_2013_05_09 = pd.to_datetime("2013-05-09")  # NTK Equityusd: 77100000, Marketcap: 1096777192

    assert sf1_art_featured.loc[(sf1_art_featured["ticker"] == "AAPL") & (sf1_art_featured["datekey"] == date_1998_12_23)].iloc[-1]["bm"] == pytest.approx(1642000000/5387080402)
    assert sf1_art_featured.loc[(sf1_art_featured["ticker"] == "NTK") & (sf1_art_featured["datekey"] == date_2013_05_09)].iloc[-1]["bm"] == pytest.approx(77100000/1096777192)


def test_add_sf_features_age():
    global sf1_art_featured
    
    date_1998_12_23 = pd.to_datetime("1998-12-23")
    aapl_firstpricedate = pd.to_datetime("1986-01-01")
    date_2013_05_09 = pd.to_datetime("2013-05-09")
    ntk_firstpricedate = pd.to_datetime("2010-06-15")
 
    aapl_age = round((date_1998_12_23 - aapl_firstpricedate).days / 365)
    ntk_age = round((date_2013_05_09 - ntk_firstpricedate).days / 365)

    assert sf1_art_featured.loc[(sf1_art_featured["ticker"] == "AAPL") & (sf1_art_featured["datekey"] == date_1998_12_23)].iloc[-1]["age"] == aapl_age
    assert sf1_art_featured.loc[(sf1_art_featured["ticker"] == "NTK") & (sf1_art_featured["datekey"] == date_2013_05_09)].iloc[-1]["age"] == ntk_age



def test_add_sf1_features_cash():
    global sf1_art_featured

    date_1998_12_23 = pd.to_datetime("1998-12-23") # Cashequsd: 1481000000, Assetsavg: 4104750000
    date_2013_05_09 = pd.to_datetime("2013-05-09") # Cashequsd: 201600000, Assetsavg: 1935175000 

    assert sf1_art_featured.loc[(sf1_art_featured["ticker"] == "AAPL") & (sf1_art_featured["datekey"] == date_1998_12_23)].iloc[-1]["cash"] == pytest.approx(1481000000/4104750000)
    assert sf1_art_featured.loc[(sf1_art_featured["ticker"] == "NTK") & (sf1_art_featured["datekey"] == date_2013_05_09)].iloc[-1]["cash"] == pytest.approx(201600000/1935175000)


def test_add_sf1_features_cashdebt():
    global sf1_art_featured

    date_1998_12_23 = pd.to_datetime("1998-12-23") # Cashequsd: 1481000000, Assetsavg: 4104750000
    date_2013_05_09 = pd.to_datetime("2013-05-09") # Cashequsd: 201600000, Assetsavg: 1935175000 

    # ASSERTIONS


def test_add_sf1_features_cashpr():
    pass

def test_add_sf1_features_cfp():
    pass

def test_add_sf1_features_cash_chcsho():
    pass

def test_add_sf1_features_chinv():
    pass

def test_add_sf1_features_profitmargin(): 
    pass

def test_add_sf1_features_chprofitmargin():
    # Used in the calculation of industry adjusted change in profit margin
    pass

def test_add_sf1_features_chtx():
    pass

def test_add_sf1_features_currat():
    pass

def test_add_sf1_features_depr():
    pass


def test_add_sf1_features_egr():
    pass

def test_add_sf1_features_ep():
    pass

def test_add_sf1_features_grcapx():
    pass

def test_add_sf1_features_invest():
    pass

def test_add_sf1_features_ipo():
    pass

def test_add_sf1_features_lev():
    pass

def test_add_sf1_features_lgr():
    pass

def test_add_sf1_features_operprof():
    pass

def test_add_sf1_features_pchcurrat():
    pass

def test_add_sf1_features_pchdepr():
    pass


def test_add_sf1_features_pchgm_pchsale():
    pass

def test_add_sf1_features_pchsale_pchinvt():
    pass

def test_add_sf1_features_pchsale_pchrect():
    pass

def test_add_sf1_features_pchsale_pchxsga():
    pass

def test_add_sf1_features_pchsaleinv():
    pass

def test_add_sf1_features_quick():
    pass

def test_add_sf1_features_rd():
    pass

def test_add_sf1_features_rd_mve():
    pass

def test_add_sf1_features_rd_sale():
    pass

def test_add_sf1_features_roaq():
    pass

def test_add_sf1_features_roic():
    pass

def test_add_sf1_features_roeq():
    pass


"""
salecash
saleinv
salerec
sin
SP
tang
tb
cinvest

sgr
rsup
sue
stdacc
stdcf

chtl_lagat
chlt_laginvcap
chlct_lagat
debtc_sale
chint_lagat
eqt_marketcap
dep_ppne
ppne0_ppne
pchppne
chinvt_lagsale
chint_lagsgna
chltc_laginvcap
chint_laglt
chdebtnc_ppne
chdebtc_sale
chdebtnc_lagat
chinvt_lagcor
chppne_laglt
tangibles_marketcap
chpay_lagact
chint_laginvcap
chinvt_lagact
pchlt
pchint
"""




"""
ticker,dimension,calendardate,datekey,reportperiod,lastupdated,accoci,assets,assetsavg,assetsc,assetsnc,assetturnover,bvps,capex,cashneq,cashnequsd,cor,consolinc,currentratio,de,debt,debtc,debtnc,debtusd,deferredrev,depamor,deposits,divyield,dps,ebit,ebitda,ebitdamargin,ebitdausd,ebitusd,ebt,eps,epsdil,epsusd,equity,equityavg,equityusd,ev,evebit,evebitda,fcf,fcfps,fxusd,gp,grossmargin,intangibles,intexp,invcap,invcapavg,inventory,investments,investmentsc,investmentsnc,liabilities,liabilitiesc,liabilitiesnc,marketcap,ncf,ncfbus,ncfcommon,ncfdebt,ncfdiv,ncff,ncfi,ncfinv,ncfo,ncfx,netinc,netinccmn,netinccmnusd,netincdis,netincnci,netmargin,opex,opinc,payables,payoutratio,pb,pe,pe1,ppnenet,prefdivis,price,ps,ps1,receivables,retearn,revenue,revenueusd,rnd,roa,roe,roic,ros,sbcomp,sgna,sharefactor,sharesbas,shareswa,shareswadil,sps,tangibles,taxassets,taxexp,taxliabilities,tbvps,workingcapital
AAPL,ART,1997-09-30,1997-12-05,1997-09-26,2019-01-30,0.0,4233000000.0,,3424000000.0,809000000.0,,0.34,-53000000.0,1230000000.0,1230000000.0,5713000000.0,-1045000000.0,1.883,2.527,976000000.0,25000000.0,951000000.0,976000000.0,0.0,118000000.0,0.0,0.0,0.0,-1045000000.0,-927000000.0,-0.131,-927000000.0,-1045000000.0,-1045000000.0,-0.29600000000000004,-0.29600000000000004,-0.29600000000000004,1200000000.0,,1200000000.0,1769575844.0,-2.0,-1.909,135000000.0,0.038,1.0,1368000000.0,0.193,0.0,0.0,2161000000.0,,437000000.0,229000000.0,229000000.0,0.0,3033000000.0,1818000000.0,1215000000.0,2023575844.0,-322000000.0,-384000000.0,34000000.0,-161000000.0,0.0,23000000.0,-533000000.0,-36000000.0,188000000.0,0.0,-1045000000.0,-1045000000.0,-1045000000.0,0.0,0.0,-0.14800000000000002,2438000000.0,-1070000000.0,685000000.0,0.0,1.686,-1.936,-1.908,486000000.0,0.0,0.565,0.28600000000000003,0.281,1035000000.0,589000000.0,7081000000.0,7081000000.0,485000000.0,,,,-0.14800000000000002,0.0,1286000000.0,1.0,3583815536.0,3529736000.0,3529736000.0,2.006,4233000000.0,259000000.0,0.0,264000000.0,1.199,1606000000.0
AAPL,ART,1998-09-30,1998-12-23,1998-09-25,2019-01-30,0.0,4289000000.0,4104750000.0,3698000000.0,591000000.0,1.4469999999999998,0.434,43000000.0,1481000000.0,1481000000.0,4462000000.0,309000000.0,2.4330000000000003,1.612,954000000.0,0.0,954000000.0,954000000.0,0.0,111000000.0,0.0,0.0,0.0,329000000.0,440000000.0,0.07400000000000001,440000000.0,329000000.0,329000000.0,0.084,0.075,0.084,1642000000.0,1440000000.0,1642000000.0,4860080402.0,15.0,11.046,818000000.0,0.221,1.0,1479000000.0,0.249,0.0,0.0,2242000000.0,2288500000.0,78000000.0,819000000.0,819000000.0,0.0,2647000000.0,1520000000.0,1127000000.0,5387080402.0,251000000.0,0.0,41000000.0,-22000000.0,0.0,19000000.0,-543000000.0,-566000000.0,775000000.0,0.0,309000000.0,309000000.0,309000000.0,0.0,0.0,0.052000000000000005,1218000000.0,261000000.0,719000000.0,0.0,3.281,17.434,16.926,348000000.0,0.0,1.422,0.907,0.884,955000000.0,898000000.0,5941000000.0,5941000000.0,303000000.0,0.075,0.215,0.14400000000000002,0.055,0.0,908000000.0,1.0,3788953812.0,3695272000.0,4701676000.0,1.608,4289000000.0,182000000.0,20000000.0,173000000.0,1.135,2178000000.0
"""


# Test shape of result to see that features have been calculated for "all" rows