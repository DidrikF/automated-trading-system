"""
In this script samples (based on SEP and SF1 data) is combined together into a final complete dataset, ready for ML.
Some samples might have to be dropped due to missing data in SF1_ART or SF1_ARQ.

The most recent row from sep_featured.csv, and the most recent SF1 row based on datekey is added to sep_sampled.csv.
- If SF1 row is too old compared to the sample date, the sample is dropped.
- If too much fabricated data was used in the generation of the SF1 features, the samples using this row
  is dropped.
"""

import datetime
import os
import pandas as pd


from processing.engine import pandas_chaining_mp_engine
from helpers.helpers import get_calendardate_x_quarters_later


from sampling import extend_sep_for_sampling, rebase_at_each_filing_sampling
from sep_features import add_sep_features, dividend_adjusting_prices_backwards, \
    add_weekly_and_12m_stock_returns, add_equally_weighted_weekly_market_returns, add_indmom
from sf1_features import add_sf1_features
from sf1_industry_features import add_industry_sf1_features
from labeling import add_labels_via_triple_barrier_method, equity_risk_premium_labeling


sep_features = [
    "industry", "sector", "siccode", "mom12m_actual", "indmom", "mom1w", "mom1w_ewa_market", "return_1m",
    "return_2m", "return_3m", "mom1m", "mom6m", "mom12m", "mom24m", "mom12m_to_7m", "chmom", "mve", "beta", "betasq",
    "idiovol", "ill", "dy", "turn", "dolvol", "maxret", "retvol", "std_dolvol", "std_turn", "zerotrade"
]

sf1_features = [
    "roaq", "chtx", "rsup", "sue", "cinvest", "nincr", "roavol", "cashpr", "cash", "bm", "currat", "depr", "ep", "lev", "quick",
    "rd_sale", "roic", "salecash", "saleinv", "salerec", "sp", "tb", "sin", "tang", "debtc_sale", "eqt_marketcap", "dep_ppne",
    "tangibles_marketcap", "agr", "cashdebt", "chcsho", "chinv", "egr", "gma", "invest", "lgr", "operprof", "pchcurrat", "pchdepr",
    "pchgm_pchsale", "pchquick", "pchsale_pchinvt", "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv", "rd", "roeq", "sgr",
    "grcapx", "chtl_lagat", "chlt_laginvcap", "chlct_lagat", "chint_lagat", "chinvt_lagsale", "chint_lagsgna", "chltc_laginvcap",
    "chint_laglt", "chdebtnc_lagat", "chinvt_lagcor", "chppne_laglt", "chpay_lagact", "chint_laginvcap", "chinvt_lagact",
    "pchppne", "pchlt", "pchint", "chdebtnc_ppne", "chdebtc_sale", "age", "ipo", "profitmargin", "chprofitmargin",
    "industry", "change_sales", "ps"
]

industry_sf1_features = ["bm_ia", "cfp_ia", "chatoia", "mve_ia", "pchcapex_ia", "chpmia", "herf", "ms"]

"""
sf1_arq_cols = ticker,dimension,calendardate,datekey,reportperiod,lastupdated,accoci,assets,assetsavg,\
assetsc,assetsnc,assetturnover,bvps,capex,cashneq,cashnequsd,cor,consolinc,currentratio,de,debt,\
    debtc,debtnc,debtusd,deferredrev,depamor,deposits,divyield,dps,ebit,ebitda,ebitdamargin,\
        ebitdausd,ebitusd,ebt,eps,epsdil,epsusd,equity,equityavg,equityusd,ev,evebit,evebitda,\
            fcf,fcfps,fxusd,gp,grossmargin,intangibles,intexp,invcap,invcapavg,inventory,investments,\
                investmentsc,investmentsnc,liabilities,liabilitiesc,liabilitiesnc,marketcap,ncf,ncfbus,\
                    ncfcommon,ncfdebt,ncfdiv,ncff,ncfi,ncfinv,ncfo,ncfx,netinc,netinccmn,netinccmnusd,\
                        netincdis,netincnci,netmargin,opex,opinc,payables,payoutratio,pb,pe,pe1,ppnenet,\
                            prefdivis,price,ps,ps1,receivables,retearn,revenue,revenueusd,rnd,roa,roe,roic,\
                                ros,sbcomp,sgna,sharefactor,sharesbas,shareswa,shareswadil,sps,tangibles,\
                                    taxassets,taxexp,taxliabilities,tbvps,workingcapital
"""

def generate_sep_featured(num_processes, cache_dir, tb_rate, sep_path, sf1_art_path, metadata_path, resume):
    """
    Wanted tickers length:  14138
    sf1_art length:  433417
    Purged sep length:  31971372
    Purged metadata length:  14135
    """

    atoms_configs = {
        "sep": { # atoms_info
            "disk_name": "sep",
            "csv_path": sep_path, # "./datasets/sharadar/SEP_PURGED.csv", # These paths are relative to what?, I think the engine...
            # "csv_path": "./datasets/testing/sep.csv",
            "parse_dates": ["date"],
            "index_col": "date",
            "report_every": 500000,
            "length": 31971372,
            "sort_by": ["ticker", "date"],
            "cache": True,
        },
        "sf1_art": {
            "disk_name": "sf1_art",
            "csv_path": sf1_art_path, # "./datasets/sharadar/SHARADAR_SF1_ART.csv",
            # "csv_path": "./datasets/testing/sf1_art.csv",
            "parse_dates": ["calendardate", "datekey"],
            "index_col": "calendardate",
            "report_every": 20000,
            "length": 433417,
            "sort_by": ["ticker", "calendardate", "datekey"],
            "cache": True,
        },
        "metadata": {
            "disk_name": "metadata",
            "csv_path": metadata_path, # "./datasets/sharadar/METADATA_PURGED.csv",
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
            "sort_by": ["ticker", "date"], # Sorting parameters, used both for molecules individually and when combined
            "cache_result": True,  # Whether to cache the resulting molecules, because they are needed later in the chain
            "disk_name": "sep_extended", # Name of molecules saved as pickle in cache_dir or as one csv file in save_dir
        },
        # Dividend adjustment...
        {
            "name": "Dividend adjusting close prices",
            "callback": dividend_adjusting_prices_backwards,
            "molecule_key": "sep",
            "data": None,
            "kwargs": {},
            "split_strategy": "ticker",
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
            "cache_result": True,
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
            "cache_result": True,
            "disk_name": "sep_sampled",
        },
        { # Sorted values
            "name": "Add sep features",
            "callback": add_sep_features, # adj_close keyerror
            "molecule_key": "sep_sampled",
            "data": {
                "sep": "sep_extended_divadj_ret_market_ind", # I need to get the updated sep df
                "sf1_art": "sf1_art"
            },
            "kwargs": {},
            "split_strategy": "ticker",
            "cache_result": True,
            "disk_name": "sep_featured",
        },
        { # Labeling
            "name": "Label the dataset for side prediction using the tripple barrier method",
            "callback": add_labels_via_triple_barrier_method,
            "molecule_key": "sep_featured",
            "data": {
                "sep": "sep_extended_divadj_ret_market_ind",
            },
            "kwargs": {
                "ptSl": [1, -1],
                "min_ret": None,
            },
            "split_strategy": "ticker",
            "cache_result": True,
            "disk_name": "tbm_labeled_sep"
        },
        {
            "name": "Label the dataset for regressions on monthly equity risk premums",
            "callback": equity_risk_premium_labeling,
            "molecule_key": "sep_featured",
            "data": None,
            "kwargs": {
                "tb_rate": tb_rate, 
            },
            "split_strategy": "ticker",
            "cache_result": True,
            "disk_name": "erp_labeled_sep"
        }
    ]

    sep_featured = pandas_chaining_mp_engine(tasks=sep_tasks, primary_atoms="sep", atoms_configs=atoms_configs, \
        split_strategy="ticker", num_processes=num_processes, cache_dir=cache_dir, sort_by=["ticker", "date"], \
            molecules_per_process=2, resume=resume)
    
    return sep_featured

def generate_sf1_featured(num_processes, cache_dir, sf1_art_path, sf1_arq_path, metadata_path, resume):
    sf1_atoms_configs = {
        "sf1_art": {
            "disk_name": "sf1_art",
            "csv_path": sf1_art_path,# "./datasets/sharadar/SHARADAR_SF1_ART.csv",
            # "csv_path": "./datasets/testing/sf1_art_no_duplicates.csv",
            "parse_dates": ["calendardate", "datekey"],
            "index_col": "calendardate",
            "report_every": 20000,
            "length": 433417,
            "sort_by": ["ticker", "calendardate", "datekey"],
            "cache": True,
        },
        "sf1_arq": {
            "disk_name": "sf1_arq",
            "csv_path": sf1_arq_path, # "./datasets/sharadar/SHARADAR_SF1_ARQ.csv",
            # "csv_path": "./datasets/testing/sf1_arq_no_duplicates.csv",
            "parse_dates": ["calendardate", "datekey"],
            "index_col": "calendardate",
            "report_every": 20000,
            "length": 433417,
            "sort_by": ["ticker", "calendardate", "datekey"],
            "cache": True,
        },
        "metadata": {
            "disk_name": "metadata",
            "csv_path": metadata_path, # "./datasets/sharadar/METADATA_PURGED.csv",
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
            "sort_by": ["ticker", "calendardate", "datekey"], # Sorting parameters, used both for molecules individually and when combined
            "cache_result": True,  # Whether to cache the resulting molecules, because they are needed later in the chain
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
            "sort_by": ["ticker", "calendardate", "datekey"], # Sorting parameters, used both for molecules individually and when combined
            "cache_result": False,
            "disk_name": "sf1_art_featured_plus_ind",
        }
    ]

    sf1_featured = pandas_chaining_mp_engine(tasks=sf1_tasks, primary_atoms="sf1_art", atoms_configs=sf1_atoms_configs, \
        split_strategy="ticker", num_processes=num_processes, cache_dir=cache_dir, sort_by=["ticker", "calendardate", "datekey"], \
            molecules_per_process=5, resume=resume)

    sf1_featured = sf1_featured.sort_values(by=["ticker", "calendardate", "datekey"])

    return sf1_featured



if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = "./datasets/ml_ready_live/dataset_" + timestamp
    os.mkdir(save_path)
    
    cache_dir = "./datasets/molecules_cache_live"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


    if True:
        
        tb_rate = pd.read_csv("./datasets/macro/t_bill_rate_3m.csv", parse_dates=["date"], index_col="date")
        sep_featured = generate_sep_featured(
            num_processes=32,
            cache_dir=cache_dir,
            tb_rate=tb_rate, 
            sep_path="./datasets/sharadar/SEP_PURGED.csv",
            sf1_art_path="./datasets/sharadar/SHARADAR_SF1_ART.csv",
            metadata_path="./datasets/sharadar/METADATA_PURGED.csv",
            resume=True
        )
        
        sep_featured.to_csv(save_path + "/sep_featured.csv")


    if False:
        """ Do I need to do this? I've done this in the test sets???
        sf1_art.drop_duplicates(subset=["ticker", "calendardate", "datekey"], keep="last", inplace=True)
        sf1_arq.drop_duplicates(subset=["ticker", "calendardate", "datekey"], keep="last", inplace=True)
        """

        # Come Runntime Warnings occure when running this...

        sf1_featured = generate_sf1_featured(
            num_processes=32,
            cache_dir=cache_dir,
            sf1_art_path="./datasets/sharadar/SHARADAR_SF1_ART.csv",
            sf1_arq_path="./datasets/sharadar/SHARADAR_SF1_ARQ.csv",
            metadata_path="./datasets/sharadar/METADATA_PURGED.csv",
            resume=True
        )


        sf1_featured.to_csv(save_path + "/sf1_featured.csv")


"""
STDOUT when running generate_sf1_featured:

# Errors might have something todo with me forgetting to activate the conda environment (ran on python 3.7)


/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:140: RuntimeWarning: Degrees of freedom <= 0 for slice
  keepdims=keepdims)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:110: RuntimeWarning: invalid value encountered in true_divide
  arrmean, rcount, out=arrmean, casting='unsafe', subok=False)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:132: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:140: RuntimeWarning: Degrees of freedom <= 0 for slice
  keepdims=keepdims)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:110: RuntimeWarning: invalid value encountered in true_divide
  arrmean, rcount, out=arrmean, casting='unsafe', subok=False)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:132: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:140: RuntimeWarning: Degrees of freedom <= 0 for slice
  keepdims=keepdims)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:110: RuntimeWarning: invalid value encountered in true_divide
  arrmean, rcount, out=arrmean, casting='unsafe', subok=False)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:132: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:140: RuntimeWarning: Degrees of freedom <= 0 for slice
  keepdims=keepdims)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:110: RuntimeWarning: invalid value encountered in true_divide
  arrmean, rcount, out=arrmean, casting='unsafe', subok=False)
/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:132: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
2019-05-04 16:21:24.591384 100.0% 14138/14138 - add_sf1_features done after 6.5 minutes. Remaining 0.0 minutes...
Cacheing as pickle result from task:  Add sf1 features
Step  2  of  2  - Add industry sf1 features - Time elapsed:  9.46  minutes.
/home/ubuntu/pycode/automated-trading-system/dataset_development/sf1_industry_features.py:163: RuntimeWarning: invalid value encountered in double_scalars
  sum_sqrd_percent_of_revenue += (company_row["revenueusd"] / sum_industry_revenue)**2
/home/ubuntu/pycode/automated-trading-system/dataset_development/sf1_industry_features.py:163: RuntimeWarning: invalid value encountered in double_scalars
  sum_sqrd_percent_of_revenue += (company_row["revenueusd"] / sum_industry_revenue)**2
Connection reset by 52.215.29.43 port 226 - add_industry_sf1_features done after 6.33 minutes. Remaining 0.13 minutes.

"""


"""
Traceback (most recent call last):
  File "/home/ubuntu/anaconda3/envs/master/lib/python3.6/processing/pool.py", line 119, in worker
    result = (True, func(*args, **kwds))
  File "/home/ubuntu/pycode/automated-trading-system/packages/processing/engine.py", line 272, in expandCall_fast
    out = callback(**kwargs)
  File "/home/ubuntu/pycode/automated-trading-system/sep_preparation.py", line 74, in add_weekly_and_12m_stock_returns
    date_index = pd.date_range(sep.index.min(), sep.index.max()) # [0], [1]
  File "/home/ubuntu/anaconda3/envs/master/lib/python3.6/site-packages/pandas/core/indexes/datetimes.py", line 1524, in date_range
    closed=closed, **kwargs)
  File "/home/ubuntu/anaconda3/envs/master/lib/python3.6/site-packages/pandas/core/arrays/datetimes.py", line 421, in _generate_range
    raise ValueError("Neither `start` nor `end` can be NaT")
ValueError: Neither `start` nor `end` can be NaT


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "final_dataset_generation.py", line 232, in <module>
    molecules_per_process=2, resume=True)
  File "/home/ubuntu/pycode/automated-trading-system/packages/processing/engine.py", line 430, in pandas_chaining_mp_engine
"""
