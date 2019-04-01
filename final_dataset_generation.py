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


from packages.multiprocessing.engine import pandas_chaining_mp_engine


from sampling import extend_sep_for_sampling, rebase_at_each_filing_sampling
from sep_preparation import dividend_adjusting_prices_backwards, add_weekly_and_12m_stock_returns, add_equally_weighted_weekly_market_returns
from sep_industry_features import add_indmom
from sep_features import add_sep_features
from sf1_features import add_sf1_features
from sf1_industry_features import add_industry_sf1_features
from feature_selection import selected_industry_sf1_features, selected_sep_features, selected_sf1_features
from packages.helpers.helpers import get_calendardate_x_quarters_later

# update imports and functions to new engine

if __name__ == "__main__":

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


    # Script configuration
    generate_sep_features = True

    generate_sf1_features = False

    write_to_disk = True

    num_processes = 64
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = "./datasets/ml_ready_live/dataset_" + timestamp
    # read_path = "./datasets/ml/dataset_20190327-153153"
    cache_dir = "./datasets/molecules_cache_live"
    save_dir = "./datasets/molecules_save_live"
    start_time = datetime.datetime.now()

    if write_to_disk == True:
        # create folder to write files to
        os.mkdir(save_path)


    if generate_sep_features == True:
    
        """
        Wanted tickers length:  14138
        sf1_art length:  433417
        Purged sep length:  31971372
        Purged metadata length:  14135
        """

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
                "disk_name": "sep_featured",
            }
        ]

        

        sep_featured = pandas_chaining_mp_engine(tasks=sep_tasks, primary_atoms="sep", atoms_configs=atoms_configs, \
            split_strategy="ticker", num_processes=num_processes, cache_dir=cache_dir, save_dir=save_dir, sort_by=["ticker", "date"], \
                molecules_per_process=5)

        
        if write_to_disk == True:
            sep_featured.to_csv(save_path + "/sep_featured.csv")

    """
    I think I need to simulate the pipeline in a testing script using the old engine, one function at a time, like I have tested it.
    Being very carefull to use the same input files!
    Then compare the two resulting dataframes!


    Improvements to the engine:
    1. Sort outputs from callbacks (in the callbacks or in the process_jobs_faster function)
    """




    if generate_sf1_features == True:

        sf1_atoms_configs = {
            "sf1_art": {
                "cache_name": "sf1_art",
                "csv_path": "./datasets/sharadar/SHARADAR_SF1_ART.csv",
                # "csv_path": "./datasets/testing/sf1_art_no_duplicates.csv",
                "parse_dates": ["calendardate", "datekey"],
                "index_col": "calendardate",
                "report_every": 20000,
                "length": 433417,
                "sort_by": ["ticker", "calendardate", "datekey"],
                "cache": True,
            },
            "sf1_arq": {
                "cache_name": "sf1_arq",
                "csv_path": "./datasets/sharadar/SHARADAR_SF1_ARQ.csv",
                # "csv_path": "./datasets/testing/sf1_arq_no_duplicates.csv",
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
                "save_result_to_disk": False,
                "sort_by": ["ticker", "calendardate", "datekey"], # Sorting parameters, used both for molecules individually and when combined
                "cache_result": False,
                "disk_name": "sf1_art_featured_plus_ind",
            }
        ]


        cache_dir = "./datasets/molecules_cache_live"
        save_dir = "./datasets/molecules_save_live"

        sf1_featured = pandas_chaining_mp_engine(tasks=sf1_tasks, primary_atoms="sf1_art", atoms_configs=sf1_atoms_configs, \
            split_strategy="ticker", num_processes=num_processes, cache_dir=cache_dir, save_dir=save_dir, sort_by=["ticker", "calendardate", "datekey"], \
                molecules_per_process=5)

        sf1_featured = sf1_featured.sort_values(by=["ticker", "calendardate", "datekey"])

        
        sf1_featured.to_csv(save_path + "/sf1_featured.csv")

        
        if write_to_disk == True:
            sf1_featured.to_csv(save_path + "/sf1_featured.csv")

   






"""

        if generate_sf1_features == True:

            sf1_art_featured = pandas_mp_engine(callback=add_sf1_features, atoms=sf1_art, \
                data={"sf1_arq": sf1_arq, 'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'ticker', \
                    num_processes=num_processes, molecules_per_process=1)


            sf1_art_featured = pandas_mp_engine(callback=add_industry_sf1_features, atoms=sf1_art_featured, \
                data={'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'industry', \
                    num_processes=num_processes, molecules_per_process=1)

            if write_to_disk == True:
                sf1_art_featured.to_csv(save_path + "/sf1_art_featured.csv")
                
                # Select only features and save 
                cols = list(set(sf1_features + industry_sf1_features).intersection(set(sf1_art_featured.columns)))
                sf1_art_only_features = sf1_art_featured[cols]

                sf1_art_only_features.to_csv(save_path + "/sf1_art_only_features.csv")

        else:
            sf1_art_featured = pd.read_csv(read_path)
            
        print("Step 5 of xxx: SF1 Calculations - COMPLETED")
    
    else: # Do not calculate features
        sf1_art_featured = pd.read_csv(read_path + "/sf1_art_featured.csv", parse_dates=["calendardate", "datekey", "reportperiod"],\
            index_col="calendardate")

        sep_featured = pd.read_csv(read_path + "/sep_featured.csv", parse_dates=["date", "datekey"], index_col=["date"])


    # At this point we have all features calculated, but they need to be assembled and features selected.

    # 1. Merge features from SEP and SF1 into the samples data frame
    sep_featured = sep_featured.reset_index()
    sf1_art_featured = sf1_art_featured.reset_index()

    dataset = pd.merge(sep_featured, sf1_art_featured,  how='left', on=["ticker", "datekey"], suffixes=("sep", ""))



    # 2. Select features from SEP, SF1 etc.
    selected_features = ["ticker", "date", "calendardate", "datekey"] + selected_sf1_features + selected_industry_sf1_features + selected_sep_features
    dataset = dataset[selected_features]


    dataset.to_csv(save_path + "/dataset.csv", index=False)


    # 3. Remove or amend row with missing/NAN values (the strategy must be consistent with that for SEP data)
    
    # MORE EFFORT SHOULD GO INTO THIS STEP, BUT I KEEP IT SIMPLE FOR NOW, DROPPING ROWS WITH ONE OR MORE NAN VALUES

    # Drop first two (one of calendardate) years
    dataset.sort_values(by=["ticker", "calendardate"])

    result = pd.DataFrame()

    for ticker in list(dataset.ticker.unique()):
        ticker_dataset = dataset.loc[dataset.ticker == ticker]

        min_caldate = ticker_dataset.calendardate.min()            
        calendardate_1y_after = get_calendardate_x_quarters_later(min_caldate, 4)

        ticker_dataset = ticker_dataset[ticker_dataset.calendardate >= calendardate_1y_after]

        result = result.append(ticker_dataset)

    dataset = result


    dataset.to_csv(save_path + "/dataset_dropped_first_year.csv", index=False)


    dataset_no_nan = dataset.dropna(axis=0)

    # 4. Write the almost ML ready dataset to disk

    dataset_no_nan.to_csv(save_path + "/dataset_no_nan.csv")

    # 5. Print statistics:
    time_elapsed = datetime.datetime.now() - start_time

    print("Dataset length: ", len(dataset))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataset.isna().sum())

    print("Dataset no nan length: ", len(dataset_no_nan))
    print("Dropped: ", len(dataset) -  len(dataset_no_nan))
    print("Time elapsed: ", time_elapsed)

"""




