import sys
from packages.dataset_builder.dataset import Dataset, merge_datasets_simple
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError
import pandas as pd
import numpy as np
import logging
import datetime
from dateutil.relativedelta import *
from os import listdir
from os.path import isfile, join

"""
Each step is performed for each industry separately

Step-by-Step Dataset Construction:
1. Extend the SEP dataset with information usefull for sampling (most recent 10-K filing date, Industry classifications)
2. Use different sampling techniques to get monthly observations
    1. At first use timebars (sampling at a fixed time interval), but try to respect the different fiscal years
3. Calculate the various price and volume based features
(4). Add inn SF1 and DAILY data
5. Compute features based on SF1
6. Select the features you want and combine into one ML ready dataset
"""

if __name__ == "__main__":

    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = "./logs/price_volume_features_" + date_time + ".log"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[logging.FileHandler(log_filename, mode="a"), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()


    samples_with_features_folder = "./datasets/timebar_samples_with_features/"
    filenames = [f for f in listdir(samples_with_features_folder) if isfile(join(samples_with_features_folder, f))]
    filenames.remove("None.csv")
    filenames.remove("Infrastructure Operations.csv")
    
    # filenames = ["Building Materials.csv"]

    # THE REST OF THE SCRIPT SHOULD BE DONE ONCE PER FILE
    for filename in filenames:
        try:
            samples_with_features_path = "./datasets/timebar_samples_with_features/" + filename
            sf1_art_path = "./datasets/industry_sf1_art/" + filename
            industry_daily_path = "./datasets/industry_daily/" + filename

            samples = pd.read_csv(samples_with_features_path)
            sf1_art = pd.read_csv(sf1_art_path)
            # daily = pd.read_csv(industry_daily_path) # daily data is not used as it is not based on ART (don't know exactly what, but I think MRT or ARY)

        except Exception as e:
            print(e)
            logger.error("# Failed to read a csv file for file name: {}, skipping this industry and trying the next.".format(filename))
            continue

        # Add SF1_ART
        result = pd.merge(samples, sf1_art, on=["ticker", "datekey"], how="left", suffixes=("_sample", "_sf1_art"), indicator=True)
        logger.debug("# File: {} - Sample and SF1_ART merge results: \n{}".format(filename, result._merge.value_counts()))
        result = result.drop(columns=["_merge"])
        
        
        # Add (create) DAILY
        for index, row in result.iterrows():
            result.at[index, "lastupdated_daily"] = row["lastupdated_sample"] # Should be the SEP date
            marketcap_daily = row["sharesbas"] * row["close"] * row["sharefactor"] #SharesBas * Price * ShareFactor
            result.at[index, "marketcap_daily"] = marketcap_daily
            ev_daily = marketcap_daily + row["debtusd"] + row["cashnequsd"] # MasketCap + DebtUSD - CashEqUSD
            result.at[index, "ev_daily"] = ev_daily
            
            if row["ebitusd"] != 0:
                result.at[index, "evebit_daily"] = ev_daily / row["ebitusd"] # EV / EBITUSD
            if row["ebitdausd"] != 0:
                result.at[index, "evebitda_daily"] = ev_daily / row["ebitdausd"] # EV / EBITDAUSD
            if row["equityusd"] != 0:
                result.at[index, "pb_daily"] = marketcap_daily / row["equityusd"] # MarketCap / EquityUSD
            if row["netinccmnusd"] != 0:
                result.at[index, "pe_daily"] = marketcap_daily / row["netinccmnusd"] # MaketCap / NetIncCmnUSD
            if row["revenueusd"] != 0:
                result.at[index, "ps_daily"] = marketcap_daily / row["revenueusd"] # MarketCap / RevenueUsd


        """ NOT USED BECAUSE DAILY DATA IS CALCULATED FROM SCRATCH USING SF1_ART DATA
        I don't use daily data...
        result = pd.merge(result, daily, on=["ticker", "date"], how="left", suffixes=("_result", "_daily"), indicator=True)
        logger.debug("# Result and DAILY merge results for file: {}: \n{}".format(filename, result._merge.value_counts()))
        result = result.rename(columns={"lastupdated": "lastupdated_daily"})


        for index, row in result.iterrows():
            if row["_merge"] == "left_only":
                # Need to add features that are missing from DAILY.

                result.at[index, "lastupdated_daily"] = row["lastupdated_sample"]
                marketcap_daily = row["sharesbas"] * row["close"] * row["sharefactor"] #SharesBas * Price * ShareFactor
                result.at[index, "marketcap_daily"] = marketcap_daily
                ev_daily = marketcap_daily + row["debtusd"] + row["cashnequsd"] # MasketCap + DebtUSD - CashEqUSD
                result.at[index, "ev_daily"] = ev_daily
                
                result.at[index, "evebit_daily"] = ev_daily / row["ebitusd"] # EV / EBITUSD
                result.at[index, "evebitda_daily"] = ev_daily / row["ebitdausd"] # EV / EBITDAUSD
                result.at[index, "pb_daily"] = marketcap_daily / row["equityusd"] # MarketCap / EquityUSD
                result.at[index, "pe_daily"] = marketcap_daily / row["netinccmnusd"] # MaketCap / NetIncCmnUSD
                result.at[index, "ps_daily"] = marketcap_daily / row["revenueusd"] # MarketCap / RevenueUsd

            elif row["_merge"] == "right_only":
                logger.warning("# File: {}, ticker: {} and Row: {} did not have a sample in sep (via samples_with_features)".format(filename, row["index"], index))
            else:
                # All good, dont need to do anything, but I can test my filler code
                marketcap_daily = row["sharesbas"] * row["close"] * row["sharefactor"]
                ev_daily = marketcap_daily + row["debtusd"] + row["cashnequsd"]
                
                if marketcap_daily != row["marketcap_daily"]:
                    logger.warning("# Marketcap was not equal. Calculated: {}, Given: {}".format(marketcap_daily, row["marketcap_daily"]))
                if ev_daily != row["ev_daily"]:
                    logger.warning("# EV was not equal. Calculated: {}, Given: {}".format(ev_daily, row["ev_daily"]))
        """
        
        # Drop indexes, this needs to be more sophisticated (when to drop and when to keep?)
        # result = result.dropna(axis=0)
 
        # Sort values by ticker, then date. Not sure if I need to, but...
        # result = result.sort_values(by=["ticker", "date"]) 

        logger.debug("In file: {} there are total rows: {}, market_cap number filled: {}".format(filename, len(result), result["marketcap_daily"].count()))
        logger.debug("In file: {} there are total rows: {}, ev_daily number filled: {}".format(filename, len(result), result["ev_daily"].count()))
        logger.debug("In file: {} there are total rows: {}, evebit_daily number filled: {}".format(filename, len(result), result["evebit_daily"].count()))
        logger.debug("In file: {} there are total rows: {}, evebitda_daily number filled: {}".format(filename, len(result), result["evebit_daily"].count()))
        logger.debug("In file: {} there are total rows: {}, pb_daily number filled: {}".format(filename, len(result), result["pb_daily"].count()))
        logger.debug("In file: {} there are total rows: {}, pe_daily number filled: {}".format(filename, len(result), result["pe_daily"].count()))
        logger.debug("In file: {} there are total rows: {}, ps_daily number filled: {}".format(filename, len(result), result["ps_daily"].count()))


        # Save
        result_dataset = Dataset.from_df(result)
        save_path = "./datasets/sep_sf1_daily_combined/" + filename
        result_dataset.to_csv(save_path)

        logger.debug("# Completed adding features for file: {}".format(filename))


"""

There is a possibility that return and momentum calculations are wrong, because I am not accounting for windows of missing pricing data.
Also I should include dividend somehow in the return calculation
Also the above code does not work, got error:

Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\envs\py36\lib\site-packages\pandas\core\indexes\base.py", line 3078, in get_loc
    return self._engine.get_loc(key)
  File "pandas\_libs\index.pyx", line 140, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 162, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1492, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1500, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'marketcap_daily'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "add_sf1_daily_features.py", line 123, in <module>
    logger.debug("In file: {} there are total rows: {}, market_cap number filled: {}".format(filename, len(result), result["marketcap_daily"].count()))
  File "C:\ProgramData\Anaconda3\envs\py36\lib\site-packages\pandas\core\frame.py", line 2688, in __getitem__
    return self._getitem_column(key)
  File "C:\ProgramData\Anaconda3\envs\py36\lib\site-packages\pandas\core\frame.py", line 2695, in _getitem_column
    return self._get_item_cache(key)
  File "C:\ProgramData\Anaconda3\envs\py36\lib\site-packages\pandas\core\generic.py", line 2489, in _get_item_cache
    values = self._data.get(item)
  File "C:\ProgramData\Anaconda3\envs\py36\lib\site-packages\pandas\core\internals.py", line 4115, in get
    loc = self.items.get_loc(item)
  File "C:\ProgramData\Anaconda3\envs\py36\lib\site-packages\pandas\core\indexes\base.py", line 3080, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas\_libs\index.pyx", line 140, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 162, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1492, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1500, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'marketcap_daily'

All Columns in the resulting dataset:
ticker
date
open
high
low
close
volume
dividends
closeunadj
lastupdated_sample
datekey
age
industry
sector
siccode
return
mom1m
mom6m
mom12m
chmom6m
dimension
calendardate
reportperiod
lastupdated_sf1_art
accoci
assets
assetsavg
assetsc
assetsnc
assetturnover
bvps
capex
cashneq
cashnequsd
cor
consolinc
currentratio
de
debt
debtc
debtnc
debtusd
deferredrev
depamor
deposits
divyield
dps
ebit
ebitda
ebitdamargin
ebitdausd
ebitusd
ebt
eps
epsdil
epsusd
equity
equityavg
equityusd
ev
evebit
evebitda
fcf
fcfps
fxusd
gp
grossmargin
intangibles
intexp
invcap
invcapavg
inventory
investments
investmentsc
investmentsnc
liabilities
liabilitiesc
liabilitiesnc
marketcap
ncf
ncfbus
ncfcommon
ncfdebt
ncfdiv
ncff
ncfi
ncfinv
ncfo
ncfx
netinc
netinccmn
netinccmnusd
netincdis
netincnci
netmargin
opex
opinc
payables
payoutratio
pb
pe
pe1
ppnenet
prefdivis
price
ps
ps1
receivables
retearn
revenue
revenueusd
rnd
roa
roe
roic
ros
sbcomp
sgna
sharefactor
sharesbas
shareswa
shareswadil
sps
tangibles
taxassets
taxexp
taxliabilities
tbvps
workingcapital
lastupdated_daily
marketcap_daily
ev_daily
evebit_daily
evebitda_daily
pb_daily
pe_daily
ps_daily

"""