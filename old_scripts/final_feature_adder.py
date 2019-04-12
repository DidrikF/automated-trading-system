import sys
from automated_trading_system.dataset_builder.dataset import Dataset, merge_datasets_simple
from automated_trading_system.logger.logger import Logger
from automated_trading_system.helpers.helpers import print_exception_info

from automated_trading_system.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from automated_trading_system.helpers.custom_exceptions import FeatureError
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
4. Add inn SF1 and DAILY data
(5). Compute features based on SF1
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


    # filenames = ["Building Materials.csv"]

    # THE REST OF THE SCRIPT SHOULD BE DONE ONCE PER FILE
    for filename in filenames:
        try:
            samples_path = "./datasets/sep_sf1_daily_combined/" + filename

            samples = pd.read_csv(samples_path)

        except Exception as e:
            print(e)
            logger.error("# Failed to read a csv file for file name: {}, skipping this industry and trying the next.".format(filename))
            continue

        # Calculate Industry averages for later use...


        # Splitting it into tickers allows for the use of .shift() to do calculations (Need to verify absence of gaps in time)
        tickers = list(samples.ticker.unique())
        for ticker in tickers:
            ticker_samples = samples.loc[samples["ticker"] == ticker]

            # This will not work, because 
            # ticker_samples["asset_growth"] = ( ticker_samples["assets"].shift(1) - ticker_samples["assets"].shift(2) ) / ticker_samples["assets"].shift(2)

            """
            for index, ticker_row in ticker_samples.iterrows():
                samples.at[index, "asset_growth"] = 
            """
            
            # Drop indexes, this needs to be more sophisticated (when to drop and when to keep?)
            # result = result.dropna(axis=0)

            # Sort values by ticker, then date. Not sure if I need to, but...
            # result = result.sort_values(by=["ticker", "date"]) 

        # Save
        sample_dataset = Dataset.from_df(samples)
        save_path = "./datasets/sep_sf1_daily_combined/" + filename
        sample_dataset.to_csv(save_path)

        logger.debug("# Completed adding features for file: {}".format(filename))


"""

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





"""
basic_columns_config = {
    'book_value': book_value,
}


# Can count on the presence of the columns specified in basic_columns_config.
columns_config = {
    'book_to_market': book_to_market,
    'cash_holdings': cash_holdings,
}

print(dataset.info())

# Add basic columns on which other new columns rely on.

for column in basic_columns_config:
    dataset.data[column] = None

for index in dataset.data.index:
    row = dataset.data.iloc[index]
    
    for column, func in basic_columns_config.items():
        # func may not be computable for some INDEXES (first year of data, missing value, etc.)
        try:
            value = func(index, row, dataset.data)
        except FeatureError as e:
            dataset.data.at[index, column] = None # Or something else? think about how it is saved and respored later
            logger.log_exc(e)
            continue

        dataset.data.at[index, column] = value




# Add features to go into the final dataset.
# Add columns
for column in columns_config:
    dataset.data[column] = None

for index in dataset.data.index:
    row = dataset.data.iloc[index]
    
    for column, func in columns_config.items():
        # func may not be computable for some INDEXES (first year of data, missing value, etc.)
        try:
            value = func(index, row, dataset.data)
        except FeatureError as e: 
            logger.log_exc(e)
            continue

        dataset.data.at[index, column] = value


"""
