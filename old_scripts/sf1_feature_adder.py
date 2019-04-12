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
3. Calculate the various price and volume based features (beta, momentum, return, change in momentum)
(4). Compute features based on SF1
5. Combine SEP, SF1 and DAILY data
6. Select the features and industries you want and combine (industry files and columns) into one ML ready dataset
"""

if __name__ == "__main__":

    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = "./logs/sf1_feature_adder_" + date_time + ".log"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[logging.FileHandler(log_filename, mode="a"), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()


    sf1_folder = "./datasets/industry_sf1_art/"
    
    filenames = [f for f in listdir(sf1_folder) if isfile(join(sf1_folder, f))]
    filenames = ["Building Materials.csv"]

    # THE REST OF THE SCRIPT SHOULD BE DONE ONCE PER FILE
    for filename in filenames:
        try:
            sf1_path = sf1_folder + filename
            sf1 = pd.read_csv(sf1_path)
        except Exception as e:
            print(e)
            logger.error("# Failed to read a csv file for file name: {}, skipping this industry and trying the next.".format(filename))
            continue

        # Parse dates
        sf1["datekey"] = pd.to_datetime(sf1["datekey"], format="%Y-%m-%d")


        # Splitting it into tickers allows for the use of .shift() to do calculations (Need to verify absence of gaps in time)
        tickers = list(sf1.ticker.unique())

        for ticker in tickers:
            # All sf1_art entries for each ticker sorted by date:
            sf1_for_ticker = sf1.loc[sf1["ticker"] == ticker]
            
            i = 0

            for sf1_index, sf1_row in sf1_for_ticker.iterrows():
                """
                i += 1
                if i > 300: 
                    print(sf1.head(300)["asset_growth"])
                    sys.exit()
                """
                # Calculate features only needing the current row
                
                # Feature: Book-to-Market:
                if sf1_row["marketcap"] != 0:
                    sf1.at[sf1_index, "bm"] = sf1_row["equity"] / sf1_row["marketcap"]




                datekey = sf1_row["datekey"]
                datekey_less_1_year = datekey - relativedelta(months=+11)
                datekey_less_2_years = datekey - relativedelta(months=+22)

                sf1_1_year_or_older = sf1_for_ticker.loc[sf1_for_ticker["datekey"] <= datekey_less_1_year]
                datekey_closest_to_1_year_ago = sf1_1_year_or_older["datekey"].max()

                sf1_2_years_or_older = sf1_for_ticker.loc[sf1_for_ticker["datekey"] <= datekey_less_2_years]
                datekey_closest_to_2_years_ago = sf1_2_years_or_older["datekey"].max()

                # print("datekey: {}, datekey_closes_to_1_year_ago: {}, datekey_closest_to_2_years_ago: {}".format(datekey, datekey_closest_to_1_year_ago, datekey_closest_to_2_years_ago))



                if datekey_closest_to_1_year_ago is not pd.NaT and datekey_closest_to_1_year_ago > (datekey - relativedelta(months=+13)):
                    # Calculate features using 1 year old data
                    one_y_old_row = sf1_for_ticker[sf1_for_ticker.datekey == datekey_closest_to_1_year_ago].iloc[0]

                    # Do calculations with current row (row) and 1 year old row (one_y_old_row)...
                    if sf1_row["assets"] != 0 and one_y_old_row["assets"] != 0:
                        sf1.at[sf1_index, "asset_growth"] = ( sf1_row["assets"] - one_y_old_row["assets"] ) / one_y_old_row["assets"]

                else:
                    # 10-K filing is to old, so cannot calculate features relying on 1 year old data
                    # logger.info("# File: {}, Ticker: {} - do not have 1 year old data at date (datekey): {}".format(filename, ticker, datekey))
                    pass

                
                
                if datekey_closest_to_2_years_ago is not pd.NaT and datekey_closest_to_2_years_ago > (datekey - relativedelta(months=+26)) and datekey_closest_to_1_year_ago is not pd.NaT and datekey_closest_to_1_year_ago > (datekey - relativedelta(months=+13)):
                    # Calculate features using 2 year old data
                    one_y_old_row = sf1_for_ticker[sf1_for_ticker.datekey == datekey_closest_to_1_year_ago].iloc[0]
                    two_y_old_row = sf1_for_ticker[sf1_for_ticker.datekey == datekey_closest_to_2_years_ago].iloc[0]

                    # NOTE: Not sure about the time indexing presented in the paper for this feature, I add in both current and shifter 1 year back...
                    if one_y_old_row["assets"] != 0 and two_y_old_row["assets"] != 0:
                        sf1.at[sf1_index, "asset_growth_1y"] = ( one_y_old_row["assets"] - two_y_old_row["assets"] ) / two_y_old_row["assets"]
                    
                else:
                    # 10-K filing is too old, so cannot calculate features relying on 2 year old data
                    # logger.info("# File: {}, Ticker: {} - do not have 2 year old data at date (datekey): {}".format(filename, ticker, datekey))
                    pass



        # Calculate Industry Adjusted Features:
        for ticker in tickers:
            # All sf1_art entries for each ticker sorted by date:
            sf1_for_ticker = sf1.loc[sf1["ticker"] == ticker]
            
            # Calculate Industry averages for later use...
            mean_bm = sf1["bm"].mean()
            
            for sf1_index, sf1_row in sf1_for_ticker.iterrows():
                sf1.at[sf1_index, "bm_ia"] = sf1_row["bm"] - mean_bm

        # Report results:
        sf1_length = len(sf1)

        logger.debug("# Total rows: {} - Nr Asset Growth (asset_growth) calculated: {}".format(sf1_length, sf1["asset_growth"].count(),))
        logger.debug("# Total rows: {} - Nr Asset Growth last year (asset_growth_1y) calculated: {}".format(sf1_length, sf1["asset_growth_1y"].count()))
        logger.debug("# Total rows: {} - Nr Book to Market (bm) calculated: {}".format(sf1_length, sf1["asset_growth_1y"].count()))
        logger.debug("# Total rows: {} - Nr Industry adjusted Book to Market (bm_ia) calculated: {}".format(sf1_length, sf1["bm_ia"].count()))
        


        # Save
        sf1_dataset = Dataset.from_df(sf1)
        save_path = "./datasets/industry_sf1_art_extended/" + filename
        sf1_dataset.to_csv(save_path)
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
