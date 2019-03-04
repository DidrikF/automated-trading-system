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
(2). Use different sampling techniques to get monthly observations
    1. At first use timebars (sampling at a fixed time interval), but try to respect the different fiscal years
3. Calculate the various price and volume based features
4. Add inn SF1 and DAILY data
5. Compute features based on SF1
6. Select the features you want and combine into one ML ready dataset
"""


"""
IF I DO THIS AGAIN, I SHOULD CODE IT TO USE MULTIPROCESSING!!
"""
if __name__ == "__main__":

    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = "./logs/sample_observation_" + date_time + ".log"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[logging.FileHandler(log_filename, mode="a"), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()


    # THE REST OF THE SCRIPT SHOULD BE DONE ONCE PER FILE
    sampleable_path = "./datasets/sampleable/"
    filenames = [f for f in listdir(sampleable_path) if isfile(join(sampleable_path, f))]
    # None.csv and Industry Operations.csv was excluded at an earlier point.

    for filename in filenames:
        try:
            file_path = "./datasets/sampleable/" + filename
            observations = pd.read_csv(file_path)
        except Exception as e:
            print(e)
            logger.error("# Failed to read csv file with path: {}, skipping this file and trying the next.".format(file_path))
            continue

        observations["date"] = pd.to_datetime(observations["date"], format="%Y-%m-%d")
        observations["datekey"] = pd.to_datetime(observations["datekey"], format="%Y-%m-%d")


        columns = list(observations.columns)
        samples = pd.DataFrame(columns=columns)
        

        # Maybe treat each company in isolation, might be easier to reason about...
        tickers = list(observations.ticker.unique())

        last_ticker = None
        sample_indexes = list()
        len_tickers = len(tickers)

        for ticker_index, ticker in enumerate(tickers):
            ticker_observations = observations.loc[observations["ticker"] == ticker]
            logger.debug("Length of ticker_observations for {}: {}".format(ticker, len(ticker_observations)))
            
            # Variable initiation
            preceding_date = None
            preceding_index = None
            preceding_datekey = None
            base_date = None
            desired_date = None

            # Add date date of last 10-K filing and age of sf1_art data
            for cur_index, row in ticker_observations.iterrows():
                """
                Sampling strategy: Monthly best effort:
                Description: 
                Base the sampling at the date of the first recorded form 10-K filing. Sample as close as possible to a monthly frequency,
                this means that if the first filing is 15. jan, this will cause a sample. Further, if 15. feb is a sunday a new sample 
                will happen monday the 16. feb (as it is closer than firday 13. feb.).

                Assumptions:
                - Rows are sorted by ticker, then date
                - Looping over one company at a time
                """
                cur_date = row["date"]
                cur_datekey = row["datekey"]
                
                if last_ticker != ticker: # maybe reset the base_date when a new 10-K filing has happend
                    sample_indexes.append(cur_index)
                    logger.debug("# {} - sampled first observation at date: {}".format(ticker, cur_date))
                    base_date = cur_date
                    desired_date = base_date + relativedelta(months=+1)
                """
                elif preceding_datekey != cur_datekey:
                    #We have  a new 10-K filing and ART (TTM) data is updated
                """


                if cur_date < desired_date:
                    # update preceding date that may be the best day to sample
                    preceding_date = cur_date
                    preceding_index = cur_index # Hope this works ok
                elif cur_date == desired_date:
                    sample_indexes.append(cur_index)
                    # logger.debug("# {} - cur_date: {}, desired_date: {}, sampled date: {}".format(ticker, cur_date, desired_date, cur_date))
                    desired_date = desired_date + relativedelta(months=+1)
                elif cur_date > desired_date:
                    # Select closes date of preceding_date and cur_date
                    distance_preceding = abs((desired_date - preceding_date).total_seconds())
                    distance_cur = abs((desired_date - cur_date).total_seconds())
                    # print("Distances: ", distance_preceding, distance_cur)
                    
                    if distance_preceding <= distance_cur:
                        sample_indexes.append(preceding_index)
                        # logger.debug("# {} - cur_date: {}, preceding_date: {}, desired_date: {}, sampled date: {}".format(ticker, cur_date, preceding_date, desired_date, preceding_date))
                    else:
                        sample_indexes.append(cur_index)
                        # logger.debug("# {} - cur_date: {}, preceding_date: {}, desired_date: {}, sampled date: {}".format(ticker, cur_date, preceding_date, desired_date, cur_date))
                        
                    desired_date = desired_date + relativedelta(months=+1)

                last_ticker = ticker
            
            progress = ((ticker_index + 1) / len_tickers) * 100
            logger.debug("# Done with ticker: {}, progress: {}".format(ticker, progress))


        # Drop all but the sampled indexes
        drop_indexes = list(set(observations.index).difference(set(sample_indexes)))
        samples = observations.drop(drop_indexes)

        """
        # Detected that sampling failed after some date for two tickers (out of the 14000), don't know why, but add code to exclude them here:
        # Exclude EUENF after 2018-10-15
        # Exclude CNST1 after 2009-05-27
        # WARNING: THIS CODE HAS NOT BEEN TESTED YET, MANUAL REMOVAL WAS DONE TO AVOID RERUN OF CODE!!!
        if "EUENF" in tickers:
            euenf_samples = samples.loc[samples["ticker"] == "EUENF"]
            euenf_to_drop = euenf_samples.loc[euenf_samples["data"] > pd.to_datetime("2018-10-15", format="%Y-%m-%d")]
            indexes_to_drop = list(euenf_to_drop.index)
            samples_dataset = Dataset.from_df(samples)
        if "NCST1" in tickers:
            ncst1_samples = samples.loc[samples["ticker"] == "NCST1"]
            ncst1_to_drop = ncst1_samples.loc[ncst1_samples["data"] > pd.to_datetime("2009-05-27", format="%Y-%m-%d")]
            indexes_to_drop = list(ncst1_to_drop.index)
        """

        # Save:
        samples_dataset = Dataset.from_df(samples)
        save_path = "./datasets/timebar_samples/" + filename
        samples_dataset.to_csv(save_path)



#1: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
#2: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28
#3: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
#4: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

