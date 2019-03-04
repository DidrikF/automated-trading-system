import pandas as pd
import sys
from packages.helpers.helpers import print_exception_info
from packages.dataset_builder.dataset import Dataset
from dateutil.relativedelta import *
from os import listdir
from os.path import isfile, join

"""
write methods for testing and production!
"""



def extend_sep_for_sampling(index_filename: tuple, testing: bool):
    file_index = index_filename[0]
    filename = index_filename[1]

    result = {
        "error": list(),
        "metadata_not_available": set(),
        "message": "",
        "success": True,
        "drop_indexes": set()
    }

    #ticker_path = "./datasets/industry_tickers/" + filename
    sep_path = "./datasets/industry_sep/" + filename
    sf1_art_path = "./datasets/industry_sf1_art/" + filename
    sf1_arq_path = "./datasets/industry_sf1_arq/" + filename
    save_path = "./datasets/sampleable/" + filename

    tickers_metadata_path = "./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv"

    if testing == True:
        #ticker_path = "./datasets/testing/ticker.csv"
        sep_path = "./datasets/testing/sep_short.csv"
        sf1_art_path = "./datasets/testing/sf1_art.csv" 
        sf1_arq_path = "./datasets/testing/sf1_arq.csv"


    try:

        sep_df = pd.read_csv(sep_path, low_memory=False)
        sf1_art_df = pd.read_csv(sf1_art_path, low_memory=False)
        sf1_arq_df = pd.read_csv(sf1_arq_path, low_memory=False)
        tickers_metadata = pd.read_csv(tickers_metadata_path, low_memory=False)
        
    except Exception as e: 
        print_exception_info(e)
        sys.exit()

    sep_df["date"] = pd.to_datetime(sep_df["date"], format="%Y-%m-%d")
    sf1_art_df["datekey"] = pd.to_datetime(sf1_art_df["datekey"], format="%Y-%m-%d")

    sep_df.sort_values(by=["ticker", "date"], inplace=True)


    """
    tickers = list(sep_df.ticker.unique())
    for ticker in tickers:
        sep_for_ticker = sep_df.loc[sep_df["ticker"] == ticker]
    """

    last_ticker = None
    length = len(sep_df.index)
    # Add date date of last 10-K filing and age of sf1_art data
    for sep_index, sep_row in sep_df.iterrows():
        
        sep_ticker = sep_row["ticker"]
        sep_date = sep_row["date"]
        
        if last_ticker != sep_ticker:
            sf1_art_for_ticker = sf1_art_df.loc[sf1_art_df["ticker"] == sep_ticker] # THIS SHOULD ONLY BE DONE ONCE PER TICKER!!!
        
        # Extract past dates
        past_sf1_art_for_ticker = sf1_art_for_ticker.loc[sf1_art_for_ticker["datekey"] <= sep_date]
        
        # Get date of latest 10-K form filing
        date_of_latest_filing = past_sf1_art_for_ticker["datekey"].max()

        last_ticker = sep_ticker

        # Capture the indexes of rows with dates earlier than the first sf1_art entry
        if date_of_latest_filing is pd.NaT:
            result["drop_indexes"].add(sep_index)
            continue

        sep_df.at[sep_index, "datekey"] = date_of_latest_filing
        sep_df.at[sep_index, "age"] = (sep_date - date_of_latest_filing)

        # Add industry classification
        metadata = tickers_metadata.loc[tickers_metadata["ticker"] == sep_ticker]

        if metadata.empty: # this is checked every iteration!
            result["metadata_not_available"].add(sep_ticker)
            result["drop_indexes"].add(sep_index)
            continue

        sep_df.at[sep_index, "industry"] = metadata.iloc[-1]["industry"]
        sep_df.at[sep_index, "sector"] = metadata.iloc[-1]["sector"]
        sep_df.at[sep_index, "siccode"] = metadata.iloc[-1]["siccode"]
        
    # Drop rows with no prior form 10-K release or missing metadata
    sep_df = sep_df.drop(list(result["drop_indexes"]))
    
    if testing == True:
        return sep_df

    sampleable_dataset = Dataset.from_df(sep_df)
    sampleable_dataset.to_csv(save_path)
    
    return None


def first_filing_based_sampling(index_filename, sep_df, testing):
    file_index = index_filename[0]
    filename = index_filename[1]

    sampleable_path = "./datasets/sampleable/" + filename
    
    if testing == True:
        observations = sep_df
    else:
        try:
            observations = pd.read_csv(sampleable_path)
        except Exception as e:
            print_exception_info(e)
            sys.exit()

    observations["date"] = pd.to_datetime(observations["date"]) # , format="%Y-%m-%d"
    observations["datekey"] = pd.to_datetime(observations["datekey"]) # , format="%Y-%m-%d"


    columns = list(observations.columns)
    samples = pd.DataFrame(columns=columns)
    

    tickers = list(observations.ticker.unique())

    last_ticker = None
    sample_indexes = list()
    len_tickers = len(tickers)

    for ticker_index, ticker in enumerate(tickers):
        ticker_observations = observations.loc[observations["ticker"] == ticker]
        
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
            # cur_datekey = row["datekey"]

            if last_ticker != ticker: # maybe reset the base_date when a new 10-K filing has happend
                sample_indexes.append(cur_index)
                base_date = cur_date
                desired_date = base_date + relativedelta(months=+1)

            if cur_date < desired_date:
                # update preceding date that may be the best day to sample
                preceding_date = cur_date
                preceding_index = cur_index # Hope this works ok
            elif cur_date == desired_date:
                sample_indexes.append(cur_index)
                desired_date = desired_date + relativedelta(months=+1)
            elif cur_date > desired_date:
                # Select closes date of preceding_date and cur_date
                distance_preceding = abs((desired_date - preceding_date).total_seconds())
                distance_cur = abs((desired_date - cur_date).total_seconds())
                
                if distance_preceding <= distance_cur:
                    sample_indexes.append(preceding_index)
                else:
                    sample_indexes.append(cur_index)
                    
                desired_date = desired_date + relativedelta(months=+1)

            last_ticker = ticker
        
    # Drop all but the sampled indexes
    drop_indexes = list(set(observations.index).difference(set(sample_indexes)))
    samples = observations.drop(drop_indexes)

    if testing == True:
        return samples

    
    # Save:
    samples_dataset = Dataset.from_df(samples)
    save_path = "./datasets/timebar_samples/" + filename
    samples_dataset.to_csv(save_path)

    return None



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


def rebase_at_each_filing_sampling(index_filename, days_of_distance, observations, testing):
    """
    Sample dates such that the sampling is based of a monthly interval since last form 10 filing
    If there are less than 20 days since last sample when a new form 10 filing is available, 
    drop that sample and sample the current date (date of form 10 filing). This ensures overlap 
    of no more than 10-11 (can be controlled by "days_of_distance") day for all samples and 
    that samples are made as close to form 10 filings as possible.

    This function assumes that all observations have a datekey. In other words: SEP dates before
    the first datekey in SF1 has been removed. This function also assumes that indexes are reset.
    """

    file_index = index_filename[0]
    filename = index_filename[1]

    sampleable_path = "./datasets/sampleable/" + filename
    
    if testing == True:
        observations = observations
    else:
        try:
            observations = pd.read_csv(sampleable_path)
        except Exception as e:
            print_exception_info(e)
            sys.exit()

    observations["date"] = pd.to_datetime(observations["date"]) # , format="%Y-%m-%d"
    observations["datekey"] = pd.to_datetime(observations["datekey"]) # , format="%Y-%m-%d"


    columns = list(observations.columns)
    samples = pd.DataFrame(columns=columns)
    

    tickers = list(observations.ticker.unique())

    last_ticker = None
    sample_indexes = list()
    len_tickers = len(tickers)

    # Variable initiation
    base_date = None
    desired_date = None
    previous_date = None
    previous_index = None
    last_sample_index = None
    last_sample = None
    last_sample_date = None

    for ticker_index, ticker in enumerate(tickers):
        ticker_observations = observations.loc[observations["ticker"] == ticker]

        for cur_index, row in ticker_observations.iterrows():
            cur_date = row["date"]
            cur_datekey = row["datekey"]


            if base_date == None:
                base_date = cur_datekey
                sample_indexes.append(cur_index) # Don't sample if datekey is not available
                desired_date = base_date + relativedelta(months=+1)
                continue
            # We have to have sampled once for the current ticker before we get this far
            elif cur_datekey != base_date: 
                # New filing!
                # I need to drop the last sample if the overlap is too great. 
                last_sample_index = sample_indexes[-1]
                last_sample = observations.iloc[last_sample_index]
                last_sample_date = last_sample["date"]
                if last_sample_date > (cur_datekey - relativedelta(days=+days_of_distance)):
                    # It is less than 20 days since last sample, so drop the old sample.
                    sample_indexes.pop(-1)

                # I want so sample immediately and base future samples of this sample
                base_date = cur_datekey
                sample_indexes.append(cur_index)
                desired_date = base_date + relativedelta(months=+1)
                continue

            if cur_date < desired_date:
                previous_date = cur_date
                previous_index = cur_index
            elif cur_date == desired_date:
                sample_indexes.append(cur_index)
                desired_date = desired_date + relativedelta(months=+1)
            elif cur_date > desired_date:
                # We need to deside wether to sample the previous date or the current date.
                distance_preceding = abs((desired_date - previous_date).total_seconds())
                distance_cur = abs((desired_date - cur_date).total_seconds())
                
                if distance_preceding <= distance_cur:
                    sample_indexes.append(previous_index)
                else:
                    sample_indexes.append(cur_index)
                    
                desired_date = desired_date + relativedelta(months=+1)

        # Reset state for next ticker
        base_date = None
        desired_date = None
        previous_date = None
        previous_index = None
        last_sample_index = None
        last_sample = None
        last_sample_date = None


    drop_indexes = list(set(observations.index).difference(set(sample_indexes)))
    samples = observations.drop(drop_indexes)

    if testing == True:
        return samples

    
    # Save:
    samples_dataset = Dataset.from_df(samples)
    save_path = "./datasets/timebar_samples/" + filename
    samples_dataset.to_csv(save_path)

    return None

def remove_gaps_from_sampled_data(samples, allowed_gap_size):
    """
    This function ensures that all sampled data do not have any gaps greater that one month 
    (can be controlled by "allowed_gap_size").

    OBS: I might not need this if I get price data directly from the source when calculating
    features based on SEP data.
    """


