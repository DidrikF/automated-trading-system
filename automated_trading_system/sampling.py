import sys

import pandas as pd
from dateutil.relativedelta import *

from automated_trading_system.dataset_builder.dataset import Dataset
from automated_trading_system.helpers.helpers import print_exception_info


def extend_sep_for_sampling(sep, sf1_art, metadata):
    """
    Data is given per ticker
    Require sep to be sorted
    sf1_art has a calendardate index
    """

    if len(metadata) == 0:
        if len(sep) > 0:
            print("No metadata for ticker: ", sep.iloc[0]["ticker"])
        else:
            print(
                "No metadata for some unknown ticker (because sep was also empty) was given to extend_sep_for_sampling. Don't know why.")

        sep = pd.DataFrame(data=None, columns=sep.columns, index=sep.index)
        sep = sep.dropna(axis=0)

        return sep

    if isinstance(metadata, pd.DataFrame):
        metadata = metadata.iloc[-1]

    ticker = metadata["ticker"]

    sep.loc[:, "industry"] = metadata["industry"]
    sep.loc[:, "sector"] = metadata["sector"]
    sep.loc[:, "siccode"] = metadata["siccode"]

    drop_indexes = set()

    # Add date date of last 10-K filing and age of sf1_art data
    for date, sep_row in sep.iterrows():

        """
        I want each sample to have associated with it, the most up to date 

        If a new report (datekey) is recorded, but it is for a 10K/Q that was released 6 months ago, it is not the one
        I want.
        I want the most up to date (datekey) for a report for the latest period

        date
        2010-08-12

        calendardate    datekey
        2010-06-30      2010-08-10 <- I want this
        2010-03-30      2010-08-11

        Need to update, but will postpone to later
        """

        # Extract past dates
        past_sf1_art = sf1_art.loc[sf1_art.datekey <= date]

        # Get date of latest 10-K form filing
        date_of_latest_filing = past_sf1_art.datekey.max()

        # Capture the indexes of rows with dates earlier than the first sf1_art entry
        if date_of_latest_filing is pd.NaT:
            drop_indexes.add(date)
            continue

        sep.at[date, "datekey"] = date_of_latest_filing
        sep.at[date, "age"] = (date - date_of_latest_filing)
        sep.at[date, "sharesbas"] = sf1_art.loc[sf1_art.datekey == date_of_latest_filing].iloc[-1][
            "sharesbas"]  # Needed when calculating std_turn in sep_features.py

    # Drop rows with no prior form 10-K release or missing metadata
    sep = sep.drop(list(drop_indexes))

    return sep


def rebase_at_each_filing_sampling(observations, days_of_distance):
    """
    NOTE: The sampling strategy should take calendardate into account, presently it does not.

    Sample dates such that the sampling is based of a monthly interval since last form 10 filing
    If there are less than 20 days since last sample when a new form 10 filing is available, 
    drop that sample and sample the current date (date of form 10 filing). This ensures overlap 
    of no more than 10-11 (can be controlled by "days_of_distance") day for all samples and 
    that samples are made as close to form 10 filings as possible.

    This function assumes that all observations have a datekey. In other words: SEP dates before
    the first datekey in SF1 has been removed. This function also assumes that indexes are reset.
    """

    # It could be that the dataframe is empty or that it is missing, maybe?
    # print(observations.head())
    observations_empty = True if (len(observations) == 0) else False

    if observations_empty == True:
        print("got empty dataframe 'observations' in rebase_at_each_filing_sampling. Don't know why.")
        return observations

    observations["datekey"] = pd.to_datetime(observations["datekey"])
    sample_indexes = list()

    # Variable initiation
    first_date = observations.index[0]
    base_date = None
    desired_date = None
    previous_date = None
    last_sample_date = None
    most_recent_filing = None

    """
    sample relative to cur_date and reset on every new filing.
    """

    for cur_date, row in observations.iterrows():
        cur_datekey = row["datekey"]
        # Initialize at the first iteration.
        if cur_date == first_date:
            base_date = cur_date
            most_recent_filing = cur_datekey
            sample_indexes.append(cur_date)
            desired_date = base_date + relativedelta(months=1)
            previous_date = cur_date
            continue

        # We have to have sampled once for the current ticker before we get this far
        if cur_datekey != most_recent_filing:
            # New filing!

            # I need to drop the last sample if the overlap is too great. 
            if len(sample_indexes) > 0:
                last_sample_date = sample_indexes[-1]
                if last_sample_date > (cur_date - relativedelta(days=days_of_distance)):
                    # It is less than 20 days since last sample, so drop the old sample.
                    sample_indexes.pop(-1)

            # I want so sample immediately and base future samples of this sample
            sample_indexes.append(cur_date)
            base_date = cur_date
            desired_date = base_date + relativedelta(months=+1)
            most_recent_filing = cur_datekey
            continue

        if cur_date == desired_date:  # We might not have data for the exact desired date, it could be a saturday...
            sample_indexes.append(cur_date)
            desired_date = desired_date + relativedelta(months=+1)
        elif cur_date > desired_date:
            """
            # It could be that SEP data is available from a data much later than SF1. In this case cur_date does not get to be
            # greater than desired_date before this code in this block runs. Therefore we need to take into account the case
            # where previous_date is None. When previous_data is None, we want only to increment desired_date and not sample.
            # We increment desired date
            if previous_date is None: # This should only trigger the very first iteration if cur_date is greater than desired_date from the beginning.
                sample_indexes.append(cur_date)
                previous_date = cur_date
                desired_date = cur_date + relativedelta(months=+1) # Shifts the base_date to the first sep date
                continue
            """

            # We need to deside wether to sample the previous date or the current date.
            distance_preceding = abs((desired_date - previous_date).total_seconds())
            distance_cur = abs((desired_date - cur_date).total_seconds())

            if distance_preceding <= distance_cur:
                sample_indexes.append(previous_date)
            else:
                sample_indexes.append(cur_date)

            desired_date = desired_date + relativedelta(months=+1)

        previous_date = cur_date

    drop_indexes = list(set(observations.index).difference(set(sample_indexes)))
    samples = observations.drop(drop_indexes)

    return samples


def rebase_at_each_filing_sampling_OLD(observations, days_of_distance):
    """
    NOTE: The sampling strategy should take calendardate into account, presently it does not.

    Sample dates such that the sampling is based of a monthly interval since last form 10 filing
    If there are less than 20 days since last sample when a new form 10 filing is available, 
    drop that sample and sample the current date (date of form 10 filing). This ensures overlap 
    of no more than 10-11 (can be controlled by "days_of_distance") day for all samples and 
    that samples are made as close to form 10 filings as possible.

    This function assumes that all observations have a datekey. In other words: SEP dates before
    the first datekey in SF1 has been removed. This function also assumes that indexes are reset.
    """

    observations["datekey"] = pd.to_datetime(observations["datekey"])
    sample_indexes = list()

    # Variable initiation
    base_date = None
    desired_date = None
    previous_date = None
    last_sample_date = None

    for cur_date, row in observations.iterrows():
        cur_datekey = row["datekey"]

        if base_date == None and type(cur_datekey) == pd.datetime:
            base_date = cur_datekey
            # sample_indexes.append(cur_date) # Don't sample if datekey is not available
            desired_date = base_date + relativedelta(months=+1)
            continue
        # We have to have sampled once for the current ticker before we get this far
        elif cur_datekey != base_date:
            # New filing!

            # I need to drop the last sample if the overlap is too great. 
            if len(sample_indexes) > 0:
                last_sample_date = sample_indexes[-1]
                if last_sample_date > (cur_datekey - relativedelta(days=+days_of_distance)):
                    # It is less than 20 days since last sample, so drop the old sample.
                    sample_indexes.pop(-1)

            # I want so sample immediately and base future samples of this sample
            base_date = cur_datekey  # This might still be long before sep data became available
            sample_indexes.append(cur_date)
            desired_date = base_date + relativedelta(months=+1)
            continue

        if cur_date < desired_date:
            previous_date = cur_date
        elif cur_date == desired_date:  # We might not have data for the exact desired date, it could be a saturday...
            sample_indexes.append(cur_date)
            desired_date = desired_date + relativedelta(months=+1)
        elif cur_date > desired_date:
            # We need to deside wether to sample the previous date or the current date.
            distance_preceding = abs((desired_date - previous_date).total_seconds())
            distance_cur = abs((desired_date - cur_date).total_seconds())

            if distance_preceding <= distance_cur:
                sample_indexes.append(previous_date)
            else:
                sample_indexes.append(cur_date)

            desired_date = desired_date + relativedelta(months=+1)

    drop_indexes = list(set(observations.index).difference(set(sample_indexes)))
    samples = observations.drop(drop_indexes)

    return samples


def remove_gaps_from_sampled_data(samples, allowed_gap_size):
    """
    This function ensures that all sampled data do not have any gaps greater that one month 
    (can be controlled by "allowed_gap_size").

    OBS: I might not need this if I get price data directly from the source when calculating
    features based on SEP data.
    """
    pass


# OBS: Need to rewrite and test, but I don't think I will end up needing this function.
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

    observations["date"] = pd.to_datetime(observations["date"])  # , format="%Y-%m-%d"
    observations["datekey"] = pd.to_datetime(observations["datekey"])  # , format="%Y-%m-%d"

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

            if last_ticker != ticker:  # maybe reset the base_date when a new 10-K filing has happend
                sample_indexes.append(cur_index)
                base_date = cur_date
                desired_date = base_date + relativedelta(months=+1)

            if cur_date < desired_date:
                # update preceding date that may be the best day to sample
                preceding_date = cur_date
                preceding_index = cur_index  # Hope this works ok
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


"""
Traceback (most recent call last):
  File "/home/ubuntu/anaconda3/envs/master/lib/python3.6/multiprocessing/pool.py", line 119, in worker
    result = (True, func(*args, **kwds))
  File "/home/ubuntu/pycode/automated-trading-system/packages/multiprocessing/engine.py", line 272, in expandCall_fast
    out = callback(**kwargs)
  File "/home/ubuntu/pycode/automated-trading-system/sampling.py", line 127, in rebase_at_each_filing_sampling
    distance_preceding = abs((desired_date - previous_date).total_seconds())
TypeError: unsupported operand type(s) for -: 'Timestamp' and 'NoneType'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "final_dataset_generation.py", line 232, in <module>
    molecules_per_process=2, resume=True)
  File "/home/ubuntu/pycode/automated-trading-system/packages/multiprocessing/engine.py", line 430, in pandas_chaining_mp_engine
    primary_molecules = process_jobs_fast(jobs, num_processes=num_processes, sort_by=sort_by) # return as list of Data Frames, I need a dict
  File "/home/ubuntu/pycode/automated-trading-system/packages/multiprocessing/engine.py", line 255, in process_jobs_fast
    for out_ in outputs:
  File "/home/ubuntu/anaconda3/envs/master/lib/python3.6/multiprocessing/pool.py", line 735, in next
    raise value
TypeError: unsupported operand type(s) for -: 'Timestamp' and 'NoneType'
"""
