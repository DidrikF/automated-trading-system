import pandas as pd
import sys
from dateutil.relativedelta import *
from os import listdir
from os.path import isfile, join

from helpers.helpers import print_exception_info


def extend_sep_for_sampling(sep, sf1_art, metadata):

    """
    NOTE: Data is given per ticker, Require sep to be sorted, sf1_art has a calendardate index

    Observations (sep rows) gets added the latest datekey for the latest reporting 
    (normalized reporting period is given by "calendardate") period.

    calendardate    datekey
    2010-06-30      2010-08-10 <- I want this
    2010-03-30      2010-08-11
    """

    if len(metadata) == 0:
        if len(sep) > 0:
            print("No metadata for ticker: ", sep.iloc[0]["ticker"])
        else:
            print("No metadata for some unknown ticker (because sep was also empty) was given to extend_sep_for_sampling. Don't know why.")
        
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

    for date, sep_row in sep.iterrows():
        # Get the date of the latest 10-K form filing with the maximum calendar date
        past_sf1_art = sf1_art.loc[sf1_art.datekey <= date]
        
        try:
            max_calendardate = past_sf1_art.index.max() # This may return NaT when past_sf1_art is empty...
            past_sf1_art = past_sf1_art.loc[[max_calendardate]]
        except KeyError:
            pass

        date_of_latest_filing = past_sf1_art.datekey.max()

        # Capture the indexes of rows with dates earlier than the first sf1_art entry
        if date_of_latest_filing is pd.NaT:
            drop_indexes.add(date)
            continue

        sep.at[date, "datekey"] = date_of_latest_filing
        sep.at[date, "age"] = (date - date_of_latest_filing)
        sep.at[date, "sharesbas"] = sf1_art.loc[sf1_art.datekey == date_of_latest_filing].iloc[-1]["sharesbas"] # Needed when calculating std_turn in sep_features.py
        
    # Drop rows with no prior form 10-K release or missing metadata
    sep = sep.drop(list(drop_indexes))

    return sep


def rebase_at_each_filing_sampling(observations, days_of_distance):
    """
    Sample dates such that the sampling is based of a monthly interval since last form 10 filing for the most recent report period.
    If there are less than 20 days since last sample when a new form 10 filing is available, 
    drop that sample and sample the current date (date of form 10 filing). This ensures overlap 
    of no more than 10-11 (can be controlled by "days_of_distance") days for all samples and 
    that samples are made as close to a relevant form 10 filings as possible.

    Because datekeys are set such that re-filings of reports for a calendardate (normalized report period) earlier than the current one 
    reports for the current period or a new period shows up as a new datekey when iterating over the 
    observations. New datekeys is therefore allways grounds to rebase the sampling process.

    NOTE: This function assumes that all observations have a datekey. In other words: SEP dates before
    the first datekey in SF1 has been removed. This function also assumes that indexes are reset.
    """

    # It could be that the dataframe is empty or that it is missing
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
    sample relative to cur_date and reset on every new datekey seen when iterating over the observations (sep rows).
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

        if cur_date == desired_date: # We might not have data for the exact desired date, it could be a saturday...
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
