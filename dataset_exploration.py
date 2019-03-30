import pandas as pd
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import numpy as np
from packages.multiprocessing.engine import pandas_mp_engine
from packages.helpers.helpers import get_calendardate_index, forward_fill_gaps

def detect_gaps(df):
    df.sort_index()
    df.index.drop_duplicates(keep="last")
    df["calendardate"] = df.index
    df['diff'] = df["calendardate"].diff(1)
    df["diff"] = df.index.to_series().diff().dt.total_seconds().fillna(0) / (60*60*24)

    df['OVER 1q'] = df["diff"] > 100
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(df["calendardate"])

    df['OVER 4q'] = df["diff"] > 370

    df1 = df.loc[df["OVER 1q"]==True]
    df4 = df.loc[df["OVER 4q"] == True]

    return (df1, df4)

def report_gaps(sf1):

    gaps_1q, gaps_4q = detect_gaps(sf1)

    report = pd.DataFrame(columns=["ticker", "gaps over 1q", "gaps over 4q"])

    if len(gaps_1q) > 0:
        report.at[0, "ticker"] = sf1.iloc[0]["ticker"]
        report.at[0, "gaps over 1q"] = len(gaps_1q)
    if len(gaps_4q) > 0:
        report.at[0, "gaps over 4q"] = len(gaps_4q)

    return report


def report_updates(sf1):
    ticker = sf1.iloc[0]["ticker"]
    vals = sf1.index.value_counts()
    vals = vals.rename("count")
    vals = vals.to_frame()
    vals["count"] = pd.to_numeric(vals["count"])
    report = pd.DataFrame()
    i = 0
    for caldate, count in vals.iterrows():
        if count[0] > 1:
            report.at[i, "calendardate"] = caldate
            report.at[i, "ticker"] = ticker
            report.at[i, "count"] = count[0]
            i += 1
    return report



def report_gaps_after_fill(sf1):
    sf1 = forward_fill_gaps(sf1, 3)

    gaps_1q, gaps_4q = detect_gaps(sf1)
    report = pd.DataFrame(columns=["ticker", "gaps over 1q", "gaps over 4q"])

    if len(gaps_1q) > 0:
        report.at[0, "ticker"] = sf1.iloc[0]["ticker"]
        report.at[0, "gaps over 1q"] = len(gaps_1q)
    if len(gaps_4q) > 0:
        report.at[0, "gaps over 4q"] = len(gaps_4q)

    return report


def report_updates_after_fill(sf1):
    sf1 = forward_fill_gaps(sf1, 3)

    ticker = sf1.iloc[0]["ticker"]
    vals = sf1.index.value_counts()
    vals = vals.rename("count")
    vals = vals.to_frame()
    vals["count"] = pd.to_numeric(vals["count"])
    report = pd.DataFrame()
    i = 0
    for caldate, count in vals.iterrows():
        if count[0] > 1:
            report.at[i, "calendardate"] = caldate
            report.at[i, "ticker"] = ticker
            report.at[i, "count"] = count[0]
            i += 1
    return report

def report_duplicate_datekeys(sf1):
    ticker = sf1.iloc[0].ticker
    
    duplicates = sf1.duplicated(subset="datekey")
    any_duplicates = any(duplicates==True)

    return pd.DataFrame(data=[any_duplicates], index=[ticker])




def report_date_relationship(sf1):
    ticker = sf1.iloc[0]["ticker"]

    sf1["datekey_after_caldate"] = sf1["datekey"] >sf1["calendardate"]

    sf1_selected = sf1.loc[sf1["datekey_after_caldate"] == False]
    report = pd.DataFrame()

    if len(sf1_selected) > 0:
        report.at[ticker, "datekey_after_caldate"] = False
    else:
        report.at[ticker, "datekey_after_caldate"] = True


    return report


if __name__ == "__main__":
    
    """
    # ./datasets/sharadar/SHARADAR_SF1_ARQ.csv
    sf1_arq = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ARQ.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"], index_col="calendardate")
    

    update_report_arq = pandas_mp_engine(callback=report_updates, atoms=sf1_arq, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    gap_report_arq = pandas_mp_engine(callback=report_gaps, atoms=sf1_arq, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    duplicate_report_arq = pandas_mp_engine(callback=report_duplicate_datekeys, atoms=sf1_arq, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    
    update_report_arq.to_csv("./datasets/testing/update_report_arq.csv", index=False)
    gap_report_arq.to_csv("./datasets/testing/gap_report_arq.csv", index=False)
    duplicate_report_arq.to_csv("./datasets/testing/duplicates_report_sf1_arq.csv")
    """


    """
    # "./datasets/sharadar/SHARADAR_SF1_ART.csv"
    # "./datasets/testing/sf1_art.csv"


    # Forward fill up to three quarters and see how it looks after
    # Needs to be done per ticker, so this work is moved into report* functions.
    


    update_report_art = pandas_mp_engine(callback=report_updates_after_fill, atoms=sf1_art, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    gap_report_art = pandas_mp_engine(callback=report_gaps_after_fill, atoms=sf1_art, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    duplicate_report_art = pandas_mp_engine(callback=report_duplicate_datekeys, atoms=sf1_art, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    # WRITTEN FOR FILLED VERSION
    update_report_art.to_csv("./datasets/testing/update_report_art_filled.csv", index=False)
    gap_report_art.to_csv("./datasets/testing/gap_report_art_filled.csv", index=False)
    duplicate_report_art.to_csv("./datasets/testing/duplicates_report_sf1_art.csv")

    """
    sf1_art = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ART.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"]) # , index_col="calendardate"


    date_relationship_report_art = pandas_mp_engine(callback=report_date_relationship, atoms=sf1_art, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    date_relationship_report_art.to_csv("./datasets/testing/date_relationship_report_art.csv")

    """
    Notes:
    sf1_arq contains 12887 tickers.
    Datekey is unique for all tickers in sf1_art!!!

    """