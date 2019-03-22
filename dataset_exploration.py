import pandas as pd
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta
from packages.multiprocessing.engine import pandas_mp_engine
from packages.helpers.helpers import get_calendardate_index

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

def fill_gaps(sf1, quarters):
    calendardate_index = get_calendardate_index(sf1)

def report_gaps_after_fill(sf1):
    sf1 = fill_gaps(sf1, 3)

    gaps_1q, gaps_4q = detect_gaps(sf1)
    report = pd.DataFrame(columns=["ticker", "gaps over 1q", "gaps over 4q"])

    if len(gaps_1q) > 0:
        report.at[0, "ticker"] = sf1.iloc[0]["ticker"]
        report.at[0, "gaps over 1q"] = len(gaps_1q)
    if len(gaps_4q) > 0:
        report.at[0, "gaps over 4q"] = len(gaps_4q)

    return report


def report_updates_after_fill(sf1):
    sf1 = fill_gaps(sf1, 3)

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


if __name__ == "__main__":
    
    """
    # ./datasets/sharadar/SHARADAR_SF1_ARQ.csv
    sf1_arq = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ARQ.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"], index_col="calendardate")

    update_report_arq = pandas_mp_engine(callback=report_updates, atoms=sf1_arq, \
        data=None, molecule_key='sf1_art', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    gap_report_arq = pandas_mp_engine(callback=report_gaps, atoms=sf1_arq, \
        data=None, molecule_key='sf1_art', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    update_report_arq.to_csv("./datasets/testing/update_report_arq.csv", index=False)
    gap_report_arq.to_csv("./datasets/testing/gap_report_arq.csv", index=False)
    """

    
    # "./datasets/sharadar/SHARADAR_SF1_ART.csv"
    # "./datasets/testing/sf1_art.csv"
    sf1_art = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ART.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"], index_col="calendardate")

    # Forward fill up to three quarters and see how it looks after

    



    update_report_art = pandas_mp_engine(callback=report_updates_after_fill, atoms=sf1_art, \
        data=None, molecule_key='sf1_art', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    gap_report_art = pandas_mp_engine(callback=report_gaps_after_fill, atoms=sf1_art, \
        data=None, molecule_key='sf1_art', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    update_report_art.to_csv("./datasets/testing/update_report_art.csv", index=False)
    gap_report_art.to_csv("./datasets/testing/gap_report_art.csv", index=False)


    """
    Notes:
    sf1_arq contains 12887 tickers.

    """