import os
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import pandas as pd
import math

def print_exception_info(e):
    """
    Takes an exception and prints where the exception ocurred, its type and its message.
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    line_number = exc_tb.tb_lineno
    print(e)
    print("Type: ", exc_type)
    print("At: ", filename, " line: ", line_number)


def get_x_past_months_of_data(df, date, months):
    """
    This function requires df has a DateTimeIndex.
    The function returns the entries in the df from $days before $date to $date.
    """
    date_in_past = date - relativedelta(months=months)
    return df.loc[(df.index <= date) & (df.index >= date_in_past)]

def get_calendardate_x_quarters_ago(date: pd.datetime, quarters: int):
    m_d_list = [[3,31],[6,30],[9, 30],[12, 31]]
    valid = False
    for m_d in m_d_list:
        if (m_d[0] == date.month) and (m_d[1] == date.day):
            valid = True

    if valid == False:
        raise ValueError("date must be a valid report period")

    quarter_list = []
    for year in range(1995, 2020):
        for month_day in m_d_list:
            quarter_list.append(datetime(year=year, month=month_day[0], day=month_day[1]))

    cur_index = quarter_list.index(date)

    return quarter_list[cur_index-quarters]



def get_most_up_to_date_10k_filing(df, date, years):
    """
    NOTE: This function requires df to have a DateTimeIndex, only contain data for one ticker and is sorted on datekey.
    Returns the the most recent 10-K filing with calendardate (normalized report period) $years number of years 
    earliar than date.
    """
    # What if 10K report is not available for the report period?
    cur_row = df.loc[date]
    calendardate = cur_row["calendardate"]

    desired_calendardate = get_calendardate_x_quarters_ago(calendardate, 4*years)
    candidates = df.loc[df.calendardate==desired_calendardate]
    candidates = candidates.loc[candidates.index <= date] # Ensure that no future information gets used

    if len(candidates) == 0:
        raise KeyError("No 10K filing for report period {}".format(desired_calendardate))

    return candidates.iloc[-1]


def get_most_up_to_date_10q_filing(df, date, quarters):
    """
    NOTE: This function requires df to have a DateTimeIndex, only contain data for one ticker and is sorted on datekey.
    Returns the most recnet 10-Q filing with calendardate (normalized report period) $quarters number 
    of quarters earlier than $date.
    """
    cur_row = df.loc[date]
    calendardate = cur_row["calendardate"]

    desired_calendardate = get_calendardate_x_quarters_ago(calendardate, quarters)
    candidates = df.loc[df.calendardate==desired_calendardate]
    candidates = candidates.loc[candidates.index <= date] # Ensure that no future information gets used

    # What if 10Q report is not available for the report period?
    if len(candidates) == 0:
        raise KeyError("No 10K filing for ticker {} and report period {}".format(df.iloc[0]["ticker"], desired_calendardate))

    return candidates.iloc[-1]




def get_calendardate_index(start: pd.datetime, end: pd.datetime):
    calendardate_index = []
    m_d_list = [[3,31],[6,30],[9, 30],[12, 31]]
    month_of_first_filing = start.month
    for i, year in enumerate(range(start.year, end.year + 1)):
        if i == 0:
            index_of_first_filing_in_m_d_list = [3,6,9,12].index(month_of_first_filing)
            for month_day in m_d_list[index_of_first_filing_in_m_d_list:]:
                calendardate_index.append(datetime(year=year, month=month_day[0], day=month_day[1]))
            continue
        for month_day in m_d_list:
            calendardate_index.append(datetime(year=year, month=month_day[0], day=month_day[1]))

    return calendardate_index


#____________________________END_______________________________________

def get_row_with_closest_date(df, date, margin):
    acceptable_dates = get_acceptable_dates(date, margin)
    candidates = df.loc[df.index.isin(acceptable_dates)]
    if len(candidates.index) == 0:
        return pd.DataFrame()
    
    best_row = select_row_closes_to_date(candidates, date)
    return best_row


def get_acceptable_dates(date, margin):
    """
    Returns a list of dates containing dates from $date-$margin days to $date+$margin days.
    """
    dates = [(date + timedelta(days=x)) for x in range(-margin, +margin + 1)]
    dates.sort()
    return dates

def select_row_closes_to_date(candidates, desired_date):
    """
    Returns the date in $candidates that are closest to $desired_date.
    """
    candidate_dates = candidates.loc[:,"datekey"].tolist()
    
    best_date = min(candidate_dates, key=lambda candidate_date: abs(desired_date - candidate_date))
    best_row = candidates.loc[candidates["datekey"] == best_date].iloc[0]

    return best_row
