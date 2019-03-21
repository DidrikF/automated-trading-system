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




def get_most_up_to_date_10k_filing(df, date, years):
    """
    NOTE: This function requires df to have a DateTimeIndex and only contain data for one ticker.
    Returns the 10-K filing $years number of years earliar than date.
    At $date, what is the most updated 10-k filing looking back, for the report period $years years ago.
    """
    cur_row = df.loc[date]
    reportperiod = cur_row["reportperiod"]

    return None

def get_report_period_x_quarters_ago(date: pd.datetime, quarters: int):
    m_d_list = [[3,31],[6,30],[9, 30],[12, 31]]
    if [date.month, date.day] in m_d_list:
        raise ValueError("date must be a valid report period")

    years = math.floor(quarters / 4)
    quarters = quarters % 4

    cur_index = m_d_list.index([date.month, date.day])

    Y = date.year - years
    m_d = m_d_list[cur_index - quarters]

    return datetime(year=Y, month=m_d[0], day=[m_d[1]])
    

def get_most_up_to_date_10q_filing(df, date, quarters):
    """
    NOTE: This function requires df to have a DateTimeIndex and only contain data for one ticker.
    Returns the most updated 10-K/10-Q filing for the report period $quarters number 
    of quarters earlier than date.
    """

    # What if 10Q report is not available for the report period?

    """
    acceptable_dates = get_acceptable_dates(date, margin)
    candidates = df.loc[df.index.isin(acceptable_dates)]
    if len(candidates.index) == 0:
        return pd.DataFrame()
    
    best_row = select_row_closes_to_date(candidates, date)
    return best_row
    """


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
