import os
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np

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
    """
    Returns the normalized report period date (calendardate) $quarters number of quarters in the past.
    """
    m_d_list = [[3,31],[6,30],[9, 30],[12, 31]]
    valid = False
    for m_d in m_d_list:
        if (m_d[0] == date.month) and (m_d[1] == date.day):
            valid = True

    if valid == False:
        raise ValueError("date must be a valid report period")

    first_year = date.year - (math.ceil(quarters/4) + 1)
    quarter_list = []
    for year in range(first_year, date.year + 1):
        for month_day in m_d_list:
            quarter_list.append(datetime(year=year, month=month_day[0], day=month_day[1]))

    cur_index = quarter_list.index(date)

    return quarter_list[cur_index-quarters]

def get_calendardate_x_quarters_later(date: pd.datetime, quarters: int):
    """
    Returns the normalized report period date (calendardate) $quarters number of quarters in the future.
    """
    m_d_list = [[3,31],[6,30],[9, 30],[12, 31]]
    valid = False
    for m_d in m_d_list:
        if (m_d[0] == date.month) and (m_d[1] == date.day):
            valid = True

    if valid == False:
        raise ValueError("date must be a valid report period")

    last_year = date.year + (math.ceil(quarters/4) + 1)
    quarter_list = []
    for year in range(date.year, last_year):
        for month_day in m_d_list:
            quarter_list.append(datetime(year=year, month=month_day[0], day=month_day[1]))

    cur_index = quarter_list.index(date)

    return quarter_list[cur_index+quarters]

def get_most_up_to_date_10k_filing(sf1_art, caldate_cur: pd.datetime, datekey_cur: pd.datetime, years):
    """
    NOTE: This function requires sf1_art to have a DateTimeIndex, only contain data for one ticker and is sorted on datekey.
    NOTE: sf1_art has a numerical index and calendardate column
    
    Returns the the most recent 10-K filing with calendardate (normalized report period) $years number of years 
    earliar than date.

    """
    desired_calendardate = get_calendardate_x_quarters_ago(caldate_cur, 4*years)
    candidates = sf1_art.loc[sf1_art.calendardate==desired_calendardate]


    candidates = candidates.loc[candidates.datekey <= datekey_cur] # Ensure that no future information gets used

    if len(candidates) == 0:
        # raise KeyError("No 10K filing for report period {}".format(desired_calendardate))
        return pd.Series(index=sf1_art.columns)

    candidates = candidates.sort_values(by="datekey", ascending=True)

    return candidates.iloc[-1]


def get_most_up_to_date_10q_filing(sf1_arq: pd.DataFrame, caldate_cur: pd.datetime, datekey_cur: pd.datetime, quarters: int):
    """
    NOTE: This function requires sf1_arq to have a DateTimeIndex, only contain data for one ticker and is sorted on datekey.
    NOTE: sf1_arq has a calendardate index

    Returns the most recnet 10-Q filing with calendardate (normalized report period) $quarters number 
    of quarters earlier than $date.
    """

    desired_calendardate = get_calendardate_x_quarters_ago(caldate_cur, quarters)
    candidates = sf1_arq.loc[sf1_arq.index == desired_calendardate]
    candidates = candidates.loc[candidates.datekey <= datekey_cur] # Ensure that no future information gets used

    if len(candidates) == 0:
        # raise KeyError("No 10K filing for ticker {} and report period {}".format(sf1_arq.iloc[0]["ticker"], desired_calendardate))
        return pd.Series(index=sf1_arq.columns)


    candidates = candidates.sort_values(by="datekey", ascending=True)

    return candidates.iloc[-1]


def get_calendardate_index(start: pd.datetime, end: pd.datetime):
    """
    Returns a DateTimeIndex containing the complete set of normalized report period dates (calendardates)
    from the date $start to the date $end.
    """
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

    # Need to drop dates after end
    for j, date in enumerate(calendardate_index):
        if date > end:
            del calendardate_index[j]

    return calendardate_index

def forward_fill_gaps(sf1, quarters):
    """
    NOTE: this function require calendardate index.
    Fill in missing data in $quarters number of quarters into the future.
    """
    sf1 = sf1.fillna(value="IAMNAN")
    sf1["calendardate_temp1"] = sf1.index # Don't know another awy to get the index value after selection

    calendardate_index = get_calendardate_index(sf1.iloc[0]["calendardate_temp1"], sf1.iloc[-1]["calendardate_temp1"])

    # sf1_reindexed = sf1.reindex(calendardate_index) # ValueError: cannot reindex from a duplicate axis

    sf1_reindexed = fill_in_missing_dates_in_calendardate_index(sf1)

    sf1_filled = sf1_reindexed.fillna(method="ffill", limit=quarters)
    
    sf1_filled = sf1_filled.drop(columns=["calendardate_temp1"])
    sf1_filled = sf1_filled.dropna(axis=0)
    sf1_filled = sf1_filled.replace(to_replace="IAMNAN", value=np.nan)

    return sf1_filled


def fill_in_missing_dates_in_calendardate_index(sf1):
    sf1["calendardate_temp2"] = sf1.index # Don't know another awy to get the index value after selection
    desired_index = get_calendardate_index(sf1.iloc[0]["calendardate_temp2"], sf1.iloc[-1]["calendardate_temp2"])

    index_difference = list(set(desired_index).difference(set(sf1.index)))

    for caldate_index in index_difference:
        # sf1.index.insert(-1, caldate_index)
        sf1.loc[caldate_index] = pd.Series()

    sf1 = sf1.drop(columns=["calendardate_temp2"])
    sf1 = sf1.sort_values(by=["calendardate", "datekey"], ascending=True)

    return sf1




#____________________________END_______________________________________

def get_row_with_closest_date(df, date, margin):
    """
    Returns the row in the df closest to the date as long as it is within the margin.
    If no such date exist it returns None.
    Assumes only one ticker in the dataframe.
    """
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



#_____ From the very beginning of the project (can probably be removed)______

def column_intersection(df1: pd.DataFrame, df2: pd.DataFrame, mapping: tuple) -> set:
    """
    Get the intersection of two columns from two different data frames.
    """
    df1_elements = set()
    df2_elements = set()
    col1 = mapping[0]
    col2 = mapping[1]

    df1_elements.add(df1[col1].tolist())
    df2_elements.add(df2[col2].tolist())
    intersection = df1_elements.intersection(df2_elements)
    return intersection


def add_derived_column(df: pd.DataFrame, column: str, calculate_value: callable) -> pd.DataFrame:
    """
    calculate_value(row, df) -> scalar
    Adds a new column to df1 and fills it with arbitrarily calculated values.
    """
    df[column] = None

    for row_index, row in enumerate(df.iterrows()):
        if row_index == 5:
            return df
        val = calculate_value(row, df)
        df.at[row_index, column] = val

    return df



def df_filter_rows(df: pd.DataFrame, filter_func: callable) -> pd.DataFrame:
    """
    filter_func(row: pd.Series, df: pd.Dataframe) -> bool
    Iterates over the rows in df and removes it, if filter_func returns False.
    """
    rows_to_drop = []
    for index, row in enumerate(df.iterrows()):
        row_passed = filter_func(row, df)
        if not row_passed:
            rows_to_drop.append(index)
        
    df.drop(rows_to_drop)

    return df



def df_filter_cols(df: pd.DataFrame, filter_func: callable) -> pd.DataFrame:
    pass