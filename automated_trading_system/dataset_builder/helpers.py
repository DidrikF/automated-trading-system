import pandas as pd

"""
class MyException(Exception):
    pass
raise MyException("")
"""

"""
    Dataset Merging on date and ticker
    - Store the data that did not make it in a df and make it viewable
    - 


    Todo:
    (!) Give info from the process
    Extend to allow custom matchers on data (not just == in "df2.loc[eval(conditional_statement)]")
        - use function to generate a list of possible values (not good enough, no control over which in the list is chosen)

"""


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


def data_availability(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    Return/prints some stats about to what degree df2 has data we want to merge into df1.
    """
    pass
    """
    fundamentals_unique_years = set()
    prices_unique_years = set()

    for date in fundamentals.data["Period Ending"].tolist():
        fundamentals_unique_years.add(date.year)

    for date in prices.data["date"].tolist():
        prices_unique_years.add(date.year)

    common_years = fundamentals_unique_years.intersection(prices_unique_years)
    
    print("Fundamentals years: ", fundamentals_unique_years)
    print("Prices years: ", prices_unique_years)
    print("Common years: ", common_years)
    """


# Not tested
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
