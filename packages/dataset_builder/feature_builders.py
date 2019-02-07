"""
Contains functions calculating various features based on the original SHARADAR datasets. 
Each of these functions receive the current index of the row in the dataset, the row itself
and a reference to the dataset data frame.
"""

import numbers
from packages.helpers.custom_exceptions import FeatureError


"""
How to get value from data frame:
df = self.indicator_descriptions.loc[self.indicator_descriptions["indicator"] == indicator]
title = df.iloc[0]['title']
"""


def book_value(index, row, df) -> float:
        errors = list()
        if not isinstance(row["equity"], numbers.Number):
            errors.append("equity was not a number")
        
        if len(errors) > 0:
            raise FeatureError("When calculating book_to_market at index {} some error(s) occurred".format(index), errors)
        
        return row["equity"]




def book_to_market(index, row, df) -> float:
    errors =  list()
    if not isinstance(row['book_value'], numbers.Number):
        errors.append("book_value was not a number")
    if not isinstance(row['marketcap'], numbers.Number):
        errors.append("marketcap was not a number")
    if row['marketcap'] == 0:
        errors.append("market_value is zero and cannot be in the devisor")        
    
    if len(errors) > 0:
        raise FeatureError("When calculating book_to_market at index {} some error(s) occurred".format(index), errors)

    return row['book_value'] / row['marketcap']



def cash_holdings(index, row, df) -> float:
    errors = list()
    if not isinstance(row['cashneq'], numbers.Number):
        errors.append("cashneq was not a number")
    # investmentcs : current invesments (I understand it to be short-term investments)
    if not isinstance(row["investmentsc"], numbers.Number):
        errors.append("investmentsc was not a number")
    
    if len(errors) > 0:
        raise FeatureError("When calculating cash_holdings at index {} some error(s) occurred".format(index), errors)

    return row['cashneq'] + row['investmentsc']


# A year is from jan to dec
# I need to respect the time which the information became available to the public.
# It does not matter if one company bases its year 15. august while another 27. november, I just care
# about the one-month, 3-month, 12-month time delta being respected for each company.

def asset_growth(index, row, df) -> float:
    errors = list()
    # total_assets is assets
    data_or_current_row = row["calendardate"]

    assets_t_1 = None
    assets_t_2 = None

    return None




"""
Probably will not include this

def return_on_capital_employed(index, row, df):
    errors = list()
    if not isinstance(row['ebit'], numbers.Number):
        errors.append("ebit was not a number")
    # dont know the column name for cap or ebit!!
    if not isinstance(row["cap"]):
        errors.append("cap was not a number")
    if row["cap"] == 0:
        errors.append("cap is zero and cannot be in the devisor")

    if len(errors) > 0:
        raise FeatureError("When calculating return_on_capital_employed at index {} some error(s) occurred".format(index), errors)

    return row['ebit'] / row['cap']
"""