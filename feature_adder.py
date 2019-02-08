import sys
from packages.dataset_builder.dataset import Dataset
from packages.logger.logger import Logger
from packages.helpers.helpers import print_exception_info

from packages.dataset_builder.feature_builders import book_to_market, book_value, cash_holdings
from packages.helpers.custom_exceptions import FeatureError


if __name__ == "__main__":

    try:
        logger = Logger('./logs')
    except Exception as e:
        print_exception_info(e)
        sys.exit()

    try:
        dataset = Dataset("...")
    except Exception as e:
        print_exception_info(e)
        sys.exit()


    basic_columns_config = {
        'book_value': book_value,
    }

    """
    Can count on the presence of the columns specified in basic_columns_config.
    """
    columns_config = {
        'book_to_market': book_to_market,
        'cash_holdings': cash_holdings,
    }

    print(dataset.info())

    """
    Add basic columns on which other new columns rely on.
    """
    for column in basic_columns_config:
        dataset.data[column] = None
    
    for index in dataset.data.index:
        row = dataset.data.iloc[index]
        
        for column, func in basic_columns_config.items():
            # func may not be computable for some INDEXES (first year of data, missing value, etc.)
            try:
                value = func(index, row, dataset.data)
            except FeatureError as e:
                dataset.data.at[index, column] = None # Or something else? think about how it is saved and respored later
                logger.log_exc(e)
                continue

            dataset.data.at[index, column] = value



    """
    Add features to go into the final dataset.
    """
    # Add columns
    for column in columns_config:
        dataset.data[column] = None
    
    for index in dataset.data.index:
        row = dataset.data.iloc[index]
        
        for column, func in columns_config.items():
            # func may not be computable for some INDEXES (first year of data, missing value, etc.)
            try:
                value = func(index, row, dataset.data)
            except FeatureError as e: 
                logger.log_exc(e)
                continue

            dataset.data.at[index, column] = value



    print(dataset.info())

    logger.close()


# Print columns with full names and descriptions 