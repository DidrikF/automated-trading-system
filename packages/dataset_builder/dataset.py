"""
Module providing functionality to manipulate large datasets
"""
import pandas as pd
import numpy as np
from tabulate import tabulate
# from helpers import print_exception_info
from ..helpers.helpers import print_exception_info
from datetime import datetime, timedelta
# from ..logger.logger import Logger
import sys

class Dataset():
    def __init__(self, path, df=None, indicator_descriptions_path=None):
        if path is not None:
            self.path = path
            try: 
                self.data = pd.read_csv(path, low_memory=False)
            except Exception as e:
                raise e
        elif df is not None:
            self.path = None
            self.data = df
        else:
            raise Exception("Unsupported arguments, must prodide path to csv file or a data frame.")

        if indicator_descriptions_path is not None:
            self.indicator_descriptions_path = indicator_descriptions_path
            try:
                self.indicator_descriptions = pd.read_csv(indicator_descriptions_path, low_memory=False)
            except Exception as e:
                raise e

        # self.logger = logger

    @classmethod
    def from_df(cls, df):
        return cls(path=None, df=df)

    
    def info(self):
        print(tabulate([
            ["File: ", self.path],
            ["Size: ", self.data.size],
            ["Shape: ", self.data.shape],
            ["Number of Dimensions: ", self.data.ndim]
        ]))

        print(self.data.info())
        
        print("Head: ", '\n', self.data.head())
            
    def print_cols(self):
        print("Columns")
        for index, indicator in enumerate(self.data.columns.values):
            print("%3i: %-15s" % (index, indicator))

    def print_cols_with_descriptions(self):
        print("Columns with descriptions")
        for index, indicator in enumerate(self.data.columns.values):
            if indicator == "None":
                continue
            #print(indicator)
            df = self.indicator_descriptions.loc[self.indicator_descriptions["indicator"] == indicator]
            title = df.iloc[0]['title']
            description = df.iloc[0]["description"]
            print("%3i: %-15s: %s" % (index, indicator, title))
            print("Description: ", description, "\n")


    def print_cols_with_title(self):
        print("Columns with titles")
        for index, indicator in enumerate(self.data.columns.values):
            if indicator == "None":
                continue
            #print(indicator)
            df = self.indicator_descriptions.loc[self.indicator_descriptions["indicator"] == indicator]
            title = df.iloc[0]['title']
            print("%3i: %-15s: %s" % (index, indicator, title))
    
    def print_col(self, indicator):
        index = self.data.columns.values.tolist().index(indicator)
        print(type(index))

        df = self.indicator_descriptions.loc[self.indicator_descriptions["indicator"] == indicator]
        title = df.iloc[0]['title']
        description = df.iloc[0]["description"]
        print("Info about indicator: ", indicator)
        print("Column Index: ", index)
        print("Title : ", title)
        print("Description: ", description)

    def drop_cols_by_index(self, cols: list):
        self.data.drop(self.data.columns[cols], axis=1, inplace=True)

    def drop_cols_except(self, cols: list):
        cols = set(self.data.columns[cols])
        all_cols = set(self.data.columns)
        diff = all_cols - cols
        self.data.drop(diff, axis=1, inplace=True)

    def sort(self, by, ascending=True):
        """
        Sorts the dataset by a column in place with N/A values first. It sorts assending by default, 
        but this can be changed by passing the ascending argument. It wraps the df.sort_values method.  
        """
        self.data.sort_values(by=by, ascending=ascending, inplace=True, na_position='first')

    """
    def split_simple(self, into):
        chunk_size = len(self.data)
        split(df, rep(1:ceiling(chunk_size/into), each=into, length.out=chunk_size))
    """

    def split_simple(self, into: int) -> list:
        """
        Split the dataset into n parts.
        """
        data_length = len(self.data)
        split_on_index = list()
        chuck_length = int(data_length/into) + 1
        cur_index = chuck_length
        while cur_index < data_length:
            split_on_index.append(cur_index)
            cur_index += chuck_length
        last_index = data_length
        split_on_index.append(last_index)

        datasets = list()
        last_index = 0
        for next_index in split_on_index:
            df = self.data.iloc[last_index:next_index]
            
            dataset = self.from_df(df)
            datasets.append(dataset)
            last_index = next_index

        return datasets
    

    def split(self, into: int) -> list:
        """
        Split the dataset into n chunks. Chunks does not split such that two different chunks contain records from the same company.
        Note: The dataset must be sorted.
        """
        data_length = len(self.data)
        chunk_length = int(data_length / into) + 1
        cur_index = chunk_length
        split_on_index = list()
        while cur_index < data_length:
            ticker = self.data.iloc[cur_index]['ticker']
            i = 1
            while True:
                next_ticker = self.data.iloc[cur_index + i]['ticker']
                if next_ticker != ticker:
                    break
                i += 1
            cur_index += + i
            split_on_index.append(cur_index)
            cur_index += chunk_length

        last_index = data_length
        split_on_index.append(last_index)

        # print(split_on_index)

        datasets = list()
        last_index = 0
        for next_index in split_on_index:
            df = self.data.iloc[last_index:next_index]
            dataset = self.from_df(df)
            datasets.append(dataset)
            last_index = next_index

        return datasets


    def parse_dates(self, cols: list):
        for col in cols:
            self.data[col] = pd.to_datetime(self.data[col])

    def to_csv(self, path):
        self.data.to_csv(path, index=False)

    """
    def log(self, exception):
        self.logger.log(exception)
    """

    def drop_rows_by_index(self):
        pass

    def filter_cols(self, column, str, filter_function: callable):
        pass

    def filter_rows(self, column: str, filter_function: callable):
        pass
 
    def get_values_only(self):
        pass

    def get_np(self):
        # return np.new(self.data)
        pass



def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, merge_on: dict, select_best: callable, columns_to_add: list) -> pd.DataFrame:
    """
    Merge two datasets on a set of columns with the ability to select the best row to merge on (for each coresponding
    row in the other dataset) with a function. 
    """
    # Step 1: Create new columns in df1
    for column in columns_to_add:
        df1[column] = None
    
    # Step 2: Extract columns and values to match on in df2
    matcher_list = []

    for df1_row_index in df1.index:
        col_val_to_match_in_df2 = {}
        # column name: value, to match on in df2
        add_to_matcher_list = True
        for merge_info in merge_on:
            col1 = merge_info["mapping"][0]
            col2 = merge_info["mapping"][1]

            match_val = df1.iloc[df1_row_index][col1]
            
            # If df1 does has a corrupt value, we cannot match on in df2, exclude that row
            if match_val in ['None', None, '']:
                add_to_matcher_list = False


            if merge_info["possible_values"]:
                col_val_to_match_in_df2[col2] = merge_info["possible_values"](match_val)
                # col_val_to_match_in_df2[col2].insert(0, match_val) # Should have a better solution here
            else:
                col_val_to_match_in_df2[col2] = match_val
        
        if add_to_matcher_list == True:
            matcher_list.append((df1_row_index, col_val_to_match_in_df2, ))
            
    # print(row_col_val_to_match)
    
    for index, matcher in enumerate(matcher_list):
        """
        if index == 10:
            break
        """
        # Find the matching row
        df1_row = matcher[0]
        col_val_to_match_in_df2 = matcher[1]

        # Create conditional statement to select candidate rows from df2
        conditional_statement = ""
        need_to_select_best = False
        j = 0
        for col, match_val in col_val_to_match_in_df2.items():
            if j != 0:
                conditional_statement += " & "

            if type(match_val) == list:
                conditional_statement += "df2['{}'].isin({})".format(col, match_val)
                need_to_select_best = True
            else:
                conditional_statement += "(df2['{}'] == '{}')".format(col, match_val)
            
            j += 1

        # print(conditional_statement)

        # Select row from df2 to merge into df1
        if len(df2.loc[eval(conditional_statement)]) > 0:
            
            if need_to_select_best == True:
                candidate_rows = df2.loc[eval(conditional_statement)]
                df2_row_index = select_best(candidate_rows, col_val_to_match_in_df2) #! need to get the index...
            else:
                # table.query('column_name == some_value | column_name2 == some_value2')
                df2_row_index = df2.loc[eval(conditional_statement)].index.item()
        else: 
            continue
        
        # Copy values over from df2 to df1 for each column in columns_to_add
        for col_to_add in columns_to_add:
            #df2_value_to_add = df2.loc[df2_row_index, col_to_add]
            df2_value_to_add = df2.at[df2_row_index, col_to_add]
            df1.at[df1_row, col_to_add] = df2_value_to_add 

    return df1


def merge_datasets_simple(df1: pd.DataFrame, df2: pd.DataFrame, on, suffixes) -> pd.DataFrame:
    """
    A wrapper around Pandas merge method but default to outer join and adds the indicator column to 
    make it clear which rows were not in the intersection. This also validates that rows are matched one to one.
    """
    return pd.merge(df1, df2, on=on, how="outer", indicator=True, suffixes=suffixes, validate="one_to_one")


def join_datasets(datasets: list) -> pd.DataFrame:
    """
    Takes a list of Dataset objects and concatenates all the underlying data frames 
    and returns the resulting data frame as a Dataset object.
    """
    dfs = list()
    for dataset in datasets:
        dfs.append(dataset.data)

    result = pd.concat(dfs)

    return Dataset.from_df(result)




# This is just for testing
if __name__ == "__main__":
    pass
