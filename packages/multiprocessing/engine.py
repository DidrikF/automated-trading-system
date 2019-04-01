import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import datetime as dt
import sys
import os.path
import pickle
from io import StringIO
import math

__all__ = ["lin_parts", "pandas_mp_engine"]

def lin_parts(num_atoms, num_threads):
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def get_jobs(atoms, data, callback, molecule_key, split_strategy, num_processes, molecules_per_process, **kwargs):
    """
    Seperating atoms into molecules based on different strategies and create jobs.
    Jobs are the arguments given to the $callback.
    """
    jobs = []

    if split_strategy == 'ticker':
        if 'ticker' not in atoms:
            raise Exception("Ticker column not in atoms")
        tickers = list(atoms["ticker"].unique())
        for ticker in tickers:
            molecule = atoms.loc[atoms["ticker"] == ticker] # CAN YOU DO THIS FASTER?
            job = {
                molecule_key: molecule,
                'callback': callback,
            }
            job.update(kwargs)

            if data is not None: # != not tested
                molecule_data = {}
                for key, df in data.items():
                    molecule_data[key] = df.loc[df["ticker"] == ticker]
                job.update(molecule_data)
                
            jobs.append(job)

    elif split_strategy == 'date':
        lowest_date = atoms.index.min()
        highest_date = atoms.index.max()
        date_index = pd.date_range(lowest_date, highest_date)
        parts = lin_parts(len(date_index), num_processes*molecules_per_process)
        # print(len(date_index))
        # print(parts)
        for i in range(1,len(parts)):
            date0 = date_index[parts[i-1]]
            date1 = date_index[parts[i] - 1]
            job = {
                molecule_key: atoms.loc[(atoms.index >= date0) & (atoms.index < date1)],
                'callback': callback
            }
            job.update(kwargs)

            if data != None:
                molecule_data = {}
                for key, df in data.items():
                    molecule_data[key] = df.loc[(df.index >= date0) & (df.index < date1)]
                job.update(molecule_data)

            jobs.append(job)

    elif split_strategy == 'industry':
        if 'industry' not in atoms:
            raise Exception("Industry column not in atoms")
        
        industries = list(atoms["industry"].unique())
        for industry in industries:
            molecule = atoms.loc[atoms["industry"] == industry]
            job = {
                molecule_key: molecule,
                'callback': callback,
            }
            job.update(kwargs)

            if data != None:
                molecule_data = {}
                for key, df in data.items():
                    if "industry" not in df:
                        raise Exception("Industry column not in dataframe")
                    molecule_data[key] = df.loc[df["industry"] == industry]
                job.update(molecule_data)
                
            jobs.append(job)

    return jobs



def pandas_mp_engine(callback, atoms, data, molecule_key, split_strategy, num_processes, molecules_per_process, **kwargs): 
    """
    callback -> function to execute in parallell
    atoms -> the data to be processed
    data -> additional data to be used in processing the atoms
    molecule_key -> the argument to use when passing a molecule to the callback
    split_strategy -> A string ('ticker'/'industry') that determines 
                    how atoms are split into molecules 
    num_processes -> the number of processes to execute in parallell
    molecules_per_process -> number of parallell jobs per core
    kwargs -> key word arguments to callback
    """

    # arts = lin_parts(len(atoms), num_processes*molecules_per_process) # subject to change
    
    jobs = get_jobs(atoms, data, callback, molecule_key, split_strategy, num_processes, molecules_per_process, **kwargs)    

    print("Number of jobs: ", len(jobs))

    if num_processes == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_processes=num_processes)

    # out is a list of dataframes, the dataframes should be able to deliver directly to the next callback

    if isinstance(out[0], pd.DataFrame):
        result = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        result = pd.Series()
    else:
        return out
    
    for i in out:
        result = result.append(i)
    
    result = result.sort_index()
    return result


def process_jobs_(jobs):
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out



def process_jobs(jobs, task=None, num_processes=8):
    if task is None:
        task = jobs[0]['callback'].__name__
        
        pool = mp.Pool(processes=num_processes)
        outputs = pool.imap_unordered(expandCall, jobs)
        out = []
        time0 = time.time()

        # Process asynchronous output, report progress
        for i, out_ in enumerate(outputs, 1):
            out.append(out_)
            report_progress(i, len(jobs), time0, task)
        
        pool.close()
        pool.join()
        return out

# Add this to the job creation process!!!
def report_progress(job_num, num_jobs, time0, task):
    # Report progress as async jobs are completed
    ratio_of_jobs_completed = float(job_num)/num_jobs
    minutes_elapsed = (time.time()-time0)/60
    minutes_remaining = minutes_elapsed*(1/ratio_of_jobs_completed - 1)
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = time_stamp + " " + str(round(ratio_of_jobs_completed*100, 2)) + "% " + str(job_num) + "/" + str(num_jobs) + \
        " - " + task + " done after " + str(round(minutes_elapsed, 2)) + " minutes. Remaining " + \
            str(round(minutes_remaining, 2)) + ' minutes.'
    
    if job_num < num_jobs: 
        sys.stderr.write(msg + '\r') # override previous line
    else: 
        sys.stderr.write(msg + '\n')
    return


def expandCall(kwargs):
    callback = kwargs['callback']
    del kwargs['callback']
    out = callback(**kwargs)
    return out



#____________________________________NEW FASTER ENGINE WITH TASK PIPELINE___________________________________

sf1_art_base = pd.read_csv("./datasets/sharadar/SF1_ART_BASE.csv", parse_dates=["calendardate", "datekey"], index_col="calendardate")
sf1_arq_base = pd.read_csv("./datasets/sharadar/SF1_ARQ_BASE.csv", parse_dates=["calendardate", "datekey"], index_col="calendardate")
sep_base = pd.read_csv("./datasets/sharadar/SEP_BASE.csv", parse_dates=["date"], index_col="date")

# atoms_name, atoms_info, split_strategy, cache_dir
#                       "sep"       "ticker"        "./datasets/mc_cache"
def split_into_molecules(disk_name, atoms_config, split_strategy, cache_dir): # , csv_file_in_memory=None
    """
    Splits atoms into molecules, if not allready availabel in the cache_dir.
    Returns dictionary of molecules with keys being "AAPL", "Electronics..."
    """
    """ PROBABLY REMOVE
    if csv_file_in_memory is not None:
        return dfs
    """


    # maybe add is some way to version or identify differnet stages of development of the molecules
    pickle_path = cache_dir + '/' + disk_name + '.pickle'

    if os.path.isfile(pickle_path):
        print("Loading pickle: " + pickle_path)
        pickle_in = open(pickle_path,"rb")
        dfs = pickle.load(pickle_in)
    else:
        csv_strings = {}
        time0 = time.time()
        cols = ""
        if split_strategy in ["ticker", "industry"]:
            with open(atoms_config["csv_path"]) as csvfile:
                for index, line in enumerate(csvfile, 1):
                    row = line.split(',')
                    if index == 1:
                        cols = line
                        row_index = row.index(split_strategy)
                        continue
                    
                    if row[row_index] not in csv_strings:
                        csv_strings[row[row_index]] = cols
                        csv_strings[row[row_index]] += line
                    else:
                        csv_strings[row[row_index]] += line

                    if (index % atoms_config["report_every"]) == 0:
                        report_progress(index, atoms_config["length"], time0, "Processing " + disk_name)
        
        elif split_strategy == "date":
            with open(atoms_config["csv_path"]) as csvfile:
                for index, line in enumerate(csvfile, 1):
                    row = line.split(',')
                    if index == 1:
                        cols = line
                        row_index = row.index(split_strategy)
                        continue

                    # Need code to split on year
                    # May not be needed as less splits can be performed


        index = 1
        time0 = time.time()
        num_iterations = len(csv_strings)
        report_every = math.ceil(num_iterations/50)
        dfs = {}
        for molecule_identifier, df_string in csv_strings.items():
            csv_file = StringIO(df_string)

            df = pd.read_csv(csv_file, parse_dates=atoms_config["parse_dates"], index_col=atoms_config["index_col"], low_memory=False)
            if atoms_config["sort_by"] is not None:
                df = df.sort_values(by=atoms_config["sort_by"])
            
            dfs[molecule_identifier] = df

            if (index % report_every) == 0:
                report_progress(index, num_iterations, time0, "Create Data Frames for " + disk_name)

            index += 1

        # Cache the results for later use
        if atoms_config["cache"] == True:
            pickle_out = open(pickle_path,"wb")
            pickle.dump(dfs, pickle_out)
            pickle_out.close()

    return dfs


def get_jobs_fast(task, primary_molecules, molecules_dict):
    """
    Need to write code for different split strategies.
    Maybe bake split strategy into the cached pickle name.
    """
    jobs = []
    # print("Type of primary_molecules: ", type(primary_molecules))
    for job_key, molecule in primary_molecules.items(): # primary_molecules is a list??
        # job_key is AAPL, Electronics etc.
        # molecule is this jobs atoms (molecule)
        job = {
            "callback": task["callback"],
            task["molecule_key"]: molecule,
            "job_key": job_key,
        }

        job.update(task["kwargs"])

        if task["data"] is not None:
            job_data = {}
            for kw_name, molecule_dict_name in task["data"].items():
                # print(job_key, kw_name, molecule_dict_name, molecules_dict.keys())
                # data_key is sep, sf1_art, sf1_arq, metadata
                # molecules_dict["sep"]["AAPL"] is the relevant df for this job

                try:
                    data_molecule = molecules_dict[molecule_dict_name][job_key]
                except Exception as e:
                    if molecule_dict_name == "sf1_art":
                        data_molecule = sf1_art_base
                    elif molecule_dict_name == "sf1_arq":
                        data_molecule = sf1_arq_base
                    elif molecule_dict_name == "sep":
                        data_molecule = sep_base
                    else:
                        data_molecule = pd.DataFrame()

                job_data[kw_name] = data_molecule
            job.update(job_data)

        jobs.append(job)

    return jobs

def process_jobs_fast(jobs, task=None, num_processes=8, sort_by=[]):
    if task is None:
        task = jobs[0]['callback'].__name__
        
        pool = mp.Pool(processes=num_processes)
        outputs = pool.imap_unordered(expandCall_fast, jobs)
        out = {}
        time0 = time.time()
        i = 1
        # Process asynchronous output, report progress
        for out_ in outputs:
            out[out_[0]] = out_[1].sort_values(by=sort_by) # .sort_values(by="date") # OBS NEED TO UPDATE
            report_progress(i, len(jobs), time0, task)
            i += 1
        
        pool.close()
        pool.join()
        return out

def expandCall_fast(kwargs):
    callback = kwargs['callback']
    job_key = kwargs['job_key']
    del kwargs['callback']
    del kwargs['job_key']
    # print(kwargs.keys())
    out = callback(**kwargs)
    return (job_key, out)


def split_df_into_molecules(atoms, split_strategy, num_molecules):
    dfs = {}
    if split_strategy == 'date':
        lowest_date = atoms.index.min()
        highest_date = atoms.index.max()
        date_index = pd.date_range(lowest_date, highest_date)
        parts = lin_parts(len(date_index), num_molecules)
        time0 = time.time()
        for i in range(1,len(parts)):
            date0 = date_index[parts[i-1]]
            date1 = date_index[parts[i] - 1]
            dfs[str(date0)] = atoms.loc[(atoms.index >= date0) & (atoms.index <= date1)]
            report_progress(i, len(parts), time0, "split_df_into_molecules according to split strategy: " + split_strategy)

    elif split_strategy == 'industry':
        if 'industry' not in atoms:
            raise Exception("Industry column not in atoms")

        grouped_molecules = atoms.groupby('industry')

        for industry, molecule in grouped_molecules:
            dfs[industry] = molecule
        
        """
        industries = list(atoms["industry"].unique())
        time0 = time.time()
        for i, industry in enumerate(industries, 1):
            dfs[industry] = atoms.loc[atoms["industry"] == industry]
            report_progress(i, len(industries), time0, "split_df_into_molecules according to split strategy: " + split_strategy)
        """
    
    elif split_strategy == 'ticker':
        if 'ticker' not in atoms:
            raise Exception("Ticker column not in atoms")
        grouped_molecules = atoms.groupby("ticker")
        for industry, molecule in grouped_molecules:
            dfs[industry] = molecule
        
        """
        tickers = list(atoms["ticker"].unique())
        time0 = time.time()
        for i, ticker in enumerate(tickers, 1):
            dfs[ticker] = atoms.loc[atoms["ticker"] == ticker]
            report_progress(i, len(tickers), time0, "split_df_into_molecules according to split strategy: " + split_strategy)
        """
    else:
        raise ValueError("split_strategy cannot be " + split_strategy + ". Only 'ticker', 'industry' and 'date' are supported.")

    return dfs


def pandas_chaining_mp_engine(tasks, primary_atoms, atoms_configs, split_strategy, num_processes, cache_dir, \
    save_dir, sort_by=None, molecules_per_process=5):
    """
    primary_atoms : What key in atoms_config to use as first molecules to create jobs from.
    """
    # Ready the data once, and then use a simple dictionary lookup to get the correct df when creating jobs
    molecules_dict = {} # { "AAPL": [df1, df2, ...], ... }

    for disk_name, atoms_config in atoms_configs.items():
        pickle_path = cache_dir + '/' + disk_name + '.pickle'

        if os.path.isfile(pickle_path):
            print("Loading pickle: " + pickle_path)
            pickle_in = open(pickle_path,"rb")
            dfs = pickle.load(pickle_in)
        else:
            print("Reading and parsing: ", atoms_config["csv_path"])
            atoms = pd.read_csv(atoms_config["csv_path"], parse_dates=atoms_config["parse_dates"], index_col=atoms_config["index_col"], low_memory=False)
            if atoms_config["sort_by"] is not None:
                atoms = atoms.sort_values(by=atoms_config["sort_by"])
            molecules_dict[disk_name] = split_df_into_molecules(atoms, split_strategy, num_processes*molecules_per_process)


            if atoms_config["cache"] == True:
                pickle_out = open(pickle_path,"wb")
                pickle.dump(molecules_dict[disk_name], pickle_out)
                pickle_out.close()

    primary_molecules = molecules_dict[primary_atoms]
    

    # Loop over tasks and pass output from each task as input to the next.
    last_index = len(tasks) - 1
    start_time_tasks = time.time()
    for index, task in enumerate(tasks):
        minutes_elapsed = (time.time()-start_time_tasks)/60
        print("Step ", str(index+1), " of ", str(len(tasks)), " - " + task["name"] + " - Time elapsed: ", str(round(minutes_elapsed, 2)), " minutes.")

        # The molecules dict is not needed for all tasks, if none is listed in task["data"] it is simply never touched.
        jobs = get_jobs_fast(task, primary_molecules, molecules_dict) # dont cache jobs

        primary_molecules = process_jobs_fast(jobs, num_processes=num_processes, sort_by=sort_by) # return as list of Data Frames, I need a dict
        
        # Save result as csv
        if task["save_result_to_disk"] == True:
            save_path = save_dir + "/" + task["disk_name"] + ".csv"
            combine_and_save_molecules(primary_molecules, save_path, sort_by=task["sort_by"])

        # Cache result as pickle of DataFrames
        if task["cache_result"] == True:
            cache_path = cache_dir + "/" + task["disk_name"] + ".pickle"
            pickle_out = open(cache_path,"wb")
            pickle.dump(primary_molecules, pickle_out)
            pickle_out.close()

        # Prepare Primary Molecules
        time0 = time.time()
        if index < last_index:
            next_task = tasks[index+1]
            if next_task["split_strategy"] != task["split_strategy"]:
                atoms = combine_molecules(primary_molecules)
                primary_molecules = split_df_into_molecules(atoms, next_task["split_strategy"], num_processes*molecules_per_process) 

                # If the next task has any other data than the primary molecules it needs, we need to split those as well according to the required strategy.
                if next_task["data"] is not None: 
                    for molecules_dict_name in next_task["data"].values():
                        atoms = combine_molecules(molecules_dict[molecules_dict_name])
                        molecules_dict[molecules_dict_name] = split_df_into_molecules(atoms, next_task["split_strategy"], num_processes*molecules_per_process)
                    
        # Add molecules to molecules_dict for later use by another task
        if task.get("add_to_molecules_dict", False) == True:
            molecules_to_add_to_molecules_dict = primary_molecules
            
            # We just split primary_molecules for the next task, 
            # if this is not the format we want to cache the molecules, they must be split in the correct way.
            if task["split_strategy_for_molecule_dict"] != next_task["split_strategy"]: 

                atoms = combine_molecules(primary_molecules)

                molecules_to_add_to_molecules_dict = split_df_into_molecules(atoms, task["split_strategy_for_molecule_dict"], num_processes*molecules_per_process)
                
            molecules_dict[task["disk_name"]] = molecules_to_add_to_molecules_dict


    print("TASKS COMPLETED SUCCESSFULLY")

    out_molecules = list(primary_molecules.values())

    if isinstance(out_molecules[0], pd.DataFrame):
        result = pd.DataFrame()
    elif isinstance(out_molecules[0], pd.Series):
        result = pd.Series()
    else:
        return out_molecules
    
    for i in out_molecules:
        result = result.append(i)
    
    if sort_by is not None:
        result = result.sort_values(by=sort_by)

    return result

def combine_and_save_molecules(molecules, path, sort_by=None):
    out_molecules = list(molecules.values())

    if isinstance(out_molecules[0], pd.DataFrame):
        result = pd.DataFrame()
    elif isinstance(out_molecules[0], pd.Series):
        result = pd.Series()
    else:
        raise TypeError("Molecules must be pd.DataFrame of pd.Series")
    
    for i in out_molecules:
        result = result.append(i)
    
    if sort_by is not None:
        result = result.sort_values(by=sort_by)

    result.to_csv(path)


def combine_molecules(molecules):
    # result = pd.DataFrame()
    molecules = list(molecules.values())
    # time0 = time.time()
    result = pd.concat(molecules)

    """
    for i, df in enumerate(molecules, 1): 
        result = result.append(df)
        report_progress(i, len(molecules), time0, "combining molecules")
    """

    return result

"""
def pandas_chaining_mp_engine_continue(tasks, primary_molecules, molecules_dict, num_processes, sort_by=None, return_molecules=False):

    # primary_atoms : What key in atoms_config to use as first molecules to create jobs from.

    # Loop over tasks and pass output from each task as input to the next.
    for task in tasks:
        jobs = get_jobs_fast(task, primary_molecules, molecules_dict) # dont cache jobs

        primary_molecules = process_jobs_fast(jobs, num_processes=num_processes) # return as list of Data Frames, I need a dict
        # Cache primary_molecules ? I can then skip whole tasks...

    print("TASKS COMPLETED SUCCESSFULLY")

    if return_molecules == True:
        return (primary_molecules, molecules_dict)

    out_molecules = list(primary_molecules.values())

    if isinstance(out_molecules[0], pd.DataFrame):
        result = pd.DataFrame()
    elif isinstance(out_molecules[0], pd.Series):
        result = pd.Series()
    else:
        return out_molecules
    
    for i in out_molecules:
        result = result.append(i)
    
    if sort_by is not None:
        result = result.sort_values(by=sort_by)

    return result

"""



""" KLIP
    # combine molecules
    print("before combine molecules")
    atoms = combine_molecules(primary_molecules)

    print("after combine_molecules")

    # split molecules according to strategy required by next task
    if next_task["split_strategy"] == "ticker":
        print("before resplitting on ticker")

        # Write df to csv on disk 
        temp_path = cache_dir + "/temp_" + task["disk_name"] + ".csv"
        atoms.to_csv(temp_path) # Does it handle the index correctly?
        
        # Use old function to read it back...
        disk_name = "temp_" + task["disk_name"] # This should be irrelevant as the file should not exist in the cache
        
        atoms_config = atoms_configs[next_task["atoms_key"]]
        atoms_config["csv_path"] = temp_path
        atoms_config["cache"] = False

        primary_molecules = split_into_molecules(disk_name, atoms_config, split_strategy=next_task["split_strategy"], \
            cache_dir=cache_dir)

        os.remove(temp_path)
        print("after resplitting on ticker")

    else:
        print("before split_df_into_molecules")
        primary_molecules = split_df_into_molecules(atoms, next_task["split_strategy"], num_processes*molecules_per_process) 
        print("before split_df_into_molecules")
"""
                