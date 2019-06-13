import numpy as np
import pandas as pd
import multiprocessing as mp

import time
import datetime as dt
import sys
import os.path
from os import listdir
from os.path import isfile, join
import pickle
from io import StringIO
import math
import re
from dateutil.parser import parse

def is_date(string, fuzzy=False):
    """
    Return whether the string can be parsed to a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


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
            molecule = atoms.loc[atoms["ticker"] == ticker] # Use group by instead
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


    elif split_strategy == "ticker_new":
        if 'ticker' not in atoms:
            raise Exception("Ticker column not in atoms")
        
        # Prepare data molecules
        molecule_data = {}
        if data is not None:
            for key, df in data.items():
                grouped_df = df.groupby("ticker")
                
                dfs = {}
                for ticker, molecule in grouped_df:
                    dfs[ticker] = molecule

                molecule_data[key] = dfs


        # Group atoms and make jobs
        grouped_molecules = atoms.groupby("ticker")
        for ticker, molecule in grouped_molecules:
            job = {
                molecule_key: molecule,
                'callback': callback,
            }
            job.update(kwargs)

            if data is not None:
                ticker_data = {}
                for key, df in data.items():
                    ticker_data[key] = molecule_data[key][ticker]

                job.update(ticker_data)

            jobs.append(job)


    # This split strategy is only used in testing
    elif split_strategy == "linspace":
        parts = lin_parts(len(atoms), num_processes*molecules_per_process)
        for i in range(1, len(parts)):
            index0 = parts[i-1]
            index1 = parts[i]

            job = {
                molecule_key: atoms[index0:index1],
                "callback": callback,
            }
            job.update(kwargs)
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
                molecule_key: atoms.loc[(atoms.index >= date0) & (atoms.index <= date1)],
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

    elif split_strategy == "industry_new":
        if 'industry' not in atoms:
            raise Exception("Industry column not in atoms")
        
        # Prepare data molecules
        molecule_data = {}
        if data is not None:
            for key, df in data.items():
                grouped_df = df.groupby("industry")
                
                dfs = {}
                for industry, molecule in grouped_df:
                    dfs[industry] = molecule

                molecule_data[key] = dfs


        # Group atoms and make jobs
        grouped_molecules = atoms.groupby("industry")
        for industry, molecule in grouped_molecules:
            job = {
                molecule_key: molecule,
                'callback': callback,
            }
            job.update(kwargs)

            if data is not None:
                industry_data = {}
                for key, df in data.items():
                    industry_data[key] = molecule_data[key][industry]

                job.update(industry_data)

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

    result = combine_molecules(out)

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

dir_path = os.path.dirname(os.path.realpath(__file__))

sf1_art_base = pd.read_csv(os.path.join(dir_path, "../datasets/sharadar/SF1_ART_BASE.csv"), parse_dates=["calendardate", "datekey"], index_col="calendardate")
sf1_arq_base = pd.read_csv(os.path.join(dir_path, "../datasets/sharadar/SF1_ARQ_BASE.csv"), parse_dates=["calendardate", "datekey"], index_col="calendardate")
sep_base = pd.read_csv(os.path.join(dir_path, "../datasets/sharadar/SEP_BASE.csv"), parse_dates=["date"], index_col="date")


def get_jobs_fast(task, primary_molecules, molecules_dict):
    """
    Create jobs for new engine.
    """
    jobs = []
    for job_key, molecule in primary_molecules.items():
        # job_key is AAPL, Electronics etc.
        # molecule is this jobs atoms
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
                    print("Did not find molecule_dict_name: ", molecule_dict_name, " with job_key: ", job_key)
                    print("Keys of molecules_dict: ", molecules_dict.keys())
                    print("Keys of desired ({}) molecules_dict: {}".format(molecule_dict_name, molecules_dict[molecule_dict_name].keys()))

                    if molecule_dict_name == "sf1_art":
                        data_molecule = sf1_art_base
                    elif molecule_dict_name == "sf1_arq":
                        data_molecule = sf1_arq_base
                    elif re.match(".*sep.*", molecule_dict_name):
                        data_molecule = sep_base
                    else:
                        data_molecule = pd.DataFrame()

                job_data[kw_name] = data_molecule
            job.update(job_data)

        jobs.append(job)

    return jobs

def process_jobs_fast(jobs, task=None, num_processes=8, sort_by=[]):
    """
    Use multiprocessing to process jobs.
    """
    if task is None:
        task = jobs[0]['callback'].__name__
        
        pool = mp.Pool(processes=num_processes)
        outputs = pool.imap_unordered(expandCall_fast, jobs)
        out = {}
        time0 = time.time()
        i = 1
        # Process asynchronous output, report progress
        for out_ in outputs:
            # print("Job name: ", out_[0])
            # print(out_[1].head(1))
            out[out_[0]] = out_[1].sort_values(by=sort_by)
            report_progress(i, len(jobs), time0, task)
            i += 1
        
        pool.close()
        pool.join()
        return out

def expandCall_fast(kwargs):
    """
    Extract callback from job object
    """
    callback = kwargs['callback']
    job_key = kwargs['job_key']
    del kwargs['callback']
    del kwargs['job_key']
    # print(kwargs.keys())
    out = callback(**kwargs)
    return (job_key, out)


def split_df_into_molecules(atoms, split_strategy, num_molecules):
    """
    Splits atoms into smaller molecules for concurrent processing.
    """
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
        
    elif split_strategy == 'ticker':
        if 'ticker' not in atoms:
            raise Exception("Ticker column not in atoms")
        grouped_molecules = atoms.groupby("ticker")
        for ticker, molecule in grouped_molecules:
            dfs[ticker] = molecule
    else:
        raise ValueError("split_strategy cannot be " + split_strategy + ". Only 'ticker', 'industry' and 'date' are supported.")

    return dfs


def combine_and_save_molecules(molecules, path, sort_by=None):
    """
    Combine molecules together into a single dataframe and write the result to disk.
    """
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
    """
    Combine molecules in an effective manner.
    """
    if isinstance(molecules, dict):
        molecules = list(molecules.values())
    
    result = pd.concat(molecules, sort=True)
    return result


def pandas_chaining_mp_engine(tasks, primary_atoms, atoms_configs, split_strategy, num_processes, cache_dir, \
    sort_by=None, molecules_per_process=5, resume=False):
    """
    Multiprocessing engine that is able to process a chain of tasks. Usefull for more complex dataprocessing pipelines.
    """
    split_strategy_changed = False
    primary_molecules = None # Might resume from cache, or set from parsed atoms found in molecules_dict
    # You might want to resume at a later task, if that is the case set primary_molecules from the cache and skip/ foregoing tasks
    if resume == True:
        """ Resume from the task that produced the latest cached molecules """

        # Get index of task to resume from:
        files_in_cache = [f for f in listdir(cache_dir) if isfile(join(cache_dir, f))]
        task_cache_files = [(i, task["disk_name"] + ".pickle") for i, task in enumerate(tasks)]
        task_cache_files.reverse()
        for i_file in task_cache_files:
            task_index = i_file[0]
            file_name = i_file[1]
            if file_name in files_in_cache:
                # resume at $task_index with file $file
                print("resuming at task index: ", task_index, " with file: ", file_name)
                pickle_path = cache_dir + "/" + file_name
                pickle_in = open(pickle_path,"rb")
                primary_molecules = pickle.load(pickle_in)
                # It might be that primary_molecules is not split correctly!
                task_we_are_loading_output_from = tasks[task_index]
                new_first_task = tasks[task_index + 1]
                if task_we_are_loading_output_from["split_strategy"] != new_first_task["split_strategy"]:
                    print("Resplitting primary_molecules when resuming because they were cached in the wrong format, needed: ", new_first_task["split_strategy"])
                    atoms = combine_molecules(primary_molecules)
                    primary_molecules = split_df_into_molecules(atoms, new_first_task["split_strategy"], num_processes*molecules_per_process)

                tasks = tasks[(task_index+1):] # Drop the task associated with the task-output we are loading
                if new_first_task["split_strategy"] != split_strategy:
                    print("changing split strategy to : ", new_first_task["split_strategy"])
                    split_strategy = new_first_task["split_strategy"] # Will change how we parse atoms
                    split_strategy_changed = True
                break

    # Ready the data one time, and then use a simple dictionary lookup to get the correct df when creating jobs
    molecules_dict = {} # { "AAPL": [df1, df2, ...], ...

    for disk_name, atoms_config in atoms_configs.items():
        pickle_path = cache_dir + '/' + disk_name + '.pickle'

        if os.path.isfile(pickle_path):
            print("Loading pickle: " + pickle_path)
            pickle_in = open(pickle_path,"rb")
            molecules_dict[disk_name] = pickle.load(pickle_in) 
            # If we are resuming a task we don't know if this is split correctly. This is amended below.
            # The split is only incorrect, if split_strategy changed from the original value given as a function argument!
            # And this only matters if the new first task has a "data" value that is not none in its config!
            if (split_strategy_changed == True) and (tasks[0]["data"] is not None):
                new_first_task = tasks[0]
                for molecules_dict_name in new_first_task["data"].values():
                    atoms = combine_molecules(molecules_dict[molecules_dict_name])
                    molecules_dict[molecules_dict_name] = split_df_into_molecules(atoms, new_first_task["split_strategy"], num_processes*molecules_per_process)
        else:
            print("Reading and parsing: ", atoms_config["csv_path"])
            atoms = pd.read_csv(atoms_config["csv_path"], parse_dates=atoms_config["parse_dates"], index_col=atoms_config["index_col"], low_memory=False)
            if atoms_config["sort_by"] is not None:
                atoms = atoms.sort_values(by=atoms_config["sort_by"])
            molecules_dict[disk_name] = split_df_into_molecules(atoms, split_strategy, num_processes*molecules_per_process) # Split strategy here might have changed

            if (atoms_config["cache"]) == True: # and (resume == False): 
                # Only cache parsed atoms, when we are not resuming (which could have changed the split_strategy, making the contents 
                # of the cache harder to reason about)
                pickle_out = open(pickle_path,"wb")
                pickle.dump(molecules_dict[disk_name], pickle_out)
                pickle_out.close()
    
    # Any data needed by future tasks as data in the molecules_dict, that are avialable in the cache, should also be loaded!!

    if primary_molecules is None: # If we are not resuming, this is split correctly.
        primary_molecules = molecules_dict[primary_atoms]
    

    # Loop over tasks and pass output from each task as input to the next.
    last_index = len(tasks) - 1
    start_time_tasks = time.time()
    for index, task in enumerate(tasks):
        """ 
        NOTE: For each iteration of the loop primary_molecules and required parts of molecules_dict must
        be split correctly before starting.
        """
        minutes_elapsed = (time.time()-start_time_tasks)/60
        print("Step ", str(index+1), " of ", str(len(tasks)), " - " + task["name"] + " - Time elapsed: ", str(round(minutes_elapsed, 2)), " minutes.")

        # The molecules dict is not needed for all tasks, if none is listed in task["data"] it is simply never touched.
        jobs = get_jobs_fast(task, primary_molecules, molecules_dict)

        primary_molecules = process_jobs_fast(jobs, num_processes=num_processes, sort_by=sort_by) # return as list of Data Frames, I need a dict

        # Cache result as pickle of DataFrames
        if task["cache_result"] == True:
            print("Cacheing as pickle result from task: ", task["name"])
            molecules_to_cache = primary_molecules
            if (task.get("add_to_molecules_dict", False) == True) and (task["split_strategy"] != task["split_strategy_for_molecule_dict"]):
                # In this case we want to cache in the format we are requiring in the molecules_dict. That way, this will be split
                # correctly when resuming.
                atoms = combine_molecules(primary_molecules)
                molecules_to_cache = split_df_into_molecules(atoms, task["split_strategy_for_molecule_dict"], num_processes*molecules_per_process)


            cache_path = cache_dir + "/" + task["disk_name"] + ".pickle"
            pickle_out = open(cache_path,"wb")
            pickle.dump(molecules_to_cache, pickle_out)
            pickle_out.close()


        # Prepare Primary Molecules for next iteration
        if index < last_index:
            next_task = tasks[index+1]
            if next_task["split_strategy"] != task["split_strategy"]:
                atoms = combine_molecules(primary_molecules)
                primary_molecules = split_df_into_molecules(atoms, next_task["split_strategy"], num_processes*molecules_per_process) 

        # Prepare data molecule for the next iteration
        if index < last_index:
            next_task = tasks[index+1]
            # If the next task has any other data than the primary molecules it needs, we need to split those as well according to the required strategy.
            if next_task["data"] is not None: 
                for molecules_dict_name in next_task["data"].values():
                    
                    curret_split_strategy = infer_split_strategy(molecules_dict[molecules_dict_name])
                    
                    if curret_split_strategy != next_task["split_strategy"]:
                        atoms = combine_molecules(molecules_dict[molecules_dict_name])
                        molecules_dict[molecules_dict_name] = split_df_into_molecules(atoms, next_task["split_strategy"], num_processes*molecules_per_process)


        # Add molecules to molecules_dict for later use by another task
        # This addresses the constraint that all needed molecules must be split correctly before the start of each iteration of this loop.
        if task.get("add_to_molecules_dict", False) == True: 
            molecules_to_add_to_molecules_dict = primary_molecules
            
            # We just split primary_molecules for the next task, 
            # if this is not the format we want to cache the molecules, they must be split in the correct way.
            if task["split_strategy_for_molecule_dict"] != next_task["split_strategy"]:
                atoms = combine_molecules(primary_molecules)
                molecules_to_add_to_molecules_dict = split_df_into_molecules(atoms, task["split_strategy_for_molecule_dict"], num_processes*molecules_per_process)

            molecules_dict[task["disk_name"]] = molecules_to_add_to_molecules_dict


    print("TASKS COMPLETED SUCCESSFULLY")

    result = combine_molecules(primary_molecules)

    if sort_by is not None:
        result = result.sort_values(by=sort_by)

    return result


def infer_split_strategy(molecule_dict: dict):
    """
    Detect what split strategy was used to create $molecule_dict.
    """
    first_key: str = list(molecule_dict.keys())[0]
    if first_key.capitalize() == first_key:
        return "ticker"
    elif is_date(first_key):
        return "date"
    else:
        return "industry"
