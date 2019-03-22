import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import datetime as dt
import sys

__all__ = ["lin_parts", "pandas_mp_engine"]

def lin_parts(num_atoms, num_threads):
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts

def get_jobs(atoms, data, callback, molecule_key, split_strategy, num_processes, molecules_per_process, **kwargs):
    """
    Seperating atoms into molecules based on different strategies and create jobs.
    """
    jobs = []

    if split_strategy == 'ticker':
        if 'ticker' not in atoms:
            raise Exception("Ticker column not in atoms")
        tickers = list(atoms["ticker"].unique())
        for ticker in tickers:
            molecule = atoms.loc[atoms["ticker"] == ticker]
            job = {
                molecule_key: molecule,
                'callback': callback,
            }
            job.update(kwargs)

            if data != None:
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
                    molecule_data[key] = df.loc[df["industry"] == ticker]
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
    
    # Use Dataset class to split the data.
    # arts = lin_parts(len(atoms), num_processes*molecules_per_process) # subject to change
    
    jobs = get_jobs(atoms, data, callback, molecule_key, split_strategy, num_processes, molecules_per_process, **kwargs)    

    print("Number of jobs: ", len(jobs))

    """
    Creating jobs is a time-consuming task.
    It would be nice if one could chain together data transformations
    Maybe not use time on this...
    """


    if num_processes == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_processes=num_processes)

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

def report_progress(job_num, num_jobs, time0, task):
    # Report progress as async jobs are completed
    ratio_of_jobs_completed = float(job_num)/num_jobs
    minutes_elapsed = (time.time()-time0)/60
    minutes_remaining = minutes_elapsed*(1/ratio_of_jobs_completed - 1)
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = time_stamp + " " + str(round(ratio_of_jobs_completed*100, 2)) + "% " + task + \
        " done after " + str(round(minutes_elapsed, 2)) + " minutes. Remaining " + \
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