import numpy as np
import pytest
import timeit
import multiprocessing as mp
from engine import pandas_mp_engine
import time
  

# Experimentation, not used
def heavy_task(process_length=1000, nr_processes=10000):
    normal_draws = np.random.normal(0, 0.01, size=(process_length, nr_processes))
    index_of_barrier_touches = barrier_touch(normal_draws, barrier_width=0.2)
    return index_of_barrier_touches


# Experimentation, not used
def heavy_task_using_mp(process_length=1000, nr_processes=10000):
    normal_draws = np.random.normal(0, 0.01, size=(process_length, nr_processes))
    num_threads=8
    parts = np.linspace(0, normal_draws.shape[0], min(num_threads, normal_draws.shape[0])+1)
    parts = np.ceil(parts).astype(int)
    jobs = []
    for i in range(1, len(parts)):
        jobs.append(normal_draws[:parts[i-1]:parts[i]]) # parallel jobs

    pool = mp.Pool(processes=num_threads)
    out = []
    outputs = pool.imap_unordered(barrier_touch, jobs) #jobs are parts of normal_draws
    for out_ in outputs:
        out.append(out_) # asynchronous response

    pool.close()
    pool.join()
    return out



def barrier_touch(normal_draws, barrier_width=0.5):
    #Find index of the earliest barrier touch
    touches = {}
    # np.log() -> natural logarithm element-wise
    # np.cumprod(axis 0) -> returns the comulative product of the elements in each row 
    gausian_processes = np.log((1+normal_draws).cumprod(axis=0)) 

    for j in range(normal_draws.shape[1]):
        for i in range(normal_draws.shape[0]):
            if gausian_processes[i, j] >= barrier_width or gausian_processes[i, j] <= -barrier_width:
                touches[j] = i
                continue
            
    return touches


@pytest.fixture(scope='module', autouse=True)
def setup():
    # Will be executed before the first test in the module
    

    yield
    # Will be executed after the last test in the module
    


def test_pandas_mp_engine():
    # global sep_extended 
    # assert sep_featured.loc[(sep_featured["ticker"] == "AAPL") & (sep_featured["date"] == date_1999_01_04)].iloc[-1]["return"] == pytest.approx((1.3530000000000002 / 1.473) -1)

    process_length = 1000
    nr_processes = 50000
    normal_draws = np.random.normal(0, 0.01, size=(process_length, nr_processes))

    time00 = time.time()
    touches0 = barrier_touch(normal_draws, barrier_width=0.2)
    time01 = time.time()

    time_standard = time01 - time00

    time10 = time.time()
    touches1 = pandas_mp_engine(barrier_touch, atoms=normal_draws, molecule_key="normal_draws", num_processes=8, \
        molecules_per_process=1, barrier_width=0.02)
    time11 = time.time()
    
    time_mp = time11 - time10

    assert time_standard/2 > time_mp