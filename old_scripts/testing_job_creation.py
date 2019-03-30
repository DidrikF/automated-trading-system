import pandas as pd
import time
import sys
import datetime as dt
"""
df = pd.DataFrame()

df.loc[0, "col1"] = "a"
df.loc[1, "col1"] = "b"
df.loc[2, "col1"] = "c"
df.loc[3, "col1"] = "d"
df.loc[4, "col1"] = "e"

for index, row in df.iterrows():
    if index < 2:
        continue
    print(row["col1"])
"""


# Splitting large dataframe into peaces:
"""
sep = pd.read_csv("./datasets/sharadar/SHARADAR_SEP.csv")


sep.sort_values(by=["ticker", "date"], inplace=True)




frames = []
first_index = None 
last_index = None
cur_ticker = None

num_rows = len(sep)
time0 = time.time()


# This takes 40 minutes

# I need an effective way to split on time and ticker... and it needs to handle running multiple functions
# over the molecules without having to separate them each time.

for index, row in sep.iterrows():
    if first_index is None:
        first_index = index
        cur_ticker = row["ticker"]

    if cur_ticker != row["ticker"]:
        # New ticker!!
        last_index = index
        ticker_df = sep.iloc[first_index:last_index]
        frames.append(ticker_df)

        # Reset
        first_index = index
        cur_ticker = row["ticker"]
        last_index = None

    if ((index % 50000) == 0) and (index != 0):
        report_progress(index, num_rows, time0, "Split SEP by ticker")

print("Length of frame: ", len(frames))
print(frames[0])
print(frames[100])

"""
from io import StringIO
import pandas as pd
import pickle


# Add this to the job creation process!!!
def report_progress(job_num, num_jobs, time0, task):
    # Report progress as async jobs are completed
    ratio_of_jobs_completed = float(job_num)/num_jobs
    minutes_elapsed = (time.time()-time0)/60
    minutes_remaining = minutes_elapsed*(1/ratio_of_jobs_completed - 1)
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = time_stamp + " " + str(round(ratio_of_jobs_completed*100, 2)) + "% " + str(job_num) + "/"+ str(num_jobs) + " " + task + \
        " done after " + str(round(minutes_elapsed, 2)) + " minutes. Remaining " + \
            str(round(minutes_remaining, 2)) + ' minutes.'
    
    if job_num < num_jobs: 
        sys.stderr.write(msg + '\r') # override previous line
    else: 
        sys.stderr.write(msg + '\n')
    return





store = {}

sep_path = "./datasets/sharadar/SHARADAR_SEP.csv"
test_path = "./datasets/testing/sep.csv"

cols = "ticker,date,open,high,low,close,volume,dividends,closeunadj,lastupdated\n"
time0 = time.time()

with open(sep_path) as csvfile:
    for index, line in enumerate(csvfile, 1):
        row = line.split(',')
        
        if row[0] not in store:
            store[row[0]] = cols
            store[row[0]] += line
        else:
            store[row[0]] += line

        if (index % 250000) == 0:
            report_progress(index, 34000000, time0, "Split SEP")
# print(store["AAPL"])

pickle_out = open("store.pickle","wb")
pickle.dump(store, pickle_out)
pickle_out.close()

pickle_in = open("store.pickle","rb")
store = pickle.load(pickle_in)

jobs = {}
time0 = time.time()
index = 1

for ticker, df_string in store.items():
    csv_file = StringIO(df_string)
    jobs[ticker] = pd.read_csv(csv_file, parse_dates=["date"], index_col="date", low_memory=False)

    if (index % 500) == 0:
            report_progress(index, 14000, time0, "Create Data Frame")

    index += 1


pickle_out = open("jobs.pickle","wb")
pickle.dump(jobs, pickle_out)
pickle_out.close()
