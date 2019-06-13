import pandas as pd
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import numpy as np
from processing.engine import pandas_mp_engine
from helpers.helpers import get_calendardate_index, forward_fill_gaps

def detect_gaps(df):
    df.sort_index()
    df.index.drop_duplicates(keep="last")
    df["calendardate"] = df.index
    df['diff'] = df["calendardate"].diff(1)
    df["diff"] = df.index.to_series().diff().dt.total_seconds().fillna(0) / (60*60*24)

    df['OVER 1q'] = df["diff"] > 100
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(df["calendardate"])

    df['OVER 4q'] = df["diff"] > 370

    df1 = df.loc[df["OVER 1q"]==True]
    df4 = df.loc[df["OVER 4q"] == True]

    return (df1, df4)

def report_gaps(sf1):

    gaps_1q, gaps_4q = detect_gaps(sf1)

    report = pd.DataFrame(columns=["ticker", "gaps over 1q", "gaps over 4q"])

    if len(gaps_1q) > 0:
        report.at[0, "ticker"] = sf1.iloc[0]["ticker"]
        report.at[0, "gaps over 1q"] = len(gaps_1q)
    if len(gaps_4q) > 0:
        report.at[0, "gaps over 4q"] = len(gaps_4q)

    return report


def report_updates(sf1):
    ticker = sf1.iloc[0]["ticker"]
    vals = sf1.index.value_counts()
    vals = vals.rename("count")
    vals = vals.to_frame()
    vals["count"] = pd.to_numeric(vals["count"])
    report = pd.DataFrame()
    i = 0
    for caldate, count in vals.iterrows():
        if count[0] > 1:
            report.at[i, "calendardate"] = caldate
            report.at[i, "ticker"] = ticker
            report.at[i, "count"] = count[0]
            i += 1
    return report



def report_gaps_after_fill(sf1):
    sf1 = forward_fill_gaps(sf1, 3)

    gaps_1q, gaps_4q = detect_gaps(sf1)
    report = pd.DataFrame(columns=["ticker", "gaps over 1q", "gaps over 4q"])

    if len(gaps_1q) > 0:
        report.at[0, "ticker"] = sf1.iloc[0]["ticker"]
        report.at[0, "gaps over 1q"] = len(gaps_1q)
    if len(gaps_4q) > 0:
        report.at[0, "gaps over 4q"] = len(gaps_4q)

    return report


def report_updates_after_fill(sf1):
    sf1 = forward_fill_gaps(sf1, 3)

    ticker = sf1.iloc[0]["ticker"]
    vals = sf1.index.value_counts()
    vals = vals.rename("count")
    vals = vals.to_frame()
    vals["count"] = pd.to_numeric(vals["count"])
    report = pd.DataFrame()
    i = 0
    for caldate, count in vals.iterrows():
        if count[0] > 1:
            report.at[i, "calendardate"] = caldate
            report.at[i, "ticker"] = ticker
            report.at[i, "count"] = count[0]
            i += 1
    return report

def report_duplicate_datekeys(sf1):
    ticker = sf1.iloc[0].ticker
    
    duplicates = sf1.duplicated(subset="datekey")
    any_duplicates = any(duplicates==True)

    return pd.DataFrame(data=[any_duplicates], index=[ticker])




def report_date_relationship(sf1):
    ticker = sf1.iloc[0]["ticker"]

    sf1["datekey_after_caldate"] = sf1["datekey"] >sf1["calendardate"]

    sf1_selected = sf1.loc[sf1["datekey_after_caldate"] == False]
    report = pd.DataFrame()

    if len(sf1_selected) > 0:
        report.at[ticker, "datekey_after_caldate"] = False
    else:
        report.at[ticker, "datekey_after_caldate"] = True


    return report


if __name__ == "__main__":
    
    """
    # ./datasets/sharadar/SHARADAR_SF1_ARQ.csv
    sf1_arq = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ARQ.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"], index_col="calendardate")
    

    update_report_arq = pandas_mp_engine(callback=report_updates, atoms=sf1_arq, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    gap_report_arq = pandas_mp_engine(callback=report_gaps, atoms=sf1_arq, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    duplicate_report_arq = pandas_mp_engine(callback=report_duplicate_datekeys, atoms=sf1_arq, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    
    update_report_arq.to_csv("./datasets/testing/update_report_arq.csv", index=False)
    gap_report_arq.to_csv("./datasets/testing/gap_report_arq.csv", index=False)
    duplicate_report_arq.to_csv("./datasets/testing/duplicates_report_sf1_arq.csv")
    """


    """
    # "./datasets/sharadar/SHARADAR_SF1_ART.csv"
    # "./datasets/testing/sf1_art.csv"


    # Forward fill up to three quarters and see how it looks after
    # Needs to be done per ticker, so this work is moved into report* functions.
    


    update_report_art = pandas_mp_engine(callback=report_updates_after_fill, atoms=sf1_art, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    gap_report_art = pandas_mp_engine(callback=report_gaps_after_fill, atoms=sf1_art, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    duplicate_report_art = pandas_mp_engine(callback=report_duplicate_datekeys, atoms=sf1_art, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    # WRITTEN FOR FILLED VERSION
    update_report_art.to_csv("./datasets/testing/update_report_art_filled.csv", index=False)
    gap_report_art.to_csv("./datasets/testing/gap_report_art_filled.csv", index=False)
    duplicate_report_art.to_csv("./datasets/testing/duplicates_report_sf1_art.csv")

    sf1_art = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ART.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"]) # , index_col="calendardate"


    date_relationship_report_art = pandas_mp_engine(callback=report_date_relationship, atoms=sf1_art, \
        data=None, molecule_key='sf1', split_strategy= 'ticker', \
            num_processes=8, molecules_per_process=1)

    date_relationship_report_art.to_csv("./datasets/testing/date_relationship_report_art.csv")
    """

    """
    Notes:
    sf1_arq contains 12887 tickers.
    Datekey is unique for all tickers in sf1_art!!!

    dataset = pd.read_csv("./datasets/ml_ready_live/dataset_with_nans.csv", parse_dates=["date", "datekey", "timeout", "calendardate"], index_col="date")
    sf1_art = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ART")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataset.isnull().sum())
    
    print("dataset shape: ", dataset.shape)


    sep = pd.read_csv("./datasets/sharadar/SEP_PURGED.csv", parse_dates=["date"])


    print(sep.loc[(sep.ticker=="TXU") & (sep.date==pd.to_datetime("2007-02-26"))])

    """

    dataset = pd.read_csv("./datasets/ml_ready_live/dataset_with_nans.csv", parse_dates=["date", "datekey", "timeout", "calendardate"], index_col="date")



"""

Nan Status After fixing Nans:
New dataset length:  917147
Percentage dropped:  29.263322702679773 # Dropped due to requiring 24 months of sep history

30% of 917147 = 275144

4528197 / (94*917147) = 5,25 % data must be filled

Dropped columns:  set()
ticker                      0
date                        0
calendardate                0
datekey                     0
roaq                    47415
chtx                   201977
rsup                    45852
sue                     45886
cinvest                 50365
nincr                   40624
roavol                  50491
cashpr                 205584
cash                     2876
bm                       2060
currat                 200210
depr                    39584
ep                       5534
lev                      2063
quick                  200213
rd_sale                 24891
roic                   201153
salecash                 8849
saleinv                394269 # Drop
salerec                145686
sp                       5487
tb                       3678
sin                         0
tang                      228
debtc_sale             223143
eqt_marketcap            2091
dep_ppne                39584
tangibles_marketcap      2104
agr                      1003
cashdebt                 5748
chcsho                    690
chinv                    3202
egr                       868
gma                      4323
invest                    990
lgr                      1170
operprof                 4249
pchcurrat              202680
pchdepr                 57827
pchgm_pchsale           34518
pchquick               202689
pchsale_pchinvt        399868 # Drop
pchsale_pchrect        158569
pchsale_pchxsga         57044
pchsaleinv             406702 # Drop
rd                     600633 # Drop
roeq                     4275
sgr                     31402
grcapx                  44055
chtl_lagat                976
chlt_laginvcap           1007
chlct_lagat            202139
chint_lagat              9781
chinvt_lagsale          28769
chint_lagsgna           35706
chltc_laginvcap        202124
chint_laglt              9980
chdebtnc_lagat         202126
chinvt_lagcor          191040
chppne_laglt             1206
chpay_lagact           199032
chint_laginvcap          9794
chinvt_lagact          199053
pchppne                 36017
pchlt                    1170
pchint                   8790
chdebtnc_ppne          209860
chdebtc_sale           225027
age                         0
ipo                         0
ps                      25685
bm_ia                    2060
cfp_ia                   6553
chatoia                 10504
mve_ia                   1938
pchcapex_ia             44055
chpmia                  34564
herf                   678342 # Drop
ms                         67
industry                    0
indmom                      0
mom1m                       0
mom6m                       0
mom12m                      0
mom24m                      0
chmom                       0
mve                        66
beta                      604
betasq                    604
idiovol                   604
ill                     10093
dy                         66
turn                       37
dolvol                   1598
maxret                      0
retvol                      0
std_dolvol                  0
std_turn                   44
zerotrade                2528
return_1m                   0
return_2m                6987
return_3m               15637
timeout                     0
ewmstd_2y_monthly           0
return_tbm                  0
primary_label_tbm           0
take_profit_barrier         0
stop_loss_barrier           0

"""

