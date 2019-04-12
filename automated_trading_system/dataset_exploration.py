import pandas as pd

from automated_trading_system.helpers.helpers import forward_fill_gaps


def detect_gaps(df):
    df.sort_index()
    df.index.drop_duplicates(keep="last")
    df["calendardate"] = df.index
    df['diff'] = df["calendardate"].diff(1)
    df["diff"] = df.index.to_series().diff().dt.total_seconds().fillna(0) / (60 * 60 * 24)

    df['OVER 1q'] = df["diff"] > 100
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(df["calendardate"])

    df['OVER 4q'] = df["diff"] > 370

    df1 = df.loc[df["OVER 1q"] == True]
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
    any_duplicates = any(duplicates == True)

    return pd.DataFrame(data=[any_duplicates], index=[ticker])


def report_date_relationship(sf1):
    ticker = sf1.iloc[0]["ticker"]

    sf1["datekey_after_caldate"] = sf1["datekey"] > sf1["calendardate"]

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
    """

    dataset = pd.read_csv("./datasets/ml_ready_live/dataset_with_nans.csv",
                          parse_dates=["date", "datekey", "timeout", "calendardate"], index_col="date")
    sf1_art = pd.read_csv("./datasets/sharadar/SHARADAR_SF1_ART")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataset.isnull().sum())

    print("dataset shape: ", dataset.shape)

"""
dataset shape:  (1296547, 111)

Count of nans in all feature columns:
ticker                      0
calendardate                0
datekey                     0
roaq                   261368 : Fill
chtx                   463953 : Drop
rsup                   256912 : Fill
sue                    257080 : Fill
cinvest                271076 : Fill 
nincr                  234028 : Fill
roavol                 309735 : Fill
cashpr                 392202 : Fill
cash                   145779 : Definitely Fill
bm                     144270 : Definitely Fill
currat                 381946 : Fill
depr                   197338
ep                     155925
lev                    144278
quick                  381956
rd_sale                183760
roic                   353647
salecash               157995
saleinv                644457
salerec                329036
sp                     155696
tb                     150457
sin                    138407
tang                   138808
debtc_sale             422667 : Drop
eqt_marketcap          144356
dep_ppne               197338
tangibles_marketcap    144356
agr                    140499
cashdebt               155955
chcsho                 145498
chinv                  146387
egr                    140154
gma                    151784
invest                 140479
lgr                    140691
operprof               151575
pchcurrat              386805 : Fill
pchdepr                295039
pchgm_pchsale          261878
pchquick               386847
pchsale_pchinvt        686932 : Drop
pchsale_pchrect        403557
pchsale_pchxsga        288060
pchsaleinv             695748
rd                     841075 : Drop
roeq                   151717
sgr                    258156
grcapx                 275454
chtl_lagat             140523
chlt_laginvcap         140575
chlct_lagat            385857
chint_lagat            227662
chinvt_lagsale         251215
chint_lagsgna          258016
chltc_laginvcap        385797
chint_laglt            227893
chdebtnc_lagat         385819
chinvt_lagcor          438497 : Drop
chppne_laglt           140818
chpay_lagact           380435
chint_laginvcap        227672
chinvt_lagact          380516
pchppne                187057
pchlt                  140691
pchint                 225331
chdebtnc_ppne          397868
chdebtc_sale           425878 : Drop
age                         0
ipo                    138407
ps                     126832
bm_ia                   13655
cfp_ia                 109803
chatoia                228545
mve_ia                  13398
pchcapex_ia            275454
chpmia                 262065
herf                   998763 : Drop
ms                        127
industry                    0
indmom                  26039
mom1m                   17141
mom6m                   86388
mom12m                 162535
mom24m                 294142
chmom                  162535
mve                      3549
beta                   294746
betasq                 294746
idiovol                294746
ill                    176138
dy                     163005
turn                    47230
dolvol                  36287
maxret                  16506
retvol                  16507
std_dolvol              16507
std_turn                19670
zerotrade               26426
return_1m               14706
return_2m               26933
return_3m               41058
timeout                     0
ewmstd_2y_monthly       38744
return_tbm                  0
primary_label_tbm           0
take_profit_barrier     38744
stop_loss_barrier       38744

dtype: int64

"""
