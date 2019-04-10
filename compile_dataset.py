import pandas as pd
import sys

# The order sets the order of columns in the final dataset
selected_sf1_features = [
    "roaq", 
    "chtx",
    "rsup",
    "sue",
    "cinvest",
    "nincr",
    "roavol",
    "cashpr", 
    "cash", 
    "bm",
    "currat",
    "depr",
    "ep",
    "lev",
    "quick",
    "rd_sale", 
    "roic",
    "salecash",
    "saleinv",
    "salerec",
    "sp",
    "tb",
    "sin",
    "tang",
    "debtc_sale",
    "eqt_marketcap",
    "dep_ppne",
    "tangibles_marketcap",
    "agr",
    "cashdebt",
    "chcsho",
    "chinv",
    "egr",
    "gma",
    "invest",
    "lgr",
    "operprof",
    "pchcurrat",
    "pchdepr",
    "pchgm_pchsale",
    "pchquick",
    "pchsale_pchinvt",
    "pchsale_pchrect",
    "pchsale_pchxsga",
    "pchsaleinv",
    "rd",
    "roeq",
    "sgr",
    "grcapx",
    "chtl_lagat",
    "chlt_laginvcap",
    "chlct_lagat",
    "chint_lagat",
    "chinvt_lagsale",
    "chint_lagsgna",
    "chltc_laginvcap",
    "chint_laglt",
    "chdebtnc_lagat",
    "chinvt_lagcor",
    "chppne_laglt", 
    "chpay_lagact",
    "chint_laginvcap",
    "chinvt_lagact",
    "pchppne", "pchlt",
    "pchint",
    "chdebtnc_ppne",
    "chdebtc_sale",
    "age",
    "ipo",
    # "profitmargin",
    # "chprofitmargin",
    # "industry",
    # "change_sales",
    "ps"
]


selected_industry_sf1_features = [
    "bm_ia",
    "cfp_ia",
    "chatoia",
    "mve_ia",
    "pchcapex_ia",
    "chpmia",
    "herf",
    "ms"
]

selected_sep_features = [
    "industry",
    # "sector",
    # "siccode",
    # Need for industry calculation
    # "mom12m_actual",
    "indmom",
    # Needed for beta calculation
    # "mom1w",
    # "mom1w_ewa_market", # This is used for idiovol
    # Calculated using forward filling and matrix multiplication
    "mom1m",
    "mom6m",
    "mom12m",
    "mom24m",
    # "mom12m_to_7m",
    "chmom",
    # Calculated only for samples
    "mve",
    "beta",
    "betasq",
    "idiovol",
    "ill",
    "dy",
    "turn",
    "dolvol",
    "maxret",
    "retvol",
    "std_dolvol",
    "std_turn",
    "zerotrade",
    # Labels
    "return_1m",
    "return_2m",
    "return_3m",
]

"""
Notes:
1. You need to filter out samples that rely on very old financial statements (see the age column)

"""


if __name__ == "__main__":

    # 1. Combine sep_featured and sf1_featured
    sep_featured = pd.read_csv("./datasets/ml_ready_live/sep_featured_done.csv", parse_dates=["date", "datekey"])
    
    sf1_featured = pd.read_csv("./datasets/ml_ready_live/sf1_featured.csv", parse_dates=["calendardate", "datekey"])

    sf1_featured = sf1_featured.drop_duplicates(subset=["ticker", "datekey"], keep="last")

    # sf1_featured_duplicates = sf1_featured[sf1_featured.duplicated(subset=["ticker", "datekey"])]
    # print(sf1_featured_duplicates)

    print(sep_featured.columns)
    print(sf1_featured.columns)

    print("sep_featured shape: ", sep_featured.shape)
    print("sf1_featured shape: ", sf1_featured.shape)

    dataset = sep_featured.merge(sf1_featured, on=["datekey", "ticker"], suffixes=("", "_sf1")) # Only the rows with matching datekey and ticker will be kept

    datekey_ticker = sep_featured[["ticker", "datekey"]]

    # 2. Select features from SEP, SF1 etc.
    selected_features = ["ticker", "date", "calendardate", "datekey"] + selected_sf1_features + selected_industry_sf1_features + selected_sep_features
    
    dataset = dataset[selected_features]
    
    print("Length of selected features: ", len(selected_features))
    print("Length of columns in dataset: ", len(dataset.columns))
    print("dataset shape: ", dataset.shape)
    print(dataset.head())


    # Drop first two (one of calendardate) years
    dataset.sort_values(by=["ticker", "calendardate"])

    dataset.to_csv("./datasets/ml_ready_live/dataset_with_nans.csv", index=False)


    # 3. Remove or amend row with missing/NAN values (the strategy must be consistent with that for SEP data)

    """
    Here is the start of some code to do some more complicated things to amend presence of NaNs.

    # MORE EFFORT SHOULD GO INTO THIS STEP, BUT I KEEP IT SIMPLE FOR NOW, DROPPING ROWS WITH ONE OR MORE NAN VALUES
    result = pd.DataFrame()

    for ticker in list(dataset.ticker.unique()):
        ticker_dataset = dataset.loc[dataset.ticker == ticker]

        min_caldate = ticker_dataset.calendardate.min()            
        calendardate_1y_after = get_calendardate_x_quarters_later(min_caldate, 4)

        ticker_dataset = ticker_dataset[ticker_dataset.calendardate >= calendardate_1y_after]

        result = result.append(ticker_dataset)

    dataset = result

    dataset.to_csv(save_path + "/dataset_dropped_first_year.csv", index=False)
    """


    dataset_no_nans = dataset.dropna(axis=0) # takes a long time

    # 4. Write the almost ML ready dataset to disk
    dataset_no_nans.to_csv("./datasets/ml_ready_live/dataset_without_nans.csv", index=False)


    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataset.isna().sum())

    """

    print("Dataset length: ", len(dataset))
    print("Dataset no nan length: ", len(dataset_no_nans))
    print("Dropped: ", len(dataset) -  len(dataset_no_nans))

    print("COMPLETED!")





