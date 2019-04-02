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





# 2. Select features from SEP, SF1 etc.
selected_features = ["ticker", "date", "calendardate", "datekey"] + selected_sf1_features + selected_industry_sf1_features + selected_sep_features
dataset = dataset[selected_features]


dataset.to_csv(save_path + "/dataset.csv", index=False)


# 3. Remove or amend row with missing/NAN values (the strategy must be consistent with that for SEP data)

# MORE EFFORT SHOULD GO INTO THIS STEP, BUT I KEEP IT SIMPLE FOR NOW, DROPPING ROWS WITH ONE OR MORE NAN VALUES

# Drop first two (one of calendardate) years
dataset.sort_values(by=["ticker", "calendardate"])

result = pd.DataFrame()

for ticker in list(dataset.ticker.unique()):
    ticker_dataset = dataset.loc[dataset.ticker == ticker]

    min_caldate = ticker_dataset.calendardate.min()            
    calendardate_1y_after = get_calendardate_x_quarters_later(min_caldate, 4)

    ticker_dataset = ticker_dataset[ticker_dataset.calendardate >= calendardate_1y_after]

    result = result.append(ticker_dataset)

dataset = result


dataset.to_csv(save_path + "/dataset_dropped_first_year.csv", index=False)


dataset_no_nan = dataset.dropna(axis=0)

# 4. Write the almost ML ready dataset to disk

dataset_no_nan.to_csv(save_path + "/dataset_no_nan.csv")

# 5. Print statistics:
time_elapsed = datetime.datetime.now() - start_time

print("Dataset length: ", len(dataset))

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(dataset.isna().sum())

print("Dataset no nan length: ", len(dataset_no_nan))
print("Dropped: ", len(dataset) -  len(dataset_no_nan))
print("Time elapsed: ", time_elapsed)





