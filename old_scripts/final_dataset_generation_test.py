"""
In this script samples (based on SEP and SF1 data) is combined together into a final complete dataset, ready for ML.
Some samples might have to be dropped due to missing data in SF1_ART or SF1_ARQ.

The most recent row from sep_featured.csv, and the most recent SF1 row based on datekey is added to sep_sampled.csv.
- If SF1 row is too old compared to the sample date, the sample is dropped.
- If too much fabricated data was used in the generation of the SF1 features, the samples using this row
  is dropped.

"""

import datetime
import os
import pandas as pd
from packages.multiprocessing.engine import pandas_mp_engine
from sampling import extend_sep_for_sampling, rebase_at_each_filing_sampling
from sep_preparation import dividend_adjusting_prices_backwards, add_weekly_and_12m_stock_returns, add_equally_weighted_weekly_market_returns
from sep_industry_features import add_indmom
from sep_features import add_sep_features
from sf1_features import add_sf1_features
from sf1_industry_features import add_industry_sf1_features
from feature_selection import selected_industry_sf1_features, selected_sep_features, selected_sf1_features
from packages.helpers.helpers import get_calendardate_x_quarters_later


if __name__ == "__main__":

    sep_features = [
        "industry", "sector", "siccode", "mom12m_actual", "indmom", "mom1w", "mom1w_ewa_market", "return_1m",
        "return_2m", "return_3m", "mom1m", "mom6m", "mom12m", "mom24m", "mom12m_to_7m", "chmom", "mve", "beta", "betasq",
        "idiovol", "ill", "dy", "turn", "dolvol", "maxret", "retvol", "std_dolvol", "std_turn", "zerotrade"
    ]

    sf1_features = [
        "roaq", "chtx", "rsup", "sue", "cinvest", "nincr", "roavol", "cashpr", "cash", "bm", "currat", "depr", "ep", "lev", "quick",
        "rd_sale", "roic", "salecash", "saleinv", "salerec", "sp", "tb", "sin", "tang", "debtc_sale", "eqt_marketcap", "dep_ppne",
        "tangibles_marketcap", "agr", "cashdebt", "chcsho", "chinv", "egr", "gma", "invest", "lgr", "operprof", "pchcurrat", "pchdepr",
        "pchgm_pchsale", "pchquick", "pchsale_pchinvt", "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv", "rd", "roeq", "sgr",
        "grcapx", "chtl_lagat", "chlt_laginvcap", "chlct_lagat", "chint_lagat", "chinvt_lagsale", "chint_lagsgna", "chltc_laginvcap",
        "chint_laglt", "chdebtnc_lagat", "chinvt_lagcor", "chppne_laglt", "chpay_lagact", "chint_laginvcap", "chinvt_lagact",
        "pchppne", "pchlt", "pchint", "chdebtnc_ppne", "chdebtc_sale", "age", "ipo", "profitmargin", "chprofitmargin",
        "industry", "change_sales", "ps"
    ]

    industry_sf1_features = ["bm_ia", "cfp_ia", "chatoia", "mve_ia", "pchcapex_ia", "chpmia", "herf", "ms"]



    # Script configuration
    calculate_features = True
    write_to_disk = True
    sampling = True
    generate_sep_features = True
    prepare_sep = True
    generate_sf1_features = True


    num_processes = 4
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = "./datasets/testing/dataset_" + timestamp
    read_path = "./datasets/testing/dataset_20190327-153153"
    start_time = datetime.datetime.now()

    if write_to_disk == True:
        # create folder to write files to
        os.mkdir(save_path)

    if calculate_features == True:
        sep = pd.read_csv("./datasets/testing/sep.csv", parse_dates=["date"], index_col="date", low_memory=False)
        
        sf1_art = pd.read_csv("./datasets/testing/sf1_art.csv", parse_dates=["calendardate", "datekey", "reportperiod"],\
            index_col="calendardate", low_memory=False)
        
        sf1_arq = pd.read_csv("./datasets/testing/sf1_arq.csv", parse_dates=["calendardate", "datekey", "reportperiod"],\
            index_col="calendardate", low_memory=False)
        
        metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", parse_dates=["firstpricedate"], low_memory=False)

        print("Step 1 of xxx: Read Base Files - COMPLETED")

        # Sampling:
        if sampling == True:
        
            sep.sort_values(by=["ticker", "date"], inplace=True)

            sep_extended = pandas_mp_engine(callback=extend_sep_for_sampling, atoms=sep, \
                data={"sf1_art": sf1_art, "metadata": metadata}, \
                    molecule_key='sep', split_strategy='ticker', \
                        num_processes=num_processes, molecules_per_process=1)

            sep_extended.sort_values(by=["ticker", "date"], ascending=True, inplace=True)

            sep_sampled = pandas_mp_engine(callback=rebase_at_each_filing_sampling, atoms=sep_extended, data=None, \
                molecule_key='observations', split_strategy='ticker', num_processes=num_processes, molecules_per_process=1, \
                    days_of_distance=20)

            sep_sampled.sort_values(by=["ticker", "date"], ascending=True, inplace=True)
            

            if write_to_disk == True:
                sep_extended.to_csv(save_path + "/sep_extended.csv")
                sep_sampled.to_csv(save_path + "/sep_sampled.csv")

        else:
            # Dont think i need, or... It might be what I want to give to the sep preparation process
            sep_extended = pd.read_csv("./datasets/testing/sep_extended.csv", parse_dates=["date"], index_col="date")
            sep_sampled = pd.read_csv(read_path + "/sep_sampled.csv", parse_dates=["date"], index_col="date")

        print("Step 2 of xxx: Sampling - COMPLETED")

        # SEP
        if generate_sep_features == True:

            if prepare_sep == True:

                sep_adjusted = pandas_mp_engine(callback=dividend_adjusting_prices_backwards, atoms=sep_extended, data=None, \
                    molecule_key='sep', split_strategy= 'ticker', \
                        num_processes=num_processes, molecules_per_process=1)
                    
                sep_adjusted_plus_returns = pandas_mp_engine(callback=add_weekly_and_12m_stock_returns, atoms=sep_adjusted, data=None, \
                    molecule_key='sep', split_strategy= 'ticker', \
                        num_processes=num_processes, molecules_per_process=1)

                if write_to_disk == True:
                    sep_adjusted_plus_returns = sep_adjusted_plus_returns.sort_values(by=["ticker", "date"])
                    sep_adjusted_plus_returns.to_csv(save_path + "/sep_prepared_almost.csv")

                sep_prepared = pandas_mp_engine(callback=add_equally_weighted_weekly_market_returns, atoms=sep_adjusted_plus_returns, data=None, \
                    molecule_key='sep', split_strategy= 'date', \
                        num_processes=num_processes, molecules_per_process=1)

                if write_to_disk == True:
                    sep_prepared.to_csv(save_path + "/sep_prepared.csv")

            else:
                sep_prepared = pd.read_csv(read_path + "/sep_prepared.csv", parse_dates=["date"], index_col="date")

            print("Step 3 of xxx: Prepare for SEP calculations - COMPLETED")

            # Takes a long time... (or no)
            sep_prepared_plus_indmom = pandas_mp_engine(callback=add_indmom, atoms=sep_prepared, data=None, \
                molecule_key='sep', split_strategy= 'industry', \
                    num_processes=num_processes, molecules_per_process=1)


            sep_prepared_plus_indmom.sort_values(by=["ticker", "date"], inplace=True)


            sep_featured = pandas_mp_engine(callback=add_sep_features, atoms=sep_sampled, \
                data={'sep': sep_prepared_plus_indmom, "sf1_art": sf1_art}, molecule_key='sep_sampled', split_strategy= 'ticker', \
                    num_processes=num_processes, molecules_per_process=1)


            sep_featured.sort_values(by=["ticker", "date"], inplace=True)

            if write_to_disk == True:
                sep_featured.to_csv(save_path + "/sep_featured.csv")

        else:
            sep_featured = pd.read_csv(read_path + "/sep_featured.csv", parse_dates=["date"], index_col="date")

        print("Step 4 of xxx: SEP Calculations - COMPLETED")


        if generate_sf1_features == True:

            sf1_art_featured = pandas_mp_engine(callback=add_sf1_features, atoms=sf1_art, \
                data={"sf1_arq": sf1_arq, 'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'ticker', \
                    num_processes=num_processes, molecules_per_process=1)


            sf1_art_featured = pandas_mp_engine(callback=add_industry_sf1_features, atoms=sf1_art_featured, \
                data={'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'industry', \
                    num_processes=num_processes, molecules_per_process=1)

            if write_to_disk == True:
                sf1_art_featured.to_csv(save_path + "/sf1_art_featured.csv")
                
                # Select only features and save 
                cols = list(set(sf1_features + industry_sf1_features).intersection(set(sf1_art_featured.columns)))
                sf1_art_only_features = sf1_art_featured[cols]

                sf1_art_only_features.to_csv(save_path + "/sf1_art_only_features.csv")

        else:
            sf1_art_featured = pd.read_csv(read_path)
            
        print("Step 5 of xxx: SF1 Calculations - COMPLETED")
    
    else: # Do not calculate features
        sf1_art_featured = pd.read_csv(read_path + "/sf1_art_featured.csv", parse_dates=["calendardate", "datekey", "reportperiod"],\
            index_col="calendardate")

        sep_featured = pd.read_csv(read_path + "/sep_featured.csv", parse_dates=["date", "datekey"], index_col=["date"])


    """ At this point we have all features calculated, but they need to be assembled and features selected. """

    # 1. Merge features from SEP and SF1 into the samples data frame
    sep_featured = sep_featured.reset_index()
    sf1_art_featured = sf1_art_featured.reset_index()

    dataset = pd.merge(sep_featured, sf1_art_featured,  how='left', on=["ticker", "datekey"], suffixes=("sep", ""))



    # 2. Select features from SEP, SF1 etc.
    selected_features = ["ticker", "date", "calendardate", "datekey"] + selected_sf1_features + selected_industry_sf1_features + selected_sep_features
    dataset = dataset[selected_features]


    dataset.to_csv(save_path + "/dataset.csv", index=False)


    # 3. Remove or amend row with missing/NAN values (the strategy must be consistent with that for SEP data)

    # Drop first two years
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






