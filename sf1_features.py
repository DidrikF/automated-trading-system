import pandas as pd
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta

from packages.helpers.helpers import print_exception_info, get_most_up_to_date_10k_filing,\
    get_most_up_to_date_10q_filing, get_calendardate_index
from packages.multiprocessing.engine import pandas_mp_engine
"""
Step-by-Step Dataset Construction:
1. Extend the SEP dataset with information usefull for sampling (most recent 10-K filing date, Industry classifications)
2. Use different sampling techniques to get monthly observations
    1. At first use timebars (sampling at a fixed time interval), but try to respect the different fiscal years
3. Calculate the various price and volume based features
4. Compute features based on SF1
5. Add inn SF1 and OTHER data
6. Select the features you want and combine into a ML ready dataset
"""

# Rewrite for multiprocessing engine

def add_sf1_features(sf1_art, sf1_arq, metadata):
    ticker = metadata["ticker"]
    first_calendardate = None

    """
    Approximately 4300 stocks are missing 1 or more consecutive quarter updates.
    Of these 4300 stocks, 1800 of them are missing more the 1 year worth of consecutive quarters.
    Quarterly reports are only available for approximately 12000 stocks, 
    """
    """
    Over 1800 stocks have a gap of more than one year of data. I consider it not appropriate 
    to forward fill this much. 
    I think the best (and simplest) solution is to amend the data as much as possible by forward filling up to
    3 missing quarters of SF1_ART. This ensures that when calculating features based on one year of data
    two of the same reports are not used.

    When it comes to filling in SF1_ARQ it comes with the disadvantage that seasonality properties are 
    distorted. 

    I think I will accept up to three quarters being forward filled. 

    What about gaps greater 3 quarters? It does not make sense to fabricate data into the future after the company
    was delisted. Also it does not make sense to calculate features for rows that where forward filled 

    Solutions to the problems outlined above:
    1. As long as the most recent (current) 10q included in a calculation is not forward filled, it is ok.
    2. Rows that are calculated using foreward filled data is marked according to how many fabricated rows 
       where used
    3. Keep track of the original calendardate index so that when feature calculation is done, the
       dataset can get downsampled again. This will ensure that:
        - all "observations used in trading" is based on a "just released" statement.
        - after being delisted, forward filled statements are not used.
    
    Strategy:
    1. Take in unaltered sf1_art and sf1_arq for a ticker.
    2. If no data in sf1_arq -> return pd.DataFrame()
      - Or just skip calculating any features using quarterly data (I like this more, give more flexibility 
        when putting together ML ready datasets later)
    3. Capture current index (calendardates) of sf1_art and sf1_arq (needed to downsample later)
    4. Forward fill up to three consecutive missing rows (in terms of calendardate)
    5. Drop the still unfilled rows
    6. Extract most recent and 1 year old sf1_art rows (representing the report periods one year appart, but use most recent update (highest datekey))
    7. If the one year old sf1_art row is not available, almost none of the features can be calculated.
       - SKIP CALCULATIONS (AND DROP ROW?)
    7. Extract most recent 8 (9???) rows from sf1_arq (representing the 8 preceding report periods, but use the most recent update (highest datekey))
    8. If a quarter is missing, it is not a deal breaker. Do checks on the level of each individual feature and
       calculate as much as possible.
    8. Capture metadata on the rows (reports) used which later can be used to select features or drop some timeframes.
        1. datekey of each sf1_art and arq row
        2. which rows where filled and not original
        3. How many missing timeframes/row/??? where used in the construction of the features for the row.

    9. Downsample to the original index (dropping the forward filled rows)
    10. Return result.
    """

    # SF1_ART PREPARATION_____________________________________________________________
    # Reindex
    calendardate_index = get_calendardate_index(sf1_art.iloc[0]["calendardate"], sf1_art.iloc[-1]["calendardate"])
    sf1_art_reindexed = sf1_art.reindex(calendardate_index)
    
    # Count occurences of the same datekey
    sf1_art_reindexed["datekey"].value_counts()

    # If more than 4 consecutive rows have the same datekey, there is missing data for a year
    # this disqualifies the stock, and is dropped from the dataset.
        
    # Foreward fill gaps
    # I only want to fill the added rows by the reindex operation.
    sf1_art_filled = sf1_art_reindexed.fillna(method="ffill") # This might fill in too much...



    # SF1_ARQ PREPARATION_____________________________________________________________
    # SF1_ARQ may not even be available (how is this handled in multiprocessing engine?)

    # If SF1_ARQ is not available, drop the stock from the dataset.
    # Could it be that arq is only reported each year???

    # If a row is missing, use reindex and foreward fill




    # MAKE SURE THE ABOVE IS ROBUST THROUGH TESTING

    print_once = True

    for caldate_cur, art_row_cur in sf1_art.iterrows():
        datekey_cur = art_row_cur["datekey"]
        # The row might be an update of a earlier report. But this is ok, because I simply
        # want to calculate the most up to date features for the given datekey. It is up to 
        # the extend_sep_for_sampling function to find the appropriate datekey (filing or update)
        # for each sample.
        
        if first_calendardate is None:
            first_calendardate = art_row_cur["calendardate"]
        
        if art_row_cur["calendardate"] <= first_calendardate + relativedelta(months=23):
            continue

        #What to do when a 10K is missing?
        art_row_1y_ago = get_most_up_to_date_10k_filing(sf1_art, datekey_cur, 1)
        art_row_2y_ago = get_most_up_to_date_10k_filing(sf1_art, datekey_cur, 2) # Only used for capex growth...

        # What to do when a 10Q i missing?
        arq_row_cur = get_most_up_to_date_10q_filing(sf1_arq, datekey_cur, 0)
        arq_row_1q_ago = get_most_up_to_date_10q_filing(sf1_arq, datekey_cur, 1)
        arq_row_2q_ago = get_most_up_to_date_10q_filing(sf1_arq, datekey_cur, 2) 
        arq_row_3q_ago = get_most_up_to_date_10q_filing(sf1_arq, datekey_cur, 3)
        arq_row_4q_ago = get_most_up_to_date_10q_filing(sf1_arq, datekey_cur, 4)
        arq_row_5q_ago = get_most_up_to_date_10q_filing(sf1_arq, datekey_cur, 5)
        arq_row_6q_ago = get_most_up_to_date_10q_filing(sf1_arq, datekey_cur, 6)
        arq_row_7q_ago = get_most_up_to_date_10q_filing(sf1_arq, datekey_cur, 7)
        # this is the ninth quarter, is it needed? want to set the limit at 2 years of unbroken 
        # history of data available...
        arq_row_8q_ago = get_most_up_to_date_10q_filing(sf1_arq, datekey_cur, 8) 
        
        arq_rows = [arq_row_cur, arq_row_1q_ago, arq_row_2q_ago, arq_row_3q_ago, arq_row_4q_ago, arq_row_5q_ago, arq_row_6q_ago, arq_row_7q_ago, arq_row_8q_ago]


        if print_once == True:
            print("Current datekey:", datekey_cur)
            for row in arq_rows:
                print(row["calendardate"])
            print_once = False


        """
        At this point, if the above is robust, calculations should be fairly straight forward.
        Some filings might have a missing value use in a calculation, this will cause the feature to be missing/NAN
        for this date. Whether to exclude the rows plagued by missing values or try to amend them needs to be 
        decided. 
        If they are to be amended, i guess forward filling is the simplest strategy.

        """

         # _____________________QUARTER FILING BASED FEATURES START_______________________

        # I might want to implement approximations for those companies that do not have quarterly statements
        if (not arq_row_cur.empty) and (not arq_row_1q_ago.empty):
            # CALCULATE QUARTER TO QUARTER FEATURES
            
            # Return on assets (roaq), Formula: SF1[netinc]q-1 / SF1[assets]q-2
            if arq_row_1q_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "assets"] = arq_row_cur["netinc"] / arq_row_1q_ago["assets"]

        if (not arq_row_cur.empty) and (not arq_row_4q_ago.empty):
            # CALCULATE FEATURES BASED ON THE SAME QUARTER FOR THAT LAST TWO YEARS
            
            # Change in tax expense (chtx), Formula: (SF1[taxexp]q-1 / SF1[taxexp]q-5) - 1
            if arq_row_4q_ago["taxexp"] != 0:
                sf1_art.at[datekey_cur, "chtx"] = (arq_row_cur["taxexp"] / arq_row_4q_ago["taxexp"]) - 1

            # Revenue surprise (rsup), Formula: ( SF1[revenueusd]q-1 - SF1[revenueusd]q-5 ) / SF1[marketcap]q-1
            if arq_row_1q_ago["marketcap"] != 0:
                sf1_art.at[datekey_cur, "rsup"] = (arq_row_1q_ago["revenueusd"] - arq_row_4q_ago["revenueusd"]) / arq_row_1q_ago["marketcap"]

            # Earnings Surprise (sue), Formula: (SF1[netinc]q-1 - SF1[netinc]q-5) / SF1[marketcap]q-1
            if arq_row_1q_ago["marketcap"] != 0:
                sf1_art.at[datekey_cur, "sue"] = (arq_row_1q_ago["netinc"] - arq_row_4q_ago["netinc"]) / arq_row_1q_ago["marketcap"]
            
            

        # MORE ADVANCED MULTI-QUARTER CALCULATIONS

        # Corporate investment (cinvest), 
        # "Change over one quarter in net PP&E (ppentq) divided by sales (saleq) - average of this variable for prior 3 quarters; if saleq = 0, then scale by 0.01."
        # Formula: 
        # (SF1[ppnenet]q-1 - SF1[ppnenet]q-2) / SF1[revenueusd]q-1 - avg((SF1[ppnenet]q-i - SF1[ppnenet]q-i-1) / SF1[revenueusd]q-i, i=[2,3,4]) NB: if sales is zero scale change in ppenet by 0.01
        if (not arq_row_cur.empty) and (not arq_row_1q_ago.empty) and (not arq_row_2q_ago.empty) and (not arq_row_3q_ago.empty) and (not arq_row_4q_ago.empty):
            # Most recent quarter's chppne/sales
            if arq_row_cur["revenueusd"] != 0:
                chppne_sales_cur = (arq_row_cur["ppnenet"] - arq_row_1q_ago["ppnenet"]) / arq_row_cur["revenueusd"]
            else:
                chppne_sales_cur = (arq_row_cur["ppnenet"] - arq_row_1q_ago["ppnenet"]) * 0.01
            # Previous three quarters of chppne/sales
            if arq_row_1q_ago["revenueusd"] != 0:
                chppne_sales_q_1 = (arq_row_1q_ago["ppnenet"] - arq_row_2q_ago["ppnenet"]) / arq_row_1q_ago["revenueusd"]
            else:
                chppne_sales_q_1 = (arq_row_1q_ago["ppnenet"] - arq_row_2q_ago["ppnenet"]) * 0.01

            if arq_row_2q_ago["revenueusd"] != 0:
                chppne_sales_q_2 = (arq_row_2q_ago["ppnenet"] - arq_row_3q_ago["ppnenet"]) / arq_row_2q_ago["revenueusd"]
            else:
                chppne_sales_q_2 = (arq_row_2q_ago["ppnenet"] - arq_row_3q_ago["ppnenet"]) * 0.01

            if arq_row_3q_ago["revenueusd"] != 0:
                chppne_sales_q_3 = (arq_row_3q_ago["ppnenet"] - arq_row_4q_ago["ppnenet"]) / arq_row_3q_ago["revenueusd"]
            else:
                chppne_sales_q_3 = (arq_row_3q_ago["ppnenet"] - arq_row_4q_ago["ppnenet"]) * 0.01
            
            sf1_art.at[datekey_cur, "cinvest"] = chppne_sales_cur - ( (chppne_sales_q_1 + chppne_sales_q_2 + chppne_sales_q_3) / 3 )
        

        if arq_row_cur["revenueusd"] != 0:
            chppne_sales_q_1 = (arq_row_cur["ppnenet"] - arq_row_1q_ago["ppnenet"]) / arq_row_cur["revenueusd"]
        else: 
            chppne_sales_q_1 = (arq_row_cur["ppnenet"] - arq_row_1q_ago["ppnenet"]) * 0.01

        
        # Number of earnings increases (nincr)	Barth, Elliott & Finn 	1999, JAR 	"Number of consecutive quarters (up to eight quarters) with an increase in earnings
        # (ibq) over same quarter in the prior year."	for (i = 1, i++, i<=8) { if(SF1[netinc]q-i > SF1[netinc]q-i-4): counter++; else: break }
        nr_of_earnings_increases = 0
        for i in range(4):
            cur_row = arq_rows[i]
            prev_row = arq_rows[i+4]
            if (not cur_row.empty) and (not prev_row.empty):
                cur_netinc = cur_row["netinc"]
                prev_netinc = prev_row["netinc"]
                if cur_netinc > prev_netinc:
                    nr_of_earnings_increases += 1
                else:
                    break
            else:
                break

        sf1_art.at[datekey_cur, "nincr"] = nr_of_earnings_increases

        # Earnings volatility (roavol)	Francis, LaFond, Olsson & Schipper 	2004, TAR 	
        # "Standard deviaiton for 16 quarters of income before extraordinary items (ibq) divided by average total assets (atq)."	
        # Formula: std(SF1[netinc]q) / avg(SF1[assets]q) for 8 - 16 quarters
        
        # OBS: Maybe be a litt less strict...
        if (not arq_row_cur.empty) and (not arq_row_1q_ago.empty) and (not arq_row_2q_ago.empty) \
            and (not arq_row_3q_ago.empty) and (not arq_row_4q_ago.empty) and (not arq_row_5q_ago.empty) \
            and (not arq_row_6q_ago.empty) and (not arq_row_7q_ago.empty) and (not arq_row_8q_ago.empty):
            sum_netinc_assets = 0
            for row in arq_rows:
                sum_netinc_assets += row["netinc"] / row["assetsavg"]
            mean_netinc_assets = sum_netinc_assets / len(arq_rows)
            sum_sqrd_distance_netinc_assets = 0
            for row in arq_rows:
                netinc_assets = row["netinc"] / row["assetsavg"]
                sum_sqrd_distance_netinc_assets += ((netinc_assets - mean_netinc_assets))**2

            std_netinc_assets = sum_sqrd_distance_netinc_assets / len(arq_rows)
            
            sf1_arq.at[datekey_cur, "roavol"] = std_netinc_assets


        # _____________________QUARTER FILING BASED FEATURES END_______________________

        # _______________________YEARLY FILING BASED FEATURES START_________________________

        # CALCULATIONS IN PREPARATION FOR LATER


        # CALCULATIONS USING ONLY CURRENT SF1_ART ROW

        # Cash productivity (cashpr), Formula: (SF1[marketcap]t-1 + SF1[debtnc]t-1 - SF1[assets]t-1) / SF1[cashneq]t-1
        if art_row_cur["cashneq"] != 0:
            sf1_art.at[datekey_cur, "cashneq"] = (art_row_cur["marketcap"] + art_row_cur["debtnc"] - art_row_cur["assets"]) / art_row_cur["cashneq"]
        # Cash (cash), Formula: SF1[cashnequsd]t-1 / SF1[assetsavg]t-1
        if art_row_cur["assetsavg"] != 0:
            sf1_art.at[datekey_cur, "cash"] = art_row_cur["cashnequsd"] / art_row_cur["assetsavg"]

        # Book to market (bm), Formula: SF1[equityusd]t-1 / SF1[marketcap]t-1
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[datekey_cur, "bm"] = art_row_cur["equityusd"] / art_row_cur["marketcap"]

        # Cash flow to price ratio (cfp), Formula: SF1[ncfo]t-1 / SF1[marketcap]t-1
        if art_row_cur["marketcap"] != 0:
            art_row_cur["cfp"] = art_row_cur["ncfo"] / art_row_cur["marketcap"]

        # Current ratio (currat), Formula: SF1[assetsc]t-1 / SF1[liabilitiesc]t-1
        if art_row_cur["liabilitiesc"] != 0:
            sf1_art.at[datekey_cur, "liabilitiesc"] = art_row_cur["assetsc"] / art_row_cur["liabilitiesc"]

        # Depreciation over PP&E (depr), Formula: SF1[depamor]t-1 / SF1[ppnenet]t-1
        if art_row_cur["ppnenet"] != 0:
            sf1_art.at[datekey_cur, "depr"] = art_row_cur["depamor"] / art_row_cur["ppnenet"]

        # Earnings to price (ep), Formula: SF1[netinc]t-1 / SF1[marketcap]t-1 
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[datekey_cur, "ep"] = art_row_cur["netinc"] / art_row_cur["marketcap"]

        # Leverage (lev), Formula: SF1[liabilities]t-1 / SF1[marketcap]t-1
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[datekey_cur, "lev"] = art_row_cur["liabilities"] / art_row_cur["marketcap"]
        
        # Quick ratio (quick), Formula: (SF1[assetsc]t-1 - SF1[inventory]t-1) / SF1[liabilitiesc]t-1
        if (art_row_cur["liabilitiesc"] != 0):
            sf1_art.at[datekey_cur, "quick"] = (art_row_cur["assetsc"] - art_row_cur["inventory"]) / art_row_cur["liabilitiesc"]

        # R&D to market capitalization (rd_mve), Formula: SF1[rnd]t-1 / SF1[marketcap]t-1
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[datekey_cur, "rd_mve"] = art_row_cur["rnd"] / art_row_cur["marketcap"]
        
        # R&D to sales (rd_sale), Formula: SF1[rnd]t-1 / SF1[revenueusd]t-1
        if art_row_cur["revenueusd"] != 0:
            sf1_art.at[datekey_cur, "rd_sale"] = art_row_cur["rnd"] / art_row_cur["revenueusd"]
        
        # Return on invested capital (roic), Formula: (SF1[ebit]t-1 - [nopinc]t-1) / (SF1[equity]t-1 + SF1[liabilities]t-1 + SF1[cashneq]t-1 - SF1[investmentsc]t-1)
        if (art_row_cur["equity"] + art_row_cur["liabilities"] + art_row_cur["cashneq"] - art_row_cur["investmentsc"]) != 0:
            # Non-iperating income = SF1[revenueusd]t-1 - art_row_cur["cor"] - SF1[opinc]t-1
            nopic_t_1 = art_row_cur["revenueusd"] - art_row_cur["cor"] - art_row_cur["opinc"]
            sf1_art.at[datekey_cur, "roic"] = (art_row_cur["ebit"] - nopic_t_1) / (art_row_cur["equity"] + art_row_cur["liabilities"] + art_row_cur["cashneq"] - art_row_cur["investmentsc"])


        # Sales to cash (salecash), Formula: SF1[revenueusd]t-1 / SF1[cashneq]t-1
        if art_row_cur["cashneq"] != 0:
            sf1_art.at[datekey_cur, "salecash"] = art_row_cur["revenueusd"] / art_row_cur["cashneq"]

        # Sales to inventory (saleinv), Formula: SF1[revenueusd]t-1 / SF1[inventory]t-1
        if art_row_cur["inventory"] != 0:
            sf1_art.at[datekey_cur, "saleinv"] = art_row_cur["revenueusd"] / art_row_cur["inventory"]


        # Sales to receivables (salerec), Formula: SF1[revenueusd]t-1 / SF1[receivables]t-1
        if art_row_cur["receivables"] != 0:
            sf1_art.at[datekey_cur, "salerec"] = art_row_cur["revenueusd"] / art_row_cur["receivables"]

        # Sales to price (sp)	SF1[revenueusd]t-1 / SF1[marketcap]t-1
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[datekey_cur, "sp"] = art_row_cur["revenueusd"] / art_row_cur["marketcap"]

        # Tax income to book income (tb), Formula: (SF1[taxexp]t-1 / 0.21) / SF1[netinc]t-1
        if art_row_cur["netinc"] != 0:
            sf1_art.at[datekey_cur, "tb"] = art_row_cur["taxexp"] / art_row_cur["netinc"]
        
        # Sin stocks (sin)	if TICKER[industry].isin(["Beverages - Brewers", "Beverages - Wineries & Distilleries", "Electronic Gaming & Multimedia", "Gambling", "Tobacco"]): 1; else: 0
        industry_cur = metadata.loc[metadata["ticker"] == ticker].iloc[-1]["industry"]
        if industry_cur in ["Beverages - Brewers", "Beverages - Wineries & Distilleries", "Gambling", "Tobacco"]: # "Electronic Gaming & Multimedia"
            sf1_art.at[datekey_cur, "sin"] = 1
        else:
            sf1_art.at[datekey_cur, "sin"] = 0

        # Debt capacity/firm tangibility (tang), Formula: SF1[cashnequsd]t-1 + 0.715*SF1[recievables]t-1 + 0.547*SF1[inventory]t-1 + 0.535*(SF1[ppnenet]t-1 / SF1[assets]t-1)
        if art_row_cur["assets"] != 0:
            sf1_art.at[datekey_cur, "tang"] = (art_row_cur["cashnequsd"] + 0.715*art_row_cur["receivables"] + 0.547*art_row_cur["inventory"] + 0.535*art_row_cur["ppnenet"]) / art_row_cur["assets"]


        # DLC/SALE  (debtc_sale), Formula: SF1[debtc]t-1 / SF1[revenueusd]t-1
        if art_row_cur["revenueusd"] != 0:
            sf1_art.at[datekey_cur, "debtc_sale"] = art_row_cur["debtc"] / art_row_cur["revenueusd"]

        # CEQT/MKTCAP (eqt_marketcap), Formula: (SF1[equity]t-1 - SF1[intangibles]t-1) / SF1[marketcap]t-1
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[datekey_cur, "eqt_marketcap"] = ( art_row_cur["equityusd"] - art_row_cur["intangibles"]) / art_row_cur["marketcap"]


        # DPACT/PPENT	(dep_ppne), Formula: SF1[depamor]t-1 / sf1[ppnenet]t-1
        if art_row_cur["ppnenet"] != 0:
            sf1_art.at[datekey_cur, "dep_ppne"] = art_row_cur["depamor"] / art_row_cur["ppnenet"]


        # CEQL/MKTCAP	(tangibles_marketcap), Formula: SF1[tangibles]t-1 / SF1[marketcap]t-1
        if art_row_cur["marketcap"] != 0:
            sf1_art.at[datekey_cur, "tangibles_marketcap"] = art_row_cur["tangibles"] / art_row_cur["marketcap"]

        

        if not art_row_1y_ago.empty:
            # Calculate regular year over year SF1 features.
            # Asset Growth (arg), Formula: (SF1[assets]t-1 / SF1[assets]t-2) - 1
            if art_row_1y_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "agr"] = (art_row_cur["assets"] / art_row_1y_ago["assets"] ) - 1

            # Cash flow to debt (cashdebt), Formula: (SF1[revenueusd]t-1+SF1[depamor]t-1) / ((SF1[liabilities]t-1 - SF1[liabilities]t-2) / 2)
            if (art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) != 0:
                sf1_art.at[datekey_cur, "cashdebt"] = (art_row_cur["revenueusd"] + art_row_cur["depamor"]) /  ((art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) / 2)

            # Change in shared outstanding (chcsho), Formula: (SF1[sharesbas]t-1 - SF1[sharesbas]t-2) - 1
            if art_row_1y_ago["sharesbas"] != 0:
                sf1_art.at[datekey_cur, "chcsho"] = (art_row_cur["sharesbas"] / art_row_1y_ago["sharesbas"]) - 1

            # Change in inventory (chinv), Formula: (SF1[inventory]t-1 - SF1[inventory]t-2) / SF1[assetsavg]t-1
            if art_row_cur["assetsavg"] != 0:
                sf1_art.at[datekey_cur, "chinv"] = ( art_row_cur["inventory"] - art_row_1y_ago["inventory"] ) / art_row_cur["assetsavg"]

            # Growth in common shareholder equity (egr), Formula: (SF1[equityusd]t-1 / SF1[equityusd]t-2) - 1
            if art_row_1y_ago["equityusd"] != 0:
                sf1_art.at[datekey_cur, "egr"] = (art_row_cur["equityusd"] / art_row_1y_ago["equityusd"]) - 1

            # Gross profitability (gma), Formula: (SF1[revenueusd]t-1 - SF1[cor]t-1) / SF1[assets]t-2
            if art_row_1y_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "gma"] = (art_row_cur["revenueusd"] - art_row_cur["cor"]) / art_row_1y_ago["assets"]

            
            # Capital expenditures and inventory (invest), Formula: ((SF1[ppnenet]t-1 - SF1[ppnenet]t-2) + (SF1[inventory]t-1 - SF1[inventory]t-2)) / SF1[assets]t-2
            if art_row_1y_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "invest"] = ((art_row_cur["ppnenet"] - art_row_1y_ago["ppnenet"]) + (art_row_cur["inventory"] - art_row_1y_ago["inventory"])) / art_row_1y_ago["assets"]

            # Growth in long-term debt (lgr), Formula: (SF1[liabilities]t-1 / SF1[liabilities]t-2) - 1
            if art_row_1y_ago["liabilities"] != 0:
                sf1_art.at[datekey_cur, "lgr"] = ( art_row_cur["liabilities"] / art_row_1y_ago["liabilities"] ) - 1
            
            # Operating profitability (operprof), Formula: (SF1[revenueusd]t-1 - SF1[cor]t-1 - SF1[sgna]t-1 - SF1[intexp]t-1) / SF1[equityusd]t-2 
            if art_row_1y_ago["equityusd"] != 0:
                sf1_art.at[datekey_cur, "operprof"] = ( art_row_cur["revenueusd"] - art_row_cur["cor"] - art_row_cur["sgna"] - art_row_cur["intexp"] ) / art_row_1y_ago["equityusd"]


            # Percent change in current ratio (pchcurrat), Formula: (SF1[assetsc]t-1 / SF1[liabilitiesc]t-1) / (SF1[assetsc]t-2 / SF1[liabilitiesc]t-2) - 1
            if (art_row_cur["liabilitiesc"] != 0) and (art_row_1y_ago["liabilitiesc"] != 0):
                sf1_art.at[datekey_cur, "pchcurrat"] = ( (art_row_cur["assetsc"] / art_row_cur["liabilitiesc"]) / (art_row_1y_ago["assetsc"] / art_row_1y_ago["liabilitiesc"]) ) - 1

            # Percent chang ein depreciation (pchdepr), Formula: (SF1[depamor]t-1 / SF1[ppnenet]t-1) / (SF1[depamor]t-2 / SF1[ppnenet]t-2) - 1
            if (art_row_cur["ppnenet"] != 0) and (art_row_1y_ago["ppnenet"] != 0):
                sf1_art.at[datekey_cur, "pchdepr"] = ( (art_row_cur["depamor"]/ art_row_cur["ppnenet"]) / (art_row_1y_ago["depamor"] / art_row_1y_ago["ppnenet"]) ) - 1


            # Percent change in gross margin - Percent change in sales (pchgm_pchsale), Formula: ( ([gross_margin]t-1 / [gross_margin]t-2) - 1 ) - ( (SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1 ) 
            # gross_margin = (SF1[revenueusd]t-1 - SF1[cor]t-1) / SF1[revenueusd]t-1
            if (art_row_cur["revenueusd"] != 0) and (art_row_1y_ago["revenueusd"] != 0):
                gross_margin_t_1 = (art_row_cur["revenueusd"] - art_row_cur["cor"]) / art_row_cur["revenueusd"]
                gross_margin_t_2 = (art_row_1y_ago["revenueusd"] - art_row_1y_ago["cor"]) / art_row_cur["revenueusd"]

                sf1_art.at[datekey_cur, "pchgm_pchsale"] = ((gross_margin_t_1 / gross_margin_t_2) - 1) - ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1)
            
            # Percent change in quick ratio (pchquick), Formula: ([quick_ratio]t-1 / [quick_ratio]t-2) - 1
            # Quick ratio = (SF1[assetsc]t-1 - SF1[inventory]t-1) / SF1[liabilitiesc]t-1
            if (art_row_cur["liabilitiesc"] != 0) and (art_row_1y_ago["liabilitiesc"] != 0):
                quick_ratio_cur = ( art_row_cur["assetsc"] - art_row_cur["inventory"] ) / art_row_cur["liabilitiesc"]
                quick_ratio_1y_ago = ( art_row_1y_ago["assetsc"] - art_row_1y_ago["inventory"] ) / art_row_1y_ago["liabilitiesc"]
                sf1_art.at[datekey_cur, "pchquick"] = (quick_ratio_cur / quick_ratio_1y_ago) - 1

            # Percent change in sales - percent change in inventory (pchsale_pchinvt), Formula: ((SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1) - ((SF1[inventory]t-1 / SF1[inventory]t-2) - 1)
            if (art_row_1y_ago["revenueusd"] != 0) and (art_row_1y_ago["inventory"] != 0):
                sf1_art.at[datekey_cur, "pchsale_pchinvt"] = ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1) - ((art_row_cur["inventory"] / art_row_1y_ago["inventory"]) - 1)

            # % change in sales - % change in A/R (pchsale_pchrect), Formula: ((SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1) - ((SF1[receivables]t-1 / SF1[receivables]t-2) - 1)
            if (art_row_1y_ago["revenueusd"] != 0) and (art_row_1y_ago["inventory"] != 0):
                sf1_art.at[datekey_cur, "pchsale_pchrect"] = ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1) - ((art_row_cur["receivables"] / art_row_1y_ago["receivables"]) - 1)

            # % change in sales - % change in SG&A (pchsale_pchxsga ), Formula: ((SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1) - ((SF1[sgna]t-1 / SF1[sgna]t-2) - 1)
            if (art_row_1y_ago["revenueusd"] != 0) and (art_row_1y_ago["sgna"] != 0):
                sf1_art.at[datekey_cur, "pchsale_pchxsga"] = ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1) - ((art_row_cur["sgna"] / art_row_1y_ago["sgna"]) - 1)
            
            # % change sales-to-inventory (pchsaleinv), Formula: ((SF1[revenueusd]t-1 / SF1[inventory]t-1) / (SF1[revenueusd]t-2 / SF1[inventory]t-2)) - 1
            if (art_row_cur["inventory"] != 0) and (art_row_1y_ago["inventory"] != 0):
                sf1_art.at[datekey_cur, "pchsaleinv"] = ((art_row_cur["revenueusd"] / art_row_cur["inventory"]) / (art_row_1y_ago["revenueusd"] / art_row_1y_ago["inventory"])) - 1

            # R&D increase (rd), Formula: if (((SF1[rnd]t-1 / SF1[assets]t-1) - 1) - ((SF1[rnd]t-2 / SF1[assets]t-2) - 1)) > 0.05: 1; else: 0;
            if (art_row_cur["assets"] != 0) and (art_row_1y_ago["assets"] != 0):
                rd_cur = (art_row_cur["rnd"] / art_row_cur["assets"]) - 1
                rd_1y_ago = (art_row_1y_ago["rnd"] / art_row_1y_ago["assets"]) - 1
                if rd_cur - rd_1y_ago > 0.05:
                    sf1_art.at[datekey_cur, "rd"] = 1
                else:
                    sf1_art.at[datekey_cur, "rd"] = 0

            # Return on equity (roeq), Formula: SF1[netinc]t-1 / SF1[equity]t-2
            if art_row_1y_ago["equityusd"] != 0:
                sf1_art.at[datekey_cur, "roeq"] = art_row_cur["netinc"] / art_row_1y_ago["equityusd"]


            # Sales growth (sgr), Formula: (SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1
            if art_row_1y_ago["revenueusd"] != 0:
                sf1_art.at[datekey_cur, "sgr"] = art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]

            
            # Growth in capital expenditure (grcapx), Formula: (SF1[capex]t-1 / SF1[capex]t-2) - 1
            if art_row_1y_ago["capex"] != 0:
                sf1_art.at[datekey_cur, "grcapx"] = (art_row_cur["capex"] / art_row_1y_ago["capex"]) - 1

            # ΔLT/LAGAT (chtl_lagat), Formula: (SF1[liabilities]t-1 - SF1[liabilities]t-2) / SF1[assets]t-2
            if art_row_1y_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "chtl_lagat"] = (art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) / art_row_1y_ago["assets"]

            # ΔLT/LAGICAPT (chlt_laginvcap), Formula: (SF1[liabilities]t-1 - SF1[liabilities]t-2) / SF1[invcap]t-2
            if art_row_1y_ago["invcap"] != 0:
                sf1_art.at[datekey_cur, "chlt_laginvcap"] = (art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) / art_row_1y_ago["invcap"]

            # ΔLCT/LAGAT (chlct_lagat), Formula: (SF1[liabilitiesc]t-1 - SF1[liabilitiesc]t-2) / SF1[assets]t-2
            if art_row_1y_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "chlct_lagat"] = (art_row_cur["liabilitiesc"] - art_row_1y_ago["liabilitiesc"]) / art_row_1y_ago["assets"]

            # ΔXINT/LAGAT	(chint_lagat), Formula: (SF1[intexp]t-1  - SF1[intexp]t-2)/SF1[assets]t-2
            if art_row_1y_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "chint_lagat"] = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["assets"]


            # ΔINVT/LAGSALE (chinvt_lagsale), Formula: (SF1[inventory]t-1 - SF1[inventory]t-2) / SF1[revenueusd]t-2
            if art_row_1y_ago["revenueusd"] != 0:
                sf1_art.at[datekey_cur, "chinvt_lagsale"] = (art_row_cur["inventory"] - art_row_1y_ago["inventory"]) / art_row_1y_ago["revenueusd"]

            # ΔXINT/LAGXSGA (chint_lagsgna), Formula: (SF1[intexp]t-1  - SF1[intexp]t-2) / SF1[sgna]t-2
            if art_row_1y_ago["sgna"] != 0:
                sf1_art.at[datekey_cur, "chint_lagsgna"] = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["sgna"]

            # ΔLCT/LAGICAPT (chltc_laginvcap), Formula: (SF1[liabilitiesc]t-1 - SF1[liabilitiesc]t-2) / SF1[invcap]t-2
            if art_row_1y_ago["invcap"] != 0:
                sf1_art.at[datekey_cur, "chltc_laginvcap"] = (art_row_cur["liabilitiesc"] - art_row_1y_ago["liabilitiesc"]) / art_row_1y_ago["invcap"]

            # ΔXINT/LAGLT	(chint_laglt), Formula: (SF1[intexp]t-1  - SF1[intexp]t-2) / SF1[liabilities]t-2
            if art_row_1y_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "chint_laglt"] = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["liabilities"]

            # ΔDLTT/LAGAT (chdebtnc_lagat), Formula: (SF1[debtnc]t-1 - SF1[debtnc]t-2) / SF1[assets]t-2
            if art_row_1y_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "chdebtnc_lagat"] = (art_row_cur["debtnc"] - art_row_1y_ago["debtnc"]) / art_row_1y_ago["assets"]

            # ΔINVT/LAGCOGS (chinvt_lagcor), Formula:	(SF1[inventory]t-1 - SF1[inventory]t-2) / SF1[cor]t-2
            if art_row_1y_ago["assets"] != 0:
                sf1_art.at[datekey_cur, "chinvt_lagcor"] = (art_row_cur["inventory"] - art_row_1y_ago["inventory"]) / art_row_1y_ago["cor"]

            # ΔPPENT/LAGLT (chppne_laglt), Formula: (SF1[ppnenet]t-1 - SF1[ppnenet]t-2) / SF1[liabilities]t-2
            if art_row_1y_ago["liabilities"] != 0:
                sf1_art.at[datekey_cur, "chppne_laglt"] = (art_row_cur["ppnenet"] - art_row_1y_ago["ppnenet"]) / art_row_1y_ago["liabilities"]

            # ΔAP/LAGACT (chpay_lagact), Formula: (SF1[payables]t-1 - SF1[payables]t-2) / SF1[assetsc]t-2
            if art_row_1y_ago["assetsc"] != 0:
                sf1_art.at[datekey_cur, "chpay_lagact"] = (art_row_cur["payables"] - art_row_1y_ago["payables"]) / art_row_1y_ago["assetsc"]

            # ΔXINT/LAGICAPT (chint_laginvcap), Formula: (SF1[intexp]t-1 - SF1[intexp]t-2) / SF1[invcap]t-2
            if art_row_1y_ago["invcap"] != 0:
                sf1_art.at[datekey_cur, "chint_laginvcap"] = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["invcap"]

            #  ΔINVT/LAGACT (chinvt_lagact), Formula:	(SF1[inventory]t-1 - SF1[inventory]t-2) / SF1[assetsc]t-2
            if art_row_1y_ago["assetsc"] != 0:
                sf1_art.at[datekey_cur, "chinvt_lagact"] = (art_row_cur["inventory"] - art_row_1y_ago["inventory"]) / art_row_1y_ago["assetsc"]

            # %Δ in PPENT	(pchppne), Formula: (SF1[ppnenet]t-1 / SF1[ppnenet]t-2) - 1
            if art_row_1y_ago["ppnenet"] != 0:
                sf1_art.at[datekey_cur, "pchppne"] = (art_row_cur["ppnenet"] / art_row_1y_ago["ppnenet"]) - 1

            # %Δ in LT (pchlt), Formula: (SF1[liabilities]t-1 / SF1[liabilities]t-2) - 1
            if art_row_1y_ago["liabilities"] != 0:
                sf1_art.at[datekey_cur, "pchlt"] = (art_row_cur["liabilities"] / art_row_1y_ago["liabilities"]) - 1

            # %Δ in XINT (pchint), Formula: (SF1[intexp]t-1 - SF1[intexp]t-2) - 1
            if art_row_1y_ago["intexp"] != 0:
                sf1_art.at[datekey_cur, "pchint"] = (art_row_cur["intexp"] / art_row_1y_ago["intexp"]) - 1

            # DLTIS/PPENT	(chdebtnc_ppne), Formula: (SF1[debtnc]t-1 - SF1[debtnc]t-2) / SF1[ppnenet]t-1
            if art_row_cur["ppnenet"] != 0:
                sf1_art.at[datekey_cur, "chdebtnc_ppne"] = (art_row_cur["debtnc"] - art_row_1y_ago["debtnc"]) / art_row_cur["ppnenet"]
            
            # NP/SALE	(chdebtc_sale), Formula: (SF1[debtc]t-1 - SF1[debtc]t-2) / SF1[revenueusd]t-1
            if art_row_cur["revenueusd"] != 0:
                sf1_art.at[datekey_cur, "chdebtc_sale"] = (art_row_cur["debtc"] - art_row_1y_ago["debtc"]) / art_row_cur["revenueusd"]


        
        # _______________________YEARLY FILING BASED FEATURES END_________________________

       

        # _________________________________OTHER_______________________________________
        # Age (age): Formula: SF1[datekey]t-1 - TICKERS[firstpricedate]
        sf1_art.at[datekey_cur, "age"] = round((art_row_cur["datekey"] - metadata.loc[metadata["ticker"] == ticker].iloc[-1]["firstpricedate"]).days / 365)

        # Initial public offering (ipo), Formula: if (SF1[datekey]t-1 - TICKERS[firstpricedate]) <= 1 year: 1; else: 0
        days_since_ipo = (art_row_cur["datekey"] - metadata.loc[metadata["ticker"] == ticker].iloc[-1]["firstpricedate"]).days
        if days_since_ipo <= 365:
            sf1_art.at[datekey_cur, "ipo"] = 1
        else:
            sf1_art.at[datekey_cur, "ipo"] = 0



        #_________________IN PREPARATION FOR INDUSTRY ADJUSTED VALUES___________________

        # Profit margin (profitmargin), Formula: SF1[netinc]t-1 / SF1[revenueusd]t-1
        if art_row_cur["revenueusd"] != 0:
            sf1_art.at[datekey_cur, "profitmargin"] = art_row_cur["netinc"] / art_row_cur["revenueusd"]
            
        if not art_row_1y_ago.empty:
            # Change in profit margin (chprofitmargin), Formula: SF1[netinc]t-1 - SF1[netinc]t-2) / SF1[revenueusd]t-1
            if art_row_cur["revenueusd"] != 0:
                sf1_art.at[datekey_cur, "chprofitmargin"] = (art_row_cur["netinc"] - art_row_1y_ago["netinc"]) / art_row_cur["revenueusd"]

            
    return sf1_art


if __name__ == "__main__":
    sf1_art = pd.read_csv("./datasets/testing/sf1_art.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"], index_col="calendardate")
    sf1_art = sf1_art.sort_values(by="datekey", ascending=True)
    
    sf1_art["datekey"] = sf1_art.index

    sf1_arq = pd.read_csv("./datasets/testing/sf1_arq.csv", parse_dates=["datekey", \
        "calendardate", "reportperiod"], index_col="calendardate")
    sf1_arq = sf1_arq.sort_values(by="datekey", ascending=True)

    metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", \
        parse_dates=["firstpricedate"])


    sf1_art_featured = pandas_mp_engine(callback=add_sf1_features, atoms=sf1_art, \
        data={"sf1_arq": sf1_arq, 'metadata': metadata}, molecule_key='sf1_art', split_strategy= 'ticker', \
            num_processes=1, molecules_per_process=1)
    
    sf1_art_aapl = sf1_art_featured.loc[sf1_art_featured.ticker=="AAPL"]
    sf1_art_ntk = sf1_art_featured.loc[sf1_art_featured.ticker=="NTK"]

    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(sf1_art_aapl)