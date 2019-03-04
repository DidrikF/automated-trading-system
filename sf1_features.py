import pandas as pd
from packages.helpers.helpers import print_exception_info
import sys
from dateutil.relativedelta import *
from datetime import datetime, timedelta

"""
Each step is performed for each industry separately

Step-by-Step Dataset Construction:
1. Extend the SEP dataset with information usefull for sampling (most recent 10-K filing date, Industry classifications)
2. Use different sampling techniques to get monthly observations
    1. At first use timebars (sampling at a fixed time interval), but try to respect the different fiscal years
3. Calculate the various price and volume based features
4. Compute features based on SF1
5. Add inn SF1 and DAILY data
6. Select the features you want and combine into one ML ready dataset
"""

def add_sf1_features(index_filename, testing):

    if testing == True:
        try:
            sf1_art = pd.read_csv("./datasets/testing/sf1_art.csv", low_memory=False)
            sf1_arq = pd.read_csv("./datasets/testing/sf1_arq.csv", low_memory=False)
            metadata = pd.read_csv("./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv", low_memory=False)
        except Exception as e:
            print_exception_info(e)
            sys.exit()

    else:
        # Need to adapt to take filename and use with multiprocesseing
        pass
    

    sf1_art["datekey"] = pd.to_datetime(sf1_art["datekey"])
    sf1_arq["datekey"] = pd.to_datetime(sf1_arq["datekey"])
    metadata["firstpricedate"] = pd.to_datetime(metadata["firstpricedate"])

    tickers = list(sf1_art.ticker.unique())
    print(tickers)

    for ticker in tickers:
        sf1_art_ticker = sf1_art.loc[sf1_art["ticker"] == ticker]
        sf1_arq_ticker = sf1_arq.loc[sf1_arq["ticker"] == ticker]

        # Iterating sampled rows
        for index, art_row_cur in sf1_art_ticker.iterrows():
            date_cur = art_row_cur["datekey"]

            date_1y_ago = date_cur - relativedelta(years=+1)
            date_2y_ago = date_cur - relativedelta(years=+2)

            art_row_1y_ago = get_row_with_closest_date(sf1_art_ticker, date_1y_ago, 30)
            art_row_2y_ago = get_row_with_closest_date(sf1_art_ticker, date_2y_ago, 30)
            
            date_1q_ago = date_cur - relativedelta(months=+3)
            date_2q_ago = date_cur - relativedelta(months=+6)
            date_3q_ago = date_cur - relativedelta(months=+9)
            date_4q_ago = date_cur - relativedelta(years=+1)
            date_5q_ago = date_cur - relativedelta(months=+15)
            date_6q_ago = date_cur - relativedelta(months=+18)
            date_7q_ago = date_cur - relativedelta(months=+22)
            date_8q_ago = date_cur - relativedelta(years=+2)

            arq_row_cur = get_row_with_closest_date(sf1_arq_ticker, date_cur, 14)
            arq_row_1q_ago = get_row_with_closest_date(sf1_arq_ticker, date_1q_ago, 14)
            arq_row_2q_ago = get_row_with_closest_date(sf1_arq_ticker, date_2q_ago, 14)
            arq_row_3q_ago = get_row_with_closest_date(sf1_arq_ticker, date_3q_ago, 21)
            arq_row_4q_ago = get_row_with_closest_date(sf1_arq_ticker, date_4q_ago, 30)
            arq_row_5q_ago = get_row_with_closest_date(sf1_arq_ticker, date_5q_ago, 30)
            arq_row_6q_ago = get_row_with_closest_date(sf1_arq_ticker, date_6q_ago, 30)
            arq_row_7q_ago = get_row_with_closest_date(sf1_arq_ticker, date_7q_ago, 30)
            arq_row_8q_ago = get_row_with_closest_date(sf1_arq_ticker, date_8q_ago, 30)

            arq_rows = [arq_row_cur, arq_row_1q_ago, arq_row_2q_ago, arq_row_3q_ago, arq_row_4q_ago, arq_row_5q_ago, arq_row_6q_ago, arq_row_7q_ago, arq_row_8q_ago]
            
            # CALCULATIONS IN PREPARATION FOR LATER
            
            
            # CALCULATIONS USING ONLY CURRENT SF1_ART ROW

            # Cash productivity (cashpr), Formula: (SF1[marketcap]t-1 + SF1[debtnc]t-1 - SF1[assets]t-1) / SF1[cashneq]t-1
            if art_row_cur["cashneq"] != 0:
                sf1_art.at[index, "cashneq"] = (art_row_cur["marketcap"] + art_row_cur["debtnc"] - art_row_cur["assets"]) / art_row_cur["cashneq"]
            # Cash (cash), Formula: SF1[cashnequsd]t-1 / SF1[assetsavg]t-1
            if art_row_cur["assetsavg"] != 0:
                sf1_art.at[index, "cash"] = art_row_cur["cashnequsd"] / art_row_cur["assetsavg"]

            # Book to market (bm), Formula: SF1[equityusd]t-1 / SF1[marketcap]t-1
            if art_row_cur["marketcap"] != 0:
                sf1_art.at[index, "bm"] = art_row_cur["equityusd"] / art_row_cur["marketcap"]

            # Cash flow to price ratio (cfp), Formula: SF1[ncfo]t-1 / SF1[marketcap]t-1
            if art_row_cur["marketcap"] != 0:
                art_row_cur["cfp"] = art_row_cur["ncfo"] / art_row_cur["marketcap"]

            # Current ratio (currat), Formula: SF1[assetsc]t-1 / SF1[liabilitiesc]t-1
            if art_row_cur["liabilitiesc"] != 0:
                sf1_art.at[index, "liabilitiesc"] = art_row_cur["assetsc"] / art_row_cur["liabilitiesc"]

            # Depreciation over PP&E (depr), Formula: SF1[depamor]t-1 / SF1[ppnenet]t-1
            if art_row_cur["ppnenet"] != 0:
                sf1_art.at[index, "depr"] = art_row_cur["depamor"] / art_row_cur["ppnenet"]

            # Earnings to price (ep), Formula: SF1[netinc]t-1 / SF1[marketcap]t-1 
            if art_row_cur["marketcap"] != 0:
                sf1_art.at[index, "ep"] = art_row_cur["netinc"] / art_row_cur["marketcap"]

            # Leverage (lev), Formula: SF1[liabilities]t-1 / SF1[marketcap]t-1
            if art_row_cur["marketcap"] != 0:
                sf1_art.at[index, "lev"] = art_row_cur["liabilities"] / art_row_cur["marketcap"]
            
            # Quick ratio (quick), Formula: (SF1[assetsc]t-1 - SF1[inventory]t-1) / SF1[liabilitiesc]t-1
            if (art_row_cur["liabilitiesc"] != 0):
                sf1_art.at[index, "quick"] = (art_row_cur["assetsc"] - art_row_cur["inventory"]) / art_row_cur["liabilitiesc"]

            # R&D to market capitalization (rd_mve), Formula: SF1[rnd]t-1 / SF1[marketcap]t-1
            if art_row_cur["marketcap"] != 0:
                sf1_art.at[index, "rd_mve"] = art_row_cur["rnd"] / art_row_cur["marketcap"]
            
            # R&D to sales (rd_sale), Formula: SF1[rnd]t-1 / SF1[revenueusd]t-1
            if art_row_cur["revenueusd"] != 0:
                sf1_art.at[index, "rd_sale"] = art_row_cur["rnd"] / art_row_cur["revenueusd"]
            
 	        # Return on invested capital (roic), Formula: (SF1[ebit]t-1 - [nopinc]t-1) / (SF1[equity]t-1 + SF1[liabilities]t-1 + SF1[cashneq]t-1 - SF1[investmentsc]t-1)
            if (art_row_cur["equity"] + art_row_cur["liabilities"] + art_row_cur["cashneq"] - art_row_cur["investmentsc"]) != 0:
                # Non-iperating income = SF1[revenueusd]t-1 - art_row_cur["cor"] - SF1[opinc]t-1
                nopic_t_1 = art_row_cur["revenueusd"] - art_row_cur["cor"] - art_row_cur["opinc"]
                sf1_art.at[index, "roic"] = (art_row_cur["ebit"] - nopic_t_1) / (art_row_cur["equity"] + art_row_cur["liabilities"] + art_row_cur["cashneq"] - art_row_cur["investmentsc"])


            # Sales to cash (salecash), Formula: SF1[revenueusd]t-1 / SF1[cashneq]t-1
            if art_row_cur["cashneq"] != 0:
                sf1_art.at[index, "salecash"] = art_row_cur["revenueusd"] / art_row_cur["cashneq"]

            # Sales to inventory (saleinv), Formula: SF1[revenueusd]t-1 / SF1[inventory]t-1
            if art_row_cur["inventory"] != 0:
                sf1_art.at[index, "saleinv"] = art_row_cur["revenueusd"] / art_row_cur["inventory"]


            # Sales to receivables (salerec), Formula: SF1[revenueusd]t-1 / SF1[receivables]t-1
            if art_row_cur["receivables"] != 0:
                sf1_art.at[index, "salerec"] = art_row_cur["revenueusd"] / art_row_cur["receivables"]

            # Sales to price (sp)	SF1[revenueusd]t-1 / SF1[marketcap]t-1
            if art_row_cur["marketcap"] != 0:
                sf1_art.at[index, "sp"] = art_row_cur["revenueusd"] / art_row_cur["marketcap"]

            # Tax income to book income (tb), Formula: (SF1[taxexp]t-1 / 0.21) / SF1[netinc]t-1
            if art_row_cur["netinc"] != 0:
                sf1_art.at[index, "tb"] = art_row_cur["taxexp"] / art_row_cur["netinc"]
            
            # Sin stocks (sin)	if TICKER[industry].isin(["Beverages - Brewers", "Beverages - Wineries & Distilleries", "Electronic Gaming & Multimedia", "Gambling", "Tobacco"]): 1; else: 0
            industry_cur = metadata.loc[metadata["ticker"] == ticker].iloc[-1]["industry"]
            if industry_cur in ["Beverages - Brewers", "Beverages - Wineries & Distilleries", "Gambling", "Tobacco"]: # "Electronic Gaming & Multimedia"
                sf1_art.at[index, "sin"] = 1
            else:
                sf1_art.at[index, "sin"] = 0

            # Debt capacity/firm tangibility (tang), Formula: SF1[cashnequsd]t-1 + 0.715*SF1[recievables]t-1 + 0.547*SF1[inventory]t-1 + 0.535*(SF1[ppnenet]t-1 / SF1[assets]t-1)
            if art_row_cur["assets"] != 0:
                sf1_art.at[index, "tang"] = (art_row_cur["cashnequsd"] + 0.715*art_row_cur["recievables"] + 0.547*art_row_cur["inventory"] + 0.535*art_row_cur["ppnenet"]) / art_row_cur["assets"]


            # DLC/SALE  (debtc_sale), Formula: SF1[debtc]t-1 / SF1[revenueusd]t-1
            if art_row_cur["revenueusd"] != 0:
                sf1_art.at[index, "debtc_sale"] = art_row_cur["debtc"] / art_row_cur["revenueusd"]

            # CEQT/MKTCAP (eqt_marketcap), Formula: (SF1[equity]t-1 - SF1[intangibles]t-1) / SF1[marketcap]t-1
            if art_row_cur["marketcap"] != 0:
                sf1_art.at[index, "eqt_marketcap"] = ( art_row_cur["equityusd"] - art_row_cur["intangibles"]) / art_row_cur["marketcap"]


            # DPACT/PPENT	(dep_ppne), Formula: SF1[depamor]t-1 / sf1[ppnenet]t-1
            if art_row_cur["ppnenet"] != 0:
                sf1_art.at[index, "dep_ppne"] = art_row_cur["depamor"] / art_row_cur["ppnenet"]


            # CEQL/MKTCAP	(tangibles_marketcap), Formula: SF1[tangibles]t-1 / SF1[marketcap]t-1
            if art_row_cur["marketcap"] != 0:
                sf1_art.at[index, "tangibles_marketcap"] = art_row_cur["tangibles"] / art_row_cur["marketcap"]

            

            if not art_row_1y_ago.empty:
                # Calculate regular year over year SF1 features.
                # Asset Growth (arg), Formula: (SF1[assets]t-1 / SF1[assets]t-2) - 1
                if art_row_1y_ago["assets"] != 0:
                    sf1_art.at[index, "agr"] = (art_row_cur["assets"] / art_row_1y_ago["assets"] ) - 1

                # Cash flow to debt (cashdebt), Formula: (SF1[revenueusd]t-1+SF1[depamor]t-1) / ((SF1[liabilities]t-1 - SF1[liabilities]t-2) / 2)
                if (art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) != 0:
                    sf1_art.at[index, "cashdebt"] = (art_row_cur["revenueusd"] + art_row_cur["depamor"]) /  ((art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) / 2)

                # Change in shared outstanding (chcsho), Formula: (SF1[sharesbas]t-1 - SF1[sharesbas]t-2) - 1
                if art_row_1y_ago["sharesbas"] != 0:
                    sf1_art.at[index, "chcsho"] = (art_row_cur["sharesbas"] / art_row_1y_ago["sharesbas"]) - 1

                # Change in inventory (chinv), Formula: (SF1[inventory]t-1 - SF1[inventory]t-2) / SF1[assetsavg]t-1
                if art_row_cur["assetsavg"] != 0:
                    sf1_art.at[index, "chinv"] = ( art_row_cur["inventory"] - art_row_1y_ago["inventory"] ) / art_row_cur["assetsavg"]

                # Growth in common shareholder equity (egr), Formula: (SF1[equityusd]t-1 / SF1[equityusd]t-2) - 1
                if art_row_1y_ago["equityusd"] != 0:
                    sf1_art.at[index, "egr"] = (art_row_cur["equityusd"] / art_row_1y_ago["equityusd"]) - 1

                # Gross profitability (gma), Formula: (SF1[revenueusd]t-1 - SF1[cor]t-1) / SF1[assets]t-2
                if art_row_1y_ago["assets"] != 0:
                    sf1_art.at[index, "gma"] = (art_row_cur["revenueusd"] - art_row_cur["cor"]) / art_row_1y_ago["assets"]

                
                # Capital expenditures and inventory (invest), Formula: ((SF1[ppnenet]t-1 - SF1[ppnenet]t-2) + (SF1[inventory]t-1 - SF1[inventory]t-2)) / SF1[assets]t-2
                if art_row_1y_ago["assets"] != 0:
                    sf1_art.at[index, "invest"] = ((art_row_cur["ppnenet"] - art_row_1y_ago["ppnenet"]) + (art_row_cur["inventory"] - art_row_1y_ago["inventory"])) / art_row_1y_ago["assets"]

                # Growth in long-term debt (lgr), Formula: (SF1[liabilities]t-1 / SF1[liabilities]t-2) - 1
                if art_row_1y_ago["liabilities"] != 0:
                    sf1_art.at[index, "lgr"] = ( art_row_cur["liabilities"] / art_row_1y_ago["liabilities"] ) - 1
                
                # Operating profitability (operprof), Formula: (SF1[revenueusd]t-1 - SF1[cor]t-1 - SF1[sgna]t-1 - SF1[intexp]t-1) / SF1[equityusd]t-2 
                if art_row_1y_ago["equityusd"] != 0:
                    sf1_art.at[index, "operprof"] = ( art_row_cur["revenueusd"] - art_row_cur["cor"] - art_row_cur["sgna"] - art_row_cur["intexp"] ) / art_row_1y_ago["equityusd"]


                # Percent change in current ratio (pchcurrat), Formula: (SF1[assetsc]t-1 / SF1[liabilitiesc]t-1) / (SF1[assetsc]t-2 / SF1[liabilitiesc]t-2) - 1
                if (art_row_cur["liabilitiesc"] != 0) and (art_row_1y_ago["liabilitiesc"] != 0):
                    sf1_art.at[index, "pchcurrat"] = ( (art_row_cur["assetsc"] / art_row_cur["liabilitiesc"]) / (art_row_1y_ago["assetsc"] / art_row_1y_ago["liabilitiesc"]) ) - 1

                # Percent chang ein depreciation (pchdepr), Formula: (SF1[depamor]t-1 / SF1[ppnenet]t-1) / (SF1[depamor]t-2 / SF1[ppnenet]t-2) - 1
                if (art_row_cur["ppnenet"] != 0) and (art_row_1y_ago["ppnenet"] != 0):
                    sf1_art.at[index, "pchdepr"] = ( (art_row_cur["depamor"]/ art_row_cur["ppnenet"]) / (art_row_1y_ago["depamor"] / art_row_1y_ago["ppnenet"]) ) - 1


                # Percent change in gross margin - Percent change in sales (pchgm_pchsale), Formula: ( ([gross_margin]t-1 / [gross_margin]t-2) - 1 ) - ( (SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1 ) 
                # gross_margin = (SF1[revenueusd]t-1 - SF1[cor]t-1) / SF1[revenueusd]t-1
                if (art_row_cur["revenueusd"] != 0) and (art_row_1y_ago["revenueusd"] != 0):
                    gross_margin_t_1 = (art_row_cur["revenueusd"] - art_row_cur["cor"]) / art_row_cur["revenueusd"]
                    gross_margin_t_2 = (art_row_1y_ago["revenueusd"] - art_row_1y_ago["cor"]) / art_row_cur["revenueusd"]

                    sf1_art.at[index, "pchgm_pchsale"] = ((gross_margin_t_1 / gross_margin_t_2) - 1) - ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1)
                
                # Percent change in quick ratio (pchquick), Formula: ([quick_ratio]t-1 / [quick_ratio]t-2) - 1
                # Quick ratio = (SF1[assetsc]t-1 - SF1[inventory]t-1) / SF1[liabilitiesc]t-1
                if (art_row_cur["liabilitiesc"] != 0) and (art_row_1y_ago["liabilitiesc"] != 0):
                    quick_ratio_cur = ( art_row_cur["assetsc"] - art_row_cur["inventory"] ) / art_row_cur["liabilitiesc"]
                    quick_ratio_1y_ago = ( art_row_1y_ago["assetsc"] - art_row_1y_ago["inventory"] ) / art_row_1y_ago["liabilitiesc"]
                    sf1_art.at[index, "pchquick"] = (quick_ratio_cur / quick_ratio_1y_ago) - 1

                # Percent change in sales - percent change in inventory (pchsale_pchinvt), Formula: ((SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1) - ((SF1[inventory]t-1 / SF1[inventory]t-2) - 1)
                if (art_row_1y_ago["revenueusd"] != 0) and (art_row_1y_ago["inventory"] != 0):
                    sf1_art.at[index, "pchsale_pchinvt"] = ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1) - ((art_row_cur["inventory"] / art_row_1y_ago["inventory"]) - 1)

                # % change in sales - % change in A/R (pchsale_pchrect), Formula: ((SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1) - ((SF1[receivables]t-1 / SF1[receivables]t-2) - 1)
                if (art_row_1y_ago["revenueusd"] != 0) and (art_row_1y_ago["inventory"] != 0):
                    sf1_art.at[index, "pchsale_pchrect"] = ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1) - ((art_row_cur["receivables"] / art_row_1y_ago["receivables"]) - 1)

                # % change in sales - % change in SG&A (pchsale_pchxsga ), Formula: ((SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1) - ((SF1[sgna]t-1 / SF1[sgna]t-2) - 1)
                if (art_row_1y_ago["revenueusd"] != 0) and (art_row_1y_ago["sgna"] != 0):
                    sf1_art.at[index, "pchsale_pchxsga"] = ((art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]) - 1) - ((art_row_cur["sgna"] / art_row_1y_ago["sgna"]) - 1)
                
                # % change sales-to-inventory (pchsaleinv), Formula: ((SF1[revenueusd]t-1 / SF1[inventory]t-1) / (SF1[revenueusd]t-2 / SF1[inventory]t-2)) - 1
                if (art_row_cur["inventory"] != 0) and (art_row_1y_ago["inventory"] != 0):
                    sf1_art.at[index, "pchsaleinv"] = ((art_row_cur["revenueusd"] / art_row_cur["inventory"]) / (art_row_1y_ago["revenueusd"] / art_row_1y_ago["inventory"])) - 1

                # R&D increase (rd), Formula: if (((SF1[rnd]t-1 / SF1[assets]t-1) - 1) - ((SF1[rnd]t-2 / SF1[assets]t-2) - 1)) > 0.05: 1; else: 0;
                if (art_row_cur["assets"] != 0) and (art_row_1y_ago["assets"] != 0):
                    rd_cur = (art_row_cur["rnd"] / art_row_cur["assets"]) - 1
                    rd_1y_ago = (art_row_1y_ago["rnd"] / art_row_1y_ago["assets"]) - 1
                    if rd_cur - rd_1y_ago > 0.05:
                        sf1_art.at[index, "rd"] = 1
                    else:
                        sf1_art.at[index, "rd"] = 0

                # Return on equity (roeq), Formula: SF1[netinc]t-1 / SF1[equity]t-2
                if art_row_1y_ago["equityusd"] != 0:
                    sf1_art.at[index, "roeq"] = art_row_cur["netinc"] / art_row_1y_ago["equityusd"]


                # Sales growth (sgr), Formula: (SF1[revenueusd]t-1 / SF1[revenueusd]t-2) - 1
                if art_row_1y_ago["revenueusd"] != 0:
                    sf1_art.at[index, "sgr"] = art_row_cur["revenueusd"] / art_row_1y_ago["revenueusd"]


                # ΔLT/LAGAT (chtl_lagat), Formula: (SF1[liabilities]t-1 - SF1[liabilities]t-2) / SF1[assets]t-2
                if art_row_1y_ago["assets"] != 0:
                    sf1_art.at[index, "chtl_lagat"] = (art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) / art_row_1y_ago["assets"]

                # ΔLT/LAGICAPT (chlt_laginvcap), Formula: (SF1[liabilities]t-1 - SF1[liabilities]t-2) / SF1[invcap]t-2
                if art_row_1y_ago["invcap"] != 0:
                    sf1_art.at[index, "chlt_laginvcap"] = (art_row_cur["liabilities"] - art_row_1y_ago["liabilities"]) / art_row_1y_ago["invcap"]

                # ΔLCT/LAGAT (chlct_lagat), Formula: (SF1[liabilitiesc]t-1 - SF1[liabilitiesc]t-2) / SF1[assets]t-2
                if art_row_1y_ago["assets"] != 0:
                    sf1_art.at[index, "chlct_lagat"] = (art_row_cur["liabilitiesc"] - art_row_1y_ago["liabilitiesc"]) / art_row_1y_ago["assets"]

                # ΔXINT/LAGAT	(chint_lagat), Formula: (SF1[intexp]t-1  - SF1[intexp]t-2)/SF1[assets]t-2
                if art_row_1y_ago["assets"] != 0:
                    sf1_art.at[index, "chint_lagat"] = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["assets"]


                # ΔINVT/LAGSALE (chinvt_lagsale), Formula: (SF1[inventory]t-1 - SF1[inventory]t-2) / SF1[revenueusd]t-2
                if art_row_1y_ago["revenueusd"] != 0:
                    sf1_art.at[index, "chinvt_lagsale"] = (art_row_cur["inventory"] - art_row_1y_ago["inventory"]) / art_row_1y_ago["revenueusd"]

                # ΔXINT/LAGXSGA (chint_lagsgna), Formula: (SF1[intexp]t-1  - SF1[intexp]t-2) / SF1[sgna]t-2
                if art_row_1y_ago["sgna"] != 0:
                    sf1_art.at[index, "chint_lagsgna"] = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["sgna"]

                # ΔLCT/LAGICAPT (chltc_laginvcap), Formula: (SF1[liabilitiesc]t-1 - SF1[liabilitiesc]t-2) / SF1[invcap]t-2
                if art_row_1y_ago["invcap"] != 0:
                    sf1_art.at[index, "chltc_laginvcap"] = (art_row_cur["liabilitiesc"] - art_row_1y_ago["liabilitiesc"]) / art_row_1y_ago["invcap"]

                # ΔXINT/LAGLT	(chint_laglt), Formula: (SF1[intexp]t-1  - SF1[intexp]t-2) / SF1[liabilities]t-2
                if art_row_1y_ago["assets"] != 0:
                    sf1_art.at[index, "chint_laglt"] = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["liabilities"]

                # ΔDLTT/LAGAT (chdebtnc_lagat), Formula: (SF1[debtnc]t-1 - SF1[debtnc]t-2) / SF1[assets]t-2
                if art_row_1y_ago["assets"] != 0:
                    sf1_art.at[index, "chdebtnc_lagat"] = (art_row_cur["debtnc"] - art_row_1y_ago["debtnc"]) / art_row_1y_ago["assets"]

                # ΔINVT/LAGCOGS (chinvt_lagcor), Formula:	(SF1[inventory]t-1 - SF1[inventory]t-2) / SF1[cor]t-2
                if art_row_1y_ago["assets"] != 0:
                    sf1_art.at[index, "chinvt_lagcor"] = (art_row_cur["inventory"] - art_row_1y_ago["inventory"]) / art_row_1y_ago["cor"]

                # ΔPPENT/LAGLT (chppne_laglt), Formula: (SF1[ppnenet]t-1 - SF1[ppnenet]t-2) / SF1[liabilities]t-2
                if art_row_1y_ago["liabilities"] != 0:
                    sf1_art.at[index, "chppne_laglt"] = (art_row_cur["ppnenet"] - art_row_1y_ago["ppnenet"]) / art_row_1y_ago["liabilities"]

                # ΔAP/LAGACT (chpay_lagact), Formula: (SF1[payables]t-1 - SF1[payables]t-2) / SF1[assetsc]t-2
                if art_row_1y_ago["assetsc"] != 0:
                    sf1_art.at[index, "chpay_lagact"] = (art_row_cur["payables"] - art_row_1y_ago["payables"]) / art_row_1y_ago["assetsc"]

                # ΔXINT/LAGICAPT (chint_laginvcap), Formula: (SF1[intexp]t-1 - SF1[intexp]t-2) / SF1[invcap]t-2
                if art_row_1y_ago["invcap"] != 0:
                    sf1_art.at[index, "chint_laginvcap"] = (art_row_cur["intexp"] - art_row_1y_ago["intexp"]) / art_row_1y_ago["invcap"]

                #  ΔINVT/LAGACT (chinvt_lagact), Formula:	(SF1[inventory]t-1 - SF1[inventory]t-2) / SF1[assetsc]t-2
                if art_row_1y_ago["assetsc"] != 0:
                    sf1_art.at[index, "chinvt_lagact"] = (art_row_cur["inventory"] - art_row_1y_ago["inventory"]) / art_row_1y_ago["assetsc"]

                # %Δ in PPENT	(pchppne), Formula: (SF1[ppnenet]t-1 / SF1[ppnenet]t-2) - 1
                if art_row_1y_ago["ppnenet"] != 0:
                    sf1_art.at[index, "pchppne"] = (art_row_cur["ppnenet"] / art_row_1y_ago["ppnenet"]) - 1

                # %Δ in LT (pchlt), Formula: (SF1[liabilities]t-1 / SF1[liabilities]t-2) - 1
                if art_row_1y_ago["liabilities"] != 0:
                    sf1_art.at[index, "pchlt"] = (art_row_cur["liabilities"] / art_row_1y_ago["liabilities"]) - 1

                # %Δ in XINT (pchint), Formula: (SF1[intexp]t-1 - SF1[intexp]t-2) - 1
                if art_row_1y_ago["intexp"] != 0:
                    sf1_art.at[index, "pchint"] = (art_row_cur["intexp"] / art_row_1y_ago["intexp"]) - 1

                # DLTIS/PPENT	(chdebtnc_ppne), Formula: (SF1[debtnc]t-1 - SF1[debtnc]t-2) / SF1[ppnenet]t-1
                if art_row_cur["ppnenet"] != 0:
                    sf1_art.at[index, "chdebtnc_ppne"] = (art_row_cur["debtnc"] - art_row_1y_ago["debtnc"]) / art_row_cur["ppnenet"]
                
                # NP/SALE	(chdebtc_sale), Formula: (SF1[debtc]t-1 - SF1[debtc]t-2) / SF1[revenueusd]t-1
                if art_row_cur["revenueusd"] != 0:
                    sf1_art.at[index, "chdebtc_sale"] = (art_row_cur["debtc"] - art_row_1y_ago["debtc"]) / art_row_cur["revenueusd"]


            if not art_row_2y_ago.empty:
                # Growth in capital expenditure (grcapx), Formula: (SF1[capex]t-1 / SF1[capex]t-3) - 1
                if art_row_2y_ago["capex"] != 0:
                    sf1_art.at[index, "grcapx"] = (art_row_cur["capex"] / art_row_2y_ago["capex"]) - 1



            # I might want to implement approximations for those companies that do not have quarterly statements
            if (not arq_row_cur.empty) and (not arq_row_1q_ago.empty):
                # CALCULATE QUARTER TO QUARTER FEATURES
                
                # Return on assets (roaq), Formula: SF1[netinc]q-1 / SF1[assets]q-2
                if arq_row_1q_ago["assets"] != 0:
                    sf1_art.at[index, "assets"] = arq_row_cur["netinc"] / arq_row_1q_ago["assets"]

            if (not arq_row_cur.empty) and (not arq_row_4q_ago.empty):
                # CALCULATE FEATURES BASED ON THE SAME QUARTER FOR THAT LAST TWO YEARS
                
                # Change in tax expense (chtx), Formula: (SF1[taxexp]q-1 / SF1[taxexp]q-5) - 1
                if arq_row_4q_ago["taxexp"] != 0:
                    sf1_art.at[index, "chtx"] = (arq_row_cur["taxexp"] / arq_row_4q_ago["taxexp"]) - 1

                # Revenue surprise (rsup), Formula: ( SF1[revenueusd]q-1 - SF1[revenueusd]q-5 ) / SF1[marketcap]q-1
                if arq_row_1q_ago["marketcap"] != 0:
                    sf1_art.at[index, "rsup"] = (arq_row_1q_ago["revenueusd"] - arq_row_4q_ago["revenueusd"]) / arq_row_1q_ago["marketcap"]

                # Earnings Surprise (sue), Formula: (SF1[netinc]q-1 - SF1[netinc]q-5) / SF1[marketcap]q-1
                if arq_row_1q_ago["marketcap"] != 0:
                    sf1_art.at[index, "sue"] = (arq_row_1q_ago["netinc"] - arq_row_4q_ago["netinc"]) / arq_row_1q_ago["marketcap"]
                
                



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
                
                sf1_art.at[index, "cinvest"] = chppne_sales_cur - ( (chppne_sales_q_1, chppne_sales_q_2, chppne_sales_q_3) / 3 )
            

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

            sf1_art.at[index, "nincr"] = nr_of_earnings_increases

            # Earnings volatility (roavol)	Francis, LaFond, Olsson & Schipper 	2004, TAR 	
            # "Standard deviaiton for 16 quarters of income before extraordinary items (ibq) divided by average total assets (atq)."	
            # Formula: std(SF1[netinc]q) / avg(SF1[assets]q) for 8 - 16 quarters
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
                
                sf1_arq.at[index, "roavol"] = std_netinc_assets

            # OTHER
            # Age (age): Formula: SF1[datekey]t-1 - TICKERS[firstpricedate]
            sf1_art.at[index, "age"] = round((art_row_cur["datekey"] - metadata.loc[metadata["ticker"] == ticker].iloc[-1]["firstpricedate"]).days / 365)

            # Initial public offering (ipo), Formula: if (SF1[datekey]t-1 - TICKERS[firstpricedate]) <= 1 year: 1; else: 0
            days_since_ipo = (art_row_cur["datekey"] - metadata.loc[metadata["ticker"] == ticker].iloc[-1]["firstpricedate"]).days
            if days_since_ipo <= 365:
                sf1_art.at[index, "ipo"] = 1
            else:
                sf1_art.at[index, "ipo"] = 0



            # IN PREPARATION FOR INDUSTRY ADJUSTED VALUES
            # Profit margin (profitmargin), Formula: SF1[netinc]t-1 / SF1[revenueusd]t-1
            if art_row_cur["revenueusd"] != 0:
                sf1_art.at[index, "profitmargin"] = art_row_cur["netinc"] / art_row_cur["revenueusd"]
                

            if not art_row_1y_ago.empty:
                # Change in profit margin (chprofitmargin), Formula: SF1[netinc]t-1 - SF1[netinc]t-2) / SF1[revenueusd]t-1
                if art_row_cur["revenueusd"] != 0:
                    sf1_art.at[index, "chprofitmargin"] = (art_row_cur["netinc"] - art_row_1y_ago["netinc"]) / art_row_cur["revenueusd"]

            


    if testing == True:
        return sf1_art

    # Save
    # OBS need to update the filepath
    sf1_art.to_csv("./datasets/testing/sf1_art_featured.csv")


def get_row_with_closest_date(df, date, margin):
    """
    Returns the row in the df closest to the date as long as it is within the margin.
    If no such date exist it returns None.
    Assumes only one ticker in the dataframe.
    """
    acceptable_dates = get_acceptable_dates(date, margin)
    candidates = df.loc[df["datekey"].isin(acceptable_dates)]
    if len(candidates.index) == 0:
        return pd.DataFrame()
    
    best_row = select_row_closes_to_date(candidates, date)
    return best_row


def get_acceptable_dates(date, margin):
    dates = [(date + timedelta(days=x)).isoformat() for x in range(-margin, +margin)]
    dates.insert(0, date.isoformat())
    return dates

def select_row_closes_to_date(candidates, desired_date):
    candidate_dates = candidates.loc[:,"datekey"].tolist()
    
    best_date = min(candidate_dates, key=lambda candidate_date: abs(desired_date - candidate_date))
    best_row = candidates.loc[candidates["datekey"] == best_date].iloc[0]

    return best_row
