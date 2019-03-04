import pandas as pd
import sys
from packages.helpers.helpers import print_exception_info
from packages.dataset_builder.dataset import Dataset
from dateutil.relativedelta import *
from os import listdir
from os.path import isfile, join
import numpy as np



sf1_art_path = "./datasets/testing/industry_sf1_art.csv"
save_path = "./datasets/testing/sf1_art_synthetic.csv"


try:

    sf1_art = pd.read_csv(sf1_art_path, low_memory=False)

    
except Exception as e: 
    print_exception_info(e)
    sys.exit()


sf1_art_columns_to_keep = ["ticker", "calendardate", "datekey", "assets", "assetsavg", "capex", "cor", "debtnc", "equityusd", "liabilities", 
    "liabilitiesc", "marketcap", "ncfo", "netinc", "ppnenet", "revenueusd", "rnd", "sgna", "sharesbas"]

sf1_art_synthetic = sf1_art.loc[:, sf1_art.columns.intersection(sf1_art_columns_to_keep)]# pd.DataFrame(columns=sf1_art_columns_to_keep)

sf1_art_synthetic["assets"] = np.nan
sf1_art_synthetic["assetsavg"] = np.nan
sf1_art_synthetic["capex"] = np.nan
sf1_art_synthetic["cor"] = np.nan
sf1_art_synthetic["debtnc"] = np.nan
sf1_art_synthetic["equityusd"] = np.nan
sf1_art_synthetic["liabilities"] = np.nan
sf1_art_synthetic["liabilitiesc"] = np.nan
sf1_art_synthetic["marketcap"] = np.nan
sf1_art_synthetic["ncfo"] = np.nan
sf1_art_synthetic["marketcap"] = np.nan
sf1_art_synthetic["netinc"] = np.nan
sf1_art_synthetic["ppnenet"] = np.nan
sf1_art_synthetic["revenueusd"] = np.nan
sf1_art_synthetic["rnd"] = np.nan
sf1_art_synthetic["sgna"] = np.nan
sf1_art_synthetic["sharesbas"] = np.nan





sf1_art_synthetic.to_csv(save_path, index=False)