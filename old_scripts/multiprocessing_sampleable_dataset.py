import time
import pandas as pd
import sys
import logging
from os import listdir
from os.path import isfile, join
import concurrent.futures
from collections import namedtuple
import pickle
import datetime

from automated_trading_system.dataset_builder.dataset import Dataset, merge_datasets_simple
from automated_trading_system.helpers.helpers import print_exception_info

# from tqdm import tqdm

"""
Each step is performed for each industry separately

Step-by-Step Dataset Construction:
(1). Extend the SEP dataset with information usefull for sampling (most recent 10-K filing date, Industry classifications)
2. Use different sampling techniques to get monthly observations
    1. At first use timebars (sampling at a fixed time interval), but try to respect the different fiscal years
3. Calculate the various price and volume based features
    1. I will end up with maybe 5-10 GB of data at that point
4. Add inn SF1 and DAILY data
5. Compute features based on SF1
6. Select the features you want and combine into one ML ready dataset
"""


def create_sampleable_dataset(index_filename: tuple) -> dict:
    file_index = index_filename[0]
    filename = index_filename[1]

    result = {
        "error": list(),
        "metadata_not_available": set(),
        "message": "",
        "success": True,
        "drop_indexes": set()
    }

    ticker_path = "./datasets/industry_tickers/" + filename
    sep_path = "./datasets/industry_sep/" + filename
    daily_path = "./datasets/industry_daily/" + filename
    sf1_art_path = "./datasets/industry_sf1_art/" + filename
    tickers_metadata_path = "./datasets/sharadar/SHARADAR_TICKERS_METADATA.csv"
    save_path = "./datasets/sampleable/" + filename
    pickle_path = "./datasets/sampleable/results/" + filename.split(".")[0] + ".p"


    try:
        sep_df = pd.read_csv(sep_path, low_memory=False)
        sf1_art_df = pd.read_csv(sf1_art_path, low_memory=False)
        tickers_metadata = pd.read_csv(tickers_metadata_path, low_memory=False)
        """
        LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = "./logs/multiprocessing_sampleable_sep_" + filename.split(".")[0] + ".log"
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[logging.FileHandler(log_filename, mode="a"), logging.StreamHandler(sys.stdout)])
        logger = logging.getLogger()
        """

    except Exception as e:
        print_exception_info(e)
        sys.exit()

    sep_df["date"] = pd.to_datetime(sep_df["date"], format="%Y-%m-%d")
    sf1_art_df["datekey"] = pd.to_datetime(sf1_art_df["datekey"], format="%Y-%m-%d")

    sep_df.sort_values(by=["ticker", "date"], inplace=True)
    
    """
    tickers = list(sep_df.ticker.unique())
    for ticker in tickers:
        sep_for_ticker = sep_df
    """
    last_ticker = None
    length = len(sep_df.index)
    # Add date date of last 10-K filing and age of sf1_art data
    for sep_index, sep_row in sep_df.iterrows():
        
        sep_ticker = sep_row["ticker"]
        sep_date = sep_row["date"]
        
        if sep_index % 5000 == 0:
            # logger.debug("# Done {} iterations out of {}".format(sep_index, len(sep_df.index)))
            progress = int((sep_index/length) * 100)
            log_message = "Filename: {} - Done {} iterations out of {} ({}%) - Success so far?: {}".format(filename, sep_index, length, progress, result["success"])
            print(log_message)
            sys.stdout.flush() # In child processes stdout seems to be buffered
                
        """
        Get the row in SF1_ART with 10-K filing date (datekey) as close in the past as possible
            1. Get candidate rows
            2. Select best by choosing the closet past observation (based on datekey)
        """
        if last_ticker != sep_ticker:
            sf1_art_for_ticker = sf1_art_df.loc[sf1_art_df["ticker"] == sep_ticker] # THIS SHOULD ONLY BE DONE ONCE PER TICKER!!!
        
        # Extract past dates
        past_sf1_art_for_ticker = sf1_art_for_ticker.loc[sf1_art_for_ticker["datekey"] <= sep_date]
        
        # Get date of latest 10-K form filing
        date_of_latest_filing = past_sf1_art_for_ticker["datekey"].max()

        last_ticker = sep_ticker

        # WHAT IF NO DATE OF LAST FILING, I WANT TO CAPTURE WITH INDFORMATION
        if date_of_latest_filing is pd.NaT:
            result["drop_indexes"].add(sep_index)
            continue

        sep_df.at[sep_index, "datekey"] = date_of_latest_filing
        sep_df.at[sep_index, "age"] = (date_of_latest_filing - sep_date)

        # Add industry classification
        metadata = tickers_metadata.loc[tickers_metadata["ticker"] == sep_ticker]

        if metadata.empty: # this is checked every iteration!
            result["metadata_not_available"].add(sep_ticker)
            result["drop_indexes"].add(sep_index)
            result["success"] = False
            continue

        sep_df.at[sep_index, "industry"] = metadata.iloc[-1]["industry"]
        sep_df.at[sep_index, "sector"] = metadata.iloc[-1]["sector"]
        sep_df.at[sep_index, "siccode"] = metadata.iloc[-1]["siccode"]
        
    # Drop rows with no prior form 10-K release or missing metadata
    sep_df = sep_df.drop(list(result["drop_indexes"]))
    
    sampleable_dataset = Dataset.from_df(sep_df)
    sampleable_dataset.to_csv(save_path)

    if not result["success"]:
        result["message"] = "Some ticker(s) was missing metadata"
    else:
        result["message"] = "Completed successfully"

    # Pickle the result
    filehandler = open(pickle_path, 'wb')
    pickle.dump(result, filehandler)
    filehandler.close()
    

    return result


if __name__ == "__main__":

    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = "./logs/multiprocessing_sampleable_sep_" + date_time + ".log"


    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[logging.FileHandler(log_filename, mode="a"), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()

    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:  
        ticker_path = "./datasets/industry_tickers/"
        filenames = [f for f in listdir(ticker_path) if isfile(join(ticker_path, f))]
        filenames.remove("None.csv")
        filenames.remove("Infrastructure Operations.csv")

        # Completed: All
        start_index = 30
        stop_index = 149

        filenames = filenames[start_index:]
        print("About to process: ", filenames)


        index_filenames = [(index, filename) for index, filename in enumerate(filenames)]

        for filename, result in zip(filenames, executor.map(create_sampleable_dataset, index_filenames)):
            log_message = "# Completed processing: " + filename, " with result: " + result["message"]
            print(log_message) 
            sys.stdout.flush()



"""
Industries:

['Advertising Agencies.csv', 'Aerospace & Defense.csv', 'Agricultural Inputs.csv', 'Airlines.csv', 'Airports & Air Services.csv', 
'Aluminum.csv', 'Apparel Manufacturing.csv', 'Apparel Stores.csv', 'Asset Management.csv', 'Auto & Truck Dealerships.csv', 
'Auto Manufacturers.csv', 'Auto Parts.csv', 'Banks - Global.csv', 'Banks - Regional - Asia.csv', 'Banks - Regional - Europe.csv', 
'Banks - Regional - Latin America.csv', 'Banks - Regional - US.csv', 'Beverages - Brewers.csv', 'Beverages - Soft Drinks.csv', 
'Beverages - Wineries & Distilleries.csv', 'Biotechnology.csv', 'Broadcasting -Radio.csv',
'Broadcasting - TV.csv', 'Building Materials.csv', 'Business Equipment.csv', 'Business Services.csv', 'Capital Markets.csv', 
'Chemicals.csv', 'Coal.csv', 'Communication Equipment.csv', 'Computer Distribution.csv', 'Computer Systems.csv',
'Confectioners.csv', 'Conglomerates.csv', 'Consumer Electronics.csv', 'Contract Manufacturers.csv', 'Copper.csv', 
'Credit Services.csv', 'Data Storage.csv', 'Department Stores.csv', 'Diagnostics & Research.csv', 'Discount Stores.csv',
'Diversified Industrials.csv', 'Drug Manufacturers - Major.csv', 'Drug Manufacturers - Specialty & Generic.csv', 
'Education & Training Services.csv', 'Electronic Components.csv', 'Electronic Gaming & Multimedia.csv', 
'Electronics Distribution.csv', 'Engineering & Construction.csv', 'Farm & Construction Equipment.csv', 'Farm Products.csv',
'Financial Exchanges.csv', 'Food Distribution.csv', 'Footwear & Accessories.csv', 'Gambling.csv', 'Gold.csv', 
'Grocery Stores.csv', 'Health Care Plans.csv', 'Health Information Services.csv', 'Home Furnishings & Fixtures.csv', 
'Home Improvement Stores.csv', 'Household & Personal Products.csv', 'Industrial Distribution.csv', 'Industrial Metals & Minerals.csv',
 'Information Technology Services.csv', 'Infrastructure Operations.csv', 'Insurance - Diversified.csv', 'Insurance - Life.csv', 
 'Insurance - Property & Casualty.csv', 'Insurance - Reinsurance.csv', 'Insurance - Specialty.csv', 'Insurance Brokers.csv', 'Integrated Shipping & Logistics.csv', 'Internet Content & Information.csv', 'Leisure.csv', 'Lodging.csv', 'Long-Term Care Facilities.csv', 'Lumber & Wood Production.csv', 'Luxury Goods.csv', 'Marketing Services.csv', 'Media - Diversified.csv', 'Medical Care.csv', 'Medical Devices.csv', 'Medical Distribution.csv', 'Medical Instruments & Supplies.csv', 'Metal Fabrication.csv', 'None.csv', 'Oil & Gas Drilling.csv', 'Oil & Gas E&P.csv',
'Oil & Gas Equipment & Services.csv', 'Oil & Gas Integrated.csv', 'Oil & Gas Midstream.csv', 'Oil & Gas Refining & Marketing.csv',
 'Packaged Foods.csv', 'Packaging & Containers.csv', 'Paper & Paper Products.csv', 'Pay TV.csv', 'Personal Services.csv', 'Pharmaceutical Retailers.csv', 'Pollution & Treatment Controls.csv', 'Publishing.csv', 'Railroads.csv', 'Real Estate - General.csv', 'Real Estate Services.csv', 'Recreational Vehicles.csv', 'REIT - Diversified.csv', 'REIT - Healthcare Facilities.csv', 'REIT - Hotel & Motel.csv', 'REIT - Industrial.csv', 'REIT - Office.csv', 'REIT - Residential.csv', 'REIT - Retail.csv', 'Rental & Leasing Services.csv', 'Residential Construction.csv', 'Resorts & Casinos.csv', 'Restaurants.csv', 'Rubber & Plastics.csv', 'Savings & Cooperative Banks.csv', 'Scientific & Technical Instruments.csv',
'Security & Protection Services.csv', 'Semiconductor Equipment & Materials.csv', 'Semiconductor Memory.csv',
 'Semiconductors.csv', 'Shipping & Ports.csv', 'Silver.csv', 'Software - Application.csv', 'Software - Infrastructure.csv', 
 'Solar.csv', 'Specialty Chemicals.csv', 'Specialty Finance.csv', 'Specialty Retail.csv', 'Staffing & Outsourcing Services.csv',
  'Steel.csv', 'Telecom Services.csv', 'Textile Manufacturing.csv', 'Tobacco.csv', 'Tools & Accessories.csv',
   'Truck Manufacturing.csv', 'Trucking.csv', 'Utilities - Diversified.csv', 'Utilities - Independent Power Producers.csv', 
   'Utilities - Regulated Electric.csv', 'Utilities - Regulated Gas.csv', 'Utilities - Regulated Water.csv', 'Waste Management.csv']

"""

"""
Used:
industry
sector
siccode

Not Used:
sicsector
famaindustry
famasector
sicindustry
"""