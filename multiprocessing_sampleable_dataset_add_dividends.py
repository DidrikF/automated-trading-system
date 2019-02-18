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
from dateutil.relativedelta import *

from packages.dataset_builder.dataset import Dataset, merge_datasets_simple
from packages.helpers.helpers import print_exception_info


"""
The idea is to add the sum of the dividends received the last month for each date. This number can then be used directly in the calculation of total return 
when working with the sampled dataset. 
There is however a slight risk of sampling twice in a timewindow used to calculate the cumulative (sum) dividend, effectively applying one dividend payment in 
two consecutive months.
On the other hand the dividend would strictly speaking be part of the monthly return for both samples, its just unfortunate that they both are to close in time, 
causing the total return to have higher serial correlation.

Also: maybe I should use total return for momentum calculations???
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

    # ticker_path = "./datasets/industry_tickers/" + filename
    sampleable_path = "./datasets/sampleable/" + filename
    save_path = "./datasets/sampleable/" + filename
    pickle_path = "./datasets/sampleable/results/" + filename.split(".")[0] + "_add_dividends.p"


    try:
        sampleable = pd.read_csv(sampleable_path, low_memory=False)

    except Exception as e:
        print_exception_info(e)
        sys.exit()

    sampleable["date"] = pd.to_datetime(sampleable["date"], format="%Y-%m-%d")
    sampleable["datekey"] = pd.to_datetime(sampleable["datekey"], format="%Y-%m-%d")

    # sampleable.sort_values(by=["ticker", "date"], inplace=True)
    
    length = len(sampleable.index)

    tickers = list(sampleable.ticker.unique())
    counter = 0
    for ticker in tickers:
        ticker_sampleable = sampleable.loc[sampleable["ticker"] == ticker]
        
        for samplesable_index, sampleable_row in ticker_sampleable.iterrows():
            counter += 1

            if counter % 5000 == 0:
                # logger.debug("# Done {} iterations out of {}".format(samplesable_index, len(sep_df.index)))
                progress = int((counter/length) * 100)
                log_message = "Filename: {} - Done {} iterations out of {} ({}%) - Success so far?: {}".format(filename, counter, length, progress, result["success"])
                print(log_message)
                sys.stdout.flush() # In child processes stdout seems to be buffered
                    

            ticker = sampleable_row["ticker"]
            date = sampleable_row["date"]

            date_less_one_month = date - relativedelta(months=+1)
            
            # Get the last months observations:
            last_months_rows = ticker_sampleable.loc[(date_less_one_month <= sampleable["date"]) & (sampleable["date"] <= date) ]
            total_divedend = last_months_rows["dividends"].sum()

            sampleable.at[samplesable_index, "last_months_dividends"] = total_divedend

    print(sampleable.head())

    # Save
    sampleable_dataset = Dataset.from_df(sampleable)
    sampleable_dataset.to_csv(save_path)


    # Pickle the result
    """
    filehandler = open(pickle_path, 'wb')
    pickle.dump(result, filehandler)
    filehandler.close()
    """

    return result


if __name__ == "__main__":
    start_time = time.time()

    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = "./logs/multiprocessing_sampleable_sep_" + date_time + ".log"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[logging.FileHandler(log_filename, mode="a"), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()

    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:  
        sampleable_path = "./datasets/sampleable/"
        filenames = [f for f in listdir(sampleable_path) if isfile(join(sampleable_path, f))]

        # Completed: 0
        start_index = 50
        stop_index = "all"

        filenames = filenames[start_index:]
        print("About to process: ", filenames)


        index_filenames = [(index, filename) for index, filename in enumerate(filenames)]

        for filename, result in zip(filenames, executor.map(create_sampleable_dataset, index_filenames)):
            log_message = "# Completed processing: " + filename, " with result: " + result["message"]
            print(log_message) 
            sys.stdout.flush()

        
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken: {}".format(total_time))



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