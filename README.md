# Dataset construction

Terminology:
Observation - a row (bar) in a dataset representing a vector of data points regarding a specific firm at a specific day. 


Here I outline the overall strategy with regards to the production of the financial dataset of informative features for stock return prediction (the stock prediction dataset). The dataset is also used for other purposes towards the end of developing a complete automated trading system and backtester.

The basis for each observation in the dataset is trailing twelve months (TTM) company data. To supplement this data and build various features related to price and volume (such as 6-month momentum) a separate dataset set with daily observations is used. This points out the porblme of having to merge multiple datasets with different reporting periods.

Some features are industry adjusted numbers, and also industry classification is modeled using dummy variables. This requires the introduction of a third dataset which provides information related to each ticker. This data only needs to be merged into the stock prediction dataset on the ticker and not the date.

Observations in the dataset are arguable of unequal importance (containing different amounts of relevant information) to training stock return predicting ML models. This fact will not be dealt with upfront. This means that monthly observations are constructed with the most updated information available at that point in time, but no effort will be made to exclude or assign different weights to certain observations. Samping (selecting observations on some criteria from the total pool of observations) will be done at a later and separate stage, if at all. Sampling will only be explored if time allows it. Yet, sampling schemes mey be explored and must therefore be facilitated by the stock prediction dataset.
The features that, at this time, is most likely to be needed for observation sampling is daily price and volume data as well as observation age (or other measure of information uncertainty??? we want to train on new information, or observations that contain more information then other, train on observations with subsequent significant price movements). 


Volume and price sampling with require the construction of a completely new dataset, if only monthly observations of fundamentals are used in the original stock return dataset (the one not considering advanced observation sampling). A solution to this, which would allow the construction of both "best effort" monthly observation sampling, as well as exploring sampling schemes later on, is to construct a dataset with daily observations including all features. From this massive dataset, different sampling schemes can easily be tested, including the simple "best effort" monthly scheme.

The use of TTM data has some important implications. The available alternatives are "as reported quarterly" and "as reported yearly" data. Sharadar's Core US Fundamentals dataset SHOULD contain quarterly statements for all US companies (as this is a requirement for public US based companies), but using these data introduces seasonality into the dataset, which for some industries can be substantial and thus introduce noise for other companies and the overall model. Including industry and "time of year" as dummy variables may let some ML algorithms (like artificial neural networks) to detect such patters. I want to avoid such complications by using yearly data. Using only 10-K filings would increase the sampling period to one year, even if more recent data is available in 10-Q filings. By using TTM data, I get yearly data updated every quarter and is therefore the the best option given my criteria and goals. Using TTM data will reduce serial correlation between observations compared to "as reported yearly" data. I do not naively assume that all companies have their fundamentals update quarterly. I also no not naively assume that all companies do their form 10 filings every 3 months like clockwork. I assume rather that some companies do not file quarterly and that their form 10 filings may be significantly skewed or may not be present at all, in which case I deal with data older than 3 months. Observations are constructed on a best effort basis, meaning that the most resent data is used to form the observation of each day (which is later sampled for monthly observations). The important thing is to capture the "datekey" (date of filing to the SEC) for the fundamental data for each daily observation. This can then later be used when sampling down to monthly data.


Exactly what data is updated quarterly and what is updated yearly? 
- This is important to label the frequency correctly in the spreadsheet outlining the features I use.


## "Big picture" strategy
1. Construct a massive dataset of all consivebly relevant features from the SHARADAR tables
2. Add as many informative features from the literature on cross-sectional and time-series stock return prediction/pricing literature based on the available SHARADAR data
3. Construct various labels for the "daily observation" dataset
4. Construct sampled datasets by sampling the data according to some sampling scheme (only "best effort" monthly observation sampling is considered at first)
5. Construct datasets almost ready to use with ML models by selecting the features and labels you want to include from the sampled dataset

6. Perform one-hot encoding or other similar data preparation
7. Perform feature scaling, if any
8. Split the dataset into training/validation and testing datasets
9. Separate out validation set, if not more advanced cross-validation techniques are used
10. At this point the dataset is ready use for training and testing ML models 



### Step by step instructions for creating dataset of daily observations from SHARADAR tables:
1. Use SHARADAR_DAILY as a base and merge it with SHARADAR_SEP on date and ticker
    1. Keep all columns (except redundant once, including: ticker and date)
    2. How to deal with missing data?
2. Find SHARADAR_SF1_ART most resent observation based on "datekey" (which is data or form 10 filing to the SEC) for each ticker and date and merge the SF1 data into that row
    1. Remember to keep the "datekey", it will be important for later sampling
3. Add in data associated with each ticker in each row from SHARADAR_TICKERS_METADATA

This concludes the most important stages of constructing a dataset from SHARADAR tables.
The inclusion of SF2 and SF3 data involves the complication of only having respectively 12 and 5 years of data.


### "Best Effort" monthly observation sampling:
This is the first sampling scheme used to build stock return predicting ML models.

- The date of the form 10 regulatory filing to the SEC is different for different firms, due to different fiscal year endings.
- Information disclosed may have been available to the public days or in some rare cases weeks before the form 10 regulatory filings, in the case of the firm having done a form 8 filing to the SEC.
- In any case, the date of the form 10 regulatory filing to the SEC is the latest time the information was available to the public. It is therefore safe to assume that as long this fact is respected when constructing the dataset, no information about the future will creep into the observations. 
- 


### "Dollar volume" based observation sampling


# Building a stock return predictor


# Building an automated trading system


# Building a backtester

