# Dataset construction

Terminology:
Observation - a row (bar) in a dataset representing a vector of data points regarding a specific firm at a specific day. 


Here I outline the overall strategy with regards to the production of the financial dataset of informative features for stock return prediction (the stock prediction dataset). The dataset is also used for other purposes towards the end of developing a complete automated trading system and backtester.

The basis for each observation in the dataset is trailing twelve months (TTM) company data. To supplement this data and build various features related to price and volume (such as 6-month momentum) a separate dataset set with daily observations is used. This points out the problem of having to merge multiple datasets with different reporting periods.

Some features are industry adjusted numbers. This requires each ticker to be labeled with its industry. This information comes from a third dataset.

...

Observations in the dataset are arguable of unequal importance (containing different amounts of relevant information) to training stock return predicting ML models. This fact will not be dealt with upfront. This means that monthly observations are constructed with the most updated information available at that point in time, but no effort will be made to exclude or assign different weights to certain observations. Sampling (selecting observations from the total pool of observations) will be done at a later and separate stage.

...

The features that, at this time, is most likely to be needed for observation sampling is daily price and volume data as well as observation age (or other measure of information uncertainty??? we want to train on new information, or observations that contain more information then other, train on observations with subsequent significant price movements). 

...

The use of TTM data has some important implications. The available alternatives are "as reported quarterly" and "as reported yearly" data. Sharadar's Core US Fundamentals dataset SHOULD contain quarterly statements for all US companies (as this is a requirement for public US based companies), but using these data introduces seasonality into the dataset, which for some industries can be substantial and thus introduce noise for other companies and the overall model. Including industry and "time of year" as dummy variables may let some ML algorithms (like artificial neural networks) to detect such patters. I want to avoid such complications by using yearly data. Using only 10-K filings would increase the sampling period to one year, even if more recent data is available in 10-Q filings. By using TTM data, I get yearly data updated every quarter and is therefore the the best option given my criteria and goals. Using TTM data will reduce serial correlation between observations compared to "as reported yearly" data. I do not naively assume that all companies have their fundamentals update quarterly. I also no not naively assume that all companies do their form 10 filings every 3 months like clockwork. I assume rather that some companies do not file quarterly and that their form 10 filings may be significantly skewed or may not be present at all, in which case I deal with data older than 3 months. 

Sampling...

Exactly what data is updated quarterly and what is updated yearly? 
- This is important to label the frequency correctly in the spreadsheet outlining the features I use.


## "Big picture" strategy


### Step by step instructions for creating dataset of daily observations from SHARADAR tables:


### "Best Effort" monthly observation sampling:
This is the first sampling scheme used to build stock return predicting ML models.

- The date of the form 10 regulatory filing to the SEC is different for different firms, due to different fiscal years.
- Information disclosed may have been available to the public days or in some rare cases weeks before the form 10 regulatory filings, in the case of the firm having done a form 8 filing to the SEC.
- In any case, the date of the form 10 regulatory filing to the SEC is the latest time the information was available to the public. It is therefore safe to assume that as long this fact is respected when constructing the dataset, no information about the future will creep into the observations. 
- 


### "Dollar volume" based observation sampling


# Building a stock return predictor


# Building an automated trading system


# Building a backtester

