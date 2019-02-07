## Dataset construction

Terminology:
Observation - a row (bar) in a dataset representing a vector of data points regarding a specific firm at a specific day. 


Here I outline the overall strategy with regards to the production of the financial dataset of informative features for stock return prediction (the stock prediction dataset). The dataset is also used for other purposes towards the end of developing a complete automated trading system and backtester.

The basis for each observation in the dataset is trailing twelve months (TTM) company data. To supplement this data and build various features related to price and volume (such as 6-month momentum) a separate dataset set with daily observations is used. This points out the porblme of having to merge multiple datasets with different reporting periods.

Some features are industry adjusted numbers, and also industry classification is modeled using dummy variables. This requires the introduction of a third dataset which provides information related to each ticker. This data only needs to be merged into the stock prediction dataset on the ticker and not the date.

Observations in the dataset are arguable of unequal importance (containing different amounts of relevant information) to training stock return predicting ML models. This fact will not be dealt with upfront. This means that monthly observations are constructed with the most updated information available at that point in time, but no effort will be made to exclude or assign different weights to certain observations. Samping (selecting observations on some criteria from the total pool of observations) will be done at a later and separate stage, if at all. Sampling will only be explored if time allows it. Yet, sampling schemes mey be explored and must therefore be facilitated by the stock prediction dataset.
The features that, at this time, is most likely to be needed for observation sampling is daily price and volume data as well as observation age (or other measure of information uncertainty??? we want to train on new information, or observations that contain more information then other, train on observations with subsequent significant price movements). 


Volume and price sampling with require the construction of a completely new dataset, if only monthly observations of fundamentals are used in the original stock return dataset (the one not considering advanced observation sampling). A solution to this, which would allow the construction of both "best effort" monthly observation sampling, as well as exploring sampling schemes later on, is to construct a dataset with daily observations including all features. From this massive dataset, different sampling schemes can easily be tested, including the simple "best effort" monthly scheme.

The use of TTM data has some important implications. The available alternatives are "as reported quarterly" and "as reported yearly" data. Sharadar's Core US Fundamentals dataset SHOULD contain quarterly statements for all companies, but using these data introduces seasonality into the dataset, which for some industries can be substatial and thus introduce noise for other companies and the overall model. Including industry and "time of year" as dummy variables may let some ML algorithms (like artificial neural networks) to detect such patters. 




# "Best Effort" monthly observation sampling:
This is the first sampling scheme used to build stock return predicting ML models.

- The date of the form 10 regulatory filing to the SEC is different for different firms, due to different fiscal year endings.
- Information disclosed may have been available to the public days or in some rare cases weeks before the form 10 regulatory filings, in the case of the firm having done a form 8 filing to the SEC.
- In any case, the date of the form 10 regulatory filing to the SEC is the latest time the information was available to the public. It is therefore safe to assume that as long this fact is respected when constructing the dataset, no information about the future will creep into the observations. 
- 



## Building a backtester

