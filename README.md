# Stock Return Prediction, Automated Trading System and Backtester
The aim of this project is to develop an automated trading system (ATS) that uses machine learning to inform the generation of orders. The ATS's performance is measured by a custom built backtester that simulates the execution of trades, while keeping track of transaction costs and portfolio return. The project involves five main tasks.
1. Development of a dataset of approximatly 100 features and 1 million observations for stock return prediction and classification.
2. Build regression models for monthly stock returns using random forests, deep neural networks, multiple linear regression and principal component regression.
3. Build classification models to be used to identify good investment opportunities.
4. Use the classification models in the development of an automated trading system.
5. Build a backtester to measure the historical performance of the automated trading system.

## Project structure
- backtester -> Includes source code for the backtester and trading strategeis.
- dataset_development -> Includes source code involved in the development of the projects dataset. Most notably are the genereate_features.py and finalize_dataset.py files, which does most of the heavy lifting.
-   datasets
- ml_strategy_models -> Includes the pickled random forest model objects used to make prediction in the ML-dirven ATS
- models -> Includes regression and classification models developed for the project
- tests

## Usage
Running backtest using ML-driven automated trading system:
> python backtester/run_backtest.py
Running backtest using randomly trading automated trading system
> python backtester/run_random_backtest.py

To change ATS configuration, change constructor arguments in source files. 

Inspect backtest results in dashboard
> python backtester/dashboard.py

By default the dashboard loads the most recent backtest-result and log files.
To specify what backtest-result and log files to load, change variables in the dashboard's source file (backtester/dashboard.py)

## Dataset
Due to copyrights, the source datasets or developed dataset can not be shared. However, the structure of the source files can be
explored by inspecting by opening csv-files under dataset_development/datasets/testing/.


## 1. Dataset Development
All factors were produced from datasets available in the “Core US Equities Bundle” delivered by Sharadar Co. These are referred to as “source datasets” going forward. The source datasets included fundamental data from approximately 14,000 companies and spanned 21 years from January 1998 to February 2019. Pricing, volume, and dividend data were available for 16,000 companies, covering the same period. The 14,000 companies included 5000 listed companies and 9000 delisted companies. The datasets were largely void of survivorship bias, making them suitable for backtesting. 
In short, the type of data available for feature construction was:
-	Fundamental data from 10-K and 10-Q filings
-	Some popular financial ratios, like book-to-market, price-to-sales etc.
-	Daily pricing data (open, high, low, close) adjusted for splits and dividends
-	Dividends paid per share reported on ex-dividend dates
-	Daily volume for each stock
-	Company related events, such as bankruptcies reported at the date of filing with the SEC.
The sheer size of the datasets being handed and the fact that feature calculations often required data from multiple source datasets, covering different time ranges and sampling frequencies, made dataset development a demanding process. Computation of features relied heavily on multiprocessing. Implementation details are not covered in the report as the source code can be consulted for this information. 

A total of 86 predictor variables from the academic literature was produced.

## 2. Regressions on Monthly 

Regressions are performed on monthly equity risk premiums using the following algorithms:
- Multiple linear regression
- Principal component regression
- Random forest
- Deep neural net


## 3. ATS Classification Models

ATS classification models are built using random forest algorithms and use the triple barrier method and meta-labeling to train one model to set the side of trades and a second model to set the size of trades.

## 4. ML-Drivne Automated Trading System

The major components of the ATS are:
- A dataset of over 340,000 samples spanning over 8000 companies and 7 years (2012-03-01 to 2019-02-01), each with 86 features. 
- A classification model predicting the sign of the return at the first barrier touch (see Triple Barrier Method, section 3.1.4.2). The output of this model is used to set the side of trades. This model is henceforth called the primary model. 
- A classification model predicting whether following the primary model’s advice will yield a positive or negative return. This classification model is henceforth called “the secondary model” or “the certainty model.” Further, the probability of positive return is from here on referred to as the level of certainty the secondary model has in a trade’s ability to generate positive returns. 
- An algorithm generating trading signals from ML model predictions
- An algorithm using trading signals to produce orders under a set of restrictions

In short, the system operates as follows:
The primary and secondary ML models make predictions based on samples from the test set. The predictions are used by a signal generation algorithm to generate trading signals. The Trading signals are fed to an order generation algorithm, which produces market orders for the best trading signals with the aim of maximizing the level of investment in the market and maintain the maximum allowed number of trades. The order generation algorithm is also subject to other restrictions. The resulting orders are given to a broker object, which simulates their execution and maintenance.


## 5. Backtester

The major components of the backtester are:
- A backtester object with an event loop, looping over each day between the backtester’s start date and end date.
- A broker object with logic to handle:
    - Order processing
    - Trade management, which involves:
	    - Detecting when stop-loss, take-profit or timeout limits are exceeded
	    - Calculate close prices and trade returns
	    - Give proceeds from closed trades to the portfolio’s cash balance
	    - Logic to handle corporate action like dividends, bankruptcies and delisting’s
	    - Logic to calculate margin requirements and issue margin
	    - Logic to deal with interest payments on both cash and margin accounts.
- A portfolio object with properties representing the cash and margin account balance, logic to log all inn and out flows from each account and logic to calculate return and cost metrics.
- An object representing market data, which includes pricing data, interest rates and corporate actions. This object also has a property which represents the current date of the backtest, all objects refer to this property to stay in sync with the rest of the system. Only having a single point of refence for the  backtest’s current date also helps avoid lookahead bias. All the objects involved in the backtester software query the market data object for information. 

The backtester’s event loop goes through the following main steps at each iteration, in order:
- Create market-data-event, initiating the next day of the event loop.
- If it is a business day: 
    - The portfolio/strategy generate trading signals (see ATS description for details)
    - The portfolio/strategy generate orders based on the trading signals (see ATS description for details)
    - The broker processes the orders by calculating commissions, fill price and performs various checks. 
    - If all succeeds a trade-object is created and added to the broker’s blotter
    - If something fails, a cancelled-order-object is created, which then can be handled by the portfolio/strategy if desired
    - The broker manages the active trades by checking if any exit conditions are met, if they are; the respective trade is closed. Exit conditions include take-profit, stop-loss or timeout being exceeded, the company declaring bankruptcy, or the company being delisted.
        - If take-profit was exceeded (verified by doing checks against the day’s high and low prices) the close price of the trade is set to be the price which would make the total return (including dividends) equal to the take profit limit
        - If stop-loss was exceeded the close price of the trade is set to be the price which would make the total return (including dividends) equal to the stop-loss limit.
        - If timeout was reached, the close price of the trade is set to the close price at the timeout date. 
        - If a company files for bankruptcy, the trade is closed at the end of the day when bankruptcy was filed. The closing price is set to zero, meaning that no proceeds from the liquidation is received by the portfolio.
        - If the company is delisted, the close price of the trade is set to the close price of the stock on the date of delisting.
        - The broker pays out any dividends received on long trades and claims payment to cover dividends on short trades.
- Regardless if it is a business day or not:
    - The broker pays out interest on margin and cash accounts equal to the risk-free rate (calculated as a daily compounding rate from the 3-month treasury rate)
    - The broker charges interest on negative balances on margin and/or cash accounts equal to the risk-free rate (calculated as a daily compounding rate from the 3-month treasury rate). 
    - The broker charges interests on cash borrowed to enter short trades.
    - Various broker and portfolio state information is captured, including the currently active trades and the total value of the portfolio. 
- At completion:
    - Data that was generated during the backtest is captured and written to disk
    - Various backtest and performance statistics are calculated and written to disk
