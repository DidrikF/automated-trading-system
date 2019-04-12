"""
General approach for producing signals for automated trading system with ML:


I need to have a model for every year from 2008 (2010) - 20019 (might only end up backtesting from 2010-2018).

Each time a new model is made, I run validation with the last 30% of the data that is available at the time. This leaves 70% of data for training.

Once the optimal hyperparameters are found, I train each model on all the data that is available at the given time.

I might later run more advanced validation schemes (custom cross validation).

The model is tested on data from the end of the validation set and to the end of the dataset.
This makes it so that more and more data is available for training and validation and less and less is available for testing

"""
