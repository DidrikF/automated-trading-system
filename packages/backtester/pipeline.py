

# WTF is PIPELINE?? will it help
# its probably something to make complex calculatesion based on data faster adn more expressive.
# I probably need something to serve a similar purpose...

# Zipline links factors to pipelines? I need a way to compute statistics and results for each trading day...
# Zipline has built in Factors that compute daily return, EWMA, max drawdown and other things. Don't know if this abstraction is suited for me.
# See: zipline.pipeline.factors.XYZ
# Zipline has a type of object; Filter, that can controll data through pipelines or something.
# Various data transformation (general timeseries math etc.) are made easier through pipelines, dont know if ill need this..
# Zipline must address the problem I am having of running computations access large datasets, where the factors have non trivial relationships between them.


class DataSet():
  """
  Base class for describing inputs to Pipeline expressions.
  A DataSet is a collection of zipline.pipeline.data.Column that describes a collection of logically-related inputs to the Pipeline API.
  To create a new Pipeline dataset, subclass from this class and create columns at class scope for each attribute of your dataset. 
  Each column requires a dtype that describes the type of data that should be produced by a loader for the dataset. Integer columns 
  must also provide a missing_value to be used when no value is available for a given asset/date combination.
  """

  # Seems like Zipline has landed on using a MultiIndex of (date, asset) pairs (in some places at least.)

