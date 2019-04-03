import pytest
import pandas as pd 
import os
from backtester import Backtester
from algorithm import BuyAppleAlgorithm
from broker import Broker
from data_handler import DailyBarsDataHander, MLFeaturesDataHandler
from utils import EquityCommissionModel, EquitySlippageModel

sep = None

@pytest.setup()
def setup():

  sep = pd.read_csv("./datasets/sharadar/sep.csv")

  if not os.path.isfile("./datasets/"):
    # set up dir
    pass

  yield



def test_backtester():
  global sep

  def handle_data(context, time_context, perf):
    print("HANDLE DATA HOOK RUNNING")


  def initialize(context, time_context, perf):
    print("INITIALIZE HOOK RUNNING")

  def analyze(context, time_context, perf):
    print("ANALYZE HOOK RUNNING")


  data_handler = DailyBarsDataHander()
  feature_handler = MLFeaturesDataHandler()


  backtester = Backtester(data_handler=data_handler, feature_handler=feature_handler, )

  algorithm = BuyAppleAlgorithm()

  slippage_model = EquitySlippageModel()

  commission_model = EquityCommissionModel()

  
  backtester.set_slippage_model(slippage_model)
  backtester.set_commission_model(commission_model)
  backtester.set_algorithm(algorithm)

  performance = backtester.run()

  print(performance.head())