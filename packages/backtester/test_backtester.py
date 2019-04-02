import pytest
import pandas as pd 


sep = None

@pytest.setup()
def setup():

  sep = pd.read_csv("./datasets/sharadar/sep.csv")

  if not os.isfile("./datasets/")

  yield



def test_backtester():
  global sep

  backtester = Backtester()

  algorithm = BuyAppleAlgorithm()

  slippage_model = FixedBasisPointSlippageModel()

  commission_model = IBCommissionModel()

  
  backtester.set_slippage_model(slippage_model)
  backtester.set_commission_model(commission_model)
  backtester.set_algorithm(algorithm)

  performance = backtester.run()

  print(performance.head())