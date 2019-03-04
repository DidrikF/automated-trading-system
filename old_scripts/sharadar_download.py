import quandl
import sys

quandl.ApiConfig.api_key = "Jz9zQ2ZK49CXoWLKQkkk"




# data = quandl.get_table('SHARADAR/SF1', dimension='ARQ', ticker='AAPL')

if __name__ == "__main__":
    arguments = sys.argv
    table = arguments[1]

    if table == "fundamentals_demo":
        data = quandl.get_table('SHARADAR/SF1')
        print(data.head())
        data.to_csv("./datasets/sharadar/fundamentals_MRY_demo.csv")
    elif table == "fundamentals_ARQ_demo":
        data = quandl.get_table('SHARADAR/SF1', dimension='ARQ')
        print(data.head())
        data.to_csv("./datasets/sharadar/fundamentals_ARQ_demo.csv")
    elif table == "fundamentals_ART_demo":
        data = quandl.get_table('SHARADAR/SF1', dimension='ART')
        print(data.head())
        data.to_csv("./datasets/sharadar/fundamentals_ART_demo.csv")
    elif table == "price_demo":
        data = quandl.get_table('SHARADAR/SEP')
        print(data.head())
        data.to_csv("./datasets/sharadar/price_demo.csv")
    elif table == "indicator_descriptions_demo":
        data = quandl.get_table('SHARADAR/INDICATORS')
        data.to_csv("./datasets/sharadar/indicators_demo.csv")
    else:
        print(table + " not a valid option!")
























































