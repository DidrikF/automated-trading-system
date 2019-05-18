import logging
import datetime

"""
def singleton(cls):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()

@singleton
class SingletonLogger():
    def __init__(self):
        logr = logging.getLogger(__name__)

        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        
        fileHandler = logging.FileHandler("./logs/backtest{}.log".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        fileHandler.setFormatter(logFormatter)
        
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        
        logr.addHandler(fileHandler)
        logr.addHandler(consoleHandler)

        self.logr = logr
"""


class Logger():
    def __init__(self, file_path: str):
        logr = logging.getLogger(__name__)

        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        
        fileHandler = logging.FileHandler(file_path)
        fileHandler.setFormatter(logFormatter)
        
        logr.addHandler(fileHandler)

        self.logr = logr