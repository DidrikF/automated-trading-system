import logging
import datetime


class Logger():
    def __init__(self, file_path: str):
        logr = logging.getLogger(__name__)

        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        
        fileHandler = logging.FileHandler(file_path)
        fileHandler.setFormatter(logFormatter)
        
        logr.addHandler(fileHandler)

        self.logr = logr