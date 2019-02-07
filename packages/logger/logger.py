from ..helpers.helpers import print_exception_info
import datetime
import os
import time

class Logger():
    def __init__(self, path_to_log_folder):

        timestr = time.strftime("%Y%m%d-%H%M%S")

        if path_to_log_folder is not None:
            self.path = path_to_log_folder + "/log_{}.txt".format(timestr)
        else:
            self.path = "./log_{}.txt".format(timestr)
        
        directory_path = os.path.dirname(self.path)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        try:
            print(self.path)
            self.fh = open(self.path, "w+")
            # maybe do with with statement and avoid problems of closing the file, if the program should crash
        except Exception as e:
            raise e

    def log_exc(self, exception: Exception):
        self.fh.write(exception.__str__()+"\n")

    def write(self, *msgs):
        for index, msg in enumerate(msgs):
            self.fh.write(msg)

    def close(self):
        self.fh.close()
        