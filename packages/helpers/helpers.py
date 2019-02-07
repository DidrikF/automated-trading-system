import os
import sys

def print_exception_info(e):
    """
    Takes an exception and prints where the exception ocurred, its type and its message.
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    line_number = exc_tb.tb_lineno
    print(e)
    print("Type: ", exc_type)
    print("At: ", filename, " line: ", line_number)

