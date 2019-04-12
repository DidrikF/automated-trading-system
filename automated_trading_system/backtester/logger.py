import logging


def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance()


@singleton
class Logger:
    def __init__(self):
        log = logging.getLogger(__name__)

        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

        file_handler = logging.FileHandler("./logs/backtest.log")
        file_handler.setFormatter(logFormatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logFormatter)

        log.addHandler(file_handler)
        log.addHandler(console_handler)

        self.log = log
