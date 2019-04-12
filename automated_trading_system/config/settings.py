import configparser


class Settings:
    def __init__(self, env: str = "PROD", root: str = "../"):
        cfg = configparser.ConfigParser()
        cfg._interpolation = configparser.ExtendedInterpolation()
        cfg.read(root + "automated_trading_system/config/config.ini")

        self.env = env
        self.cfg = cfg
        self.root = root

    def __get(self, key: str):
        return self.cfg.get(self.env, key, raw=False)

    def __get_path(self, key: str):
        return self.root + self.__get(key)

    def get_data_set(self):
        return self.__get_path("data_set")

    def get_save_location(self):
        return self.__get_path("save_location")
