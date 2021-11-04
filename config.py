import yaml


class ConfigBase:
    def __init__(self, data):
        for key, value in data.items():
            if type(value) is dict:
                self.__setattr__(key, ConfigBase(value))
            else:
                self.__setattr__(key, value)


def load_config(path):
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.load(fp, yaml.FullLoader)

    return ConfigBase(data)
