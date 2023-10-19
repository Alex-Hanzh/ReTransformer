import yaml


class YamlHandler:
    def __init__(self, file):
        self.file = file

    def read_yaml(self):
        with open(self.file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)
        