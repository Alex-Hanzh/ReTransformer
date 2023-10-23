import torch
from .config_handler import YamlHandler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = YamlHandler(r"./config/config.yaml").read_yaml()
