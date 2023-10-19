import torch
import yaml
from config_handler import YamlHandler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = YamlHandler("../config/config.yml").read_yaml()
