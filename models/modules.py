import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, head_num, model_dim, ffn_dim, drop_out=0.1, post_norm=True
    ) -> None:
        super().__init__()
        
