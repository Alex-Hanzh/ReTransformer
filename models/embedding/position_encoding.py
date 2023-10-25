import torch
import torch.nn as nn


class PositionEncoding(nn.Module):
    """
    compute the position encoding(sinusoid)
    """

    def __init__(self, d_model, max_len, device):
        super(PositionEncoding, self).__init__()
        self.encoding = torch.zeros(size=(max_len, d_model), device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 计算每个位置的每个维度的PE
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x: torch.Tensor):
        _, seq_len = x.size()
        return self.encoding[:seq_len, :]
