import torch
import torch.nn as nn
from typing import Optional, Union
import math
from modules import *


def Embedding(num_embeddings, embedding_dim, padding_idx) -> nn.Embedding:
    embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(
        embed.weight, mean=0, std=embedding_dim**-0.5
    )  # xavier initialization
    nn.init.constant_(embed.weight[padding_idx], 0)
    return embed


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        head_num: int,
        encoder_layer_num: int,
        post_norm: bool = True,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        if not post_norm:
            self.layer_norm = nn.LayerNorm(model_dim)  #  pre norm最后一层需要再做一次LN

        self.layers.extend(
            [
                TransformerDecoderLayer(
                    head_num, model_dim, ffn_dim, post_norm=post_norm
                )
                for _ in range(encoder_layer_num)
            ]
        )

    def forward(
        self,
        net_input: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = net_input

        for layer in self.layers:
            x, _ = layer(x, padding_mask, attn_mask)

        if hasattr(self, "layer_norm"):
            x = self.layer_norm(x)

        return x


if __name__ == "__main__":
    pass
