import torch
import torch.nn as nn
from .token_embedding import TokenEmbedding
from .position_encoding import PositionEncoding


class TransformerEmbedding(nn.Module):
    def __init__(
        self, vocab_size, d_model, max_len, device, paddind_idx=1, drop_prob=0.01
    ):
        super(TransformerEmbedding, self).__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model, padding_idx=paddind_idx)
        self.pos_embed = PositionEncoding(d_model, max_len, device)
        # embedding层后接dropout
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor):
        token_emb = self.token_embed(x)
        pos_emb = self.pos_embed(x)
        return self.drop_out(token_emb + pos_emb)
