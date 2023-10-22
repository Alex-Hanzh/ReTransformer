import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=1):
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx
        )
        nn.init.xavier_normal(self.embedding.weight)

    def forward(self, x: torch.Tensor):
        output = self.embedding(x)
        output = output * (self.d_model**0.5)
        return output
