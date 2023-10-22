import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx,
        tgt_pad_idx,
        tgt_sos_idx,
        enc_voc_size,
        dec_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        drop_prob,
        device,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = tgt_pad_idx
        self.trg_sos_idx = tgt_sos_idx

        self.device = device
