import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatRec(nn.Module):
    def __init__(self, in_dim, d_model=256, nhead=8, dropout=0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        self.linear = nn.Linear(in_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=self.dropout)

        self.linear1 = nn.Linear(d_model, in_dim)
        self.norm1 = nn.LayerNorm(in_dim)

    def forward(self, f_t, condition=None):
        f_t1 = self.linear(f_t)
        f_t = self.norm(f_t + self.dropout(f_t1))

        f_t2 = self.self_attn(f_t, f_t, f_t)[0]
        f_t = f_t + self.dropout(f_t2)

        f_t3 = self.linear1(f_t)
        f_t = self.norm1(f_t + self.dropout(f_t3))

        return f_t




