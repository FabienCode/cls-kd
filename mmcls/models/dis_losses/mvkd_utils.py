import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatRec(nn.Module):
    def __init__(self, in_dim, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.linear = nn.Linear(in_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, in_dim)
        self.norm1 = nn.LayerNorm(in_dim)

        time_dim = in_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(in_dim),
            nn.Linear(in_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, in_dim*2)
        )

    def forward(self, f_t, t, condition=None):
        f_t1 = self.linear(f_t)
        f_t1 = self.norm(self.dropout(f_t1))

        f_t2 = self.self_attn(f_t1, f_t1, f_t1)[0]
        f_t2 = f_t1 + self.dropout(f_t2)

        f_t3 = self.linear1(f_t2)
        f_t = self.norm1(f_t + self.dropout(f_t3))

        temb = self.time_mlp(t)
        scale, shift = temb.chunk(2, dim=-1)
        f_t = f_t * (scale + 1) + shift

        return f_t


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings




