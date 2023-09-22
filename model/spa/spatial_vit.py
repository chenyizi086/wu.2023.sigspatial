import torch
import torch.nn as nn

from einops import rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SpatialVit(nn.Module):
    def __init__(
        self,
        ratio,
        in_channels=128,
        n_head=16,
        dropout=0.1,
        d_model=256,
        positional_encoding=True,
    ):
        super(SpatialVit, self).__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.pos_encoding = positional_encoding

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.attention_heads = Attention(dim=self.d_model, num_heads=self.n_head, attn_drop=dropout, proj_drop=dropout, sr_ratio=ratio)

        self.fuse_spatial = nn.Sequential(
            nn.Conv2d(9, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LayerNorm(self.d_model),
        )

    def forward(self, x):
        sz_b, seq_len, d, h, w = x.shape

        center = rearrange(x[:, 4:5], 'b s c h w -> (b s) (h w) c')

        out_list = []
        attn_list = []
        for i in range(9):
            neighbor = rearrange(x[:, i:i+1], 'b s c h w -> (b s) (h w) c')
            out, attn = self.attention_heads(center, neighbor, h, w) # Non-self attention for center patch and all the neighbouring patches
            out_list.append(out)
            attn_list.append(attn)
        out = torch.stack(out_list, dim=1)
        attn = torch.stack(attn_list, dim=1)
        out = self.fuse_spatial(out).squeeze(1)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return out, attn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, center, neighbor, H, W):
        B, N, C = center.shape
        q = self.q(center).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = rearrange(neighbor, 'bs (h w) c -> bs c h w', h=H, w=W).clone()
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn