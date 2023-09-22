import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from einops import rearrange

import copy

import pdb


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SpatialVit(nn.Module):
    def __init__(
        self,
        image_size,
        ratio,
        in_channels=128,
        n_head=16,
        d_k=8,
        mlp=[256, 256],
        dropout=0.0,
        d_model=256,
        return_att=False,
        positional_encoding=True,
    ):
        super(SpatialVit, self).__init__()
        self.in_channels = in_channels
        self.return_att = return_att
        self.n_head = n_head
        self.mlp = copy.deepcopy(mlp)
        self.pos_encoding = positional_encoding

        # for patch encoding ViT
        image_height, image_width = pair(image_size)

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        window_size = (image_height*image_width, 9*(image_height//ratio)*(image_width//ratio))
        self.attention_heads = MultiHeadViTNonSelf(
            n_head=n_head, 
            d_k=d_k, 
            d_in=self.d_model, 
            window_size=window_size,
            ratio= ratio,
            pos_encoding=positional_encoding
        )

        # Best practice for multi-layer perceptron
        # Linear -> 
        # GELU (non-linear relu) ->
        # Dropout -> 
        # Linear -> 
        # Dropout
        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i+1], self.mlp[i+1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.mlp[i+1], self.mlp[i+1]),
                    nn.Dropout(dropout)
                ]
            )

        # Merge different layer to MLP module
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Pre-LN transformer layer: without requiring warm-up strategy for self-attention module
        sz_b, seq_len, d, h, w = x.shape

        out, attn = self.attention_heads(x) # Non-self attention for center patch and all the neighbouring patches
        out = rearrange(out, 'b h w c -> b c h w', b=sz_b, h=h, w=w)

        pdb.set_trace()
        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadViTNonSelf(nn.Module):
    def __init__(self, n_head, d_k, d_in, window_size, ratio, pos_encoding=True, dropout=0.1):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadViTNonSelf, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.ratio=ratio

        # patch embeddings of ViT
        self.to_patch_embedding_1 = nn.Sequential(
            Rearrange('b c h w -> b (h w) c')
        )
        
        self.to_patch_embedding_2 = nn.Sequential(
            Rearrange('b s c h w -> (b s) c h w'),
            nn.Conv2d(d_in, d_in, kernel_size=self.ratio, stride=self.ratio),
            nn.GroupNorm(n_head, d_in)
        )

        d_k = d_in//n_head
        self.d_k = d_k
        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        self.fc1_kv = nn.Linear(d_in, 2 * n_head * d_k)

        self.temperature = d_k ** -0.5

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(d_in, d_in)
        self.proj_drop = nn.Dropout(dropout)

        # Positional bias added with attention
        if pos_encoding:
            pos_bias = nn.Parameter(torch.zeros((n_head, window_size[0],window_size[1]))).requires_grad_(True).cuda()  # n_head, 1, 4
            nn.init.normal_(pos_bias, mean=0, std=0.02)
            self.pos_bias = pos_bias.unsqueeze(1)
        else:
            self.pos_bias = 0

    def forward(self, x):
        sz_b, seq_len, d, h, w = x.shape
        v_self = x[:,4]

        v_nonself = x
        v_nonself = rearrange(v_nonself, 'b s d h w -> (b s) d h w')
        v_nonself = rearrange(v_nonself, '(b s) d h w -> b s d h w', b=sz_b, s=seq_len)
        
        v_self = self.to_patch_embedding_1(v_self)
        v_nonself = self.to_patch_embedding_2(v_nonself)

        v_nonself = rearrange(v_nonself, '(b s) c h w -> b (s h w) c', b=sz_b, s=seq_len)
        query = self.fc1_q(v_self).view(sz_b, -1, self.n_head, self.d_k).permute(2,0,1,3)
        kv = self.fc1_kv(v_nonself).view(sz_b, -1, 2, self.n_head, self.d_in//self.n_head).permute(2,3,0,1,4) 
        key, value = kv[0], kv[1]

        dots = torch.matmul(query, key.transpose(-1, -2)) * self.temperature
        dots = dots + self.pos_bias

        attn = self.softmax(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value)
        out = rearrange(out, 'n b (h w) c -> b h w (n c)', h=h, w=w)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out, attn