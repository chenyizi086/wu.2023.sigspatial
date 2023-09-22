import copy

import torch
import torch.nn as nn

from einops import rearrange

class TEMP(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        return_att=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(TEMP, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.attention_heads = MultiHeadAttentionNonSelf(
            n_head=n_head, d_k=d_k, d_in=self.d_model, pos_encoding=positional_encoding, dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sz_b, seq_len, d, h, w = x.shape
        
        out = rearrange(x, 'b t n h w -> (b h w) t n')
        out, attn = self.attention_heads(out)
        out = rearrange(out, '(b h w) x  -> b x h w', b=sz_b, h=h, w=w)
        attn = rearrange(attn, 'n (b h w) t -> n b t h w', b=sz_b, h=h, w=w)

        if self.return_att:
            return out, attn
        else:
            return out

class MultiHeadAttentionNonSelf(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    Non-self attention
    """

    def __init__(self, n_head, d_k, d_in, pos_encoding, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        d_k = d_in//n_head
        self.d_k = d_k
        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        # nn.init.normal_(self.fc1_q.weight, mean=0, std=0.02)   # std=np.sqrt(2.0 / (d_k))

        self.fc1_kv= nn.Linear(d_in, 2*n_head*d_k)
        # nn.init.normal_(self.fc1_kv.weight, mean=0, std=0.02) # std=np.sqrt(2.0 / (d_k))

        self.attention = ScaledDotProductAttentionNonSelf(temperature=d_k ** -0.5, n_head=n_head, pos_encoding=pos_encoding, attn_dropout=dropout)

        self.proj = nn.Linear(d_in, d_in)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head

        sz_b, seq_len, _ = v.size()
        v_self = v[:,0:1,:]
        v_nonself = v[:,0:]

        seq_len = v_nonself.shape[1]
        q = self.fc1_q(v_self).view(sz_b, 1, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous()

        kv = self.fc1_kv(v_nonself).view(sz_b, seq_len, 2, n_head, d_k)
        kv = kv.permute(2, 3, 0, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        output, attn = self.attention(q, k, v)

        attn = attn.view(self.n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)
        
        output = output.view(self.n_head, sz_b, 1, self.d_in // self.n_head)
        output = output.squeeze(dim=2)
        output = rearrange(output, 'n x c -> x (n c)')

        output = self.proj(output)
        output = self.proj_drop(output)

        return output, attn


class ScaledDotProductAttentionNonSelf(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1, window_size=(1,5), n_head=8, pos_encoding=True):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
        if pos_encoding:
            pos_bias = nn.Parameter(torch.zeros((n_head, window_size[0], window_size[1]))).requires_grad_(True).cuda()  # n_head, 1, 4
            nn.init.normal_(pos_bias, std=0.02)
            self.pos_bias = pos_bias.unsqueeze(1)
        else:
            self.pos_bias = 0
            print("no pos bias")

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / self.temperature
        attn = attn + self.pos_bias
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn