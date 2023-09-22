from torch import nn
from model.unet.unet_parts import *
from einops import rearrange
from model.tem.temporal_vit import TemporalVit


class UNETTem(nn.Module):
    def __init__(self, n_channels, n_classes, \
                 channel_list=[16, 32, 64, 128, 256], \
                 n_head=16, dropout=0.1, ratio=4, \
                 d_model=128, bilinear=True, mode='train'):
        super(UNETTem, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.ratio = ratio

        self.channel_list = channel_list
        print('Channel:{}'.format(self.channel_list))
        # Default: channel_list=[16, 32, 64, 128, 256]

        self.inc = DoubleConv(n_channels, self.channel_list[0])
        self.down1 = Down(self.channel_list[0], self.channel_list[1])
        self.down2 = Down(self.channel_list[1], self.channel_list[2])
        self.down3 = Down(self.channel_list[2], self.channel_list[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(self.channel_list[3], self.channel_list[4] // factor)
        self.up1 = Up(self.channel_list[4], self.channel_list[3] // factor, bilinear)
        self.up2 = Up(self.channel_list[3], self.channel_list[2] // factor, bilinear)
        self.up3 = Up(self.channel_list[2], self.channel_list[1] // factor, bilinear)
        self.up4 = Up(self.channel_list[1], self.channel_list[0], bilinear)
        self.outc = OutConv(self.channel_list[0], n_classes)
        self.sigmoid = nn.Sigmoid()

        self.temporal_vit = TemporalVit(
            ratio=ratio,
            in_channels = 128,
            n_head = n_head,
            d_model=None,
            return_att=True,
            dropout= dropout,
            positional_encoding=False
            )

        self.norm_pre = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        self.linear_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

        if mode == 'train':
            self._initialize_weights()
        else:
            pass

    def forward(self, input_temporal, input_spatial, return_att=False):
        sz_b, seq_len, d, h, w = input_temporal.shape
        x = rearrange(input_temporal, 'b t d h w -> (b t) d h w')

        # Downsample
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Pre-norm
        _, _ , h, w = x5.shape
        x5_out = x5.flatten(2).transpose(1,2)
        x5_out = self.norm_pre(x5_out)
        x5_out = rearrange(x5_out, 'b (h w) d -> b d h w', h=h, w=w)
        x5_out = rearrange(x5_out,  '(b t) d h w -> b t d h w', b=sz_b, t=seq_len)

        # Spatial multi-head attention
        temp_attn_out, temp_attn = self.temporal_vit(x5_out)

        # Residual connection
        out = rearrange(x5, '(b t) c h w -> b t c h w', b=sz_b, t=seq_len)[:, 0]
        # out[:, 0] = temp_attn_out + out[:, 0]
        out = temp_attn_out + out

        # Feed Forward network
        # All the feedforward architecture should follow the following structure
        # Layer norm ->
        # Linear layer ->
        # Residual connection
        out = rearrange(out, 'b c h w -> b (h w) c')
        out = out + self.linear_net(self.norm_out(out)) # Residual connection
        out = rearrange(out, 'b (h w) c -> b c h w', b=sz_b, h=h, w=w)

        # Upsample
        x4 = rearrange(x4, '(b s) c h w -> b s c h w', b=sz_b, s=seq_len)[:, 0]
        x = self.up1(out, x4) # Only center patch is used

        x3 = rearrange(x3, '(b s) c h w -> b s c h w', b=sz_b, s=seq_len)[:, 0]
        x = self.up2(x, x3)

        x2 = rearrange(x2, '(b s) c h w -> b s c h w', b=sz_b, s=seq_len)[:, 0]
        x = self.up3(x, x2)

        x1 = rearrange(x1, '(b s) c h w -> b s c h w', b=sz_b, s=seq_len)[:, 0]
        x = self.up4(x, x1)

        out = self.outc(x)
        logits = self.sigmoid(out)

        if return_att:
            temp_attn = temp_attn.max(3)
            sz_b, seq_len, n_head, xy = temp_attn.shape
            temp_attn = temp_attn.reshape(sz_b, seq_len, n_head, 4, 4)
            return logits, temp_attn
        else:
            return logits

    def _initialize_weights(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        # If no pretrain, apply KAIMING initialization
        self.apply(weights_init)
        print('Kaiming initliazation done.')
