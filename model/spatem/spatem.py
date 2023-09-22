from torch import nn
from model.unet.unet_parts import *
from einops import rearrange
from model.spatem.tem import TEMP
from model.spatem.spatial_vit import SpatialVit


class UNETSpaTem(nn.Module):
    def __init__(self, n_channels, n_classes, \
                 channel_list=[16, 32, 64, 128, 256], \
                 n_head=16, dropout=0.1, ratio=4, \
                 d_model=128, bilinear=True, mode='train'):
        super(UNETSpaTem, self).__init__()
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
        self.up1 = UpConv(self.channel_list[4], self.channel_list[3] // factor, bilinear)
        self.up2 = UpConv(self.channel_list[3], self.channel_list[2] // factor, bilinear)
        self.up3 = UpConv(self.channel_list[2], self.channel_list[1] // factor, bilinear)
        self.up4 = UpConv(self.channel_list[1], self.channel_list[0], bilinear)
        self.outc = OutConv(self.channel_list[0], n_classes)
        self.sigmoid = nn.Sigmoid()

        self.temporal_encoder = TEMP(
            in_channels=128,
            d_model=None,
            n_head=n_head,
            return_att=True,
            d_k=8,
            positional_encoding=False,
            dropout=dropout
        )

        self.spa_vit = SpatialVit(
            image_size = 16,
            ratio=ratio,
            in_channels = 128,
            n_head = n_head,
            d_k = 8,
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
        
        self.spatialtemporalaggregator = SpatialTemporalAggregator()

        if mode == 'train':
            self._initialize_weights()
        else:
            pass

    def forward(self, input_temporal, input_spatial, return_att=False):
        x = torch.concat([input_temporal,input_spatial], axis=1)
        sz_b, seq_len, d, h, w = x.shape
        seq_len_tem = input_temporal.shape[1]
        seq_len_spa = input_spatial.shape[1]
        x = rearrange(x, 'b t d h w -> (b t) d h w')

        # Downsample
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Pre-norm
        _, _ , h, w = x5.shape
        out = x5.flatten(2).transpose(1,2)
        out = self.norm_pre(out)
        out = rearrange(out, 'b (h w) d -> b d h w', h=h, w=w)
        out = rearrange(out,  '(b t) d h w -> b t d h w', b=sz_b, t=seq_len)
        self_out = out[:,0]
        temp_out = out[:,0:seq_len_tem]
        spatial_out = out[:,seq_len_tem:]

        # Temporal multi-head attention
        temp_attn_out, temp_attn = self.temporal_encoder(
            temp_out
        )

        # Spatial multi-head attention
        spa_attn_out, spa_attn = self.spa_vit(spatial_out)

        # Fuse
        out = spa_attn_out + self_out + temp_attn_out

        # Feed Forward network
        # All the feedforward architecture should follow the following structure
        # Layer norm ->
        # Linear layer ->
        # Residual connection
        out = rearrange(out, 'b n h w -> b (h w) n')
        out = self.norm_out(out)
        linear_out = self.linear_net(out)
        out = out + linear_out # Residual connection
        out = rearrange(out, 'b (h w) n -> b n h w', b=sz_b, h=h, w=w)

        # Upsample attention and fuse features
        x4 = self.spatialtemporalaggregator(x4, sz_b, self.ratio, seq_len_tem, seq_len_spa, temp_attn, spa_attn)
        x3 = self.spatialtemporalaggregator(x3, sz_b, self.ratio, seq_len_tem, seq_len_spa, temp_attn, spa_attn)
        x2 = self.spatialtemporalaggregator(x2, sz_b, self.ratio, seq_len_tem, seq_len_spa, temp_attn, spa_attn)
        x1 = self.spatialtemporalaggregator(x1, sz_b, self.ratio, seq_len_tem, seq_len_spa, temp_attn, spa_attn)
        
        # Upsample
        x = self.up1(out, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.sigmoid(self.outc(x))

        if return_att:
            spa_attn = spa_attn.max(2)[0]
            n_head, sz_b, seq_spa = spa_attn.shape
            spa_attn = spa_attn.reshape(n_head, sz_b, seq_spa//16, 4, 4)
            return logits, temp_attn, spa_attn
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

class SpatialTemporalAggregator(nn.Module):
    def __init__(self):
        super(SpatialTemporalAggregator, self).__init__()

    def forward(self, x, sz_b, ratio, seq_len_tem, seq_len_spa, attn_mask_temporal=None, attn_mask_spatial=None):
        x = rearrange(x, '(b t) d h w -> b t d h w', b=sz_b)
        x_tem = x[:,:seq_len_tem]
        x_spa = x[:,seq_len_tem:] # TODO: leave it for future use
        
        # temporal
        n_heads, sz_b, t, h0, w0 = attn_mask_temporal.shape
        attn_temporal = rearrange(attn_mask_temporal, 'n b t h w -> (n b) t h w')

        if x_tem.shape[-2] > w0:
            attn_temporal = nn.Upsample(
                size=x_tem.shape[-2:], mode="bilinear", align_corners=False
            )(attn_temporal)
        else:
            attn_temporal = nn.AvgPool2d(kernel_size=w // x_tem.shape[-2])(attn_temporal)

        # Assemble head and batch dimensions
        attn_temporal = rearrange(attn_temporal, '(n b) t h w -> n b t h w', n=n_heads, b=sz_b) 
        out = torch.stack(x.chunk(n_heads, dim=2)) # b,n,(t+s)*4,h,w

        out_self = out[:,:,0]
        out_temporal = out[:,:,:seq_len_tem]
        out_spatial = out[:,:,seq_len_tem:]

        out_temporal = attn_temporal[:, :, :, None, :, :] * out_temporal
        out_temporal = out_temporal.sum(dim=2)

        n_heads, b, t, d, h, w = out_spatial.shape
        out_spatial = rearrange(out_spatial, 'n b t d h w -> (n b t) d h w')
        out_spatial = nn.AvgPool2d(kernel_size=(h//h0*ratio, w//w0*ratio))(out_spatial)
        out_spatial = rearrange(out_spatial, '(n b t) d h w -> n b (t h w) d', n=n_heads, b=b, t=t)

        out_spatial = torch.matmul(attn_mask_spatial, out_spatial)
        out_spatial = rearrange(out_spatial, 'n b (h w) d -> (n b) d h w', h=h0, w=w0)

        out_spatial = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(out_spatial)
        out_spatial = rearrange(out_spatial, '(n b) d h w -> n b d h w', n=n_heads, b=b)
        
        # Fuse different attentions (self + spatial + temporal)
        out = out_self + out_spatial + out_temporal
        
        out = torch.cat([group for group in out], dim=1)
        return out
