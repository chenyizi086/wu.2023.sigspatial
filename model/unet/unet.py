from torch import nn
from model.unet.unet_parts import *


class UNET(nn.Module):
    def __init__(self, n_channels, n_classes, channel_list=[16, 32, 64, 128, 256], bilinear=True):
        super(UNET, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channel_list = channel_list

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

        self._initialize_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.sigmoid(self.outc(x))
        return logits

    def _initialize_weights(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        # If no pretrain, apply KAIMING initialization
        self.apply(weights_init)
        print('Kaiming initliazation done.')