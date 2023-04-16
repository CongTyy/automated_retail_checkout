import torch.nn as nn
from torch.nn.utils import spectral_norm
import math
from torchinfo import summary
import torch
from ultralytics.nn.gradient_reversal import GradientReversalLayer

class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size = 3, stride = 2, padding = 1, act = True) -> None:
        super().__init__()

        if act:
            self.block = nn.Sequential(
                spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias = False)),
                # nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias = False),
                # nn.InstanceNorm2d(out_chan),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias = False)),
                # nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias = False),
                # nn.InstanceNorm2d(out_chan),
            )

    def forward(self, x):
        return self.block(x)
  
class ResBlock(nn.Module):
    def __init__(self, in_chan) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(in_chan, in_chan, kernel_size=3, stride=1, padding=1, act = True),
            ConvBlock(in_chan, in_chan, kernel_size=3, stride=1, padding=1, act = False)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act((x + self.block(x))/math.sqrt(2))
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.cb1 = ConvBlock(in_chan, out_chan, stride = stride)
        self.cb2 = ConvBlock(out_chan, out_chan, act = False)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.ca = ChannelAttention(out_chan)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.cb1(x)
        out = self.cb2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.act(out)
        return out

class D_layer4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # ConvBlock(64, 128, kernel_size=3, stride=2, padding=1, act = True), # /2
            # ResBlock(128),
            # ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, act = True), #/4
            # ResBlock(256),
            # ConvBlock(256, 512, kernel_size=3, stride=2, padding=1, act = True), # /8
            # ResBlock(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
            # nn.LogSoftmax(),
            # ConvBlock(512, 1, kernel_size=1, stride=1, padding=0, act = False), # 16
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    

class D_layer6(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, act = True), # /2
            # ResBlock(256),
            # ConvBlock(256, 512, kernel_size=3, stride=2, padding=1, act = True), #/4
            # ResBlock(512),
            # ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1, act = True), # /8
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
            # nn.Linear(1024*10*10, 2),
            # nn.LogSoftmax(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class D_layer9(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # ConvBlock(256, 256, kernel_size=3, stride=2, padding=1, act = True), # /2
            # ResBlock(256),
            # ConvBlock(256, 512, kernel_size=3, stride=2, padding=1, act = True), #/4
            # ResBlock(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            # nn.Softmax(),
        )
    def forward(self, x):
        x = self.encoder(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

'''
4: 64 160 160
6: 128 80 80
9: 256 40 40 
'''