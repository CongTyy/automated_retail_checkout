# import torch.nn as nn
# from torch.nn.utils import spectral_norm
# import math
# from torchinfo import summary
# import torch
# from ultralytics.nn.gradient_reversal import GradientReversalLayer

# class ConvBlock(nn.Module):
#     def __init__(self, in_chan, out_chan, kernel_size = 3, stride = 2, padding = 1, act = True) -> None:
#         super().__init__()

#         if act:
#             self.block = nn.Sequential(
#                 spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias = False)),
#                 # nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias = False),
#                 # nn.InstanceNorm2d(out_chan),
#                 nn.LeakyLeakyReLU(0.2,0.2, inplace=True)
#             )
#         else:
#             self.block = nn.Sequential(
#                 spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias = False)),
#                 # nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias = False),
#                 # nn.InstanceNorm2d(out_chan),
#             )

#     def forward(self, x):
#         return self.block(x)
  
# class ResBlock(nn.Module):
#     def __init__(self, in_chan) -> None:
#         super().__init__()

#         self.block = nn.Sequential(
#             ConvBlock(in_chan, in_chan, kernel_size=3, stride=1, padding=1, act = True),
#             ConvBlock(in_chan, in_chan, kernel_size=3, stride=1, padding=1, act = False)
#         )
#         self.act = nn.LeakyLeakyReLU(0.2,0.2, inplace=True)

#     def forward(self, x):
#         return self.act((x + self.block(x))/math.sqrt(2))
    
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.LeakyReLU(0.2,),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
    
# class BasicBlock(nn.Module):
#     def __init__(self, in_chan, out_chan, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.cb1 = ConvBlock(in_chan, out_chan, stride = stride)
#         self.cb2 = ConvBlock(out_chan, out_chan, act = False)
#         self.act = nn.LeakyLeakyReLU(0.2,0.2, inplace=True)
#         self.ca = ChannelAttention(out_chan)
#         self.sa = SpatialAttention()

#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         out = self.cb1(x)
#         out = self.cb2(out)
#         out = self.ca(out) * out
#         out = self.sa(out) * out
#         out = self.act(out)
#         return out

# class D_layer4(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encode = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 2, 1), # 128 80 80
#             nn.LeakyReLU(0.2,inplace=True), 
#             nn.Conv2d(128, 256, 3, 2, 1), # 64 40 40
#             nn.LeakyReLU(0.2,inplace=True), 
#             nn.Conv2d(256, 512, 3, 2, 1), # 128 20 20
#             nn.LeakyReLU(0.2,inplace=True), 
#             nn.Conv2d(512, 512, 3, 2, 1), # 64 10 10
#             nn.LeakyReLU(0.2,inplace=True), 
#         )
    
#         self.last = nn.Sequential(
#             nn.Linear(51200, 10, bias=False),
#             nn.LeakyReLU(0.2,True),
#             nn.Linear(10, 1, bias=False),
#         )
#         # self.last = nn.Sequential(
#         #     nn.Conv2d(512, 1, 1, 1),
#         # )
#     def forward(self, x):
#         x = self.encode(x)
#         x = x.view(x.shape[0], -1)
#         x = self.last(x)
#         return x
    

# class D_layer6(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encode = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 2, 1), # 128 40 40
#             nn.LeakyReLU(0.2,inplace=True), 
#             nn.Conv2d(128, 256, 3, 2, 1), # 64 20 20
#             nn.LeakyReLU(0.2,inplace=True), 
#             nn.Conv2d(256, 512, 3, 2, 1), # 64 10 10
#             nn.LeakyReLU(0.2,inplace=True), 
#         )
    
#         self.last = nn.Sequential(
#             nn.Linear(51200, 10, bias=False),
#             nn.LeakyReLU(0.2,True),
#             nn.Linear(10, 1, bias=False),
#         )
#         # self.last = nn.Sequential(
#         #     nn.Conv2d(512, 1, 1, 1),
#         # )
#     def forward(self, x):
#         x = self.encode(x)
#         x = x.view(x.shape[0], -1)
#         x = self.last(x)
#         return x

# class D_layer9(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encode = nn.Sequential(
#             nn.Conv2d(256, 512, 3, 2, 1), # 512 20 20
#             nn.LeakyReLU(0.2,inplace=True), 
#             nn.Conv2d(512, 512, 3, 2, 1), # 64 10 10
#             nn.LeakyReLU(0.2,inplace=True), 
#         )
    
#         self.last = nn.Sequential(
#             nn.Linear(51200, 10, bias=False),
#             nn.LeakyReLU(0.2,True),
#             nn.Linear(10, 1, bias=False),
#         )
#         # self.last = nn.Sequential(
#         #     nn.Conv2d(512, 1, 1, 1),
#         # )
#     def forward(self, x):
#         x = self.encode(x)
#         x = x.view(x.shape[0], -1)
#         x = self.last(x)
#         return x


    
'''
4: 64 160 160
6: 128 80 80
9: 256 40 40 
'''



import torch.nn as nn
from torch.nn.utils import spectral_norm
import math
from torchinfo import summary
import torch
from ultralytics.nn.gradient_reversal import GradientReversalLayer

class D_layer4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 3, 2, 1, bias=False)), # 128 80 80
            nn.LeakyReLU(0.2,inplace=True), 
            spectral_norm(nn.Conv2d(128, 256, 3, 2, 1, bias=False)), # 64 40 40
            nn.LeakyReLU(0.2,inplace=True), 
            spectral_norm(nn.Conv2d(256, 512, 3, 2, 1, bias=False)), # 128 20 20
            nn.LeakyReLU(0.2,inplace=True), 
            spectral_norm(nn.Conv2d(512, 512, 3, 2, 1, bias=False)), # 64 10 10
            nn.LeakyReLU(0.2,inplace=True), 
        )
    
        self.last = nn.Sequential(
            nn.Linear(51200, 10, bias=False),
            nn.LeakyReLU(0.2,True),
            nn.Linear(10, 1, bias=False),
        )
        # self.last = nn.Sequential(
        #     spectral_norm(nn.Conv2d(512, 1, 1, 1),
        # )
    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.shape[0], -1)
        x = self.last(x)
        return x
    

class D_layer6(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 128, 3, 2, 1, bias=False)), # 128 40 40
            nn.LeakyReLU(0.2,inplace=True), 
            spectral_norm(nn.Conv2d(128, 256, 3, 2, 1, bias=False)), # 64 20 20
            nn.LeakyReLU(0.2,inplace=True), 
            spectral_norm(nn.Conv2d(256, 512, 3, 2, 1, bias=False)), # 64 10 10
            nn.LeakyReLU(0.2,inplace=True), 
        )
    
        self.last = nn.Sequential(
            nn.Linear(51200, 10, bias=False),
            nn.LeakyReLU(0.2,True),
            nn.Linear(10, 1, bias=False),
        )
        # self.last = nn.Sequential(
        #     spectral_norm(nn.Conv2d(512, 1, 1, 1),
        # )
    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.shape[0], -1)
        x = self.last(x)
        return x

class D_layer9(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            spectral_norm(nn.Conv2d(1792, 512, 1, 1, bias=False)), # 512 20 20
            nn.LeakyReLU(0.2,inplace=True), 
            spectral_norm(nn.Conv2d(512, 512, 1, 1, bias=False)), # 64 10 10
            nn.LeakyReLU(0.2,inplace=True), 
        )
    
        self.last = nn.Sequential(
            nn.Linear(512*40*40, 100, bias=False),
            nn.LeakyReLU(0.2,True),
            nn.Linear(10, 1, bias=False),
        )
        # self.last = nn.Sequential(
        #     spectral_norm(nn.Conv2d(512, 1, 1, 1),
        # )
    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.shape[0], -1)
        x = self.last(x)
        return x
    

# class D_layer9(Detect):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encode = nn.Sequential(
#             spectral_norm(nn.Conv2d(1792, 512, 1, 1, bias=False)), # 512 20 20
#             nn.LeakyReLU(0.2,inplace=True), 
#             spectral_norm(nn.Conv2d(512, 512, 1, 1, bias=False)), # 64 10 10
#             nn.LeakyReLU(0.2,inplace=True), 
#         )
    
#         self.last = nn.Sequential(
#             nn.Linear(512*40*40, 10, bias=False),
#             nn.LeakyReLU(0.2,True),
#             nn.Linear(10, 1, bias=False),
#         )
        
#     def forward(self, x, source =  False):
#         x = self.encode(x)
#         c = c.view(x.shape[0], -1)
#         c = self.last(c)
#         x = torch.cat((self.cv2[2](x), self.cv3[2](x)), 1)
#         if source == False:
#             return c
#         else:
#             return c, x
# class D_layer9(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encode = nn.Sequential(
#             spectral_norm(nn.Conv2d(1792, 256, 3, 2, 1, bias=False)), # 512 20 20
#             nn.LeakyReLU(0.2,inplace=True), 
#             spectral_norm(nn.Conv2d(256, 128, 3, 2, 1, bias=False)), # 64 10 10
#             nn.LeakyReLU(0.2,inplace=True), 
#         )
    
#         self.last = nn.Sequential(
#             nn.Linear(12800, 10, bias=False),
#             nn.LeakyReLU(0.2,True),
#             nn.Linear(10, 1, bias=False),
#         )
#         # self.last = nn.Sequential(
#         #     spectral_norm(nn.Conv2d(512, 1, 1, 1),
#         # )
#     def forward(self, x):
#         x = self.encode(x)
#         x = x.view(x.shape[0], -1)
#         x = self.last(x)
#         return x