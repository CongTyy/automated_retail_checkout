from ultralytics.nn.modules import Bottleneck, nn, Conv
import numpy as np
import time
import math
class CriticConv(nn.Module):
    def __init__(self, in_chan, out_chan, ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_chan),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.layer(x)
  
class CriticRes(nn.Module):
    def __init__(self, in_chan, out_chan) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_chan),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_chan),
        )
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.act((x + self.layer(x))/math.sqrt(2))

class Critic(nn.Module):
    def __init__(self, in_chan, in_res, out_res = 40) -> None:
        '''
        IN:
            in_chan: input channel
            in_res: input resolution
            out_res: output resolution

        OUT: (B, 1, out_res, out_res)
        '''
        super().__init__()
        num_layers = int(np.log2(in_res // out_res))
        self.layers = []
        for i in range(num_layers):
            self.layers.append(CriticRes(in_chan, in_chan))
            self.layers.append(CriticConv(in_chan, in_chan*2))
            in_chan *= 2
            if i == num_layers-1:
                self.layers.append(nn.Conv2d(in_chan, 1, 1))
        if num_layers == 0:
            self.layers.append(nn.Conv2d(in_chan, 1, 1))

        self.critic = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.critic(x)
if __name__ == "__main__":
    from torchsummary import summary
    import torch
    # torch.Size([1, 32, 64, 64])
    critic = Critic(256, 40).cuda()
    x = torch.rand(1, 256, 40, 40).cuda()
    y = critic(x)
    print(y.size())
        
