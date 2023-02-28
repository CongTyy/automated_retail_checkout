from ultralytics.nn.modules import Bottleneck, nn, Conv, Critic
import numpy as np
from torchsummary import summary
if __name__ == "__main__":
    import torch
    # torch.Size([1, 32, 64, 64])
    critic = Critic(1792, num_layers = 3, out_res= 40).cuda()
    print(critic)
    x = [torch.rand(1, 64, 160, 160).cuda(), torch.rand(1, 128, 80, 80).cuda(), torch.rand(1, 256, 40, 40).cuda()]
    y = critic(x)
    print(y.size())
        
