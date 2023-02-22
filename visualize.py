import matplotlib.pyplot as plt
import torch
def vis(fm, name):
    # plt.figure(figsize=(50, 10))
    # print(fm)
    layer_viz = fm.squeeze().detach().cpu()
    layer_viz = torch.sum(layer_viz, dim = 0)
    # for i, filter in enumerate(layer_viz):
    #     if i == 16:
    #         break
        # plt.subplot(2, 8, i + 1)
    plt.imshow(layer_viz)
    plt.savefig(name)
