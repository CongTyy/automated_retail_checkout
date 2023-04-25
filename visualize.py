import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
def visualize_feature(features, save_path, source_img):
    # B C H W
    features = features[0] 
    # C H W
    features = torch.sum(features, dim=0)
    features = np.array(features.cpu())
    # H W

    source_img = source_img[0].cpu()
    source_img = source_img*255
    source_img = np.array(source_img.permute(1, 2, 0), dtype='uint8')
    source_img = cv2.cvtColor(cv2.resize(source_img, (features.shape[0], features.shape[1])), cv2.COLOR_BGR2GRAY)
    

    plot_img = np.hstack((source_img, features))
    plt.imshow(plot_img)
    plt.savefig(save_path)

    plt.clf()
