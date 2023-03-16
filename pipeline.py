import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import sys
import torch
import cv2
import torch.backends.cudnn as cudnn
from numpy import random
from vit import inference
from ultralytics import YOLO
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default="AIC/wGAN_citers/train26/weights/best.pt", help='model.pt path(s)')
# parser.add_argument('--source', type=str, default='TestA/', help='source')  # file/folder, 0 for webcam
parser.add_argument('--source', type=str, default='/home/hoangtv/Desktop/Ty/ultralytics/AIC22-TEAM55-TRACK4/testA/', help='source')  # file/folder, 0 for webcam
# parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
opt = parser.parse_args()

model = YOLO(opt.weights)

video_paths = glob(f'{opt.source}/*.mp4')
for vp in video_paths:
    pred = model.predict(   source = vp,
                            conf = 0.5,
                            iou = 0.5,
                            device = 'cuda:0',
                            save = True)
    
