from __future__ import print_function, division

import subprocess, cv2
import PIL.Image as Image
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.onnx as onnx
import torchvision.models as models
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import glob
import sys
import random
import math
import argparse
from matplotlib import pyplot as plt

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

vid = cv2.VideoCapture(0)  # from webcam
start_time = time.time()
cmd = 'C:/Users/livqu/Downloads/CSU_RA/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose/bin/OpenPoseDemo.exe  --net_resolution -1x256'
cmd2 = 'bin/OpenPoseDemo.exe  --net_resolution -1x256 --disable_blending'
cmd3 = 'bin/OpenPoseDemo.exe  --image_dir input_img --net_resolution -1x256 --disable_blending --write_images output_images'
subprocess.call(cmd2)
print('after calling openpose')
output = 'input_img'
i =0
while vid.isOpened():
    current_time = time.time()
    print('running')

    ret, frame = vid.read()
    # cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    '''    
    while i < 15:
        cv2.imwrite('input_img/input.jpg', frame)
        time.sleep(1)
    '''
    if current_time-start_time >= 2:
        start_time = current_time
        cv2.imwrite('input_img/input.jpg', frame)
        subprocess.call(cmd3)   # call openpose to render the image into 2D skeleton

    # time.sleep(1.0 - time.time() + current_time)


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()