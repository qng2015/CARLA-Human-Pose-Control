from __future__ import print_function, division

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

plt.ion()  # interactive mode
# Normalize the data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])

backward = carla.VehicleControl(throttle=0.5, steer=0.0, reverse=True)
forward = carla.VehicleControl(throttle=0.6, steer=0.0)
steerLeft = carla.VehicleControl(throttle=0.3, steer=-0.33)
steerRight = carla.VehicleControl(throttle=0.315, steer=0.4)
brake = carla.VehicleControl(brake=1)
cityTurnLeft = carla.VehicleControl(throttle=0.3, steer=-0.3)
cityTurnRight = carla.VehicleControl(throttle=0.3, steer=0.4)


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):  # the 6 classes inside 'val'
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):  # the images (j) inside each of the 6 classes
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                predicted = class_names[preds[j]]
                if predicted == 'left':
                    print('my prediction: left')
                if predicted == 'forward':
                    print('my prediction: forward')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


data_transforms2 = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
data_dir2 = 'data2'
image_datasets2 = {x: datasets.ImageFolder(os.path.join(data_dir2, x),
                                           data_transforms2[x])
                   for x in ['val']}
dataloaders2 = {x: torch.utils.data.DataLoader(image_datasets2[x], batch_size=1,
                                               shuffle=True, num_workers=0)
                for x in ['val']}
dataset_sizes2 = {x: len(image_datasets2[x]) for x in ['val']}
class_names2 = image_datasets2['val'].classes


def test_single2(model):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders2['val']):  # the 6 classes inside 'val'
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                pose = class_names2[preds[j]]
                if pose == 'right':
                    print('index', preds[j])
                    print('image class: ', class_names2[preds[j]])
                    imshow(inputs.data[j], class_names2[preds[j]])
                elif pose == 'left':
                    print('index', preds[j])
                    print('image class: ', class_names2[preds[j]])
                    imshow(inputs.data[j], class_names2[preds[j]])
                elif pose == 'forward':
                    print('index', preds[j])
                    print('image class: ', class_names2[preds[j]])
                    imshow(inputs.data[j], class_names2[preds[j]])
                elif pose == 'backward':
                    print('index', preds[j])
                    print('image class: ', class_names2[preds[j]])
                    imshow(inputs.data[j], class_names2[preds[j]])
                elif pose == 'speedup':
                    print('index', preds[j])
                    print('image class: ', class_names2[preds[j]])
                    imshow(inputs.data[j], class_names2[preds[j]])
                elif pose == 'brake':
                    print('index', preds[j])
                    print('image class: ', class_names2[preds[j]])
                    imshow(inputs.data[j], class_names2[preds[j]])


def test_single(model):
    classes = ('backward', 'brake', 'forward', 'left', 'right', 'speedup')
    model.eval()
    images_so_far = 0
    imsize = 256
    # loader2 = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

    loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # test_data = 'data2/val'   
    # transform = transforms.ToTensor()
    left = 'data2/val/left/left169_rendered.png'
    left2 = 'data2/val/left/input_rendered.png'
    right = 'data2/val/right/right129_rendered.png'
    forward = 'data2/val/forward/forward95_rendered.png'

    img = Image.open(forward)
    img = loader(img).float()
    imshow(img)
    img = img.unsqueeze(0)
    img = img.cuda()  # add this line when running image from CPU on a machine with CUDA-capable GPU

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
        # _, pred = output.max(1).indices

        print('type: ', pred)
        print('predicted: ', classes[pred])


# loading a saved model
def load_model(model):
    model.load_state_dict(torch.load('posesGPU.pth'))
    return model


######################################################################
# Finetuning the convnet
# ----------------------
# Load a pretrained model and reset final fully connected layer.
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)

model_ft = model_ft.to(device)

model_ft = load_model(model_ft)

test_single(model_ft)
# visualize_model(model_ft)

# ConvNet as fixed feature extractor
# ----------------------------------
#
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad == False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 6)

model_conv = model_conv.to(device)

model_conv = load_model(model_conv)
# visualize_model(model_conv)

plt.ioff()
plt.show()


def run():
    torch.multiprocessing.freeze_support()
    # print('loop')


if __name__ == '__main__':
    # freeze_support()
    run()
