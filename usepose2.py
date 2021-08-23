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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
actor_list = []

client = carla.Client("localhost", 2000)
client.set_timeout(5.0)

# API for vehicle's direction
backward = carla.VehicleControl(throttle=0.5, steer=0.0, reverse=True)
forward = carla.VehicleControl(throttle=0.6, steer=0.0)
steerLeft = carla.VehicleControl(throttle=0.3, steer=-0.33)
steerRight = carla.VehicleControl(throttle=0.315, steer=0.4)
brake = carla.VehicleControl(brake=1)
speedup = carla.VehicleControl(throttle=0.05)
cityTurnLeft = carla.VehicleControl(throttle=0.3, steer=-0.3)
cityTurnRight = carla.VehicleControl(throttle=0.3, steer=0.4)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)
model_ft = model_ft.to(device)


vid = cv2.VideoCapture(0)  # from webcam

def test_single(model):
    classes = ('backward', 'brake', 'forward', 'left', 'right', 'speedup')
    model.eval()
    # loader2 = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

    loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # forward = 'data2/val/forward/forward95_rendered.png'  # image copied from cpu may need line img.cuda() un-commented
    capture = 'output_images/input_rendered.png'
    # capture2 = 'output_images/backward2.jpg'
    img = Image.open(capture)
    img = loader(img).float()
    # imshow(img)
    img = img.unsqueeze(0)
    img = img.cuda()  # add this line when running image from CPU on a machine with CUDA-capable GPU

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

        # print('type: ', pred)
        print('predicted: ', classes[pred])  # get the class prediction
        return classes[pred]

def load_model(model):
    model.load_state_dict(torch.load('poses.pth'))
    return model

model_ft = load_model(model_ft)

def move_car(vehicle, result):
    if result == 'left':
        vehicle.apply_control(steerLeft)
    if result == 'right':
        vehicle.apply_control(steerRight)
    if result == 'forward':
        vehicle.apply_control(forward)
    if result == 'backward':
        vehicle.apply_control(backward)
    if result == 'brake':
        vehicle.apply_control(brake)
    if result == 'speedup':
        vehicle.apply_control(speedup)

try:

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    # pick a vehicle
    car_bp = blueprint_library.filter("model3")[0]
    print(car_bp)
    # get the first spawn point
    spawn_list = world.get_map().get_spawn_points()
    spawn_point = spawn_list[1]
    # spawn_point.location.x -= 80
    # spawn_point = spawn_list[4]
    # So let's tell the world to spawn the vehicle.
    vehicle = world.spawn_actor(car_bp, spawn_point)
    # add the actor to the list to destroy later
    actor_list.append(vehicle)
    world.tick()  # wait for the world to get the vehicle actor
    world_snapshot = world.wait_for_tick()
    print(vehicle.id)

    # actor_snapshot = world_snapshot.find(vehicle.id)
    actor_snapshot = world.get_actor(vehicle.id)
    world.tick()

    spectator = world.get_spectator()
    car_spawn = vehicle.get_transform()

    # try spawning more cars
    '''
    car_bp = random.choice(blueprint_library.filter('mustang'))
    spawn_point.location.x = car_spawn.location.x + 25.0
    # spawn_point = spawn_list[3]
    npc = world.spawn_actor(car_bp, spawn_point)
    npc_spawn = npc.get_transform()
    actor_list.append(npc)
    npc_snapshot = world.get_actor(npc.id)
    npc.set_autopilot(True)
    '''

    start_time = time.time()
    cmd = 'C:/Users/livqu/Downloads/CSU_RA/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose/bin/OpenPoseDemo.exe  --net_resolution -1x256'
    cmd2 = 'bin/OpenPoseDemo.exe  --net_resolution -1x192 --disable_blending --write_images output2'
    cmd3 = 'bin/OpenPoseDemo.exe  --image_dir input_img --net_resolution -1x192 --disable_blending --write_images output_images'
    # subprocess.call(cmd2)

    while vid.isOpened():
        spectator = world.get_spectator()
        car_spawn = vehicle.get_transform()
        # set the camera on where the car was placed
        # spectator.set_transform(actor_snapshot.get_transform())
        spectator.set_transform(carla.Transform(car_spawn.location + carla.Location(z=30),
                                                carla.Rotation(pitch=280)))
        current_time = time.time()
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
            cv2.imwrite('input_img/input.jpg', frame) # write the image to disk
            # time.sleep(1)
            subprocess.call(cmd3)   # call openpose to render the image into 2D skeleton
            result = test_single(model_ft)  # let the model evaluate the image
            move_car(vehicle, result)   # move the car based on result
        # time.sleep(1.0 - time.time() + current_time)

finally:
    # clean all actors in Carla
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print("All cleaned up!")
    # After the loop release the capture object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()