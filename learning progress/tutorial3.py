import glob
import os
import sys
import random
import time
import math
import argparse
import matplotlib as plt

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def parser():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=20,
        type=int,
        help='number of vehicles (default: 30)')
    argparser.add_argument(
        '-d', '--number-of-dangerous-vehicles',
        metavar='N',
        default=1,
        type=int,
        help='number of dangerous vehicles (default: 3)')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=True,
        help='Synchronous mode execution')

    return argparser.parse_args()

actor_list = []

def speed_up(vehicle, time_in):
    flag = True
    _control = carla.VehicleControl()
    # starttime = time.time()
    while flag: 
        if _control.throttle < 0.8:      
            _control.throttle = min(_control.throttle + 0.01, 1)
            vehicle.apply_control(_control)
        else:
            flag = False
        # time.sleep(time_in)
        # time.sleep(60.0 - ((time.time() - starttime) % 60.0))
def camera_go(car_spawn): # method for camera to follow the car
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(car_spawn.location + carla.Location(z=30),
                                            carla.Rotation(pitch=280)))

try:
    # args = parser()

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)    
    world = client.get_world()  
    # world = client.load_world('Town04') 
    # create traffic manager
    traffic_manager = client.get_trafficmanager(8000)
   
    blueprint_library = world.get_blueprint_library()
    
    # pick a vehicle
    car_bp = blueprint_library.filter("model3")[0]
    print(car_bp)
    # get the first spawn point
    spawn_list = world.get_map().get_spawn_points()
    spawn_point = spawn_list[1]
    # So let's tell the world to spawn the vehicle.
    vehicle = world.spawn_actor(car_bp, spawn_point)
    # add the actor to the list to destroy later
    
    actor_list.append(vehicle)
    world.tick() # wait for the world to get the vehicle actor
    world_snapshot = world.wait_for_tick()
    print(vehicle.id)


    
    # actor_snapshot = world_snapshot.find(vehicle.id)
    actor_snapshot = world.get_actor(vehicle.id)
    
    # while True:
    spectator = world.get_spectator()
    car_spawn = vehicle.get_transform() # get our car's spawn point to compute the distance to the other car

    # set the camera on where the car was placed
    # spectator.set_transform(actor_snapshot.get_transform())
    spectator.set_transform(carla.Transform(car_spawn.location + carla.Location(z=20), 
            carla.Rotation(pitch=-90)))
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    # actor_list.append(vehicle)
    backward = carla.VehicleControl(throttle=0.5, steer=0.0, reverse=True)
    forward = carla.VehicleControl(throttle =0.5, steer = 0.0)
    steerLeft = carla.VehicleControl(throttle=0.7, steer = -1.0)
    steerRight = carla.VehicleControl(throttle=0.7, steer = 1.0)
    brake = carla.VehicleControl(brake= 1)

    # try spawning more cars
    '''
    # spawn_point.location += carla.Location(x=40, y=-3.2)
    # spawn_point.rotation.yaw = -180.0
    car_bp = random.choice(blueprint_library.filter('vehicle'))
    spawn_point.location.x = car_spawn.location.x + 15.0
    # spawn_point = spawn_list[3]
    npc = world.spawn_actor(car_bp, spawn_point)
    npc_spawn = npc.get_transform() 
    actor_list.append(npc)
    npc_snapshot = world.get_actor(npc.id)
    # print("distance before autopilot: %4d  " % car_spawn.location.distance(npc.get_location()))
    vehicle.set_autopilot()
    npc.set_autopilot()
    '''

    # get information about lanes
    waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
    print("Current lane type: " + str(waypoint.lane_type))
    # Check current lane change allowed
    print("Current Lane change:  " + str(waypoint.lane_change))
    # Left and Right lane markings
    print("L lane marking type: " + str(waypoint.left_lane_marking.type))
    print("L lane marking change: " + str(waypoint.left_lane_marking.lane_change))
    print("R lane marking type: " + str(waypoint.right_lane_marking.type))
    print("R lane marking change: " + str(waypoint.right_lane_marking.lane_change))
    #while True:
    # print("distance after calling autopilot: %4d" % car_spawn.location.distance(npc.get_location()))
    # By using try_spawn_actor, if the spot is already
    # occupied by another object, the function will return None.
    '''
    for _ in range(0, 10):
        car_bp = random.choice(blueprint_library.filter('vehicle'))
        spawn_point.location.x += 8.0
        npc = world.try_spawn_actor(car_bp, spawn_point)
        
        if npc is not None:
            actor_list.append(npc)
            npc.set_autopilot()
            print('created %s' % npc.type_id)   
    '''      
    
    va = actor_snapshot.get_velocity()
    speed = math.sqrt(va.x**2 + va.y**2 + va.z**2)  
    
    vp = actor_snapshot.get_velocity()
    speed2 = math.sqrt(vp.x**2 + vp.y**2 + vp.z**2)
    print("relative speed: %15.0f m/s" % abs(speed-speed2))
    # vehicle.apply_control(forward)

    # speed_up(vehicle,5)

    while True:
        car_spawn = vehicle.get_transform()
        camera_go(car_spawn)

        vehicle.apply_control(backward)
        time.sleep(2)

        vehicle.apply_control(steerLeft)
        time.sleep(2)

        vehicle.apply_control(steerRight)
        time.sleep(0.90)

        vehicle.apply_control(forward)
        time.sleep(5)

    '''
    v = actor_snapshot.get_velocity()
    speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)    
    print(v.x)
    print(v.y)
    print(v.z)
    print("Speed (in m/s): %15.0f m/s" % speed)   
    print("Speed (in km/h): %15.0f km/h" % (speed * 3.6))
    vehicle.apply_control(brake)
    time.sleep(2)
    vehicle.apply_control(backward)
    time.sleep(5)
    # speed_up(vehicle, 11)
    # time.sleep(11)

    vehicle.apply_control(steerRight)
    time.sleep(6)
    vehicle.apply_control(steerLeft)
    time.sleep(6)
    vehicle.apply_control(forward)
    time.sleep(7)
    '''
  
finally:
    # for actor in actor_list:
        # actor.destroy()
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print("All cleaned up!")