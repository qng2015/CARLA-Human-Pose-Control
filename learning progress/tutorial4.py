import glob
import os
import sys
import random
import time
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

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

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)    
    world = client.get_world()  
     
    blueprint_library = world.get_blueprint_library()
    # pick a vehicle
    car_bp = blueprint_library.filter("model3")[0]
    print(car_bp)
    # get a random spawn point
    spawn_point = random.choice(world.get_map().get_spawn_points())
    # So let's tell the world to spawn the vehicle.
    vehicle = world.spawn_actor(car_bp, spawn_point)
    # add the actor to the list to destroy later
    
    actor_list.append(vehicle)
    world.tick() # wait for the world to get the vehicle actor
    world_snapshot = world.wait_for_tick()
    print(vehicle.id)

    # try spawning more cars
    """spawn_point.location += carla.Location(x=40, y=-3.2)
    spawn_point.rotation.yaw = -90.0
    for _ in range(0, 10):
        spawn_point.location.x += 7.0

        car_bp = random.choice(blueprint_library.filter('vehicle'))

            # By using try_spawn_actor, if the spot is already
            # occupied by another object, the function will return None.
        npc = world.try_spawn_actor(car_bp, spawn_point)
        if npc is not None:
            actor_list.append(npc)
            npc.set_autopilot()
            print('created %s' % npc.type_id)
    """
          
    # actor_snapshot = world_snapshot.find(vehicle.id)
    actor_snapshot = world.get_actor(vehicle.id)
    # v = actor_snapshot.get_velocity()
    spectator = world.get_spectator()
    car_spawn = vehicle.get_transform()

    # set the camera on where the car was placed
    # spectator.set_transform(actor_snapshot.get_transform())
    spectator.set_transform(carla.Transform(car_spawn.location + carla.Location(z=20), 
                carla.Rotation(pitch=-90)))
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    # actor_list.append(vehicle)
    backward = carla.VehicleControl(throttle=0.5, steer=0.0, reverse=True)
    forward = carla.VehicleControl(throttle =0.5, steer = 0.0)
    steerLeft1 = carla.VehicleControl(throttle=0.3, steer=-0.1)
    steerLeft2 = carla.VehicleControl(throttle=0.3, steer= -0.5)
    steerLeft3 = carla.VehicleControl(throttle=0.3, steer = -1.0)
    steerRight = carla.VehicleControl(throttle=0.3, steer = 1.0)
    brake = carla.VehicleControl(brake= 1)

    
    # vehicle.apply_control(forward)
    # speed_up(vehicle,5)   
    # time.sleep(7)
    v = actor_snapshot.get_velocity()
    speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)    
    print(v.x)
    print(v.y)
    print(v.z)
    print("Speed: %15.0f m/s" % speed)   

    '''
    vehicle.apply_control(brake)
    time.sleep(2)
    vehicle.apply_control(backward)
    time.sleep(5)
    # speed_up(vehicle, 11)
    # time.sleep(11)
    '''
    vehicle.apply_control(steerLeft3)
    time.sleep(10)
    '''
    vehicle.apply_control(steerLeft2)
    time.sleep(5)
    vehicle.apply_control(steerLeft3)
    time.sleep(5)
    '''
   #  blueprint = 
finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")