import glob
import os
import sys
import random
import time
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

actor_list = []
# xAxis = range(15)
xAxis = []
yAxis = []

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

client = carla.Client("localhost", 2000)
client.set_timeout(5.0)  

try:
     
    world = client.get_world()  
    # world = client.load_world('Town04') 
    # create traffic manager
    # traffic_manager = client.get_trafficmanager()
    # tm_port = traffic_manager.get_port()
    
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
    world.tick() # wait for the world to get the vehicle actor
    world_snapshot = world.wait_for_tick()
    print(vehicle.id)

    # spawn a camera sensor
    camera_bp = blueprint_library.find('sensor.camera.depth')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actor_list.append(camera)
    print('created %s' % camera.type_id)
    
    # actor_snapshot = world_snapshot.find(vehicle.id)
    actor_snapshot = world.get_actor(vehicle.id)
    world.tick()
    
    # while True:
    spectator = world.get_spectator()
    car_spawn = vehicle.get_transform()

    # set the camera on where the car was placed
    # spectator.set_transform(actor_snapshot.get_transform())
    spectator.set_transform(carla.Transform(car_spawn.location + carla.Location(z=30), 
            carla.Rotation(pitch=280)))
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    # actor_list.append(vehicle)
    backward = carla.VehicleControl(throttle=0.5, steer=0.0, reverse=True)
    forward = carla.VehicleControl(throttle =0.6, steer = 0.0)
    steerLeft = carla.VehicleControl(throttle=0.3, steer = -0.33)
    steerRight = carla.VehicleControl(throttle=0.315, steer = 0.4)
    brake = carla.VehicleControl(brake= 1)
    cityTurnLeft = carla.VehicleControl(throttle=0.3, steer = -0.3)
    cityTurnRight = carla.VehicleControl(throttle=0.3, steer = 0.4)
    '''
    # try spawning more cars
    # spawn_point.location += carla.Location(x=40, y=-3.2)
    # spawn_point.rotation.yaw = -180.0
    car_bp = random.choice(blueprint_library.filter('mustang'))
    spawn_point.location.x = car_spawn.location.x + 25.0
    # spawn_point = spawn_list[3]
    npc = world.spawn_actor(car_bp, spawn_point)
    npc_spawn = npc.get_transform() 
    actor_list.append(npc)
    npc_snapshot = world.get_actor(npc.id)
    # print("distance before autopilot: %4d  " % car_spawn.location.distance(npc.get_location()))
    # npc.set_autopilot(True)
    '''

    # get information about lanes
    waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
    second_waypoint = world.get_map().get_waypoint(npc.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
    print("Current lane type: " + str(waypoint.lane_type))
    # Check current lane change allowed
    print("Current Lane change:  " + str(waypoint.lane_change))
    # Left and Right lane markings
    print("our section id: " + str(waypoint.section_id))
    # print("npc's second id: ")
    # print("L lane marking change: " + str(waypoint.left_lane_marking.lane_change))
    print("current lane id " + str(waypoint.lane_id))
    print("lane width: " + str(waypoint.lane_width))
    #while True:
    print("distance after calling autopilot: %4d" % car_spawn.location.distance(npc.get_location())) 
    
    
    # add obstacle sensor to our car
    obs_bp = world. get_blueprint_library().find('sensor.other.obstacle')
    obs_bp.set_attribute("only_dynamics", str(True))
    obs_location = carla.Location(0,0,0)
    obs_rotation = carla.Rotation(0,0,0)
    obs_transform = carla.Transform(obs_location, obs_rotation)
    vehicle_obs = world.spawn_actor(obs_bp, obs_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    actor_list.append(vehicle_obs)
    def obs_callback(obs):
        # print("Obstacle detected:\n"+str(obs)+'\n')
        # if obs.distance < 15:
        print("Obstacle detected: %4d" % obs.distance)
            # print("Obstacle detected: ", obs.other_actor)
        
    vehicle_obs.listen(lambda obs: obs_callback(obs))

    npc.set_autopilot(True)
    vehicle.apply_control(forward)
    
    # speed_up(vehicle,5)   
    # time.sleep(15)
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
    i=0
    current_waypoint = waypoint
    
        
    while i < 15:
        car_spawn = vehicle.get_transform() # get our car's spawn point to compute the distance to the other car

        # set the camera on where the car was placed
        # spectator.set_transform(actor_snapshot.get_transform())
        '''
        spectator.set_transform(carla.Transform(car_spawn.location + carla.Location(z=30), 
            carla.Rotation(pitch=-90)))
        '''
        our_waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
        npc_waypoint = world.get_map().get_waypoint(npc.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))

        car_spawn = vehicle.get_transform()
        distance = car_spawn.location.distance(npc.get_location()) # update the distance
        va = actor_snapshot.get_velocity()
        speed = math.sqrt(va.x**2 + va.y**2 + va.z**2)  

        vp = actor_snapshot.get_velocity()
        speed2 = math.sqrt(vp.x**2 + vp.y**2 + vp.z**2)
        # print("relative speed: %15.0f m/s" % abs(speed-speed2))
        
        
        if distance <= 18 and our_waypoint.lane_id == npc_waypoint.lane_id:
            
            vehicle.apply_control(cityTurnLeft)
            time.sleep(0.8)
            vehicle.apply_control(cityTurnRight)
            time.sleep(0.816)
            vehicle.apply_control(forward)
                    
        
        xAxis.append(i)
        yAxis.append(float("%4d" % distance))  # Distance 4 is enough for collision (even though it's not zero) maybe due to boundbox/hitbox
        print(float("%4d" % distance))
        time.sleep(1)
        i +=1
        if our_waypoint.lane_id != npc_waypoint.lane_id:
            print("our lane id: ", our_waypoint.lane_id)
            print("other car's lane id: ", npc_waypoint.lane_id)
    plt.plot(xAxis, yAxis, color='green', marker='o', linestyle='solid')
    plt.ylabel("distance")
    plt.xlabel("seconds")
    plt.show()
        
    time.sleep(15)

finally:
    # for actor in actor_list:
        # actor.destroy()
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print("All cleaned up!")