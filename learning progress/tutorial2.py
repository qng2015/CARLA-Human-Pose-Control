import glob
import os
import sys
import random
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

actor_list = []
try: 
	client = carla.Client("localhost", 2000)
	client.set_timeout(5.0) 
	world = client.get_world()

	# Start recording
	"""
	client.start_recorder('~/tutorial/recorder/recording01.log')
	"""

	# Spawn ego vehicle
	# --------------
	ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
	ego_bp.set_attribute('role_name','ego')
	print('\nEgo role_name is set')
	ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
	ego_bp.set_attribute('color',ego_color)
	print('\nEgo color is set')

	spawn_points = world.get_map().get_spawn_points()
	number_of_spawn_points = len(spawn_points)

	if 0 < number_of_spawn_points:
	    # random.shuffle(spawn_points)
	    ego_transform = spawn_points[1]
	    ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
	    actor_list.append(ego_vehicle)
	    print('\nEgo is spawned')
	else: 
	    logging.warning('Could not found any spawn points')


	

	# Add Obstacle sensor to ego vehicle. 
	obs_bp = world.get_blueprint_library().find('sensor.other.obstacle')
	obs_bp.set_attribute("only_dynamics",str(True))
	obs_location = carla.Location(0,0,0)
	obs_rotation = carla.Rotation(0,0,0)
	obs_transform = carla.Transform(obs_location,obs_rotation)
	ego_obs = world.spawn_actor(obs_bp,obs_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
	print("obstacle sensor is spawned")
	actor_list.append(ego_obs)
	def obs_callback(obs):
	    print("Obstacle detected:\n"+str(obs)+'\n')
	ego_obs.listen(lambda obs: obs_callback(obs))

	# Spectator on ego position
	# --------------
	spectator = world.get_spectator()
	world_snapshot = world.wait_for_tick() 
	spectator.set_transform(ego_vehicle.get_transform())	

	
	
	# spawn second car
	blueprint_library = world.get_blueprint_library()
	car_spawn = ego_vehicle.get_transform()
	car_bp = random.choice(blueprint_library.filter('vehicle'))
	spawn_point = car_spawn
	spawn_point.location.x = car_spawn.location.x + 25.0
	# spawn_point = spawn_list[3]
	npc = world.spawn_actor(car_bp, spawn_point)
	print("npc car spawned")
	npc_spawn = npc.get_transform() 
	actor_list.append(npc)
	npc_snapshot = world.get_actor(npc.id)
	# print("distance before autopilot: %4d  " % car_spawn.location.distance(npc.get_location()))
	world_snapshot = world.wait_for_tick()
	npc.set_autopilot(True)

	# Enable autopilot for ego vehicle	
	ego_vehicle.set_autopilot(True)
	print("both set to autopilot")
	# Game loop. Prevents the script from finishing.
	        # --------------
	while True:
		car_spawn = ego_vehicle.get_transform()
		spectator.set_transform(carla.Transform(car_spawn.location + carla.Location(z=30), 
			carla.Rotation(pitch=290)))
		ego_obs.listen(lambda obs: obs_callback(obs))
		world_snapshot = world.wait_for_tick()

finally:
# --------------
# Stop recording and destroy actors
# --------------
# client.stop_recorder()
	for actor in actor_list:
		actor.destroy()
