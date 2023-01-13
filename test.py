import carla 
import random 


def main():
    # connect the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    # retrieve world data 
    world = client.get_world()
    try: 
        spectator = world.get_spectator() 
        # print(help(spectator))
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        print(spawn_point) 
        spectator_transform = carla.Transform(
            carla.Location(
                x=spawn_point.location.x - 4.0,
                y=spawn_point.location.y, 
                z=spawn_point.location.z + 2.5), 
            carla.Rotation(),
        ) 
        
        spectator.set_transform(spectator_transform) 
        vehicle_blueprints = world.get_blueprint_library().filter('vehicle')
        vehicle_blueprint = random.choice(vehicle_blueprints)

        vehicle = world.spawn_actor(vehicle_blueprint, spawn_point) 
        print(vehicle.get_location()) 
        print(vehicle) 

        i = 0
        while True: 
            i += 1 
    
    finally:
        vehicle.destroy()

if __name__ == "__main__":
    main()