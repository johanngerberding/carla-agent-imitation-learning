import os
import shutil
import carla
import random
import pygame


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
        """
        spectator_transform = carla.Transform(
            carla.Location(
                x=spawn_point.location.x - 4.0,
                y=spawn_point.location.y,
                z=spawn_point.location.z + 2.5),
            carla.Rotation(),
        )
        """
        # spectator.set_transform(spectator_transform)
        vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
        vehicle_blueprint = random.choice(vehicle_blueprints)

        vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)

        ego_vehicle_bp = random.choice(vehicle_blueprints)
        ego_vehicle_sp = random.choice(spawn_points)
        print(f"ego vehicle spawn point: {ego_vehicle_sp}")
        ego_vehicle = world.spawn_actor(
            ego_vehicle_bp,
            ego_vehicle_sp,
        )
        # spectator to ego vehicle
        spectator.set_transform(ego_vehicle_sp)

        # Create a transform to place a camera on top of the vehicle
        camera_init_trans = carla.Transform(carla.Location(z=1.5))
        print(f"camera init transform: {camera_init_trans}")
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera = world.spawn_actor(
            camera_bp,
            camera_init_trans,
            attach_to=ego_vehicle,
        )
        imgs_dir = os.path.join(os.path.abspath(os.getcwd()), "imgs")
        if os.path.isdir(imgs_dir):
            shutil.rmtree(imgs_dir)
            os.makedirs(imgs_dir)

        camera.listen(
            lambda image: image.save_to_disk(
                os.path.join(imgs_dir, f'{image.frame}.png')
            )
        )

        actor_list = world.get_actors()
        print(f"All currently active actors: {actor_list}")

        i = 0
        while True:
            i += 1

    finally:
        vehicle.destroy()


if __name__ == "__main__":
    main()
