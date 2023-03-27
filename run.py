import os
import shutil
import carla
import cv2
import random
import numpy as np
# import pygame



def preprocess_img(img, cfg, image_cut: tuple = (115, 510)):
    img = img[image_cut[0]:image_cut[1], :, :]
    print(img.shape)
    img = np.reshape(img, (cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH, 3))
    print(img.shape)
    return img

def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))


def main():
    # connect the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    # retrieve world data
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    blueprints = world.get_blueprint_library()

    try:
        # print(help(spectator))
        vehicle_blueprint = blueprints.find('vehicle.tesla.model3')
        vehicle = world.try_spawn_actor(
            vehicle_blueprint,
            random.choice(spawn_points),
        )

        # spectator to ego vehicle
        spectator = world.get_spectator()
        transform = carla.Transform(
            vehicle.get_transform().transform(carla.Location(x=-5., z=2.5)),
            vehicle.get_transform().rotation
        )
        spectator.set_transform(transform)

        # Create a transform to place a camera on top of the vehicle
        camera_bp = blueprints.find('sensor.camera.rgb')
        camera_init_trans = carla.Transform(
            carla.Location(2.0, 0.0, 1.4),
            carla.Rotation(-15.0, 0, 0),
        )
        camera = world.spawn_actor(
            camera_bp,
            camera_init_trans,
            attach_to=vehicle,
        )
        image_height = camera_bp.get_attribute("image_size_x").as_int()
        image_width = camera_bp.get_attribute("image_size_y").as_int()
        print(f"image height: {image_height}")
        print(f"image width: {image_width}")
        camera_data = {'image': np.zeros((image_height, image_width, 4))}

        # imgs_dir = os.path.join(os.path.abspath(os.getcwd()), "imgs")
        # if os.path.isdir(imgs_dir):
        #     shutil.rmtree(imgs_dir)
        #     os.makedirs(imgs_dir)

        camera.listen(
           lambda image: camera_callback(image, camera_data)
        )
        # actor_list = world.get_actors()
        # print(f"All currently active actors: {actor_list}")

        i = 0
        while True:
            i += 1
            cv2.imshow("RGB Camera", camera_data['image'])
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        vehicle.destroy()
        spectator.destroy()
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
