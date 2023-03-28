import os
import shutil
import carla
import cv2
import random
import numpy as np
import torch
from config import get_cfg_defaults
from dataset import get_val_transform
from agents.navigation.basic_agent import BasicAgent


class ImitationAgent(BasicAgent):
    def __init__(self, vehicle, target_speed, debug=False):
        super().__init__(target_speed, debug)

    def run_step(self, debug=False):
        control = carla.VehicleControl()
        return control


def preprocess_img(img, cfg, image_cut: tuple = (115, 510)) -> tuple:
    display_img = img[image_cut[0]:image_cut[1], :, :]
    model_img = cv2.resize(img, (200, 88), cv2.INTER_LINEAR)
    return display_img, model_img

def camera_callback(image, data_dict, model, device, cfg):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    dimg, mimg = preprocess_img(img, cfg)
    data_dict['image'] = dimg

    # img reshape (BS, C, H, W)
    # normalize
    # img = torch.tensor(mimg).unsqueeze(0).float().to(device)
    img = get_val_transform()(image=mimg)['image']
    img = img.unsqueeze(0).float().to(device)
    print(img.shape)
    # get the current speed
    speed = None
    # get a command
    cmd = None
    with torch.no_grad():
        out = model(mimg)



def main():
    exp_dir = "/home/johann/dev/conditional-imitation-learning-pytorch/exps/2023-03-26"
    cfg = get_cfg_defaults()
    opts = None
    exp_cfg = os.path.join(exp_dir, "config.yaml")
    if os.path.isfile(exp_cfg):
        cfg.merge_from_file(exp_cfg)
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()

    model_ckpt = os.path.join(exp_dir, "checkpoints/best.pth")
    model = torch.load(model_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # keep track of all actors
    actor_list = []
    # connect the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    # retrieve world data
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    blueprints = world.get_blueprint_library()

    try:
        vehicle_blueprint = blueprints.find('vehicle.tesla.model3')
        vehicle = world.try_spawn_actor(
            vehicle_blueprint,
            random.choice(spawn_points),
        )
        actor_list.append(vehicle)
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
        actor_list.append(camera)
        image_height = camera_bp.get_attribute("image_size_x").as_int()
        image_width = camera_bp.get_attribute("image_size_y").as_int()
        print(f"image height: {image_height}")
        print(f"image width: {image_width}")
        camera_data = {'image': np.zeros((image_height, image_width, 4))}

        camera.listen(
           lambda image: camera_callback(image, camera_data, model, device, cfg)
        )
        # actor_list = world.get_actors()
        # print(f"All currently active actors: {actor_list}")
        vehicle.set_autopilot(True)

        transform = random.choice(spawn_points)
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0, 20):
            transform.location.x += 8.0
            bp = random.choice(world.get_blueprint_library().filter('vehicle'))
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print(f"Created npc id {npc.type_id}")
        while True:
            cv2.imshow("RGB Camera", camera_data['image'])
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        client.apply_batch(
            [carla.command.DestroyActor(a) for a in actor_list]
        )
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
