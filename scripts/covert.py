# changes the data format from orca robomimic codebase to act codebase
import h5py
import numpy as np
import os
from tqdm import tqdm

def convert(rbm_file_path, act_dir_path, episode_length, use_frame_stack=False):

    with h5py.File(rbm_file_path) as f_rbm:
        # actions
        actions = []
        qpos = []
        qvel = []
        images = {"camera_primary": [], "camera_wrist": []}
        for id, demo in enumerate(f_rbm["data"].keys()):
            print(f"processing {demo}")
            ###
            actions = f_rbm[f"data/{demo}/actions"][:]
            ###
            obs = []
            for s in ["ee_pos", "joint_qpos", "gripper_qpos"]:
                obs.append(f_rbm[f"data/{demo}/obs/{s}"][:])
            qpos = np.concatenate(obs, axis=1)
            ###
            images = {}
            for camera in ["camera_primary", "camera_wrist"]:
                images[camera] = f_rbm[f"data/{demo}/obs/{camera}"][:]

            # whether using frame stack
            if use_frame_stack:
                # stack with previous frame
                # for camera in ["camera_primary", "camera_wrist"]:
                #     tmp = np.concatenate([images[camera][:-1], images[camera][1:]], axis=1)
                #     images[camera] = tmp
                # stack with wrist
                tmp = np.concatenate([images["camera_primary"], images["camera_wrist"]], axis=1)
                images["camera_primary"] = tmp

            # truncate or extend to episode_length with the last element
            if len(actions) > episode_length:
                actions = actions[:episode_length]
                qpos = qpos[:episode_length]
                for camera in ["camera_primary", "camera_wrist"]:
                    images[camera] = images[camera][:episode_length]
            else:
                actions = np.concatenate([actions, actions[-1:].repeat(episode_length - len(actions), axis=0)])
                qpos = np.concatenate([qpos, qpos[-1:].repeat(episode_length - len(qpos), axis=0)])
                for camera in ["camera_primary", "camera_wrist"]:
                    images[camera] = np.concatenate([images[camera], images[camera][-1:].repeat(episode_length - len(images[camera]), axis=0)])

            with h5py.File(os.path.join(act_dir_path, f"episode_{id}.hdf5"), "w") as f_act:
                f_act.create_dataset("action", data=actions)
                obs_act = f_act.create_group("observations")
                obs_act.create_dataset("qpos", data=qpos)
                images_act = obs_act.create_group("images")
                for camera in ["camera_primary", "camera_wrist"]:
                    images_act.create_dataset(camera, data=images[camera])

if __name__ == "__main__":
    path = ''   # absolute path to orcagym colected data
    path2 = '../datasets/sim_frankapickup/'
    if not os.path.exists(path2):
        os.makedirs(path2)
    convert(path, path2, 100, use_frame_stack=False)