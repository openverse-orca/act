from policy import ACTPolicy
import torch
import os
import pickle
from types import SimpleNamespace

def make_policy(policy_class, policy_config):
    print(policy_config)
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


class PolicyForFranka:
    def __init__(self, ckpt_dir, device="cuda"):
        # load config
        with open(os.path.join(os.path.join(ckpt_dir, 'config.pkl')), 'rb') as f:
            config = pickle.load(f)
        self.model = make_policy("ACT", config["policy_config"])
        self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'policy_last.ckpt')))
        # self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'policy_epoch_200_seed_0.ckpt')))

        self.device = device

        with open(os.path.join(os.path.join(ckpt_dir, 'dataset_stats.pkl')), 'rb') as f:
            self.norm_stats = pickle.load(f)
        
        self.pre_process = lambda s_qpos: (s_qpos - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']
        self.post_process = lambda a: a * self.norm_stats['action_std'] + self.norm_stats['action_mean']

    def __call__(self, ob, goal=None):
        return self.get_action(ob, goal)

    def get_action(self, obs, goal=None):
        # qpos
        tmp = []
        for s in ["ee_pos", "joint_qpos", "gripper_qpos"]:
            tmp.append(torch.from_numpy(obs[s]).float())
        qpos = self.pre_process(torch.concat(tmp, axis=0)).unsqueeze(0).to(self.device)
        # image camera_primary (3, 128, 128), camera_wrist (3, 128, 128) => batch, num_cam, channel, height, width
        image_primary = torch.from_numpy(obs["camera_primary"]).float()
        image_wrist = torch.from_numpy(obs["camera_wrist"]).float()
        image = torch.stack([image_primary, image_wrist]).unsqueeze(0).to(self.device) / 255.0
        # action
        action = self.model(qpos, image).to("cpu").squeeze().detach().numpy()
        action = self.post_process(action)
        return action
    
    def start_episode(self):
        self.model.eval()
        self.model.cuda()