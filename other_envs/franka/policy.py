from policy import ACTPolicy
import torch
import numpy as np
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
    def __init__(self, ckpt_dir, device="cuda", temperal_agg=True, max_episode_steps=100, ACTION_STEP=5, use_frame_stack=False):
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

        self.temporal_agg = temperal_agg
        self.num_queries = 15
        self.state_dim = 8
        if temperal_agg:
            self.max_steps = int(max_episode_steps / ACTION_STEP)
            self.all_time_actions = torch.zeros([self.max_steps, self.max_steps+self.num_queries, self.state_dim])

        self.use_frame_stack = False  # hardcoded
        if self.use_frame_stack:
            self.previous_frame = None


    def __call__(self, ob, goal=None, t=None):
        return self.get_action(ob, goal, t)

    def get_action(self, obs, goal=None, t=None):
        # qpos
        tmp = []
        for s in ["ee_pos", "joint_qpos", "gripper_qpos"]:
            tmp.append(torch.from_numpy(obs[s]).float())
        qpos = self.pre_process(torch.concat(tmp, axis=0)).unsqueeze(0).to(self.device)
        # image camera_primary (3, 128, 128), camera_wrist (3, 128, 128) => batch, num_cam, channel, height, width
        ## two camera
        # image_primary = torch.from_numpy(obs["camera_primary"]).float()
        # image_wrist = torch.from_numpy(obs["camera_wrist"]).float()
        # image = torch.stack([image_primary, image_wrist]).unsqueeze(0).to(self.device) / 255.0
        ## one camera
        image_primary = torch.from_numpy(obs["camera_primary"]).float()
        if self.use_frame_stack:
            # stack with previous frame
            # if self.previous_frame is None:
            #     tmp = torch.concat([image_primary, image_primary], dim=1)
            #     self.previous_frame = image_primary
            #     image_primary = tmp
            # else:
            #     tmp = torch.concat([self.previous_frame, image_primary], dim=1)
            #     self.previous_frame = image_primary
            #     image_primary = tmp

            # stack with wrist
            image_wrist = torch.from_numpy(obs["camera_wrist"]).float()
            tmp = torch.concat([image_primary, image_wrist], dim=1)
            image_primary = tmp

        image = torch.stack([image_primary]).unsqueeze(0).to(self.device) / 255.0
        # action
        action = self.model(qpos, image).detach().cpu()

        if self.temporal_agg:
            print(f"t: {t}")
            self.all_time_actions[[t], t:t+self.num_queries] = action
            actions_for_curr_step = self.all_time_actions[:, t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.1
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
            action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        action = self.post_process(action.squeeze().numpy())
        return action
    
    def start_episode(self):
        self.model.eval()
        self.model.cuda()