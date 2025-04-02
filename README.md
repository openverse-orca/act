# How to use ACT with Orcagym
Step 1:
- Arrange your repo like this:
    - anyname
        - act (this repo)
        - checkpoints
        - datasets
            - sim_frankapickup
Step 2:
- convert teleoperation franka data from orcagym format to the format that this repo recognize
    - use scripts/convert.py (use your own path1 variable)
Step 3:
- Run training like you would in ACT repo
    - I would run this command
        python3 imitate_episodes.py \
        --task_name sim_frankapickup \
        --ckpt_dir ../checkpoints \
        --policy_class ACT --kl_weight 0.1 --chunk_size 50 --hidden_dim 512 --batch_size 16 --dim_feedforward 1600 \
        --num_epochs 2000  --lr 1e-4 \
        --seed 0
Step 4 Inference:
- Inference is harder, act/other_envs/franka/policy.py is for interence
- You should copy and cover {your path}/OrcaGym/examples/imitation/run_franka_single_arm.py and {your path}/OrcaGym/3rd_party/robomimic/robomimic/utils/train_utils.py with act/other_envs/code_for_orcagym/run_franka_single_arm.py and act/other_envs/code_for_orcagym/train_utils.py
- Then run inference in orcagym repo with use_act flag enabled, but remember to include --model_file flag in it too. You should run a training with the same data in orcagym and use the output modelfile. This is for env metainfo.


# ACT: Action Chunking with Transformers

### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    conda create -n aloha_orca2 python=3.8.10
    conda activate aloha_orca2
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments    python3 imitate_episodes.py \
    --task_name sim_frankapickup \
    --ckpt_dir ../checkpoints \
    --policy_class ACT --kl_weight 0.1 --chunk_size 50 --hidden_dim 512 --batch_size 16 --dim_feedforward 1600 \
    --num_epochs 2000  --lr 1e-4 \
    --seed 0 run:

    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_frankapickup \
    --ckpt_dir ../checkpoints \
    --policy_class ACT --kl_weight 0.1 --chunk_size 50 --hidden_dim 512 --batch_size 16 --dim_feedforward 1600 \
    --num_epochs 2000  --lr 1e-4 \
    --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.


# How to train Franka using orcagym and act?
1. do teleoperation as usual, get the data
2. convert the data into the format that act codebase recognize
```
python scripts/covert.py
```
3. run training as usual using act
4. To do inference, a folder named other_envs and use_act flag are added to adapt the act codebase to orcagym rollout.
Before inference, you should train a tiny model using orcagym because some env config are contained there.
Make sure you are using the version of orcagym that supports act.
Then run inference as usual using orcagym (the model file you specified won't be used if use_act enabled) and enable use_act flag.

