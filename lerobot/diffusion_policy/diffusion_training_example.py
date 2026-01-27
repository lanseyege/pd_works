"""This script demonstrates how to train Diffusion Policy on a real-world dataset."""

from pathlib  import Path
from datetime import datetime
import os, sys

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

batch_size = 128
num_workers = 4

training_steps = 60000
log_freq = 100

horizon = 16
n_action_steps = 8
n_obs_steps = 2
drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1

''' using pose feature or not
    False ： use pose feature
    True : drop pose feature
'''
drop_pose_feature = False

# Select your device
device = torch.device("cuda")  # or "cuda" or "cpu"

dataset_id = "batch3_right"
dataset_root = "/home/yuanye/dataset/batch3_right"

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if not drop_pose_feature:
    folder_name = f"outputs/batch3_right/diffusion_{current_time}_pose"
else:
    folder_name = f"outputs/batch3_right/diffusion_{current_time}"

def print_conf():
    print(f"batch_size = {batch_size}")
    print(f"num_workers = {num_workers}")
    print(f"training_steps = {training_steps}")
    print(f"log_freq = {log_freq}")
    print(f"device = {device}")
    print(f"model saved path is {folder_name}")
    print(f"use pose feature or not: {not drop_pose_feature}")

print_conf()

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


output_directory = Path(folder_name)
output_directory.mkdir(parents=True, exist_ok=True)

# This specifies the inputs the model will be expecting and the outputs it will produce
dataset_metadata = LeRobotDatasetMetadata(repo_id=dataset_id, root=dataset_root)
features = dataset_to_policy_features(dataset_metadata.features)

output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

if "observation.images.front_top_depth" in input_features:
    del input_features["observation.images.front_top_depth"]
#sys.exit()

if drop_pose_feature:
    if "observation.pos" in input_features:
        del input_features["observation.pos"]

print(f"input_features: {input_features}")

cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        noise_scheduler_type="DDIM",
        num_train_timesteps=100,
        num_inference_steps=10,
        prediction_type='sample', #'v_prediction',
        beta_schedule='squaredcos_cap_v2',
        optimizer_lr=5e-5,
        crop_shape=None,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        drop_n_last_frames=drop_n_last_frames,
        down_dims=(256, 512, 1024),
    )

policy = DiffusionPolicy(cfg)
preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)


policy.train()
policy.to(device)

#cfg.n_obs_steps #: int = 2
#cfg.horizon #: int = 16
#cfg.n_action_steps#: int = 8
#cfg.drop_n_last_frames#: int = 7  # horizon - n_action_steps - n_obs_steps + 1
#cfg.num_inference_steps = cfg.num_train_timesteps = 100


# To perform action chunking, ACT expects a given number of actions as targets
delta_timestamps = {
    "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
    "observation.pos": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
    "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
}

# add image features if they are present
delta_timestamps |= {
    k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps) for k in cfg.image_features
}
print(f"delta timestamps: {delta_timestamps}")
#sys.exit()
# Instantiate the dataset
dataset = LeRobotDataset(repo_id=dataset_id, root=dataset_root, delta_timestamps=delta_timestamps)

# Create the optimizer and dataloader for offline training
optimizer = cfg.get_optimizer_preset().build(policy.parameters())

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
    num_workers=num_workers,
)

# Number of training steps and logging frequency
print(f"config: {cfg}")
print(f"batch_size = {batch_size}, \n training_steps = {training_steps}")

# Run training loop
step = 0
done = False
while not done:
    for batch in dataloader:
        #print("batch raw")
        #print(batch)
        #print(batch["observation.images.front_top_rgb"].shape)
        #print(batch['observation.images.front_top_depth'].shape)
        #print(batch['observation.state'].shape)
        #print(batch['observation.pos'].shape)
        #print(batch['action'].shape)
        batch = preprocessor(batch)
        #print("batch preprocessor")
        #print(batch["observation.images.front_top_rgb"].shape)
        #print(batch['observation.images.front_top_depth'].shape)
        #print(batch['observation.state'].shape)
        #print(batch['observation.pos'].shape)
        #print(batch)
        #sys.exit()
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

print(f"step: {step} loss: {loss.item():.3f}") # 最后一步

# Save the policy checkpoint, alongside the pre/post processors
policy.save_pretrained(output_directory)
preprocessor.save_pretrained(output_directory)
postprocessor.save_pretrained(output_directory)

# Save all assets to the Hub
# policy.push_to_hub("fracapuano/robot_learning_tutorial_diffusion")
# preprocessor.push_to_hub("fracapuano/robot_learning_tutorial_diffusion")
# postprocessor.push_to_hub("fracapuano/robot_learning_tutorial_diffusion")
