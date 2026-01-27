"""This script demonstrates how to train ACT Policy on a real-world dataset."""

import argparse
import sys, os
import numpy as np
from pathlib import Path
from datetime import datetime

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

#torch.set_printoptions(threshold=float('inf')) # print all tensor content out
#np.set_printoptions(threshold=np.inf)

# Number of training steps and logging frequency
training_steps = 60000
log_freq = 100
batch_size = 32
num_workers = 2

chunk_size = 36
n_action_steps = 36

drop_pose_feature = False
drop_left = True

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"outputs/batch3_right/act_{current_time}_chunk_size_{chunk_size}_left{drop_left}"


print(f"batch_size: {batch_size}")
print(f"training_steps: {training_steps}")

# Select your device
device = torch.device("cuda")  # or "cuda" or "cpu"

dataset_id = "batch3_right"
dataset_root = "/home/yuanye/dataset/batch3_right"

'''
parser = argparse.ArgumentParser(description='act model training code')
parser.add_argument("--dataset_id", type=str, help="intput dataset id")
parser.add_argument("--dataset_root", type=str, help="input dataset root")
parser.add_argument("--device", type=str, help="input device",default="cuda")
parser.add_argument("")
'''

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]

#output_directory = Path("outputs/robot_learning_tutorial/act")
#output_directory = Path("outputs/batch3_right/act_2")
output_directory = Path(folder_name)
output_directory.mkdir(parents=True, exist_ok=True)


# This specifies the inputs the model will be expecting and the outputs it will produce
dataset_metadata = LeRobotDatasetMetadata(repo_id=dataset_id, root=dataset_root)
features = dataset_to_policy_features(dataset_metadata.features)
print()
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

print(f"output_features: {output_features}")
print(f"input_features: {input_features}")
#sys.exit()

if drop_pose_feature:
    if "observation.pos" in input_features:
        del input_features["observation.pos"]

cfg = ACTConfig(
        input_features=input_features, 
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
    )

policy = ACTPolicy(cfg)
preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

print(f"cfg cofig: {cfg}")

policy.train()
policy.to(device)

print(f"act policy trainable params: {policy.get_optim_params_name()}")
#sys.exit()

# To perform action chunking, ACT expects a given number of actions as targets
delta_timestamps = {
    "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
}

# add image features if they are present
delta_timestamps |= {
    k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps) for k in cfg.image_features
}

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


print(f"model save at: {folder_name}")

# Run training loop
step = 0
done = False

mask = torch.ones(38)#.to(device)
mask[:7] = 0.0

mask_pos = torch.ones(14)#.to(device)
mask_pos[:7] = 0.0

while not done:
    for batch in dataloader:
        #print("before preprocess batch")
        #print(batch['observation.images.front_top_rgb'].shape)
        #print(batch['observation.images.front_top_depth'].shape)
        #print(batch['observation.images.front_top_depth'][2])
        #print(batch['observation.state'][2])
        #print(batch['observation.state'].shape)
        if drop_left:
            batch['observation.state'] = batch['observation.state'] * mask
            batch['action'] = batch['action']  * mask
        if not drop_pose_feature and drop_left:
            batch['observation.pos'] = batch['observation.pos'] * mask_pos
        sys.exit()
        batch = preprocessor(batch)
        #print("after preprocess batch")
        #print(batch)
        #print(batch['observation.images.front_top_rgb'][2])
        #print(batch['observation.images.front_top_depth'][2])
        #print(batch['observation.state'][2])
        #if step > 2:
        #    sys.exit()
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #sys.exit()
        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

# Save the policy checkpoint, alongside the pre/post processors
policy.save_pretrained(output_directory)
preprocessor.save_pretrained(output_directory)
postprocessor.save_pretrained(output_directory)

# Save all assets to the Hub
#policy.push_to_hub("fracapuano/robot_learning_tutorial_act")
#preprocessor.push_to_hub("fracapuano/robot_learning_tutorial_act")
#postprocessor.push_to_hub("fracapuano/robot_learning_tutorial_act")
