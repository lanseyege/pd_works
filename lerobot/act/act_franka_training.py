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
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.train_utils import save_checkpoint, get_step_checkpoint_dir, update_last_checkpoint, load_training_state

#torch.set_printoptions(threshold=float('inf')) # print all tensor content out
#np.set_printoptions(threshold=np.inf)

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]

def main(args):
    device = torch.device(args.device)
    resume = args.resume
    temporal_ensemble_coeff = args.temporal_ensemble_coeff

    log_freq = args.log_freq
    training_steps = args.training_steps
    save_checkpoints_freq = args.save_checkpoints_freq
    use_layer_n = args.use_layer_n
    dim_model = args.dim_model

    vision_backbone=args.vision_backbone
    pretrained_backbone_weights=args.pretrained_backbone_weights

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = Path(args.output_path) / f"act_{current_time}"
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"model save at: {output_directory}")
    
    dataset_metadata = LeRobotDatasetMetadata(repo_id=args.dataset_id, root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    
    print(f"features: {features}")

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(f"output_features: {output_features}")
    print(f"input_features: {input_features}")
    #sys.exit()

    if args.drop_pose_feature:
        if "observation.pose" in input_features:
            del input_features["observation.pose"]
    if not resume:
        cfg = ACTConfig(
            input_features=input_features, 
            output_features=output_features,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            #temporal_ensemble_coeff=temporal_ensemble_coeff,
            use_layer_n=use_layer_n,
            dim_model=dim_model,
            vision_backbone=vision_backbone,
            pretrained_backbone_weights=pretrained_backbone_weights,
        )

        policy = ACTPolicy(cfg)
    else:
        model_id = Path(args.checkpoint_path) / "pretrained_model"
        policy = ACTPolicy.from_pretrained(model_id)
        cfg = policy.config
    preprocessor, postprocessor = make_pre_post_processors(
            cfg, 
            dataset_stats=dataset_metadata.stats
    )
    print(f"cfg cofig: {cfg}")

    policy.train()
    policy.to(device)

    #print(f"act policy trainable params: {policy.get_optim_params_name()}")
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
    dataset = LeRobotDataset(repo_id=args.dataset_id, root=args.dataset_root, delta_timestamps=delta_timestamps)

    # Create the optimizer and dataloader for offline training
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    lr_scheduler = None
    #optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=args.num_workers,
    )

    # Run training loop
    step = 0
    done = False
    print(f"resume: {args.resume}")
    if resume:
        step, optimizer, lr_scheduler = load_training_state(Path(args.checkpoint_path), optimizer, lr_scheduler)

    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            #print("after preprocess batch")
            #print(batch)
            #if step > 2:
            #    sys.exit()
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            if step % save_checkpoints_freq == 0:
                checkpoint_dir = get_step_checkpoint_dir(output_directory, training_steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
            if step >= training_steps:
                done = True
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='act model training code')

    parser.add_argument("--dataset_id", type=str, help="intput dataset id")
    parser.add_argument("--dataset_root", type=str, help="input dataset root")
    parser.add_argument("--device", type=str, help="input device",default="cuda")
    parser.add_argument("--chunk_size", type=int, help="action chunk size",default=100)
    parser.add_argument("--n_action_steps",type=int,help="action steps",default=100)
    parser.add_argument("--training_steps",type=int,help="training steps",default=30000)
    parser.add_argument("--log_freq",type=int,help="log freq",default=100)
    parser.add_argument("--batch_size",type=int,help="batch size",default=32)
    parser.add_argument("--num_workers",type=int,help="num workers",default=4)
    parser.add_argument("--drop_pose_feature",type=bool,help="drop pose or not",default=False)
    parser.add_argument("--output_path",type=str,help="output path",default="./outputs/franka_batch5/")
    parser.add_argument("--resume",action="store_true",help="resume last checkpoint or not",default=False)
    parser.add_argument("--checkpoint_path",type=str,help="checkpoint path",default="./outputs/franka_batch5/act_2025-12-29_08-56-49")
    parser.add_argument("--save_checkpoints_freq",type=int,help="save checkpoint freq",default=20000)

    parser.add_argument("--temporal_ensemble_coeff", type=float, help="action chunk size",default=None) #0.01
    
    # use other version resnet or ..
    parser.add_argument("--use_layer_n",type=int,help="resnet layer n's feature",default=4)
    parser.add_argument("--dim_model",type=int,help="resnet layer n's feature dim",default=512)
    parser.add_argument("--vision_backbone",type=str,help="which version resnet",default="resnet18")
    parser.add_argument("--pretrained_backbone_weights",type=str,help="resnet weight name",default="ResNet18_Weights.IMAGENET1K_V1")

    args = parser.parse_args()

    main(args)


