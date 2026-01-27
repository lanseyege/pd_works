DATASET_ID="codebase_v30_franka_batch5_wrist_grip_v2"
DATASET_ROOT="/mnt/nfs_a/cvat-data/zhw/${DATASET_ID}"
OUTPUT_PATH="./outputs/${DATASET_ID}"
#CHECKPOINT_PATH="./outputs/${DATASET_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=7 python -u act_franka_training.py \
    --dataset_id="${DATASET_ID}" \
    --dataset_root="${DATASET_ROOT}" \
    --output_path="${OUTPUT_PATH}" \
    --device="cuda:0" \
    --training_steps=60000 \
    --save_checkpoints_freq=20000 \
    --chunk_size=32 \
    --n_action_steps=32 \
    --batch_size=32 \
    --use_layer_n=4 \
    --dim_model=512 \
    --vision_backbone="resnet34" \
    --pretrained_backbone_weights="ResNet34_Weights.IMAGENET1K_V1" \
    #--temporal_ensemble_coeff=0.01 \
    #--checkpoint_path="./outputs/codebase_v30_franka_batch5_wrist_grip_v2/act_2026-01-09_16-35-36/checkpoints/last" \
    #--resume
