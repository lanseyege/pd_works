import os, sys

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import time
import numpy as np
import zmq
import json
import base64
import numpy as np
import cv2
import time

import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action, prepare_observation_for_inference

device = torch.device("cuda") # or "mps" or "cpu"

# 根据服务器端的host和port设置
model_host_ = "localhost"  # 要和服务端对的上
model_port_ = 5555  # 要和服务端对的上
arm_host = "192.168.84.77"
arm_port = 8005

#model_file = "act_2025-12-29_08-56-49"
#model_path = "/home/yuanye/work/test/outputs/franka_batch5/" 
model_file = "pretrained_model"
model_path = "./outputs/franka_batch5_120episodes/act_2026-01-05_15-05-01/checkpoints/last/" # 60k time steps
model_path = "./outputs/franka_batch5_120episodes/act_2026-01-06_15-03-03/checkpoints/last/" # 120k time steps

#dataset_id = "franka_batch5_wrist_grip"
dataset_id = "franka_batch5_120episodes"
#dataset_root = "/home/yuanye/dataset/franka_batch5_wrist_grip"
dataset_root = "/home/yuanye/dataset/franka_batch5_120episodes"

def load_checkpoints(model_file, model_path, dataset_id, dataset_root):
    model_file = model_file
    print(f"model file: {model_file}")
    model_id = model_path + model_file 

    model = ACTPolicy.from_pretrained(model_id)
    model.config.n_time_steps = model.config.chunk_size

    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(repo_id = dataset_id, root = dataset_root)
    preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

    return model, preprocess, postprocess

model, preprocess, postprocess = load_checkpoints(model_file, model_path, dataset_id, dataset_root)

# 云端接收数据并处理
def process_data(data):
    print("云端接收到数据.")
    data_dict = json.loads(data)

    image_base64 = data_dict['image']
    image_bytes = base64.b64decode(image_base64)
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    depth_base64 = data_dict['depth']
    depth_bytes = base64.b64decode(depth_base64)
    depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_COLOR)

    wrist_image_base64 = data_dict['wrist_image']
    wrist_image_bytes = base64.b64decode(wrist_image_base64)
    wrist_image = cv2.imdecode(np.frombuffer(wrist_image_bytes, np.uint8), cv2.IMREAD_COLOR)

    joint_states_base64 = data_dict['joint_states']
    joint_states_bytes = base64.b64decode(joint_states_base64)
    joint_states = np.frombuffer(joint_states_bytes, dtype=np.float64).astype(np.float32, copy=False)

    end_pose_base64 = data_dict["end_pose"]
    end_pose_bytes = base64.b64decode(end_pose_base64)
    end_pose = np.frombuffer(end_pose_bytes, dtype=np.float64).astype(np.float32, copy=False)

    task = data_dict['task']

    # 组织观测
    '''
    obs = {
        "observation.images.head_rgb": np.expand_dims(image, axis=0),
        "observation.images.head_depth": np.expand_dims(depth, axis=0),
        "observation.images.wrist_rgb": np.expand_dims(wrist_image, axis=0),
        "observation.state": np.expand_dims(joint_states, axis=0),
        "observation.pose": np.expand_dims(end_pose, axis=0),
        #"annotation.human.action.task_description": [task],
    }'''
    obs = {
        "observation.images.head_rgb": image,
        "observation.images.head_depth": depth,
        "observation.images.wrist_rgb": wrist_image,
        "observation.state": joint_states,
        "observation.pose": end_pose,
        #"annotation.human.action.task_description": [task],
    }

    print(f"images head_rgb shape: {obs['observation.images.head_rgb'].shape}")
    print(f"images head_depth shape: {obs['observation.images.head_depth'].shape}")
    print(f"images wrist_rgb shape: {obs['observation.images.wrist_rgb'].shape}")
    obs_frame = prepare_observation_for_inference(obs, device)
    obs = preprocess(obs_frame)
    # 调用 VLA 模型进行计算
    #actions = client.get_action(obs)["action.actions"]  # VLA 模型预测函数
    if model.config.temporal_ensemble_coeff is not None:
        actions.model.select_actions(obs)
    else:
        actions = model.predict_action_chunk(obs)
    actions = postprocess(actions).squeeze(0).to("cpu").numpy()
    # 将动作数据编码为 JSON 格式
    actions_bytes = actions.astype(np.float64).tobytes()  # 转为float64
    actions_base64 = base64.b64encode(actions_bytes).decode('utf-8')
    return actions_base64


# 云端 ZMQ 服务
def cloud_server():
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{arm_port}")  # 绑定云端 ZMQ 服务地址
        print("云端服务端建立成功.")
        while True:
            print("等待arm数据中...")
            start_recv = time.monotonic()
            data = socket.recv_string()
            end_recv = time.monotonic() - start_recv
            print(f"receive data cost: {int(1000*end_recv)} ms")
            
            start_infer = time.monotonic()
            actions_base64 = process_data(data)
            elapsed = time.monotonic() - start_infer
            print(f"inference time: {int(elapsed * 1000)} ms")
            socket.send_string(actions_base64)

    except KeyboardInterrupt:  # 捕获 Ctrl+C
        print("\n检测到退出信号，正在关闭端口...")
    except Exception as e:  # 捕获其他异常
        print(f"云端服务端建立失败! 错误信息: {e}")
    finally:
        # 确保资源被释放
        socket.close()
        context.term()
        print("所有端口已安全关闭。")


# 主函数
if __name__ == "__main__":
    cloud_server()


