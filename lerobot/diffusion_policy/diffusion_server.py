import zmq
import json
import base64
import numpy as np
import cv2
import time
import sys, os

from datetime import datetime

import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action, prepare_observation_for_inference

HOST = "0.0.0.0"
PORT = 8688
ACTION_DIM = 38

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 100

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
figures_folder_name = f"/home/yuanye/work/diffusions/res/act_figures/images_{current_time}"
os.makedirs(figures_folder_name, exist_ok=True)


device = torch.device("cuda") # or "mps" or "cpu"

drop_pose_feature = False

if not drop_pose_feature:
    model_file = "diffusion_2025-12-05_13-53-42_pose"
    #model_file = "diffusion_2025-12-08_12-06-14_pose" # ddim, default params
    model_file = "diffusion_2025-12-08_21-16-08_pose" # ddim, lr=1e-5,predtype=sample
    model_file = "diffusion_2025-12-11_18-38-35_pose"
    model_file = "diffusion_2025-12-15_21-01-54_pose" #horizon24,actionsteps12,obssteps4
    #model_file = "diffusion_2025-12-18_13-58-31_pose"
else:
    model_file = "diffusion_2025-11-26_00-51-36"

model_id = "/home/yuanye/work/diffusions/outputs/batch3_right/" + model_file
model = DiffusionPolicy.from_pretrained(model_id)
#model.config.n_time_steps = model.config.

model.num_inference_steps = 10
model.drop_n_last_frames = 2
model.vision_backbone = "resnet34"

dataset_id = "batch3_right"
dataset_root = "/home/yuanye/dataset/batch3_right"

# This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
dataset_metadata = LeRobotDatasetMetadata(repo_id = dataset_id, root = dataset_root)
preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

def main():
    # 1. initialize zeromq context
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{HOST}:{PORT}")

    print(f"act inference server has started, listen on: tcp://{HOST}:{PORT}")
    print(f"wait for client's connection ... ")

    #socket = socket_connect()
    step = 0
    while True:
        frame = {}
        try:
            # 2. accept request (阻塞等待)
            # 对应客户端：self.socket.send_string(data)
            print("recv string ...")
            message_json = socket.recv_string()
            start_time = time.perf_counter()

            # 3. 解析 JSON 数据
            print("json load ..")
            req_data = json.loads(message_json)

            # 3.0 task
            task_text = req_data['task']
            print(f"task_text: {task_text}")

            # 3.1 rgb image
            img_b64 = req_data['image']
            #image_dtype = req_data['image_dtype']
            #image_shape = req_data['image_shape']

            # 3.2 depth image
            #depth_b64 = req_data['depth']
            #depth_dtype = req_data['depth_dtype']
            #depth_shape = req_data['depth_shape']

            # 3.3 joint states
            state_b64 = req_data['joint_states']

            # 3.4 pose
            pose_b64 = None
            if 'pose' in req_data:
                pose_b64 = req_data['pose']

            # 4. 解码图像（Base64 -> Bytes -> NumPy Image）
            print("decode rgb .. ")
            if img_b64:
                img_bytes = base64.b64decode(img_b64)
                #np_arr = np.frombuffer(img_bytes, dtype=image_dtype).reshape(image_shape)
                np_arr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                print(f"np_arr shape: {np_arr.shape}")

                frame['observation.images.front_top_rgb'] = np_arr.copy() #np.expand_dims(np_arr, axis=0)
                cv2.imwrite(f"{figures_folder_name}/cv_rgb_{step}.jpg", np_arr)
            # 5. 深度图
            '''
            print("decode depth ...")
            if depth_b64:
                depth_bytes = base64.b64decode(depth_b64)
                #np_depth_arr = np.frombuffer(depth_bytes, dtype=depth_dtype).reshape(depth_shape)
                np_depth_arr = cv2.imdecode(np.frombuffer(depth_bytes, dtype=np.uint8),cv2.IMREAD_COLOR)
                print(f"np_depth_arr shape: {np_depth_arr.shape}")

                #frame['observation.images.front_top_depth'] =np.expand_dims( np_depth_arr, axis=0)
                #frame['observation.images.front_top_depth'] =np_depth_arr.copy()
                #cv2.imwrite(f"{figures_folder_name}/cv_depth_{step}.jpg", np_depth_arr)
            '''

            # 6. decode joint state
            print("decode joint state")
            current_state = None
            if state_b64:
                state_bytes = base64.b64decode(state_b64)
                current_state = np.frombuffer(state_bytes, dtype=np.float64).astype(np.float32, copy=False)
                frame['observation.state'] = current_state.copy() #np.expand_dims(current_state, axis=0)
                print(f"current_state shape: {current_state.shape}")

            # 7. decode pose
            if pose_b64:
                pose_bytes = base64.b64decode(pose_b64)
                current_pose = np.frombuffer(pose_bytes, dtype=np.float64).astype(np.float32, copy=False)
                frame['observation.pos'] = current_pose.copy()
                print(f"current pose shape: {current_pose.shape}")

        except Exception as err:
            print(f"error in server: {err}")
            try:
                socket.send_string("")
                sys.exit()
            except:
                pass

        step += 1

        # diffusion policy model inference
        print("inference ... ")
        #print(frame)
        obs_frame = prepare_observation_for_inference(frame, device)
        #print("after prepare ... ")
        #print(obs_frame)
        obs = preprocess(obs_frame)
        #print("preprocess ... ")
        #print(obs)

        start = time.perf_counter()
        action = model.select_action(obs)
        #action = model.predict_action_chunk(obs)
        elapsed = time.perf_counter() - start
        print(f"inference cost time: {elapsed*1000:.2f} ms")
        action = postprocess(action)

        action_tensor = action.squeeze(0)
        action_tensor = action_tensor.to("cpu").numpy()
        #print(f"action_tensor.dtype: {action_tensor.dtype}")
        #print(f"action tensor shape: {action_tensor.shape}")
        #print(f"action tensor: {action_tensor}")
        actions_bytes = action_tensor.tobytes()

        actions_b64 = base64.b64encode(actions_bytes).decode('utf-8')

        try:
            socket.send_string(actions_b64)
        except Except as e:
            print(f"error {e}")


if __name__ == "__main__":
    main()


