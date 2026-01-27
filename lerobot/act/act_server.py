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
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action, prepare_observation_for_inference


device = torch.device("cuda") # or "mps" or "cpu"
#torch.set_printoptions(threshold=float('inf')) # print all tensor content out
#np.set_printoptions(threshold=np.inf)

HOST = "0.0.0.0"
PORT = 8688
ACTION_DIM = 38

drop_pose_feature = False
drop_left = True

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
figures_folder_name = f"/home/yuanye/work/test/res/act_figures/images_{current_time}"
os.makedirs(figures_folder_name, exist_ok=True)

if drop_left and not drop_pose_feature:
    model_file = "act_2025-12-13_12-30-09_chunk_size_50_noleft" # no left, 60k
    model_file = "act_2025-12-13_12-30-50_chunk_size_25_noleft" #
    #model_file = "act_2025-12-13_12-31-24_chunk_size_10_noleft" #
    #model_file = "act_2025-12-14_15-24-16_chunk_size_100_noleft" # chunk100,60k,noleft
    model_file = "act_2025-12-15_13-38-53_chunk_size_36_leftTrue" # chunk36,60k,noleft
elif not drop_left and not drop_pose_feature:
    model_file = "act_2025-12-08_10-23-44_chunk_size_50" # 60k timesteps
    #model_file = "act_2025-12-11_11-36-51_chunk_size_25" # 60k timesteps
    #model_file = "act_2025-12-14_15-25-23_chunk_size_100_hasleft" # chunk100,60k,hasleft
    #model_file = "act_2025-12-04_13-39-58_chunk_size_50" # chunk size = 50
    #model_file = "act_2025-12-04_13-39-42_chunk_size_25" # chunk size = 25
    #model_file = "act_2025-12-04_13-37-42_chunk_size_10" # chunk size = 10
    model_file = "act_2025-12-15_13-38-15_chunk_size_36_leftFalse"  # chunksize36,steps60k,hasleft
   
elif not drop_left and drop_pose_feature:
    model_file = "act" #chunk_size=100
    #model_file = "act_2025-11-20_13-29-58/" # chunk_size=100
    model_file = "act_2025-11-23_04-27-51/" # chunk_size=50
    model_file = "act_2025-11-23_04-26-14/" # chunk_size=25
    #model_file = "act_2025-11-26_00-52-14/" # chunk size=10
    model_file = "act_2025-12-08_10-23-44_chunk_size_50" # no pose with 60k training steps

print(f"drop pose feature: {drop_pose_feature}, drop_left: {drop_left}")
print(f"model file: {model_file}")

model_id = "/home/yuanye/work/test/outputs/batch3_right/" + model_file 
model = ACTPolicy.from_pretrained(model_id)
model.config.n_time_steps = model.config.chunk_size

if drop_pose_feature:
    if 'observation.pos' in model.config.input_features:
        del model.config.input_features['observation.pos']

dataset_id = "batch3_right"
dataset_root = "/home/yuanye/dataset/batch3_right"

# This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
dataset_metadata = LeRobotDatasetMetadata(repo_id = dataset_id, root = dataset_root)
preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

#mask = np.ones(38)
#mask[:7] = 0.0
#mask_pose = np.ones(14)
#mask_pose[:7] = 0.0

def main(elapseds):
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
            #print("recv string ...")
            message_json = socket.recv_string()
            start_time = time.perf_counter()

            # 3. 解析 JSON 数据
            #print("json load ..")
            req_data = json.loads(message_json)
            # 3.1. task
            task_text = req_data['task']
            #print(f"task_text: {task_text}")
            
            # 3.2 rgb image
            img_b64 = req_data['image']
            #image_dtype = req_data['image_dtype']
            #image_shape = req_data['image_shape']
            
            # 3.3 depth image
            depth_b64 = req_data['depth']
            #depth_dtype = req_data['depth_dtype']
            #depth_shape = req_data['depth_shape']

            # 3.4 joint states
            state_b64 = req_data['joint_states']

            # 3.5 pose 
            pose_b64 = None
            if 'pose' in req_data:
                pose_b64 = req_data['pose']

            # 4. RGB
            #print("decode rgb .. ")
            img_bytes = base64.b64decode(img_b64)
            #np_arr = np.frombuffer(img_bytes, dtype=image_dtype).reshape(image_shape)
            np_arr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            #print("img from buffer: ")
            #print(np_arr)

            frame['observation.images.front_top_rgb'] = np_arr 
            cv2.imwrite(f"{figures_folder_name}/cv_rgb_{step}.jpg", np_arr)

            # 5. 深度图
            #print("decode depth ...")
            depth_bytes = base64.b64decode(depth_b64)
            #np_depth_arr = np.frombuffer(depth_bytes, dtype=depth_dtype).reshape(depth_shape)
            np_depth_arr = cv2.imdecode(np.frombuffer(depth_bytes, dtype=np.uint8),cv2.IMREAD_COLOR)

            frame['observation.images.front_top_depth'] = np_depth_arr
            cv2.imwrite(f"{figures_folder_name}/cv_depth_{step}.jpg", np_depth_arr)

            # 6. decode joint state
            print("decode joint state")
            current_state = None
            if state_b64:
                state_bytes = base64.b64decode(state_b64)
                current_state = np.frombuffer(state_bytes, dtype=np.float64).astype(np.float32, copy=False)
                frame['observation.state'] = current_state #* mask
                print("current_state: ")
                #print(current_state)
                #print(current_state*mask)
                
            # 7. decode pose
            if pose_b64:
                pose_bytes = base64.b64decode(pose_b64)
                current_pose = np.frombuffer(pose_bytes, dtype=np.float64).astype(np.float32, copy=False).copy()
                frame['observation.pos'] = current_pose #* mask_pose
                #print("observation pose")
                #print(current_pose)
                #print(current_pose * mask_pose)
                #frame['observation.pos'] = np.expand_dims(current_pose, axis=0)
                #print(f"current pose: {current_pose}")

            step += 1
        except Exception as err:
            print(f"error in server: {err}")
            try:
                socket.send_string("")
                #sys.exit()
            except:
                pass

        # act model inference 
        print("inference ... ")
        #print("frame before prepare ...")
        #print(frame)
        obs_frame = prepare_observation_for_inference(frame, device)
        #print("frame after prepare ...")
        #print(obs_frame)
        
        obs = preprocess(obs_frame)
        #print("obs after preprocess ")
        #print(obs)

        start = time.perf_counter()
        #action = model.select_action(obs)
        action = model.predict_action_chunk(obs)
        elapsed = time.perf_counter() - start

        action = postprocess(action)
        
        action_tensor = action.squeeze(0)
        action_tensor = action_tensor.to("cpu").numpy()
        #print(f"action_tensor.dtype: {action_tensor.dtype}")
        #print(f"action tensor shape: {action_tensor.shape}")
        #print(f"action tensor: {action_tensor}")
        
        #sys.exit()

        actions_b64 = base64.b64encode(action_tensor.tobytes()).decode('utf-8')

        try:
            socket.send_string(actions_b64) 
            end_time = time.perf_counter() - start_time
        except Except as e:
            print(f"error {e}")

        elapseds.append((elapsed, end_time))

if __name__ == "__main__":
    elapseds = []
    try:
        main(elapseds)
    except KeyboardInterrupt:
        fw = open(f"res/inference_time_{current_time}.txt", "w")
        for i, elapsed in enumerate(elapseds):
            line = f"step: {i}, inference: {elapsed[0]*1000:.2f} ms, all: {elapsed[1]*1000:.2f} ms" + "\n" 
            fw.write(line)
        fw.close()


