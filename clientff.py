# task_ = "pick up the bottle and put it in to the box"  # 第一批数据对应的任务
# task_ = "pick up the Coca-Cola bottle and place it into the brown cardboard box"  # 第二批数据对应的任务
# task_ = "pick up the  Coca-Cola and place it into the white plastic box"  # 第二批数据对应的任务
# task_ = "pick up the cola and place it on the white tray"  # 第三批数据对应的任务
task_ = "insert the green cube into the corresponding hole of the shape sorter"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32MultiArray, Float64, Bool
from geometry_msgs.msg import Pose
import message_filters
import os
import queue
import threading
import numpy as np
import time
import cv2
import zmq
import base64
import json
from collections import deque
import argparse

print("库导入成功.")

arm_host_ = "192.168.84.77"
arm_port_ = 8005  # 8003-pi0, gr00t-8004, act-8005

queue_send_hz_ = 20  # 待发送区发送的hz

# 异步融合参数配置
agg_per = 0.5  # 聚合加权轨迹中，新轨迹的权重占比（0.0-1.0）
queue_less_pro_ = 1.0  # 异步推理：待发送区还剩下多少百分比数据开始触发新一轮的推理
action_chunk_size = 8

# for rtc
quene_less_num_ = 3  # RTC推理：待发送区还剩下多少百分比数据开始触发新一轮的推理

wrist_video = "/dev/video0"

vis_ = False

from datetime import datetime

VLA_steps = 0

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"VLA_vis/images_{current_time}"
if vis_:
    os.makedirs(folder_name, exist_ok=True)

infer_time = deque(maxlen=20)

pos_name_list = [
    "pos_xyz_x", "pos_xyz_y", "pos_xyz_z", "pose_quat_qx", "pose_quat_qy",
    "pose_quat_qz", "pose_quat_qw", "gripper_cmd"
]

# 腕部相机配置
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

CAMERA_DEVICE = wrist_video
config = OpenCVCameraConfig(
    index_or_path=CAMERA_DEVICE,
    color_mode=ColorMode.RGB,
    width=640,
    height=480,
    fps=30,
)
camera = OpenCVCamera(config)
try:
    camera.connect(warmup=True)
except Exception as e:
    print(f"相机连接失败: {e}")
    camera = None

if camera is not None:
    print(f"相机连接成功")
    print(f"分辨率: {camera.width}x{camera.height}")
    print(f"FPS: {camera.fps}")
    print(f"颜色模式: {camera.color_mode}")
else:
    print(f"相机连接失败,退出程序")


# 将每次触发推理输出的动作结果存成txt文件
def save_action_chunk(action_pre):
    global VLA_steps

    np.savetxt(f"{folder_name}/action_chunk{VLA_steps}.txt",
               action_pre,
               fmt='%.6f',
               delimiter='\t')


class SyncNode(Node):

    def __init__(self, args):
        global action_chunk_size

        super().__init__('pi')

        self.action_chunk_size = action_chunk_size
        self.cartesian = args.cartesian
        self.rtc = args.rtc

        if self.rtc:
            self.infer_start_num = quene_less_num_
            print(f"使用RTC推理模式,待发送区还剩下{quene_less_num_}个数据开始触发新一轮的推理")
        else:
            self.infer_start_num = int(action_chunk_size * queue_less_pro_)
            print(f"使用异步推理模式,待发送区还剩下{self.infer_start_num}个数据开始触发新一轮的推理")

        # 0. 创建客户端
        self.zmq_ctx = zmq.Context()
        self.socket = self.zmq_ctx.socket(zmq.REQ)
        self.socket.connect(f"tcp://{arm_host_}:{arm_port_}")  # 云端 ZMQ 服务地址
        self.infer_lock = threading.Lock()

        # 1. 创建待发送区
        self.send_queue = queue.Queue(maxsize=0)  # 无限大
        if self.cartesian:
            print("使用笛卡尔空间作为输出,发布话题: /cartesian_pose_cmd")
            self.arm_cmd_pub = self.create_publisher(Pose,
                                                     '/cartesian_pose_cmd', 10)
        else:
            print("使用关节空间作为输出,发布话题: /remote_controller/joint_angles")
            self.arm_cmd_pub = self.create_publisher(
                Float32MultiArray, '/remote_controller/joint_angles', 10)
        self.gripper_cmd_pub = self.create_publisher(Bool, '/gripper_cmd', 10)

        self.send_thread = threading.Thread(target=self._send_routine,
                                            daemon=True)
        self.send_thread.start()
        self.first_frame_ready = False  # 是否是第一次推理
        self.last_action8 = np.zeros(8)  # 缓存上一次发出的 36 维

        self.infer_start_last = 0  # 推理开始时动作队列剩余动作数量
        self.infer_end_last = 0  # 推理结束时动作队列的剩余动作数量

        # 2. 创建 Subscriber
        sub_img = message_filters.Subscriber(
            self, CompressedImage, '/camera/color/image_raw/compressed')
        sub_joint = message_filters.Subscriber(self, JointState,
                                               '/franka/joint_states')
        sub_pose = message_filters.Subscriber(self, Pose,
                                              '/franka/end_effector_pose')
        sub_grip = message_filters.Subscriber(self, Float64,
                                              '/franka/gripper_width')
        sub_depth = message_filters.Subscriber(
            self, CompressedImage, '/camera/depth/image_raw/compressedDepth')

        # 创建30Hz定时器
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        # 创建图像缓存
        self.camera_buffer = deque(maxlen=30)
        # 保证有数据进入
        time.sleep(2)

        self.wirist_img = None
        self.rgb_img = None

        # 3. 用 ApproximateTime 对齐图像和关节话题
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [sub_img, sub_joint, sub_pose, sub_grip, sub_depth],
            queue_size=1000,
            slop=0.1,
            allow_headerless=True)
        self.ats.registerCallback(self.callback)

    def timer_callback(self):
        # 获取腕部图像
        cv_wrist_cur = camera.read()
        # 缓存
        self.camera_buffer.append(cv_wrist_cur)
        if self.wirist_img is not None:
            cv2.imshow("rgb_img", cv2.cvtColor(self.rgb_img,
                                               cv2.COLOR_RGB2BGR))
            cv2.imshow("wrist_img",
                       cv2.cvtColor(self.wirist_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    # 异步发送
    def _send_routine(self):
        rate = self.create_rate(queue_send_hz_)
        while rclpy.ok():
            if not self.send_queue.empty():
                joint8 = self.send_queue.get().tolist()  # 左端 pop
                pose_msg = Pose()
                pose_msg.position.x = joint8[0]
                pose_msg.position.y = joint8[1]
                pose_msg.position.z = joint8[2]
                pose_msg.orientation.x = joint8[3]
                pose_msg.orientation.y = joint8[4]
                pose_msg.orientation.z = joint8[5]
                pose_msg.orientation.w = joint8[6]

                gripper_cmd_msg = Bool()
                # print(f"gripper value: {joint8[7]}")
                gripper_cmd_msg.data = joint8[7] > 0.7

                self.arm_cmd_pub.publish(pose_msg)  # 发送
                self.gripper_cmd_pub.publish(gripper_cmd_msg)  # 发送
                self.last_action8 = joint8.copy()  # 缓存

            rate.sleep()

    # 将图像和关节角度数据编码为 JSON 格式
    def encode_data(self, image, depth, joint_states, task, pos_list,
                    cv_wrist):
        # RGB-head
        _, image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        image_bytes = image.tobytes()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        # Depth
        _, depth = cv2.imencode('.jpg', depth, [cv2.IMWRITE_JPEG_QUALITY, 100])
        depth_bytes = depth.tobytes()
        depth_base64 = base64.b64encode(depth_bytes).decode('utf-8')
        # RGB-wrist
        _, wrist_image = cv2.imencode('.jpg', cv_wrist,
                                      [cv2.IMWRITE_JPEG_QUALITY, 100])
        wrist_image_bytes = wrist_image.tobytes()
        wrist_image_base64 = base64.b64encode(wrist_image_bytes).decode(
            'utf-8')
        # 6D-pose
        pose6D_base64 = base64.b64encode(
            np.array(pos_list).tobytes()).decode('utf-8')
        # Joints
        joint_states_base64 = base64.b64encode(
            joint_states.tobytes()).decode('utf-8')

        return json.dumps({
            'image': image_base64,
            'depth': depth_base64,
            'joint_states': joint_states_base64,
            'task': task,
            'end_pose': pose6D_base64,
            'wrist_image': wrist_image_base64
        })

    def _infer_and_refill(self, img_msg, joint_msg, pose_msg, griper_msg,
                          depth_msg):

        global VLA_steps
        global camera
        global infer_time

        try:
            start_ready = time.perf_counter()

            # 获取头部RGB
            cv_rgb = np.frombuffer(img_msg.data, np.uint8)
            cv_rgb = cv2.imdecode(cv_rgb, cv2.IMREAD_COLOR)  # BGR
            cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2RGB)  # 需要转换为RGB
            self.rgb_img = cv_rgb
            # 获取头部Depth
            cv_depth = np.frombuffer(depth_msg.data[12:], np.uint8)
            cv_depth = cv2.imdecode(cv_depth, cv2.IMREAD_ANYDEPTH)
            cv_depth = cv2.normalize(cv_depth,
                                     None,
                                     0,
                                     255,
                                     cv2.NORM_MINMAX,
                                     dtype=cv2.CV_8U)
            cv_depth = cv2.equalizeHist(cv_depth)
            cv_depth = cv2.cvtColor(cv_depth, cv2.COLOR_GRAY2BGR)
            # 获取腕部RGB
            if self.camera_buffer:
                cv_wrist = self.camera_buffer[-1]
                self.wirist_img = cv_wrist
            # 获取6D-pose
            pos_list = [
                pose_msg.position.x, pose_msg.position.y, pose_msg.position.z,
                pose_msg.orientation.x, pose_msg.orientation.y,
                pose_msg.orientation.z, pose_msg.orientation.w, griper_msg.data
            ]

            # 可视化
            if vis_:
                cv_rgb_save = np.copy(cv_rgb)
                cv_rgb_save = cv2.cvtColor(cv_rgb_save,
                                           cv2.COLOR_RGB2BGR)  # 需要转换为BGR
                cv2.imwrite(f"{folder_name}/cv_rgb_{VLA_steps}.jpg",
                            cv_rgb_save)
                cv_depth_save = np.copy(cv_depth)
                cv2.imwrite(f"{folder_name}/cv_depth_{VLA_steps}.jpg",
                            cv_depth_save)
                cv_wrist_save = np.copy(cv_wrist)
                cv_wrist_save = cv2.cvtColor(cv_wrist_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{folder_name}/cv_wrist_save_{VLA_steps}.jpg",
                            cv_wrist_save)

            arm8 = np.array([
                joint_msg.position[0], joint_msg.position[1],
                joint_msg.position[2], joint_msg.position[3],
                joint_msg.position[4], joint_msg.position[5],
                joint_msg.position[6]
            ])

            # 组织总观测
            data = self.encode_data(cv_rgb, cv_depth, arm8, task_, pos_list,
                                    cv_wrist)

            elapsed_ready = time.perf_counter() - start_ready
            print("elapsed_ready:", elapsed_ready * 1000)

            # 模型推理（发送 + 推理 + 接收）
            start = time.perf_counter()

            self.socket.send_string(data)
            elapsed_1 = time.perf_counter() - start
            print("elapsed_1:", elapsed_1 * 1000)

            actions_base64 = self.socket.recv_string()
            elapsed_2 = time.perf_counter() - start
            print("elapsed_2:", elapsed_2 * 1000)

            actions_bytes = base64.b64decode(actions_base64)
            elapsed_3 = time.perf_counter() - start
            print("elapsed_3:", elapsed_3 * 1000)

            actions8 = np.frombuffer(actions_bytes, dtype=np.float64).reshape(
                (-1, 8))
            elapsed_4 = time.perf_counter() - start
            print("elapsed_4:", elapsed_4 * 1000)

            if self.rtc:
                actions8_copy = actions8.copy()
            else:
                actions8_copy = actions8.copy()[:self.action_chunk_size]

            elapsed = time.perf_counter() - start

            if vis_:
                # 保存推理的action_chunk
                save_action_chunk(actions8)
                # 触发记录:执行开始
                self.record_pos_txt_flag = True  # 开始记录机械臂末端pos
                # 创建pos记录txt文件
                with open(f"{folder_name}/6D_pose{VLA_steps}.txt",
                          'w',
                          encoding='utf-8') as f:
                    pass

            # 非首次推理
            if self.first_frame_ready and not self.rtc:

                buffer = []  # 一次性全部取出
                try:
                    while True:
                        buffer.append(self.send_queue.get_nowait())
                except queue.Empty:
                    pass
                if not buffer:
                    buffer = np.empty((0, 8))
                else:
                    buffer = np.stack(buffer)

                # 丢弃流逝过的动作
                self.infer_end_last = buffer.shape[0]
                drop_num = self.infer_start_last - self.infer_end_last  # 计算丢弃数量
                actions8_copy = actions8_copy[drop_num:]
                print("推理期间已执行的动作步长：", drop_num)

                # 加权聚合动作
                actions8_agg = actions8_copy[:self.infer_end_last]
                actions8_agg = buffer * (1.0 -
                                         agg_per) + actions8_agg * agg_per
                print("需要聚合的步长：", self.infer_end_last)

                # 非加权聚合动作
                actions8_new = actions8_copy[self.infer_end_last:]
                print("新动作步长：", actions8_new.shape[0])

                # 将2个动作数组加入待发送区
                for i in range(actions8_agg.shape[0]):
                    self.send_queue.put(actions8_agg[i])  # 单步 36 维
                for i in range(actions8_new.shape[0]):
                    self.send_queue.put(actions8_new[i])  # 单步 36 维
            # 首次推理
            else:
                # 将输出加入待发送区
                for i in range(actions8_copy.shape[0]):
                    self.send_queue.put(actions8_copy[i])  # 单步 36 维

            # 统计耗时
            infer_time.append(elapsed * 1000)
            print(f'平均推理耗时 {np.mean(list(infer_time)):.2f} ms')

        finally:
            if self.first_frame_ready:
                self.infer_lock.release()

    def callback(self, img_msg: CompressedImage, joint_msg: JointState,
                 pose_msg: Pose, grip_msg: Float64,
                 depth_msg: CompressedImage):

        global VLA_steps

        # 1. 第一帧必推理
        if not self.first_frame_ready:
            VLA_steps += 1
            self._infer_and_refill(img_msg, joint_msg, pose_msg, grip_msg,
                                   depth_msg)
            self.first_frame_ready = True
            return

        # 2. 只要队列 ≤ 一定数量 就异步再推理（加锁保证不会重复调用推理）
        if self.send_queue.qsize(
        ) <= self.infer_start_num and self.infer_lock.acquire(False):
            VLA_steps += 1
            self.infer_start_last = self.send_queue.qsize()  # 记录触发推理时的剩余动作数量
            print(f"\n\n触发推理!当前待发送区数量={self.send_queue.qsize()}")
            # 开线程做，不阻塞 10 Hz 发送
            infer_th = threading.Thread(target=self._infer_and_refill,
                                        args=(img_msg, joint_msg, pose_msg,
                                              grip_msg, depth_msg),
                                        daemon=True)
            infer_th.start()


def main(args):
    rclpy.init()
    node = SyncNode(args)
    while True:
        key = input('====== 请按回车开始 ======\n')
        if key == '':
            print('程序开始!')
            break
        else:
            print('按下的不是回车，请按回车开始程序.')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Keyboard interrupt -> exit now', flush=True)
        os._exit(0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='接收运行标志位')
    parser.add_argument('-c',
                        '--cartesian',
                        action='store_true',
                        help='是否使用笛卡尔空间作为输出')
    parser.add_argument('-r', '--rtc', action='store_true', help='是否使用RTC推理模式')
    args = parser.parse_args()
    main(args)
