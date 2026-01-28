# stanford 原版 Diffusion Policy 的应用

## 1. 对数据的处理

Diffusion Policy使用的是zarr格式的数据，这个可以写个脚本将LeRobotDataset v2.1格式的数据转换成zarr格式的数据

- franka采集的数据转到zarr格式：franka_parquet2zarr.py
- flasharm采集的数据转到zarr格式：lerobot2zarr.py （程序内未添加视频转移的函数，可手动转移或使用franka_parquet2zarr.py里的get_video函数）

## 2. 需要更改的代码和配置

### 2.1 config文件

在diffusion_policy/config/下的配置文件中配置所使用的模型

- real_franka_batch5_120episodes_train_diffusion_transformer_real_hybrid_workspace.yaml
- real_franka_batch5_train_diffusion_unet_real_hybrid_workspace.yaml
- real_image_right_diffusion_policy_cnn.yaml
- real_right_train_diffusion_unet_real_hybrid_workspace.yaml

在diffusion_policy/config/task下的配置文件中配置任务的参数

- real_franka_batch5_image.yaml
- real_right_image.yaml

### 2.2 模型文件

其他的文件，比如workspace文件，数据文件需要编写一下（yuque里有详细记载，基本更改现有程序即可）

## 2. 运行训练代码

- train_real_right.sh # flasharm数据集
- train_real_franka_batch5.sh # franka数据集
- train_real_franka_batch5_120.sh # franka数据集120条数据

若传入hydra.run.dir参数，则是resume训练模型，否则从头开始训练。

## 3. 运行推理代码

flasharm的server端推理代码的运行

- inference_real_right.sh # 脚本
- dp_flasharm_server.py # server 程序

franka的server端推理代码的运行

- inference_real_franka_batch5.sh
- dp_franka_server.py
- dp_franka_util.py # 数据结构，dp输入不仅包括当前的信息，还包括历史信息

franka的伪server端，用于测试

- inference_pseduo_franka.sh
- pseduo_franka.py

