# 基于LeRobot架构的ACT模型

## act

获取的数据为lerobotdataset2.1格式的，需要将其转换到3.0格式的

- convert_data_version.bash

在FlashARM采集的数据上训练ACT，脚本文件为

- act_training_example.py
- train_act.bash

相应的部署文件为：
- act_server.py
- inference_act.bash

在Franka采集的数据上训练ACT，脚本文件为（在Franka上采集的数据的字段与lerobot框架默认的不同，在相应的yuque文档中有说明）
- act_franka_training.py
- train_franka_act.bash

相应的部署文件为：
- act_franka_server.py
- inference_franka_act.bash

作图程序，即在训练集上的动作loss图程序位于 ./act/pictures 下

## diffusion
基于lerobot框架的diffusion policy 模型，但是效果不好。

## lerobot-main
（lerobot的官方版本变动很大，此次安装的是25年11月的版本。）

1. 为了将pose信息加入训练特征，源代码处有几处相应的更改（更改处基本使用add by ye 20251212 字样标注）

在 src/lerobot/utils/constants.py 28行

- OBS_POSE = OBS_STR + ".pose" # change needed

在 src/lerobot/configs/policies.py

- @property
  def robot_pose_feature(self) -> PolicyFeature | None: # add by ye 20251203
      for ft_name, ft in self.input_features.items():
          if ft.type is FeatureType.STATE and ft_name == OBS_POSE :
              return ft
      return None

在src/lerobot/policies/act/modeling_act.py添加相应的源代码，类似对state的处理。

