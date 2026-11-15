# Go2 专用实现

- `robot.py`：Go2 电机与初始姿态；`model/`：原始 MJCF、网格与模型许可。
- `env.py`、`command.py`、`mdp.py`、`terrains.py`：三个追逐/躲避环境及地形配置。
- `target/`：追逐目标、动态障碍物、碰撞和地形采样。共享训练与 X30 世界模型也复用其中的几何/地形辅助函数。
- `driver.py`：Go2 数据驱动策略；`viz.py`：Go2 视频状态卡与渲染辅助。

[使用说明](../../../docs/go2/README.md)。旧 `robot/go2.py`、`env/*.py`、`target/*.py` 等路径仅用于兼容导入；修改代码时编辑本目录。
