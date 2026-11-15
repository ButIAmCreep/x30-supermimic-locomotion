# 共享训练核心

`state.py`、`world_model.py`、`buffer.py`、`imagine.py`、`train.py` 保留原有世界模型、状态提取、回放、想象和训练循环。`train.py` 同时保留原 Go2 Trainer 与公共命令行分派，X30 Trainer 在 `../robots/x30/train.py` 中继承它；本轮没有改变算法或类名。

Go2 数据驱动策略实现已移至 `../robots/go2/driver.py`；X30 速度任务实现已移至 `../robots/x30/`。此处的 `driver.py`、`velocity*.py` 是旧导入兼容层。
