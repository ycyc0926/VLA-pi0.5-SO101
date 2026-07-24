# VLA π0.5 LoRA Fine-tuning for SO-101

基于 Physical Intelligence OpenPI 与 Hugging Face LeRobot，在 SO-101 双相机操作数据上完成 π0.5 LoRA 微调的工程实践。

## 项目状态

- 单卡 NVIDIA RTX PRO 6000 96GB 完成 50,000-step LoRA 训练。
- 数据集包含 50 episodes、31,000 frames、双路 AV1 视频和 6-DoF state/action。
- 训练约 15 小时完成，训练 loss 从 0.0609 下降并稳定在约 0.001～0.002。
- 接入 W&B 实时监控、JAX 编译缓存和 Orbax 分层 checkpoint 管理。
- 已完成训练链路，真实机械臂闭环成功率评测仍在进行中。

> 训练 loss 不能直接等价为真实机器人任务成功率。本仓库不会在完成系统评测前宣称抓取成功率。

## 核心工作

- 将 Conda 环境、模型、数据、缓存、日志和 checkpoint 隔离到数据盘，解决小系统盘环境中的部署问题。
- 下载、校验并本地缓存约 12GB 的 π0.5 Base 权重，避免训练时重复访问慢速 GCS。
- 适配 LeRobot v3 数据：环境相机、手腕相机、6D joint state 和 6D action。
- 修复 π0.5 离散 state 输入：`discrete_state_input=True`。
- 对前 5 个机械臂关节使用 delta action，gripper 保持绝对动作语义。
- 统一训练和推理 prompt：`Grab the black cube and place it in the white cup`。
- 移除跨服务器硬编码数据路径，改为环境变量配置。
- 配置 W&B 在线记录及每 1,000 step 保存、每 5,000 step 长期保留的 checkpoint 策略。

## 数据流

```text
LeRobot parquet + 双相机视频
        ↓
字段重组与 SO-101 输入适配
        ↓
关节 delta / gripper absolute action
        ↓
分位数归一化
        ↓
224×224 图像 + prompt/state tokenization
        ↓
PaliGemma + Action Expert
        ↓
Flow Matching loss + LoRA 更新
```

## 训练配置

| 参数 | 数值 |
|---|---:|
| Model config | `pi05_so101_lora` |
| Batch size | 16 |
| Train steps | 50,000 |
| Action horizon | 10 |
| Warmup | 1,000 steps |
| Peak learning rate | 1e-4 |
| Final learning rate | 1e-6 |
| Save interval | 1,000 steps |
| Keep period | 5,000 steps |
| GPU memory fraction | 0.90 |

阶段训练指标：

| Step | Loss | Grad norm | Param norm |
|---:|---:|---:|---:|
| 0 | 0.0609 | 0.3825 | 1803.8630 |
| 5,000 | 0.0052 | 0.0571 | 1806.1942 |
| 10,000 | 0.0040 | 0.0421 | 1807.8408 |
| 30,000 | 0.0026 | 0.0349 | 1810.0790 |
| 49,990 | 0.0017 | 0.0349 | 1810.2131 |

W&B run: [pi05-so101 / blacknew_lora_50k_v1](https://wandb.ai/can498987-/pi05-so101/runs/7rshhe6z)

## 仓库结构

```text
.
├── custom_vla/        # 实际使用并修改的 OpenPI/LeRobot 源码快照
├── README.md
├── solution.md        # 完整部署、排错、训练与面试复盘文档
├── pi0训练命令.txt     # 原始实验命令参考
└── readme.txt         # 初始环境参考文档
```

## 大文件说明

以下内容不会上传 GitHub：

- π0.5 Base 权重与所有训练 checkpoint。
- LeRobot parquet、视频和其他数据集文件。
- W&B 本地运行文件与训练日志。
- Conda 环境、PIP/HF/JAX 缓存。
- 大型 CAD、模型权重、归档和媒体文件。

数据集来源：[`AlexFeng1/blacknew`](https://huggingface.co/datasets/AlexFeng1/blacknew)

π0.5 权重按 OpenPI 官方目录结构单独准备：

```text
$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params
```

## 关键修改位置

```text
custom_vla/openpi/src/openpi/training/config.py
custom_vla/openpi/scripts/train.py
custom_vla/openpi/packages/openpi-client/src/openpi_client/zpf_new_so101_client.py
custom_vla/openpi/packages/openpi-client/src/openpi_client/lkw_aysnc_so101_client.py
```

详细安装命令、训练命令、问题定位、checkpoint 策略、实验结果、简历表述和面试问答见 [`solution.md`](solution.md)。

## 后续计划

- 在真实 SO-101 上对 5k～50k checkpoint 进行相同条件的重复评测。
- 按完整 episode 划分验证集，避免相邻 frame 泄漏。
- 报告抓取、放置和完整任务成功率，并整理失败案例。
- 统一推理服务端口、prompt、action horizon 和动作队列策略。
- 增加关节限位、单步动作限制和急停保护。

## 致谢

- [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- [ROS-LiKunwei/VLA](https://github.com/ROS-LiKunwei/VLA)

