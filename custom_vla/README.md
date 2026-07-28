# VLA 机器人抓取与放置项目 (Pi0 + LoRA)

本项目记录了基于 **LeRobot** 和 **OpenPI** 框架，使用 LoRA 技术对 Pi0 模型进行微调，以实现SO101机械臂抓取并放置黑色六边体物块等任务的完整流程。

---
## 🏝️ 进度
- so101:
  - [x] 数据采集及适配: 遥操作采集Lerobot v3.0格式数据
  - [x] 模型微调: 使用OpenPI+LoRA进行微调
  - [x] 异步推理: 使用OpenPI服务端+自定义客户端进行异步推理  
  - [ ] Lerobot微调
  - [ ] Lerobot推理
- 人形实机
  - [ ] 数据采集
  - [ ] 模型微调
  - [ ] 实机异步推理

## 📊 数据集 (Dataset)
* **在线可视化**: [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)

* **数据集名称**: 
1. `VLALearner/pi0_pick_and_place`   
   * **内容描述** : 针对红色小球抓取与放置到白色杯子任务的专家演示数据。    
2. `VLALearner/pi0_pick_and_place_2`
   * **内容描述** : 针对黑色六边体抓取与放置任务的专家演示数据。

* **采集方法**:
   、、、bash
   (lerobot) likunwei@likunwei:~/lerobot$ lerobot-record    \
    --robot.type=so101_follower     \
    --robot.port=/dev/ttyACM1     \
    --robot.id=my_awesome_follower_arm     \
    --robot.cameras="{ "env": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}}"     \
    --teleop.type=so101_leader     \
    --teleop.port=/dev/ttyACM0     \
    --teleop.id=my_awesome_leader_arm     \
    --display_data=true     \
    --dataset.repo_id=datasets/record-test     \
    --dataset.num_episodes=5  \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10    \
    --dataset.single_task="Grab the black cube"   \
    --dataset.push_to_hub=false \
    --dataset.root=/home/likunwei/lerobot/datasets/test \
    --resume=false  \
    --dataset.streaming_encoding=true 
   ```
---

## 🛠️ 训练方案 A：LeRobot + LoRA

此流程适用于使用 LeRobot 官方工具链进行快速微调。建议使用 `tmux` 维持长连接。

### 1. 启动训练会话
```bash
# 创建一个新的 tmux 会话
tmux new -s pi0_train

# 激活环境并启动训练
conda activate lerobot
CUDA_VISIBLE_DEVICES=0 lerobot-train --config_path pi0_lora_train.yaml


### 2. 会话维护
* **分离会话**: 按下 `Ctrl + B`，然后按 `D`。
* **重新附着**: SSH 登录服务器后，运行：
```bash
tmux attach -t pi0_train
```

## 🛠️ 训练方案 B：OpenPI+LoRA
### 1. 参数归一化
```bash
(lerobot) likunwei@vla-device:~/pi0/openpi$ PYTHONPATH=src:/home/likunwei/lerobot/src uv run scripts/compute_norm_stats.py  --config-name=pi05_so101
```

### 2. 训练
```bash
# 2. 设置只可见卡 1（因为它最干净，没有显示输出占用）
export CUDA_VISIBLE_DEVICES=1 
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85 
export LD_LIBRARY_PATH=/home/likunwei/miniconda3/envs/pi0-zero/lib:$LD_LIBRARY_PATH


PYTHONPATH=src:/home/likunwei/lerobot/src uv run scripts/train.py pi05_so101_lora \
    --exp-name=grab_01_checkpoint \
    --checkpoint-base-dir=/home/likunwei/pi0/openpi/checkpoints
```

### 3. 异步推理
server:
```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/likunwei/miniconda3/envs/pi0-zero/lib:$LD_LIBRARY_PATH

# 启动推理服务
    PYTHONPATH=src:/home/likunwei/lerobot/src uv run scripts/serve_policy.py \
      --port 5000 \
      policy:checkpoint \
      --policy.config pi05_so101_lora \
      --policy.dir /home/likunwei/pi0/openpi/checkpoints/pi05_so101_lora/grab_01_checkpoint/33000
```

client:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/packages/openpi-client/src:/home/likunwei/lerobot/src
python packages/openpi-client/src/openpi_client/lkw_aysnc_so101_client.py
```

## 📖 参考资料
1. [同济子豪兄带你玩LeRobot具身智能](https://zihao-ai.feishu.cn/wiki/H3r1w8ALmilFJUkQTAtcAZnHnoh)