# π0.5 + SO-101 VLA 项目实践记录

> 用途：持续记录本项目做过的工作、遇到的问题、解决方案、实验结果和后续计划，便于复盘、撰写简历和准备 VLA 岗位面试。
>
> 最近更新：2026-07-27 13:01:25 UTC

## 1. 项目概览

本项目在 AutoDL 云服务器上完成了 Physical Intelligence `openpi` 的部署，并使用 LeRobot 格式的 SO-101 双相机数据，对 π0.5 进行 LoRA 微调。

当前已经完成：

- 将 Conda 环境、模型、数据集、缓存、日志和 checkpoint 放到数据盘，避免占满系统盘。
- 部署 OpenPI、LeRobot 和自定义 VLA 仓库。
- 下载并校验 π0.5 Base 权重。
- 下载完整的 LeRobot 数据集，包括 parquet、meta 和双相机视频。
- 修正 π0.5 state 输入、数据路径、训练 prompt 和客户端 prompt。
- 计算数据归一化统计。
- 配置 W&B 在线监控和 Orbax checkpoint。
- 在单张 RTX PRO 6000 96GB 上完成 50,000 step 的 π0.5 LoRA 微调。
- 保留每 5,000 step 的阶段 checkpoint 和最终 checkpoint。

尚未完成：

- 在真实 SO-101 机械臂上系统评测不同 checkpoint。
- 建立训练集/验证集划分和离线验证指标。
- 统计抓取成功率、平均完成时间和失败类型。
- 将当前未提交的代码修改整理成 Git commit/tag。

## 2. 硬件和软件环境

### 2.1 硬件

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 显存 | 97,887 MiB，约 96GB |
| 训练显存 | 约 88,163 MiB |
| GPU 利用率 | 正常训练时约 99%～100% |
| 系统盘 | 30GB |
| 数据盘 | 600GB，训练完成后使用约 367GB |

训练时配置：

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export CUDA_VISIBLE_DEVICES=0
```

JAX 会预分配接近 90% 的显存，因此约 88GB 显存占用属于正常现象。

### 2.2 Python 环境

主要训练环境：

```text
/root/autodl-tmp/envs/pi0_env
Python 3.11
```

关键依赖：

```text
torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128
numpy==1.26.4
lerobot==0.4.4
accelerate==1.13.0
wandb==0.28.1
```

另一个 LeRobot 环境：

```text
/root/autodl-tmp/envs/lerobot
Python 3.12.13
FFmpeg 7.1.1
```

## 3. 项目目录

```text
/root/autodl-tmp/VLA/
├── custom_vla/openpi/                 # 实际训练使用的 OpenPI 代码
├── custom_vla/lerobot/                # 自定义仓库中的 LeRobot
├── openpi/                             # 另一份 OpenPI 仓库
├── lerobot/                            # 另一份 LeRobot 仓库
├── datasets/AlexFeng1/blacknew/        # SO-101 训练数据
├── checkpoints/pi05_so101_lora/        # 微调 checkpoint
├── logs/                               # 本地训练日志
├── wandb/                              # W&B 本地运行数据
├── pi0训练命令.txt                      # 原始参考命令
└── solution.md                         # 本文档
```

数据盘上的公共缓存：

```text
/root/autodl-tmp/cache/openpi           # π0.5 权重和 tokenizer
/root/autodl-tmp/cache/huggingface      # Hugging Face 缓存
/root/autodl-tmp/cache/jax              # JAX 编译缓存
/root/autodl-tmp/cache/pip              # pip 缓存
/root/autodl-tmp/tmp                    # 训练临时目录
```

## 4. 数据集

数据集：`AlexFeng1/blacknew`

本地路径：

```text
/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
```

数据规模：

| 项目 | 数值 |
|---|---:|
| Episode | 50 |
| Frame | 31,000 |
| Task | 1 |
| FPS | 30 |
| 本地大小 | 约 173MB |
| State 维度 | 6 |
| Action 维度 | 6 |
| 相机 | 环境相机 + 手腕相机 |
| 视频编码 | AV1 |
| 原始分辨率 | 640×480 |

六维关节字段：

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

完整数据集必须包含：

```text
data/chunk-000/file-000.parquet
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
videos/observation.images.env/chunk-000/file-000.mp4
videos/observation.images.hand/chunk-000/file-000.mp4
norm_stats.json
```

只有 parquet 文件不能完成视觉 VLA 训练，因为图像内容实际存储在视频文件中。

## 5. π0.5 Base 权重

基础权重约 12GB，最终路径：

```text
/root/autodl-tmp/cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

PaliGemma tokenizer：

```text
/root/autodl-tmp/cache/openpi/big_vision/paligemma_tokenizer.model
```

训练时显式传入本地权重，避免再次访问慢速 GCS：

```bash
--weight-loader.params-path=/root/autodl-tmp/cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

## 6. 模型数据流

训练数据的大致流向：

```text
LeRobot parquet + 双相机视频
        ↓
RepackTransform：统一字段名称
        ↓
SO101Inputs：解析图像、state、action
        ↓
前5维关节动作转换为 delta，第6维 gripper 保持绝对值
        ↓
使用 norm_stats 做分位数归一化
        ↓
图像缩放到 224×224
        ↓
prompt tokenize + π0.5 离散 state tokenize
        ↓
PaliGemma 视觉语言主干 + Action Expert
        ↓
Flow Matching 动作预测损失
        ↓
仅更新 LoRA/允许训练的参数
```

### 6.1 LoRA 原理

LoRA 不直接完整更新大矩阵 `W`，而是学习低秩增量：

```text
W' = W + scale × A × B
```

其中 `A` 和 `B` 的秩远小于原矩阵维度。优点：

- 训练参数更少。
- 优化器状态更小。
- 在小数据集上比全量微调更容易控制过拟合。
- 保留 π0.5 预训练视觉、语言和机器人先验。

本项目使用：

```text
PaliGemma: gemma_2b_lora
Action Expert: gemma_300m_lora
Action horizon: 10
```

## 7. 关键代码修正

实际修改位于：

```text
/root/autodl-tmp/VLA/custom_vla/openpi
```

### 7.1 修正 π0.5 state 输入

文件：

```text
src/openpi/training/config.py
```

目标配置：

```python
name="pi05_so101_lora"
discrete_state_input=True
```

原因：π0.5 将 state 作为离散 token 输入。如果同时使用 `pi05=True` 和 `discrete_state_input=False`，state 虽然可能参与动作差分转换，却不会正确进入 π0.5 网络。

字段映射：

```python
"observation.state": "observation.state"
```

### 7.2 移除旧服务器硬编码路径

数据路径改为环境变量：

```python
repo_id=os.environ["SO101_DATASET_DIR"]
```

运行前设置：

```bash
export SO101_DATASET_DIR=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
```

同时删除了 `scripts/train.py` 中指向旧机器 `/home/likunwei/...` 的 PyAV 视频解码探测代码，避免训练在不存在的文件上报错。

### 7.3 修正 prompt 一致性

训练默认 prompt：

```text
Grab the black cube and place it in the white cup
```

定义位置：

```text
src/openpi/training/config.py
LeRobotSO101DataConfig.create()
```

两个旧 SO-101 客户端原先发送了截断文本：

```text
Grab the black cube and place it
```

已修改为通过字符串参数发送：

```python
"prompt": args.prompt
```

并将默认值统一为完整训练 prompt。推理时也可以显式指定：

```bash
--prompt "Grab the black cube and place it in the white cup"
```

### 7.4 JAX 缓存迁移到数据盘

```python
jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("JAX_COMPILATION_CACHE_DIR", "/root/autodl-tmp/cache/jax"),
)
```

## 8. 遇到的问题与解决方案

### 8.1 Conda 环境占满系统盘

问题：按别人的 README 使用环境名称创建 Conda 环境，会默认写入系统盘。

解决：使用绝对路径创建和激活环境。

```bash
conda create -p /root/autodl-tmp/envs/pi0_env python=3.11
conda activate /root/autodl-tmp/envs/pi0_env
```

创建新环境前可以先 `conda deactivate` 回到 base，但 Conda 也允许直接激活另一个绝对路径环境。

### 8.2 pip/PyTorch 下载慢且文件巨大

问题：PyTorch CUDA wheel 和 NVIDIA 依赖总计数 GB，单连接下载速度极慢。

解决：

- 将 `PIP_CACHE_DIR` 和 `TMPDIR` 放到数据盘。
- 对明确 URL 使用 aria2 多连接和断点续传。
- 下载完成的 wheel 通过本地路径安装。
- pip 的 `Using cached` 表示复用缓存，不是重复下载。

注意：清华 PyPI 镜像适合普通 Python 包，但 PyTorch CUDA 专用 wheel 仍需官方索引或可用镜像。`source /etc/network_turbo` 主要面向 GitHub/Hugging Face，可能让普通 pip 源更慢。

### 8.3 阿里 PyTorch 镜像返回 403

问题：aria2 请求某个阿里镜像的 wheel URL 返回 HTTP 403。

解决：换用实际可访问的直链；403 不是 aria2 本身故障。aria2 的 `.aria2` 控制文件支持对同一 URL 和输出路径继续下载。

### 8.4 SHA256 校验命令失败

问题：长文件名被终端换行，导致校验路径被截断。

解决：确保哈希和完整路径在同一个 shell 逻辑行中，或先进入 wheel 目录再校验短文件名。

### 8.5 Git submodule 下载慢

问题：OpenPI 主仓库下载完成，但 LIBERO submodule 速度很慢。

解决：主仓库和子模块分开恢复；Git clone 可重复执行 submodule update。`GIT_LFS_SKIP_SMUDGE=1` 可避免初始 clone 自动拉取大文件。

### 8.6 Hugging Face Xet 401

问题：下载 parquet 时，Xet CAS reconstruction 返回 `401 Unauthorized`。

解决思路：

- 为 Hugging Face 配置有效 token。
- 或设置 `HF_HUB_DISABLE_XET=1`，改走普通 HTTP。
- 如果加速代理 TLS 握手超时，关闭代理后重试。
- 对难以稳定下载的大文件，可在本地浏览器下载后传到数据盘。

### 8.7 heredoc 一直显示 `>` 而没有输出

问题：执行 `python - <<'PY'` 后 shell 一直显示 `>`。

原因：结束标记 `PY` 前有缩进或没有独占一行，shell 认为输入尚未结束。

解决：结束标记必须从行首开始：

```bash
python - <<'PY'
print("ok")
PY
```

### 8.8 `ModuleNotFoundError: pytest`

问题：OpenPI 的 PyTorch Gemma 文件导入了 `pytest`，环境中未安装。

解决：在 `pi0_env` 中安装 `pytest` 后重新计算归一化统计。

### 8.9 π0.5 Base 权重从 GCS 下载极慢

问题：11.6GB 权重从 `openpi-assets` GCS 下载只有 KB/s。

解决：

- 本地浏览器下载权重压缩包。
- 通过 SCP、aria2 或其他稳定通道传到数据盘。
- 解压到 OpenPI 的标准缓存结构。
- 使用 `rclone check`、文件数和 manifest 验证完整性。
- 训练命令显式传入本地 `params`，不再访问 GCS。

### 8.10 W&B 找到凭据但返回 401

报错：

```text
wandb.errors.errors.CommError: user is not logged in
HTTP 401 POST https://api.wandb.ai/graphql
```

原因：`.netrc` 中存在凭据不代表 API Key 有效；旧 Key 可能失效。

解决：

```bash
unset WANDB_API_KEY WANDB_BASE_URL WANDB_MODE WANDB_DISABLED
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
wandb login --cloud --relogin --verify
```

不要在文档、代码或聊天记录中保存 API Key。

### 8.11 GPU 周期性从 100% 掉到 0%

现象：GPU 正常时 99%～100%，每隔约 1,000 step 短暂下降后恢复。

原因：Orbax checkpoint 保存需要短暂同步和大量磁盘写入。step 2000 的实测：

```text
主线程阻塞保存约 6.93 秒
后台保存总耗时约 19.04 秒
单个 checkpoint 约 8.9GB
```

连续 12 秒 GPU 采样均为 99%～100%，说明正常训练阶段没有持续性数据瓶颈。

如果掉速明显比 checkpoint 更频繁，再检查 AV1 解码和 DataLoader worker 是否导致 GPU 等待数据。

### 8.12 系统盘持续增长

检查结果：训练相关路径均指向数据盘。系统盘的主要额外占用来自历史 `/tmp`，约 15GB，包括：

```text
/tmp/IsaacLab                                  约 6.7GB
/tmp/ankle_kinematics_fa_robot_upload...       约 1.8GB
/tmp/unitree_ros                               约 1.6GB
/tmp/hiking_in_the_wild_fa_robot_29dof...      约 1.4GB
```

这些不是本次 π0.5 训练 checkpoint。清理前必须确认没有其他任务正在使用，不能在不确定时直接递归删除。

## 9. 归一化统计

训练前必须运行与训练完全相同配置的 norm stats：

```bash
conda activate /root/autodl-tmp/envs/pi0_env

export SO101_DATASET_DIR=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
export OPENPI_DATA_HOME=/root/autodl-tmp/cache/openpi
export JAX_COMPILATION_CACHE_DIR=/root/autodl-tmp/cache/jax
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_LEROBOT_HOME=/root/autodl-tmp/VLA/datasets
export TMPDIR=/root/autodl-tmp/tmp
export PYTHONPATH=/root/autodl-tmp/VLA/custom_vla/openpi/src

cd /root/autodl-tmp/VLA/custom_vla/openpi

python -u scripts/compute_norm_stats.py \
  --config-name=pi05_so101_lora
```

生成结果：

```text
/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew/norm_stats.json
```

## 10. 最终训练配置

| 参数 | 数值 |
|---|---:|
| Config | `pi05_so101_lora` |
| Experiment | `blacknew_lora_50k_v1` |
| Batch size | 16 |
| DataLoader workers | 4 |
| Train steps | 50,000 |
| Warmup steps | 1,000 |
| Peak LR | 1e-4 |
| Decay steps | 50,000 |
| Final LR | 1e-6 |
| Gradient clip | 0.5 |
| Log interval | 10 |
| Save interval | 1,000 |
| Keep period | 5,000 |
| W&B project | `pi05-so101` |
| JAX memory fraction | 0.90 |

最终训练命令：

```bash
conda activate /root/autodl-tmp/envs/pi0_env
cd /root/autodl-tmp/VLA/custom_vla/openpi

export SO101_DATASET_DIR=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
export OPENPI_DATA_HOME=/root/autodl-tmp/cache/openpi
export JAX_COMPILATION_CACHE_DIR=/root/autodl-tmp/cache/jax
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_LEROBOT_HOME=/root/autodl-tmp/VLA/datasets
export WANDB_DIR=/root/autodl-tmp/VLA/wandb
export TMPDIR=/root/autodl-tmp/tmp
export PYTHONPATH=/root/autodl-tmp/VLA/custom_vla/openpi/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export CUDA_VISIBLE_DEVICES=0

unset WANDB_MODE WANDB_DISABLED
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python -u scripts/train.py pi05_so101_lora \
  --exp-name=blacknew_lora_50k_v1 \
  --project-name=pi05-so101 \
  --data.repo-id=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew \
  --weight-loader.params-path=/root/autodl-tmp/cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --checkpoint-base-dir=/root/autodl-tmp/VLA/checkpoints \
  --assets-base-dir=/root/autodl-tmp/VLA/assets \
  --batch-size=16 \
  --num-workers=4 \
  --num-train-steps=50000 \
  --log-interval=10 \
  --save-interval=1000 \
  --keep-period=5000 \
  --lr-schedule.warmup-steps=1000 \
  --lr-schedule.peak-lr=1e-4 \
  --lr-schedule.decay-steps=50000 \
  --lr-schedule.decay-lr=1e-6 \
  --wandb-enabled \
  --overwrite \
  2>&1 | tee /root/autodl-tmp/VLA/logs/blacknew_lora_50k_v1.log
```

## 11. W&B、日志和 tmux

W&B 在线运行：

```text
Project: pi05-so101
Run ID: 7rshhe6z
Run URL: https://wandb.ai/can498987-/pi05-so101/runs/7rshhe6z
```

本地日志：

```text
/root/autodl-tmp/VLA/logs/blacknew_lora_50k_v1.log
```

实时查看：

```bash
tail -F /root/autodl-tmp/VLA/logs/blacknew_lora_50k_v1.log \
  | grep --line-buffered -E 'TRAIN_METRICS|Step|loss|grad_norm|param_norm|WARNING|ERROR'
```

tmux：

```bash
tmux new -s pi05_train
tmux ls
tmux attach -t pi05_train
tmux attach -d -t pi05_train
```

从 tmux 分离但不中止训练：先按 `Ctrl+B`，松开后按 `D`。

## 12. Checkpoint 策略

输出目录：

```text
/root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1
```

配置：

```text
save_interval=1000
keep_period=5000
max_to_keep=1
```

含义：

- 每 1,000 step 触发一次保存，提供最新断点。
- 普通 checkpoint 在下一个 checkpoint 完成后被删除。
- 每 5,000 step 的 checkpoint 被永久保留。
- 最后一步无论是否为保存间隔，都会保存。

50,000 step 最终保留：

```text
5000
10000
15000
20000
25000
30000
35000
40000
45000
49999
```

总占用约 87GB，单个 checkpoint 约 8.9GB。

中断续训时，必须使用原来的实验名，并将 `--overwrite` 替换为：

```bash
--resume
```

同时本地日志使用 `tee -a`，避免覆盖已有日志。

## 13. 训练结果

训练状态：**已完成**。

训练开始记录：2026-07-22 21:57（日志时区 UTC+8）  
训练完成记录：2026-07-23 13:01（日志时区 UTC+8）  
总耗时：约 15 小时 4 分钟。  
平均速度：约 1.08 秒/step，约 0.92 step/s。

阶段指标：

| Step | Loss | Grad norm | Param norm |
|---:|---:|---:|---:|
| 0 | 0.0609 | 0.3825 | 1803.8630 |
| 100 | 0.0323 | 0.2390 | 1803.8635 |
| 500 | 0.0148 | 0.0971 | 1803.8997 |
| 1,000 | 0.0123 | 0.0733 | 1804.1061 |
| 5,000 | 0.0052 | 0.0571 | 1806.1942 |
| 10,000 | 0.0040 | 0.0421 | 1807.8408 |
| 20,000 | 0.0037 | 0.0456 | 1809.5106 |
| 30,000 | 0.0026 | 0.0349 | 1810.0790 |
| 40,000 | 0.0014 | 0.0259 | 1810.2001 |
| 49,990 | 0.0017 | 0.0349 | 1810.2131 |

观察：

- 训练 loss 从 step 0 的 0.0609 降到后期约 0.001～0.002。
- grad norm 总体下降并保持有限，期间存在少量尖峰，但能够快速恢复。
- param norm 平滑变化，没有出现突然爆炸或 NaN。
- 50,000 step 对 31,000 frame、batch 16 约相当于抽样 25.8 个数据集 epoch。

重要限制：

- 上述都是训练指标，没有独立 validation loss。
- 训练 loss 下降不能直接证明真实机械臂任务成功率提高。
- 简历中应写“完成训练并使训练 loss 稳定下降”，不能在没有真实评测时写“达到某抓取成功率”。

## 14. 如何阅读训练曲线

### Loss

良好趋势：前期快速下降，中期缓慢下降并小幅波动，后期稳定。单个 batch 波动正常，应关注移动平均。

危险信号：持续上升、尖峰后不恢复、NaN/Inf，或训练 loss 极低但真实任务效果差。

### Grad norm

良好趋势：warmup 阶段存在一定变化，随后总体稳定；偶发尖峰后恢复可以接受。

危险信号：长期爆炸、频繁贴近裁剪阈值、长期接近零且 loss 不改善、NaN/Inf。

### Param norm

良好趋势：平滑、缓慢变化。它不要求永久上升，绝对数值也不能跨不同实验直接比较。

危险信号：突然跳变、指数增长、NaN/Inf，尤其是同时伴随 loss 和 grad norm 异常。

### Learning rate

本实验预期：

```text
0～1000 step：从接近 0 warmup 到 1e-4
1000～50000 step：Cosine decay
最终接近 1e-6
```

## 15. 下一步实验计划

### P0：完成真实机械臂评测

依次评测：

```text
5000、10000、15000、20000、30000、40000、49999
```

每个 checkpoint 至少运行相同数量的重复实验，记录：

- 抓取成功率。
- 放置成功率。
- 完整任务成功率。
- 平均完成时间。
- 失败发生在接近、抓取、抬升还是放置阶段。
- 对光照、物体位置和背景变化的鲁棒性。

### P1：增加验证方案

- 按 episode 划分训练集和验证集，避免相邻 frame 泄漏。
- 验证集必须按完整 episode 划分，不能随机拆 frame。
- 比较不同 checkpoint 的 validation loss 和真实成功率。
- 检查 30k～50k 是否已经过拟合。

### P1：推理链路统一

- 确认服务端和客户端端口统一，例如均为 5000。
- 确认客户端 prompt 与训练 prompt 完全一致。
- 确认推理输入 state 是 6 维并与训练单位一致。
- 确认环境相机和手腕相机没有交换。
- 确认 action horizon=10 与客户端动作队列逻辑一致。
- 增加关节限位、单步最大变化和急停机制。

### P2：工程整理

- 清理无用的 `/tmp` 历史文件，但清理前确认没有其他进程使用。
- 将未提交代码拆分成独立 commit：state、path、prompt、debug cleanup。
- 给本次实验创建 Git tag。
- 保存环境依赖锁文件和服务器复现脚本。
- 为每次新实验追加本文档的实验记录。

## 16. 简历表述建议

在没有真实成功率前，可使用以下表述：

> 基于 OpenPI 和 LeRobot 搭建 SO-101 双相机 VLA 训练链路，在单张 RTX PRO 6000 96GB 上完成 π0.5 LoRA 微调；处理 50 个 episode、31k 帧的 6-DoF 视觉操作数据，完成 state 离散 token 接入、关节 delta/夹爪绝对动作变换、prompt 对齐、归一化统计、W&B 监控及 Orbax 断点管理，50k-step 训练约 15 小时完成，训练 loss 从 0.0609 降至约 0.001～0.002。

更精炼的三条：

- 搭建 π0.5 + LeRobot + SO-101 双相机 VLA LoRA 微调流水线，实现数据、权重、缓存、日志和 checkpoint 的数据盘隔离与可复现训练。
- 修复 π0.5 离散 state 输入、跨服务器硬编码路径和训练/推理 prompt 不一致问题，完成 6-DoF 动作归一化与前 5 关节 delta action 适配。
- 在 RTX PRO 6000 96GB 上完成 50k-step 训练，接入 W&B 实时监控与分层 checkpoint 保留策略，并基于日志定位 GPU 周期性掉速和系统盘增长问题。

拿到真实评测结果后，再补充：

```text
在 N 次不同物体位置/光照条件实验中达到 X% 完整任务成功率，较某基线提升 Y 个百分点。
```

## 17. 面试讲解框架

### 17.1 一分钟项目介绍

1. 任务：使用 SO-101 双相机数据微调 π0.5，完成黑色方块抓取并放入白色杯子。
2. 数据：LeRobot v3，50 episodes、31k frames、两路 AV1 视频、6D state/action。
3. 模型：π0.5 PaliGemma + Action Expert，使用 LoRA 和 flow matching。
4. 工程：解决大权重下载、系统盘隔离、state token、prompt 对齐、归一化、W&B 和 checkpoint。
5. 结果：单卡 96GB 完成 50k-step，训练稳定；下一步进行分 checkpoint 的真实机械臂成功率评测。

### 17.2 高频追问

**为什么使用 LoRA，而不是全量微调？**  
数据量只有 50 episodes，LoRA 参数更少、显存和优化器状态更小，也更不容易破坏预训练能力。

**π0.5 的 state 为什么必须特别处理？**  
π0.5 将 state 作为离散 token 输入；如果关闭 `discrete_state_input`，state 可能不会进入模型主干。

**为什么前 5 个关节用 delta，gripper 用绝对值？**  
机械臂关节轨迹适合建模相对变化，而 gripper 通常表示开合状态或绝对目标。mask 必须与数据采集语义一致。

**为什么训练 loss 不能代表任务成功率？**  
训练 loss 衡量动作预测误差，不能覆盖闭环控制误差、视觉分布变化、误差积累、接触动力学和安全限制。

**GPU 为什么周期性掉到 0？**  
每 1,000 step 保存约 8.9GB checkpoint，Orbax 短暂同步和磁盘写入会让 GPU 等待，保存后立即恢复。

**如何避免系统盘被占满？**  
Conda 环境、模型、HF/PIP/JAX 缓存、TMPDIR、W&B 和 checkpoint 全部显式指向 `/root/autodl-tmp` 数据盘，并定期用 `df/du` 审计。

## 18. 实验记录模板

以后每次实验在这里追加：

```markdown
### YYYY-MM-DD：实验名称

- Git commit/tag：
- 数据集版本：
- 数据量与划分：
- Config：
- Prompt：
- Base checkpoint：
- Batch size：
- Train steps：
- Learning rate：
- Action horizon：
- Delta/absolute action 定义：
- 图像增强：
- W&B URL：
- Checkpoint 路径：
- 训练耗时：
- 最终训练/验证指标：
- 真实机械臂测试次数：
- 成功率：
- 失败案例：
- 结论：
- 下一步：
```

## 19. 当前工程状态提醒

`custom_vla` 仓库当前仍有未提交修改：

```text
openpi/scripts/train.py
openpi/src/openpi/training/config.py
openpi/packages/openpi-client/src/openpi_client/zpf_new_so101_client.py
openpi/packages/openpi-client/src/openpi_client/lkw_aysnc_so101_client.py
```

在继续修改前应先检查 diff，并按逻辑拆分提交，避免后续无法解释训练使用的是哪个代码版本。

## 20. LeRobot 与 OpenPI 的职责及仓库关系

项目中同时存在 LeRobot 和 OpenPI，不代表训练时同时使用了两套 policy。

本项目实际链路是：

```text
LeRobot：SO-101 硬件驱动、遥操作采集、LeRobot v3 数据格式
   ↓
OpenPI：读取 LeRobot 数据，训练和推理 π0/π0.5
   ↓
OpenPI Client + LeRobot：接收模型动作并控制真实 SO-101
```

`custom_vla/README.md` 中已经标明：

```text
已完成：LeRobot v3 数据采集、OpenPI + LoRA 微调、OpenPI 异步推理
未完成：LeRobot policy 微调、LeRobot policy 推理
```

因此，本项目 50,000 step 权重是 **OpenPI π0.5 LoRA** 训练结果，不是 LeRobot policy 的训练结果。OpenPI 训练代码使用 LeRobot 作为数据集读取库，这不等于采用 LeRobot 的模型实现。

工作区有四份相关目录：

```text
/root/autodl-tmp/VLA/openpi
/root/autodl-tmp/VLA/lerobot
/root/autodl-tmp/VLA/custom_vla/openpi
/root/autodl-tmp/VLA/custom_vla/lerobot
```

SO-101 复现和部署应优先使用 `custom_vla` 下的一对代码，因为其中已经包含：

- SO-101 双相机字段映射。
- 6 维 state/action 适配。
- 前 5 关节 delta、夹爪 absolute 动作转换。
- π0.5 LoRA 配置。
- SO-101 异步推理和安全控制客户端。

根目录的 `openpi` 是另一份较新的官方仓库，不能在未做 API 和 checkpoint 兼容验证前直接替换训练时的自定义版本。

关于“LeRobot 微调实机效果不好”的判断需要区分：

- LeRobot 数据格式本身不是性能差的原因，OpenPI 也使用该格式。
- 真实表现还取决于模型实现、动作表示、归一化、相机视角、机器人校准、推理延迟和控制器。
- 当前项目没有完成 LeRobot policy 的同条件微调和实机对照，不能仅凭传言得出框架优劣结论。

## 21. 云端训练权重部署到本地 SO-101

### 21.1 推荐架构

当前推荐“AutoDL 云端 GPU 推理 + Ubuntu 本地真机控制”：

```text
SO-101 + 环境相机 + 腕部相机
              ↕
Ubuntu 本地电脑
LeRobot 驱动 + OpenPI Client
              ↕ WebSocket over SSH tunnel
AutoDL 云服务器
OpenPI + GPU + 49999 checkpoint + norm_stats
```

模型权重不需要放进机器人，也不必下载到本地电脑。云服务器负责：

- 加载 8.7GB 左右的最终 checkpoint。
- 在 GPU 上运行 π0.5 推理。
- 对输入 state 做归一化。
- 对模型输出做反归一化和 absolute action 恢复。

本地电脑负责：

- 读取两路 USB 相机。
- 读取 SO-101 六维当前状态。
- 把 observation 发给云端。
- 接收 10 步动作块。
- 做限速、平滑、过期动作拒绝和实机下发。

训练配置共运行 50,000 次循环，但保存目录从 0 计数，因此最终目录是：

```text
/root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999
```

### 21.2 云端启动 Policy Server

在 AutoDL 中继续使用训练时的 Python 环境：

```bash
tmux new -s openpi_server

conda activate /root/autodl-tmp/envs/pi0_env
cd /root/autodl-tmp/VLA/custom_vla/openpi

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export SO101_DATASET_DIR=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
export OPENPI_DATA_HOME=/root/autodl-tmp/cache/openpi
export JAX_COMPILATION_CACHE_DIR=/root/autodl-tmp/cache/jax
export PYTHONPATH=/root/autodl-tmp/VLA/custom_vla/openpi/src

python -u scripts/serve_policy.py \
  --port 5000 \
  policy:checkpoint \
  --policy.config pi05_so101_lora \
  --policy.dir /root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999
```

服务成功时应看到模型权重、归一化统计和 WebSocket Server 均加载成功。模型或 norm stats 加载失败时，不得继续启动真机客户端。

从 tmux 分离但保持服务运行：按 `Ctrl+B`，松开后按 `D`。重新进入：

```bash
tmux attach -t openpi_server
```

### 21.3 SSH 隧道

不建议把 5000 端口直接暴露到公网。本地电脑根据 AutoDL 提供的 SSH 地址建立本地转发。假设登录参数为：

```text
SSH Host: <autodl-host>
SSH Port: <autodl-ssh-port>
```

本地执行：

```bash
ssh -N \
  -L 5000:127.0.0.1:5000 \
  -p <autodl-ssh-port> \
  root@<autodl-host>
```

该终端必须保持运行。另一个本地终端检查：

```bash
nc -zv 127.0.0.1 5000
```

客户端随后连接 `127.0.0.1:5000`，流量会通过 SSH 隧道到达 AutoDL。

### 21.4 云端推理的延迟限制

当前模型：

```text
action_horizon=10
control_hz=30
```

一个动作块覆盖时间仅为：

```text
10 / 30 = 0.333 秒
```

图像采集、网络 RTT、云端推理和动作返回最好整体低于约 300ms。自适应客户端会记录：

```text
rtt_ms
server_ms
result_age_ms
adaptive_skip
rejected_stale
```

如果频繁出现 `Reject stale policy result`，表示动作抵达时已经过期。不能通过关闭安全检查或提高电机速度来掩盖问题，应优先：

- 选择网络距离更近的云服务器。
- 降低网络 RTT 和抖动。
- 优化推理耗时。
- 改用本地 GPU 推理。
- 后续重新训练更长 action horizon 的模型。

## 22. Ubuntu 22 本地 SO-101 客户端环境

本地项目路径：

```text
/home/yc/working_base/VLA/custom_vla
```

本地需要单独虚拟环境，但不需要复制云端完整 JAX/CUDA 训练环境，也不需要本地 checkpoint 或 norm stats。

### 22.1 创建 Python 3.12 环境

下载到本地的 `custom_vla/lerobot` 要求 Python 3.12，而 Ubuntu 22 默认 Python 通常为 3.10：

```bash
cd /home/yc/working_base/VLA/custom_vla

uv venv --python 3.12 .venv-so101
source .venv-so101/bin/activate

python --version
which python
```

以后每次运行客户端前：

```bash
cd /home/yc/working_base/VLA/custom_vla
source .venv-so101/bin/activate
```

本地不要在 `custom_vla/openpi` 根目录运行完整 `uv sync`，否则会安装不必要的 JAX、训练依赖和大体积 GPU 包。

### 22.2 安装 LeRobot 与 Feetech 驱动

```bash
cd /home/yc/working_base/VLA/custom_vla/lerobot
uv pip install -e ".[feetech]"
```

测试：

```bash
python -c "from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig; print('LeRobot SO101 OK')"
```

### 22.3 安装最小 OpenPI Client

当前下载的 LeRobot 要求 NumPy 2.x，而 `openpi-client` 的项目元数据仍声明 NumPy `<2.0`。客户端只使用基本 NumPy 操作，因此本地采用已有 LeRobot 的 NumPy，并跳过 `openpi-client` 的旧依赖解析：

```bash
uv pip install \
  dm-tree \
  msgpack \
  pillow \
  tree \
  "websockets>=11"

uv pip install --no-deps -e \
  /home/yc/working_base/VLA/custom_vla/openpi/packages/openpi-client
```

测试：

```bash
python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy; print('OpenPI Client OK')"

python -c "import cv2, numpy; from lerobot.robots.so_follower import SO101Follower; from openpi_client.websocket_client_policy import WebsocketClientPolicy; print('all imports OK', numpy.__version__, cv2.__version__)"
```

### 22.4 串口和摄像头权限

```bash
sudo usermod -aG dialout,video "$USER"
```

执行后注销当前 Ubuntu 账户并重新登录，然后检查：

```bash
groups
ls -l /dev/ttyACM*
ls -l /dev/video*
```

查找 SO-101 串口：

```bash
lerobot-find-port
```

查找 USB 摄像头：

```bash
lerobot-find-cameras opencv
```

需要记录：

```text
SO-101 follower 串口，例如 /dev/ttyACM0
环境相机编号，例如 2
腕部相机编号，例如 0
```

### 22.5 电机初始化和校准

如果是刚组装、从未配置电机 ID 的 SO-101，先执行：

```bash
lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0
```

已经能够正常被 LeRobot 控制的机械臂不应重复设置电机 ID。

真机推理前必须校准 follower。客户端硬编码使用 ID `my_awesome_follower_arm`，因此校准时保持一致：

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm
```

校准时按照终端提示，将各关节移到活动范围中间，再逐一缓慢覆盖完整活动范围。错误校准会造成 state/action 坐标和训练数据不一致，即使模型与网络都正常也可能导致错误动作。

### 22.6 第一次短时真机测试

推荐使用：

```text
openpi/packages/openpi-client/src/openpi_client/zpf_pi0_so101_client_adaptive_pro_v4.py
```

第一次测试应架空机械臂、清空工作区、准备随时断电，并限制运行时间和单步动作：

```bash
cd /home/yc/working_base/VLA/custom_vla
source .venv-so101/bin/activate

python \
  openpi/packages/openpi-client/src/openpi_client/zpf_pi0_so101_client_adaptive_pro_v4.py \
  --host 127.0.0.1 \
  --port 5000 \
  --serial /dev/ttyACM0 \
  --use_degrees \
  --cam_top 2 \
  --cam_wrist 0 \
  --max_run_sec 10 \
  --dq_limit_deg 0.5 \
  --max_dq_gripper 3
```

根据本机修改串口和相机编号。OpenCV GUI 正常时可增加 `--show_camera`；如果出现 Qt 或 `cv2.imshow` 错误，先移除该参数，摄像头采集和推理本身不依赖 GUI 窗口。

本项目的固定实机协议是：

```text
前 5 关节：degrees
夹爪：0～100
控制频率：30Hz
动作块长度：10
Prompt：Grab the black cube and place it in the white cup
```

不得把前 5 关节改为 radians 后直接使用现有 checkpoint。

## 23. 归一化统计的作用与 checkpoint assets 问题

### 23.1 norm_stats 是什么

归一化统计不是模型权重，也不是通过反向传播学习的参数。它是一个约 1.6KB 的 JSON 文件：

```text
/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew/norm_stats.json
```

文件包含每个 state/action 维度的：

```text
mean
std
q01
q99
```

π0.5 使用分位数归一化，把真实值映射到约 `[-1, 1]`：

```text
normalized = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
```

推理闭环：

```text
真实 state（degrees、gripper 0～100）
        ↓ Normalize
模型输入（约 -1～1）
        ↓ π0.5
模型归一化动作
        ↓ Unnormalize
前 5 维 delta action + 第 6 维 absolute gripper
        ↓ AbsoluteActions
真实六维 absolute target
```

当前 action stats 不是原始 parquet action 的简单统计，而是经过 SO-101 数据变换后的统计：

- 前 5 维统计关节 delta。
- 第 6 维统计 absolute gripper。

因此不能拿 `meta/stats.json`、其他机器人或其他动作语义的 norm stats 替代。

如果统计缺失或错误：

- 数十度的 state 可能未经缩放直接进入模型。
- 模型输出的归一化数值可能被误当成真实角度。
- 关节和夹爪动作尺度会错误。
- 实机可能不动、动作过小、动作异常甚至存在安全风险。

训练和推理必须使用同一份 `norm_stats.json`。

### 23.2 当前 checkpoint 的问题

最终 checkpoint 的目录：

```text
/root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999/assets
```

当前是空目录。原因是训练配置未显式设置 `asset_id`，于是代码把绝对 `repo_id`：

```text
/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
```

同时当作 `asset_id`。绝对路径覆盖了 checkpoint 内的相对 `assets` 路径，导致统计继续写入数据集根目录，而不是复制进 checkpoint。

在原 AutoDL 服务器上，只要设置同一个 `SO101_DATASET_DIR`，现有代码仍可能从数据集根目录找到统计；但该 checkpoint 不完整、不可独立迁移。

### 23.3 推荐修复方式

该修复不改变模型权重，也不需要重新训练。

首先在云端 `src/openpi/training/config.py` 的 `pi05_so101_lora` 中显式增加：

```python
data=LeRobotSO101DataConfig(
    repo_id=os.environ["SO101_DATASET_DIR"],
    assets=AssetsConfig(asset_id="blacknew"),
    base_config=DataConfig(prompt_from_task=False),
    extra_delta_transform=True,
    use_gaussian_noise=False,
    gaussian_std=0.05,
),
```

然后只复制统计文件，不移动或删除数据集中的原文件：

```bash
mkdir -p \
  /root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999/assets/blacknew

cp \
  /root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew/norm_stats.json \
  /root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999/assets/blacknew/norm_stats.json
```

验证：

```bash
ls -lh \
  /root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999/assets/blacknew/norm_stats.json
```

重启 Policy Server 后应看到类似：

```text
Loaded norm stats from .../49999/assets/blacknew
```

即使添加了 `asset_id`，当前 `config.py` 仍在模块导入时读取 `SO101_DATASET_DIR`，所以启动服务前仍需设置：

```bash
export SO101_DATASET_DIR=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
```

本地 Ubuntu 电脑只运行客户端，不读取训练配置和 norm stats，因此本地副本无需为云端推理而复制该 JSON。

## 24. 真机部署排错与安全清单

### 24.1 推荐启动顺序

1. 云端启动 `serve_policy.py`。
2. 确认 `params` 和 `assets/blacknew/norm_stats.json` 加载成功。
3. 本地建立 SSH 隧道。
4. 用 `nc -zv 127.0.0.1 5000` 检查端口。
5. 确认 SO-101 串口和两个相机编号。
6. 确认机械臂完成校准并位于安全初始姿态。
7. 先使用 `--max_run_sec 10`、低 `dq_limit` 运行。
8. 查看客户端 RTT、结果年龄、动作 shape 和 stale 拒绝日志。
9. 多次短时测试稳定后再逐步增加运行时间。

### 24.2 常见问题

**客户端无法连接 `127.0.0.1:5000`**

- 检查云端 Policy Server 是否仍在运行。
- 检查 SSH 隧道终端是否关闭。
- 检查 AutoDL SSH host/port 是否填写正确。
- 检查本地 5000 端口是否已被占用。

**云端报 `SO101_DATASET_DIR` 或 `KeyError`**

启动前设置：

```bash
export SO101_DATASET_DIR=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
```

**云端报 `Norm stats file not found`**

- 检查配置是否包含 `AssetsConfig(asset_id="blacknew")`。
- 检查 `49999/assets/blacknew/norm_stats.json` 是否存在。
- 不得随便使用其他 checkpoint 或其他数据集的 stats。

**客户端无法访问 `/dev/ttyACM0`**

- 检查用户是否属于 `dialout`。
- 重新插拔 USB。
- 使用 `lerobot-find-port` 重新确认设备名。
- 执行用户组修改后必须重新登录。

**两个相机打不开或顺序颠倒**

- 使用 `lerobot-find-cameras opencv` 查编号。
- 确保 `cam_top` 对应训练时的环境视角。
- 确保 `cam_wrist` 对应训练时的腕部视角。
- USB 重插后 `/dev/videoN` 编号可能变化。

**机械臂动作方向或幅度异常**

- 立即停止或断电。
- 核对 follower 校准。
- 核对前 5 关节为 degrees、夹爪为 0～100。
- 核对训练和推理使用同一 norm stats。
- 核对关节顺序与训练字段完全一致。
- 不得通过简单放宽限速继续尝试。

**模型服务正常但任务成功率低**

- 当前权重使用前人的 `blacknew` 数据，不是本机新采数据。
- 检查相机高度、角度、左右方向和训练时是否相近。
- 检查桌面背景、光照、黑色方块、白色杯子及初始姿态分布。
- 检查网络延迟是否导致动作块过期。
- 依次评测 5k～49999 checkpoint，最终 step 不一定实机最好。
- 若视觉和机器人分布差异明显，应采集本机数据继续微调。

### 24.3 当前部署状态

截至 2026-07-24：

- 已完成 50k-step OpenPI π0.5 LoRA 训练。
- 已确认推荐部署架构为云端推理、本地控制。
- 已确认本地 Ubuntu 路径为 `/home/yc/working_base/VLA/custom_vla`。
- 已确认本地应使用 Python 3.12 的独立 SO-101 客户端环境。
- 已定位最终 checkpoint `assets` 为空和绝对 `asset_id` 的根因。
- 尚未实际完成本地环境安装、SO-101 校准、SSH 隧道联调和第一次短时真机运行。

## 25. AutoDL SSH 参数与 uv run 重复安装问题

### 25.1 当前 AutoDL SSH 信息

登录命令：

~~~bash
ssh -p 36130 root@connect.westd.seetacloud.com
~~~

在 Ubuntu 本地电脑建立推理端口隧道时，必须在本地终端执行：

~~~bash
ssh -N \
  -L 5000:127.0.0.1:5000 \
  -p 36130 \
  root@connect.westd.seetacloud.com
~~~

该命令把本地 127.0.0.1:5000 转发到 AutoDL 内部的 127.0.0.1:5000。隧道终端没有正常输出是正常现象，必须保持开启。另开本地终端验证：

~~~bash
nc -zv 127.0.0.1 5000
~~~

### 25.2 为什么 uv run 又开始安装依赖

训练时实际使用的环境是：

~~~text
/root/autodl-tmp/envs/pi0_env
~~~

其中已经安装 JAX、Flax、Orbax、Tyro、WebSocket 和 OpenPI Client。2026-07-26 已重新验证这些关键包都可以导入，JAX 版本为 0.5.3。

但在项目根目录执行 uv run 时，uv 默认按当前项目的 pyproject.toml 和 uv.lock 管理独立项目环境：

~~~text
/root/autodl-tmp/VLA/custom_vla/openpi/.venv
~~~

它不会自动把训练用 Conda 环境当作已同步的项目环境。只要 .venv 不存在或不完整，uv run 就会自动解析、下载和安装固定版本依赖及 Git 形式的 LeRobot 依赖。网络较慢时看起来像“以前没有安装”，实际是另一个 Python 环境在重复安装。

本次中止后的检查结果：

~~~text
没有残留 uv、pip 或 serve_policy 进程
custom_vla/openpi/.venv 仅约 72KB，是未完成的新环境
pi0_env Python 3.11.15 正常
pi0_env 的关键推理依赖导入正常
~~~

因此最快方案不是继续安装，而是忽略该 .venv，直接复用训练环境。

### 25.3 不重新安装的云端启动命令

在 AutoDL 登录终端中执行：

~~~bash
tmux new -s openpi_server

conda activate /root/autodl-tmp/envs/pi0_env
cd /root/autodl-tmp/VLA/custom_vla/openpi

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export SO101_DATASET_DIR=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
export OPENPI_DATA_HOME=/root/autodl-tmp/cache/openpi
export JAX_COMPILATION_CACHE_DIR=/root/autodl-tmp/cache/jax
export PYTHONPATH=/root/autodl-tmp/VLA/custom_vla/openpi/src

/root/autodl-tmp/envs/pi0_env/bin/python -u scripts/serve_policy.py \
  --port 5000 \
  policy:checkpoint \
  --policy.config pi05_so101_lora \
  --policy.dir /root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999
~~~

显式使用绝对 Python 路径可以避免 shell 环境或 .venv 选择错误。

注意：

- policy.dir 必须是一条完整路径，不能在 pi05_so101_lora/ 后直接换行；如需换行，上一行末尾必须有反斜杠。
- 命令结尾不能带中文或英文逗号。
- 首次加载 8.7GB checkpoint 和首次 JAX 推理编译可能需要等待，但日志应是 Loading model、checkpoint restore 或 JAX compile，而不是 Resolving、Downloading、Installing packages。
- 服务启动失败时先看完整报错，不要同时重复启动多个安装或服务进程。

### 25.4 如果确实缺少单个包

当前关键依赖已经存在，不需要批量重装。如果以后明确报某一个 ModuleNotFoundError，可以把缺失包装入已有训练环境：

~~~bash
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv

uv pip install \
  --python /root/autodl-tmp/envs/pi0_env/bin/python \
  <missing-package>
~~~

该方式保留 uv 的下载和缓存能力，但不会创建并同步整个项目 .venv。只有看到明确缺包报错时才执行，不应预先重装全部依赖。

### 25.5 后续操作顺序

1. AutoDL 使用上述绝对 Python 命令启动 Policy Server。
2. 确认模型和 norm stats 加载成功，并看到服务监听 5000 端口。
3. Ubuntu 本地另开终端，执行准确的 SSH 隧道命令。
4. Ubuntu 本地用 nc -zv 127.0.0.1 5000 验证隧道。
5. 激活本地 .venv-so101。
6. 先运行 10 秒、低限速的 SO-101 自适应客户端。

### 25.6 Bash 手动换行导致 checkpoint 路径和子命令丢失

实际出现过两类报错：

~~~text
FileNotFoundError: ... /checkpoints/pi05_so101_lora/params/_METADATA
-bash: blacknew_lora_50k_v1/49999: No such file or directory
~~~

这表示 policy.dir 在 pi05_so101_lora/ 后被手动回车截断，Python 只收到父目录，剩余路径被 Bash 当成另一条命令。

另一类：

~~~text
ValueError: Config 'pi0_aloha_sim' not found
-bash: policy:checkpoint: command not found
~~~

这表示在 --port=5000 后手动回车，Python 没有收到 policy:checkpoint，因而使用默认 ALOHA_SIM；下一行又被 Bash 当成新命令。

为避免长命令被人为拆断，先逐行设置短变量并验证 checkpoint：

~~~bash
cd /root/autodl-tmp/VLA/custom_vla/openpi

PY=/root/autodl-tmp/envs/pi0_env/bin/python
CKPT=/root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999

test -f "$CKPT/params/_METADATA" && echo CHECKPOINT_OK
~~~

必须看到 CHECKPOINT_OK。然后把下面最后一条作为完整逻辑行执行；终端自动视觉折行不影响命令，但不能在中间按 Enter：

~~~bash
"$PY" -u scripts/serve_policy.py --port 5000 policy:checkpoint --policy.config pi05_so101_lora --policy.dir "$CKPT"
~~~

如果确实需要人工分成多行，每个未结束行末尾都必须是反斜杠，且反斜杠后不能有空格。

### 25.7 nc 端口测试触发 WebSocket InvalidMessage

实际日志：

~~~text
EOFError: connection closed while reading HTTP request line
websockets.exceptions.InvalidMessage: did not receive a valid HTTP request
~~~

原因不是模型或 checkpoint 崩溃，而是 nc -zv 只建立普通 TCP 连接后立即关闭，没有发送 WebSocket 所要求的 HTTP Upgrade 握手。WebSocket Server 会为该无效连接打印 traceback，但主服务继续运行。

2026-07-26 现场检查结果：

~~~text
serve_policy.py PID：368934
GPU 显存占用：约 83,260 MiB
checkpoint：blacknew_lora_50k_v1/49999
服务进程仍然运行
~~~

判断服务是否退出应看：

- serve_policy.py 进程是否仍存在。
- 推理终端是否回到 Bash 提示符。
- GPU 上是否仍有该 Python 进程。
- 正式 OpenPI Client 能否完成 WebSocket 握手并取得 metadata。

nc 只能粗略证明 TCP 端口可以连接，测试后产生上述 InvalidMessage 可以忽略。更准确的本地测试是在 .venv-so101 中运行：

~~~bash
python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy; c=WebsocketClientPolicy('127.0.0.1', 5000); print(c.get_server_metadata())"
~~~

执行该命令前必须保持 SSH 隧道开启。如果能打印 metadata，说明 SSH、WebSocket 和 Policy Server 三层均已打通。

### 25.8 将 Policy Server 从 5000 切换到 6006

在 base 环境执行 uv run 可能报：

~~~text
-bash: uv: command not found
~~~

原因是 uv 不在当前 shell 的 PATH；而且本项目不应再用 uv run 创建项目 .venv，应继续使用已有训练环境的绝对 Python。

切换端口前先确认旧 serve_policy.py 已停止，避免同一 GPU 同时加载两份约 83GB 的模型。2026-07-26 检查时旧进程已停止，6006 端口空闲。

云端启动命令：

~~~bash
cd /root/autodl-tmp/VLA/custom_vla/openpi

export SO101_DATASET_DIR=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
export PYTHONPATH=/root/autodl-tmp/VLA/custom_vla/openpi/src
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

PY=/root/autodl-tmp/envs/pi0_env/bin/python
CKPT=/root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999

"$PY" -u scripts/serve_policy.py --port 6006 policy:checkpoint --policy.config pi05_so101_lora --policy.dir "$CKPT"
~~~

本地 SSH 隧道同步改为：

~~~bash
ssh -N -L 6006:127.0.0.1:6006 -p 36130 root@connect.westd.seetacloud.com
~~~

本地 WebSocket 测试和真机客户端都必须使用 6006：

~~~bash
python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy; c=WebsocketClientPolicy('127.0.0.1', 6006); print(c.get_server_metadata())"
~~~

~~~bash
python openpi/packages/openpi-client/src/openpi_client/zpf_pi0_so101_client_adaptive_pro_v4.py --host 127.0.0.1 --port 6006 ...
~~~

三处必须一致：

~~~text
云端 serve_policy.py：6006
SSH 本地转发：6006 -> 6006
本地 OpenPI Client：6006
~~~

## 26. 2026-07-27：数据、配置代码阅读与工程整理

> 记录时间：2026-07-27 13:01:25 UTC

今天的目标不是重新训练模型，而是从已有工程反向理解一条完整的 VLA 链路：LeRobot v3 数据如何组织、OpenPI 如何把数据转换成模型输入、LIBERO 与 SO-101 配置有什么差别，以及如何安全整理磁盘和归档源码。

### 26.1 系统盘 100%：定位与清理

问题：系统盘 / 为 30GB，只剩约 26MB，使用率达到 100%。这会导致 Python/uv 无法写缓存、Git 操作失败、训练或推理进程异常。

排查结果：主要可清理项不是 checkpoint 或数据集，而是系统盘缓存：

~~~text
/root/.cache/uv                 约 7.2GB
/root/.cache/pip                约 106MB
/root/.cache/ov/texturecache   约 449MB
~~~

处理：删除上述可再生成缓存，以及本次分析留下的少量 /tmp 临时文件。没有删除项目源码、数据集、模型权重、环境或 checkpoint。

结果：

~~~text
清理前：30GB 已用 30GB，可用 26MB，使用率 100%
清理后：30GB 已用 23GB，可用 7.7GB，使用率 75%
~~~

经验：不要在系统盘直接执行会自动解析整套依赖的 uv run。本项目已有训练环境时，应优先使用 /root/autodl-tmp/envs/pi0_env/bin/python；确需 uv 时把 UV_CACHE_DIR 指向数据盘。

### 26.2 blacknew LeRobot v3 数据集如何理解

数据集路径：

~~~text
/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
~~~

检查结果：50 个 episode、31,000 条表格记录、1 个任务、30 FPS，包含环境相机和手腕相机两个 AV1 视频流。meta/info.json 描述全局 schema 和路径模板，meta/tasks.parquet 保存任务文本，meta/episodes 保存每个 episode 的边界和统计，meta/stats.json 保存数据集统计；真正的逐帧 state/action/index 在 data/*.parquet，图像像素在 videos/**/*.mp4。

这份数据更像是由 LeRobot 的 lerobot-record 直接采集并写成 v3 格式，而不是在本项目中通过某个“原始格式转 LeRobot”脚本离线转换。关键调用链是：

~~~text
lerobot-record
  -> lerobot/scripts/lerobot_record.py
  -> LeRobotDataset.create / add_frame / save_episode
  -> datasets/dataset_writer.py
  -> parquet + episode metadata + video 编码
~~~

已补充的阅读材料和工具：

~~~text
DOCS/blacknew_dataset_guide_zh.md
tools/inspect_lerobot_v3.py
~~~

还发现视频帧数与 parquet 行数并非完全一致：总视频帧数 31,411，而数据表为 31,000 行，差异集中在 episode 7 和 18。训练以数据表索引和时间戳采样视频，后续若出现末尾画面偏移，应优先检查这两个 episode。

### 26.3 config.py 的结构与 pi05_libero

核心文件：

~~~text
custom_vla/openpi/src/openpi/training/config.py
~~~

代码结构可以按下面顺序阅读：

~~~text
模型配置
  -> DataConfig / DataConfigFactory
  -> 机器人专用 DataConfigFactory.create()
  -> TrainConfig
  -> _CONFIGS 配置注册表
  -> get_config(name)
~~~

TrainConfig 同时描述模型结构、初始权重、优化器、学习率、冻结策略、数据变换、batch、训练步数、checkpoint、W&B 和 FSDP。name="pi05_libero" 的数据流为：

~~~text
LeRobot 原始字段
  -> RepackTransform（字段重命名）
  -> LiberoInputs（转换为 OpenPI Observation）
  -> Normalize
  -> ResizeImages(224, 224)
  -> TokenizePrompt
  -> PadStatesAndActions
  -> π0.5
~~~

推理输出反向经过：

~~~text
π0.5 的 [T, action_dim]
  -> Unnormalize
  -> LiberoOutputs
  -> LIBERO 需要的 [T, 7]
~~~

RepackTransform 的映射方向是“新键 -> 原始数据中的旧键”。LiberoInputs 在训练和推理时都负责把机器人数据适配成统一结构；LiberoOutputs 只在推理端把通用动作裁回 LIBERO 动作维度；ModelTransformFactory 负责图像缩放、prompt/state token 化和维度补齐。

### 26.4 SO-101 配置与 policy 数据流

核心文件：

~~~text
custom_vla/openpi/src/openpi/training/config.py
custom_vla/openpi/src/openpi/policies/so101_policy.py
~~~

LeRobotSO101DataConfig.create() 先把 LeRobot 字段映射为项目约定的字段：

~~~text
observation.images.env  -> observation.images.images_env
observation.images.hand -> observation.images.images_wrist
observation.state       -> observation.state
action                  -> action
~~~

随后 SO101Inputs 构造统一的双相机、state、prompt 和训练 action。前五个关节动作使用 delta 表示，第六维 gripper 保持绝对值；SO101Outputs 在推理端配合逆变换恢复六维绝对目标动作。π0.5 的内部 action 张量仍会 padding 到配置的 action_dim=32，机器人最终只消费前 6 维。

LIBERO 与 SO-101 的关键区别：LIBERO 原始 action 已符合其相对动作语义，不额外做 delta；SO-101 对前五维使用 DeltaActions，推理时通过 AbsoluteActions 加回当前 state。

完整逐段说明见：

~~~text
DOCS/openpi_config_guide_zh.md
~~~

### 26.5 阅读代码时发现的问题

1. config.py 中 _CONFIGS 被定义了两次，第二次使用 _CONFIGS = [...] 覆盖第一次注册表。因此当前实际可查到 SO-101/SYSMO 配置，但 get_config("pi05_libero") 会失败。若希望官方配置和自定义配置同时可用，应把第二次赋值改为追加或统一成一个注册表。
2. SO-101 图像噪声变换目前追加在 SO101Inputs 之后，但噪声插件读取的是变换前的顶层图像键，因此即使开启开关也可能不生效。目前开关为 False，不影响现有训练结果。
3. TASK_INDEX_TO_PROMPT 已定义但当前没有参与 prompt 选择，实际使用的是默认 prompt 或数据中的 task。
4. 当前训练查找 norm stats 的路径由 assets_base_dir / config.name / asset_id 共同决定。以 pi05_so101_lora 和 asset_id="blacknew" 为例，需要准备：

~~~text
/root/autodl-tmp/VLA/assets/pi05_so101_lora/blacknew/norm_stats.json
~~~

checkpoint 推理则从 checkpoint 自带的 assets/blacknew/norm_stats.json 加载，所以已有 checkpoint 可以独立推理。

### 26.6 已添加的中文注释和验证

为便于逐行学习，已在难点处增加中文注释：

~~~text
custom_vla/openpi/src/openpi/training/config.py
custom_vla/openpi/src/openpi/policies/libero_policy.py
custom_vla/openpi/src/openpi/policies/so101_policy.py
lerobot/src/lerobot/scripts/lerobot_record.py
lerobot/src/lerobot/datasets/dataset_writer.py
~~~

同时修复了 so101_policy._parse_image() 在坏视频占位分支中引用未定义变量 name 的明确 bug，改为显式传入相机键名。验证结果：

~~~text
三个 OpenPI Python 文件 py_compile 通过
LIBERO repack / input / output smoke test 通过
SO-101 repack / delta action / inverse output smoke test 通过
git diff --check 通过
~~~

SO-101 的数值验证示例：

~~~text
state:           [1, 2, 3, 4, 5, 6]
absolute action: [2, 4, 6, 8, 10, 9]
模型训练 action: [1, 2, 3, 4, 5, 9]
逆变换后 action: [2, 4, 6, 8, 10, 9]
~~~

### 26.7 GitHub 源码归档策略

目标仓库：ycyc0926/VLA-pi0.5-SO101，目标分支：main。

本次归档包含：

- custom_vla/ 中实际训练和推理使用的修改版代码。
- 根目录官方 openpi/ 与 lerobot/ 工作树的源码快照。
- DOCS/ 中文学习记录、根 README 和 tools/ 数据检查工具。
- uv.lock 等可复现依赖版本文件。

明确排除：

- checkpoints/、模型权重和训练参数文件。
- datasets/、parquet、视频和其他采集数据。
- logs/、wandb/、缓存、临时目录和本地虚拟环境。
- STL/USD/GLB 等大型机械 CAD 资产和压缩包。
- 各嵌套仓库自己的 .git 元数据。

这种方式保留可阅读、可比较、可复现安装的源码，同时避免 GitHub 单文件大小限制和仓库体积失控。

### 26.8 提交前凭据扫描

问题：提交前扫描在 custom_vla/lerobot/mergedModel/scripts/hfDownload.py 中发现硬编码 Hugging Face token。通过 blob 哈希对比确认，该内容已经存在于远端初始提交 600a327 中。

当前代码处理：

~~~python
HF_TOKEN = os.environ.get("HF_TOKEN")
~~~

使用前在 shell 中设置：

~~~bash
export HF_TOKEN=<新令牌>
~~~

注意：后续普通 commit 只能删除当前版本中的 token，不能清除旧 Git 历史。旧 token 必须立即在 Hugging Face 设置中撤销并重新创建；如果还要从远端历史彻底移除，需要另行执行历史重写和 force-push，不能在未确认协作者状态时擅自操作。
