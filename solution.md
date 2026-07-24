# π0.5 + SO-101 VLA 项目实践记录

> 用途：持续记录本项目做过的工作、遇到的问题、解决方案、实验结果和后续计划，便于复盘、撰写简历和准备 VLA 岗位面试。
>
> 最近更新：2026-07-24

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
