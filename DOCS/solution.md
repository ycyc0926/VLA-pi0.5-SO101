# π0.5 + SO-101 VLA 项目实践记录

> 用途：记录本项目从数据、训练到真机部署的完整过程，便于工程复盘、代码复现、简历整理和 VLA 岗位面试。
>
> 最近更新：2026-08-03 UTC
>
> 阅读说明：第 2～26 节保留项目推进过程中的阶段性记录；其中部分“下一步”后来已经完成。最终状态、结果边界和收尾结论以第 1 节与第 27 节为准。

## 1. 项目概览与最终状态

本项目已经完成 PI0.5 在 SO101 上从数据处理/采集、LoRA 微调、checkpoint 管理、策略服务到真实机械臂闭环执行的完整工程流程。工作分为两个阶段：

| 阶段 | 任务 | 数据来源 | 最终结论 |
| --- | --- | --- | --- |
| 物块放入杯子 | 抓取物块并放进杯子 | 同事已有数据 | 固定场景可完成，但改变物块或杯子位置后明显失败，暴露轨迹记忆和数据覆盖不足 |
| 长程抽屉任务 | 开抽屉，依次放入黑/白物块，再关抽屉 | 单物块数据/checkpoint 来自同事；双物块数据由本人设计并采集 | 同一 checkpoint 可按指令执行单/双物块任务，双物块任务完成真机闭环验证 |

最终完成的主要工作：

- 在 AutoDL 上部署训练环境，完成第一阶段 50,000-step PI0.5 LoRA 微调、W&B 监控和 Orbax checkpoint 管理。
- 定位 AutoDL 公网推理 1～3 秒延迟无法满足 10-step action chunk 实时性的原因，将训练/推理迁移到公司局域网 GPU 服务器。
- 在固定场景中完成物块放杯子真机推理，并通过位置扰动实验确认其泛化性不足。
- 自行设计并采集 30 条、40,355 帧双物块抽屉数据；采集时随机化黑白物块位置。
- 对齐不同采集者的 follower calibration，转换 state/action，合并单/双物块数据并重新计算 mixture norm。
- 从已验证的单物块 checkpoint `35000` 参数 warm-start，以 fresh optimizer 完成新的 50,000-step 混合任务训练，保存最终 checkpoint `49999`。
- 构建本地双相机/SO101 与公司 GPU 服务器之间的 WebSocket 闭环推理系统，完成 action chunk 延迟对齐、stale 拒绝、平滑限幅和安全 Hold。
- 通过不同 prompt 基本完成单物块和双物块两种抽屉任务；双物块关闭阶段偶尔需要多次调整后才成功。

当前结果边界：

- 已经证明工程链路可用，并完成双物块长程任务的端到端实机执行。
- 物块位置随机化后的抓取和关抽屉失败后的继续调整，说明策略不是一次固定开环轨迹播放，并呈现一定闭环视觉反馈。
- 尚未完成固定时限、随机初始条件下的 20～30 次标准化成功率统计，因此不能宣称稳定成功率或充分泛化。
- 随机化抽屉位置/接触点、补采失败恢复数据和 RL 后训练均属于后续方案，本项目收尾时尚未正式实施。

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

> 口径说明：本节记录的是第一阶段杯子任务所用 `pi05_so101_lora` 配置当时的修正，不应推广为所有 PI0.5 checkpoint 的固定设置。后期 `drawer_one_two_blocks` 混合配置明确使用 `discrete_state_input=False`。两者属于不同实验配置；面试或复现时必须以目标 checkpoint 对应的 model config、数据变换和代码版本为准。

文件：

```text
src/openpi/training/config.py
```

目标配置：

```python
name="pi05_so101_lora"
discrete_state_input=True
```

当时的判断是：该早期配置按离散 token 路径接入 state，因此需要打开此开关。后续代码和混合任务配置采用了不同设置，不能仅依据 `pi05=True` 推导该布尔值。

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

## 15. 第一阶段当时的下一步实验计划

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

## 16. 第一阶段阶段性简历表述

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

## 17. 第一阶段阶段性面试讲解框架

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

## 19. 工程归档状态提醒

本节原记录的少量文件列表已经不再代表最终工程状态。项目收尾时，工作区同时保留了 OpenPI/客户端阅读注释、推理改动、中文文档以及 `drawer_one_two_blocks/` 训练增量代码。

归档和提交时应遵守：

- 先检查 `git status` 和 `git diff`，按数据处理、训练、推理、文档拆分提交。
- 不把 datasets、checkpoint、optimizer state、训练日志、W&B 本地文件、校准原文件或凭据加入 Git。
- 以 `drawer_one_two_blocks/solution.md` 中记录的 config、asset、filter 和 checkpoint 口径作为混合训练复现依据。
- 最终项目状态和能力边界以第 27 节为准。
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

## 27. 2026-08-03：项目收尾总结

### 27.1 一句话总结

在 SO101 平台上完成 PI0.5 从数据采集/处理、跨任务 LoRA warm-start 到局域网真机闭环部署的完整链路：先通过物块放杯子任务识别固定数据造成的轨迹记忆，再自行采集位置随机化的双物块数据，将已有单物块抽屉策略扩展为“开抽屉—依次放入黑白物块—关抽屉”的长程语言条件任务。

### 27.2 第一阶段：物块放入杯子

第一阶段使用同事已经采集的 `AlexFeng1/blacknew` 数据。数据包含 50 个 episode、31,000 帧、环境/腕部两路相机和 6 维 state/action。本人负责训练环境部署、PI0.5 LoRA 微调、checkpoint 服务和 SO101 客户端接入。

在 AutoDL 上模型能够加载和返回动作，但公网 TCP 映射下单轮推理约 1～3 秒。模型每次返回 10 步动作，30 Hz 下理论覆盖时间只有：

~~~text
10 / 30 = 0.333 秒
~~~

因此公网动作到达时往往已经过期。将训练与推理迁移到公司局域网 GPU 服务器后，固定场景中的物块放杯子任务能够完成。

真正重要的失败发生在泛化测试：稍微改变物块或杯子的位置，任务就很难完成，机械臂仍倾向执行与示范相近的一条轨迹。这说明固定初始状态的数据让策略学到了场景捷径和轨迹记忆，不能因为一次成功就认为模型学会了“抓取并放入”这个任务。

该阶段直接影响了下一阶段的数据设计：采集时必须刻意改变目标物体的位置，并在训练分布内验证插值位置和边界位置。

### 27.3 第二阶段：双物块长程抽屉任务

同事已有的基础任务为：

~~~text
Open the drawer, place the block inside, and close the drawer
~~~

本人在此基础上扩展并采集双物块任务：

~~~text
Open the drawer, put the black block into the drawer,
then put the white block into the drawer, and close the drawer.
~~~

完整子任务顺序为：

1. 拉开抽屉。
2. 抓取黑色物块并放入抽屉。
3. 抓取白色物块并放入抽屉。
4. 关闭抽屉。

双物块数据由本人通过 leader/follower 遥操作采集，共 30 个 episode、40,355 帧、30 FPS。每条 episode 包含环境相机、腕部相机、6 维关节状态、6 维动作和任务文本。黑白物块在相机可见、机械臂可达范围内随机摆放，目标是迫使策略依据视觉定位物块，而不是复现固定关节轨迹。

个人贡献边界如下：

| 内容 | 来源/负责人 |
| --- | --- |
| 物块放杯子原始数据 | 同事采集；本人负责训练、部署、推理和泛化诊断 |
| 单物块抽屉数据与 35k checkpoint | 同事已有任务 |
| 双物块任务设计与数据 | 本人设计并采集 |
| calibration 对齐、数据合并、filter、norm | 本人完成 |
| 混合任务 warm-start 训练与故障恢复 | 本人完成 |
| 局域网策略服务、SO101 客户端和实机验证 | 本人完成 |

### 27.4 数据与训练闭环

最终数据链路为：

~~~text
遥操作采集双物块数据
        ↓
检查 LeRobot v3 schema、视频、任务文本和 episode 边界
        ↓
将新数据从采集校准系转换到旧单物块 calibration
        ↓
同时转换 observation.state 与 action，并重建统计
        ↓
合并 32 条单物块 + 30 条双物块数据
        ↓
保留两条 task prompt，构建 mixed sample-start filter
        ↓
按 10-step horizon、前五轴 delta、gripper absolute 重算 norm
        ↓
从单物块 35000/params warm-start，使用 fresh optimizer
        ↓
完成 50000-step 混合 LoRA 训练，保存 checkpoint 49999
~~~

合并数据统计：

| 项目 | 单物块 | 双物块 | 合计 |
| --- | ---: | ---: | ---: |
| Episodes | 32 | 30 | 62 |
| Frames | 33,478 | 40,355 | 73,833 |
| Task prompts | 1 | 1 | 2 |

mixed filter 保留 71,876 个合法 sample starts；训练实际使用与 `drop_last` 对齐的 71,872 个起点。配置名为：

~~~text
pi05_so101_drawer_one_two_blocks_calib2_lora
~~~

关键参数为 batch size 16、action horizon 10、1,000-step warmup、peak learning rate `2e-5`、cosine decay 到 `1e-7`、gradient clipping 0.5。

“从 35k 训练到约 8 万步”是便于叙述的累计训练量口径。严格工程口径是：

- `35000` 是旧单物块实验中用于初始化的模型参数。
- 新混合任务改变了数据分布、filter 和 norm，因此不能直接恢复旧 optimizer。
- 新实验 step 从 0 开始，完成 50,000 steps，最终目录为 `49999`。
- 模型承接的训练量约为 35k + 50k，即 8 万步量级；这不是同一实验从 step 35000 resume 到 step 80000。

详细可复现过程、脚本和服务器路径见 `drawer_one_two_blocks/solution.md`。

### 27.5 真机部署与结果

最终使用局域网客户端/服务器架构：

~~~text
本地 Ubuntu：
  双 USB 相机 + SO101 follower + 30 Hz Body 控制
                         │
                         │ WebSocket
                         ▼
公司 GPU 服务器：
  PI0.5 LoRA checkpoint + Policy Server
~~~

客户端持续发送最新双目图像、关节状态和语言指令；服务端返回 `(10, 6)` action chunk。客户端根据图像年龄和 RTT 跳过已经过期的前缀动作，并实现动作队列、stale action 拒绝、低通平滑、关节/夹爪限幅、异常检查与安全 Hold。

一次 90 秒双物块运行的记录为：

| 指标 | 结果 |
| --- | ---: |
| 服务端推理中位延迟 | 约 65.9 ms |
| 端到端 RTT 中位数 | 约 89.9 ms |
| 动作结果年龄中位数 | 约 131.4 ms |
| 接受动作块 | 776 |
| stale 拒绝 | 1 |
| Body missed ticks | 0 |

同一个 checkpoint 基本可以通过两条不同 prompt 完成单物块和双物块任务。双物块任务已经完整执行到最终关闭抽屉。

主要薄弱点是关闭阶段：夹爪有时先推到抽屉上方或接触位置不理想，需要重新定位几次后才能关上。对此应同时给出两方面判断：

- 积极证据：物块位置存在变化时仍能抓取，首次关闭失败后会依据后续观测产生不同动作，说明不是一次固定开环轨迹播放，并呈现闭环反馈和一定恢复行为。
- 能力边界：多次重试降低首次成功率和执行效率；单次“最终成功”不足以证明稳定自纠错，更不能表述为模型理解了失败。

### 27.6 项目中解决的关键问题

| 问题 | 诊断与处理 |
| --- | --- |
| AutoDL 能连接但真机闭环不稳定 | 公网 1～3 秒延迟超过约 333 ms action horizon；迁移到公司局域网 |
| 第一阶段固定位置成功、位置扰动失败 | 判定为数据覆盖不足和轨迹记忆；第二阶段主动随机化物块位置 |
| 两批 SO101 数据不能直接合并 | 不同 calibration 使同一物理姿态数值不一致；统一转换 state/action |
| 两个任务可能混淆 | 保留 LeRobot task 表并启用 `prompt_from_task=True` |
| 加入新任务后如何继续训练 | 参数 warm-start + fresh optimizer；只在同一混合实验中断时使用 `--resume` |
| norm 计算慢且可能与训练语义不一致 | 直接读取 parquet，并严格复现 filter、horizon、delta mask 和 drop-last |
| GPU 参数驻留但利用率为 0 | 定位 JAX/XLA/CUDA 主线程自旋，从最近完整 checkpoint 4000 恢复 |
| 推理返回动作但机械臂不明显运动 | 分层检查电源/扭矩/校准后，定位短运行时限、过强限幅和平滑参数 |
| 相机启动黑屏 | 当前 checkpoint 存在潜在视觉捷径；部署时兼容，后续数据应清理 |
| 抽屉关闭会重复尝试 | 作为闭环恢复现象记录，同时明确接触数据和空间覆盖仍不足 |

### 27.7 最重要的工程认识

1. 数据覆盖决定模型学的是任务还是轨迹。增加训练步数不能弥补固定初始状态。
2. 数据合并不是简单拼 parquet。calibration、schema、task prompt、动作语义、filter 和 norm 必须全部对齐。
3. Warm-start 与 resume 的语义不同。任务 mixture 变化时恢复旧 optimizer 会把旧训练状态错误带入新实验。
4. 训练 loss 只能说明动作拟合过程稳定，不能替代真机闭环评测。
5. 网络可达不等于控制可用。RTT、图像年龄、action horizon 和队列策略共同决定远程闭环是否实时。
6. 真机问题要按模型、数据、网络、客户端控制和机械硬件分层诊断，不能看到失败就只继续训练。
7. 恢复后成功应单独统计。它体现闭环反馈的价值，也会掩盖首次执行可靠性不足。

### 27.8 尚未实施的改进方案

本项目在完成端到端闭环后收尾，以下是明确的后续思路，而不是已经取得的结果：

- 随机化抽屉相对机械臂的左右/前后位置、开合距离和轻微角度。
- 随机化夹爪推抽屉的高度、横向接触点和推入方向。
- 补采首次推偏、顶到抽屉上缘、没有完全关严等失败状态下的恢复示范。
- 清除 episode 开头的黑屏和曝光异常帧，避免模型依赖启动特征。
- 为开抽屉、黑块、白块、首次关闭、恢复后关闭分别建立指标。
- 在固定 90 或 120 秒时限下至少测试 20～30 次，并覆盖训练分布内、插值和边界位置。
- 在更充分、更多样的行为克隆数据基础上尝试 RL 后训练，优化空间泛化和接触密集阶段的首次成功率。

其中抽屉/接触点随机化、恢复数据和 RL 后训练均尚未正式执行，不能写入已完成成果。

### 27.9 最终交付物索引

| 路径 | 内容 |
| --- | --- |
| `README.md` | 项目入口、最终结果和仓库导航 |
| `DOCS/solution.md` | 从 AutoDL 训练到项目收尾的完整时间线 |
| `DOCS/solution_local.md` | 本地/局域网推理、机械臂接入和实机排错 |
| `drawer_one_two_blocks/solution.md` | 校准、合并、norm、训练、W&B 和 checkpoint 细节 |
| `drawer_one_two_blocks/openpi/src/openpi/training/config.py` | 单/双物块混合训练配置 |
| `drawer_one_two_blocks/openpi/src/openpi/training/data_loader.py` | episode range 样本过滤 |
| `drawer_one_two_blocks/openpi/scripts/convert_so101_calibration.py` | calibration 坐标转换 |
| `drawer_one_two_blocks/openpi/scripts/build_mixed_lerobot_filter.py` | mixed filter 生成 |
| `drawer_one_two_blocks/openpi/scripts/compute_so101_norm_stats_fast.py` | parquet 快速 norm 统计 |
| `drawer_one_two_blocks/openpi/scripts/upload_openpi_training_log_to_wandb.py` | 训练日志旁路上传 |
| `DOCS/openpi_*_guide_zh.md` | OpenPI 数据流、模型与推理代码阅读笔记 |

### 27.10 最终项目表述

可以准确地将本项目总结为：

> 基于 PI0.5、OpenPI、LeRobot v3 和 SO101 完成从遥操作数据采集、跨 calibration 数据融合、LoRA 继续训练到局域网真实机械臂闭环部署的完整 VLA 工程。通过早期物块放杯子任务识别固定数据造成的轨迹记忆，随后自行采集位置随机化的双物块数据，将已有单物块抽屉策略扩展为单/双物块语言条件长程任务。最终完成开抽屉、依次放入黑白物块并关闭抽屉的实机验证，并观察到关闭失败后的视觉闭环调整；当前局限是首次关闭成功率和标准化随机场景评测仍待提高。

## 28. VLA 实习求职：简历与面试准备

本节不是新的实验记录，而是将前面的工程事实整理成可以直接用于简历和面试的材料。使用时应坚持三个原则：只写本人真正完成的工作；区分一次实机验证与标准化成功率；区分已经实现的改进和仍停留在方案阶段的工作。

### 28.1 面试前必须记住的项目事实

面试官经常会从简历中的一个数字继续追问。下面的事实必须能够不看文档直接回答：

| 项目 | 事实 |
| --- | --- |
| 基础模型 | PI0.5，OpenPI 实现，LoRA 微调 |
| 机器人 | SO101 follower，leader/follower 遥操作采集 |
| 视觉输入 | 环境相机 + 腕部相机 |
| 低维输入/输出 | 6 维 state、6 维 action |
| 控制频率 | 30 Hz |
| Action horizon | 10 步，理论覆盖约 333 ms |
| 杯子数据 | 同事采集，50 episodes、31,000 帧 |
| 双物块数据 | 本人采集，30 episodes、40,355 帧 |
| 混合抽屉数据 | 62 episodes、73,833 帧、2 个 task prompts |
| Mixed filter | 71,876 个合法 sample starts |
| Warm-start | 已验证单物块 checkpoint `35000/params` |
| 混合训练 | fresh optimizer，50,000 steps，最终 checkpoint `49999` |
| Batch size | 16 |
| 学习率 | warmup 1,000，peak `2e-5`，cosine decay 到 `1e-7` |
| 动作表示 | 前 5 个关节 delta，gripper absolute |
| 服务端延迟 | 中位数约 65.9 ms |
| 局域网 RTT | 中位数约 89.9 ms |
| 实机结果 | 同一 checkpoint 可按 prompt 执行单/双物块任务；双物块完成端到端验证 |
| 主要不足 | 关闭抽屉首次成功率不稳定；尚无标准化多轮完整成功率 |

需要特别避免混淆：`35000` 是旧单物块参数，`49999` 是新混合实验最终 checkpoint。新实验改变了数据 mixture、filter 和 norm，因此是参数 warm-start，不是带旧 optimizer 从 35000 原地 resume 到 80000。

### 28.2 简历项目名称

推荐名称：

```text
PI0.5 VLA 在 SO101 上的长程多任务操作与真机部署
```

备选名称：

```text
基于 PI0.5 LoRA 的 SO101 单/双物块抽屉操作系统
```

名称应同时体现模型、机器人平台、长程任务和真机部署，不建议只写“机械臂抓取项目”，否则无法突出 VLA 和数据/训练贡献。

### 28.3 简历项目描述：推荐四条版

下面这版适合 VLA、具身智能、机器人学习或多模态算法实习岗位：

**PI0.5 VLA 在 SO101 上的长程多任务操作与真机部署**

- 基于 OpenPI、LeRobot v3 与 SO101 搭建从 leader/follower 遥操作采集、双相机数据处理、PI0.5 LoRA 微调到真实机械臂部署的完整 VLA 流程；自采 30 个双物块 episodes、40,355 帧，并随机化黑白物块位置以降低固定轨迹过拟合。
- 将已有单物块抽屉策略扩展为“开抽屉—依次放入黑白物块—关抽屉”的长程语言条件任务；完成跨采集 calibration 的 state/action 对齐、62 个 episodes 数据融合、task prompt 保留、sample-range 过滤和 mixture normalization statistics 重算。
- 从已验证的 35k 单物块 checkpoint 参数 warm-start，以 fresh optimizer 完成 50k-step PI0.5 LoRA 混合训练；设计 Orbax checkpoint、故障恢复和 W&B 旁路监控流程，并定位 JAX/XLA 训练卡死、数据加载和共享服务器凭据冲突等问题。
- 构建本地 SO101/双相机与公司 GPU 服务器的 30 Hz WebSocket 视觉闭环系统，实现 action chunk 延迟对齐、stale action 拒绝、动作限幅和平滑及安全 Hold；实测服务端推理中位延迟约 66 ms、RTT 约 90 ms，完成单/双物块指令控制和双物块长程任务实机验证。

这四条分别回答：做了什么系统、解决了什么数据问题、如何训练、如何部署。顺序不要反过来，因为招聘者通常先判断任务价值，再看技术细节。

### 28.4 简历项目描述：精简三条版

当简历空间有限时使用：

- 基于 PI0.5、OpenPI、LeRobot v3 和 SO101 打通遥操作采集、LoRA 微调与真机部署闭环；自采 30 个双物块 episodes、40,355 帧，并通过位置随机化缓解固定场景轨迹记忆。
- 对齐不同采集 calibration 下的 state/action，融合 62 个单/双物块 episodes，重新构建 task prompt、sample filter 与 norm；从单物块 35k 参数 warm-start 完成 50k-step 多任务 LoRA 训练。
- 搭建 30 Hz 局域网视觉闭环推理系统，实现 action chunk 时延对齐、stale 拒绝、限幅与安全 Hold；服务端推理中位延迟约 66 ms、RTT 约 90 ms，完成双物块长程任务实机验证。

### 28.5 简历项目描述：一句话版

适合放在个人简介或投递邮件中：

> 在 SO101 上完成 PI0.5 从位置随机化数据采集、跨 calibration 数据融合、LoRA warm-start 到局域网真机闭环部署的完整 VLA 项目，并将单物块策略扩展为单/双物块语言条件长程抽屉任务。

### 28.6 简历关键词

ATS 或招聘者可能关注的关键词：

```text
VLA / Vision-Language-Action / PI0.5 / OpenPI / PaliGemma
LoRA / Flow Matching / Behavior Cloning / Action Chunking
LeRobot v3 / SO101 / Robot Learning / Imitation Learning
JAX / XLA / Orbax / W&B / WebSocket / OpenCV / PyArrow
Calibration / Normalization / Delta Action / Closed-loop Control
```

不要为了堆关键词写入没有使用过的算法，例如没有正式做过的 RL、Diffusion Policy、ACT 或 ROS2。可以在“后续方案”中讨论 RL，但不能列为已完成技术栈。

### 28.7 简历中不能这样写

以下表述会引出无法回答的追问或夸大结果：

- “实现稳定的长程自主操作”：没有标准化多轮成功率支持。
- “显著提高模型泛化率”：没有前后对照成功率和置信区间。
- “模型具备自主纠错和推理能力”：当前只观察到闭环重新调整。
- “从 35k resume 到 80k”：实际是 35k 参数 warm-start，新实验重新计步 50k。
- “独立完成所有数据采集”：单物块和杯子数据来自同事，双物块数据才是本人采集。
- “使用 RL 提高成功率”：RL 只是后续思路，尚未实施。

更稳妥的动词是“完成”“构建”“定位”“观察到”“提出方案”，而不是“稳定实现”“显著提升”“证明理解”。

### 28.8 30 秒面试介绍

> 我做的是 PI0.5 在 SO101 上的端到端 VLA 项目。前期我用同事采集的物块放杯子数据完成 LoRA 训练和真机部署，但发现稍微改变物体位置模型就失败，说明固定数据让模型记住了轨迹。之后我自行采集了 30 条带物块位置随机化的双物块数据，把已有的单物块抽屉策略扩展为开抽屉、依次放入黑白物块再关抽屉的长程任务。我完成了 calibration 对齐、数据合并、norm 重算、35k checkpoint warm-start 和局域网闭环部署。最终同一个 checkpoint 可以通过不同指令执行单/双物块任务，双物块任务完成了实机验证。

### 28.9 90 秒面试介绍

> 这个项目的目标是完整走通一次 VLA 从数据到真机的流程，而不只是跑通训练脚本。第一阶段我使用同事采集的 50 条物块放杯子数据，在 AutoDL 上对 PI0.5 做 LoRA 微调。模型能够加载和返回动作，但公网单轮推理需要 1 到 3 秒，而 10-step action chunk 在 30 Hz 下只有约 333 毫秒，所以无法稳定闭环。我迁移到公司局域网 GPU 服务器后把 RTT 降到约 90 毫秒，固定场景能够完成任务，但稍微改变物块或杯子位置就失败。这让我判断模型主要记住了固定轨迹。
>
> 第二阶段我在同事已有的单物块抽屉任务上扩展双物块长程任务，并自己采集了 30 条、40,355 帧数据，采集时随机化黑白物块位置。由于两批数据使用的 calibration 不同，我先同时转换 state 和 action，再合并成 62 个 episodes，并重新构建 prompt、filter 和 norm。从已经过真机验证的单物块 35k 参数 warm-start，用新 optimizer 完成 50k-step LoRA 训练。
>
> 部署时我把本地双相机和 SO101 与公司 GPU 策略服务器通过 WebSocket 连接，客户端处理 action chunk 的延迟对齐、stale 拒绝、平滑限幅和安全 Hold。实测服务端推理约 66 毫秒、RTT 约 90 毫秒。最终同一个 checkpoint 可以通过不同指令执行单物块和双物块任务。双物块任务可以完整完成，关闭抽屉偶尔需要多次调整，所以我把它描述为有一定闭环恢复表现，但不宣称已经达到稳定成功率。

### 28.10 五分钟项目介绍框架

五分钟介绍按“动机—问题—方案—结果—反思”展开，不要从安装环境开始讲。

#### 第一部分：动机与任务，约 40 秒

> 我希望完整理解 VLA 在真实机器人上的数据、模型和控制闭环，因此选择 PI0.5 和 SO101。项目包含物块放杯子和单/双物块抽屉两个阶段，第二阶段的任务有四个连续子任务，能够检验视觉定位、语言条件、长程动作衔接和接触操作。

#### 第二部分：第一阶段失败带来的数据认识，约 50 秒

> 杯子任务在固定场景成功，但物体位置变化后失败。由于机械臂仍执行相似动作，我判断它主要学习了训练分布内的轨迹，而不是稳定的目标级策略。这个失败让我在第二阶段采集时主动随机化黑白物块位置，并把“位置变化后的表现”作为判断是否只记轨迹的依据。

#### 第三部分：数据与训练，约 90 秒

> 双物块数据由 leader/follower 遥操作采集，包含环境和腕部相机、6 维 state/action 与任务文本。合并同事的单物块数据时，最大的问题不是文件格式，而是 calibration 不同。同一个物理姿态在两个坐标系中的数值不同，尤其 wrist roll 零点相差约 83.87 度，所以必须同时转换 observation.state 和 action。之后保留两条 prompt、构建 sample-start filter，并按相同 horizon 和 delta mask 重算 norm。模型使用旧 35k 参数 warm-start，但由于数据 mixture 和 norm 改变，新实验使用 fresh optimizer 训练 50k steps。

#### 第四部分：部署与控制，约 70 秒

> AutoDL 公网延迟远大于 action horizon，因此最终采用局域网客户端/服务器架构。服务器负责 PI0.5 推理，本地客户端负责相机、关节状态和 30 Hz 动作执行。Brain 与 Body 解耦，动作块到达后依据图像年龄和 RTT 跳过过期前缀，并设置 stale 拒绝、关节限幅、夹爪裁剪和异常 Hold，避免网络抖动直接变成危险动作。

#### 第五部分：结果、局限与下一步，约 50 秒

> 最终同一个 checkpoint 能根据 prompt 执行单物块和双物块任务，双物块任务完整执行到关闭抽屉。关闭阶段有时首次接触不准，但会根据新观测继续调整。这能说明系统是闭环的，却不能证明稳定自纠错。下一步最有价值的不是盲目增加训练步数，而是随机化抽屉和接触点、补充失败状态恢复示范、清理黑屏帧，并进行固定时限的标准化评测；之后才考虑 RL 后训练。

### 28.11 STAR 版本项目回答

当面试官说“讲一个你解决过的最复杂问题”时，可以使用下面的 STAR 结构：

- **Situation**：已有单物块抽屉 checkpoint，但需要扩展为依次抓取黑白两个物块的长程任务；新旧数据由不同人员采集并使用不同 calibration。
- **Task**：既要保留原策略的开关抽屉能力，又要让同一个模型根据语言指令区分单/双物块任务，并能够安全部署到真实 SO101。
- **Action**：检查 schema 和数值分布，确认 wrist roll 等关节坐标不一致；实现 state/action 联合校准转换，合并数据和 task 表，构建 sample-start filter，重算 mixture norm；从 35k 参数 warm-start，以 fresh optimizer 训练；部署时实现延迟对齐、stale 拒绝和安全 Hold。
- **Result**：得到覆盖 62 个 episodes 和两个任务指令的混合策略，同一 checkpoint 基本可以执行单/双物块任务，双物块长程任务完成端到端实机验证；同时识别出关闭抽屉首次成功率仍不足，并形成可验证的数据改进方案。

## 29. 面试官可能追问的项目细节

以下回答不是要求逐字背诵，而是提供回答中的事实骨架。建议先直接回答结论，再补原因和项目证据。

### 29.1 这个项目里哪些是你做的，哪些来自同事？

杯子任务数据和单物块抽屉数据/35k checkpoint 来自同事。本人完成杯子任务的训练、部署、真机推理与泛化分析；双物块长程任务由本人设计并采集 30 条数据；calibration 对齐、数据融合、norm、warm-start 训练、局域网部署和实机验证也由本人完成。这个边界应主动说清楚，反而能体现真实协作能力。

### 29.2 为什么要做两个阶段，而不是直接做抽屉任务？

第一阶段完成了训练和部署链路验证，同时暴露了最关键的数据问题：固定摆放下成功并不等于学会任务。第二阶段不是简单换任务，而是根据这个失败结论主动改变数据采集方式，通过位置随机化减少轨迹记忆。

### 29.3 你如何判断杯子任务学到的是轨迹，而不是任务？

在训练场景附近可以成功；物块或杯子位置稍微变化后明显失败；失败时机械臂仍倾向执行与示范相似的动作路径。这三个现象共同支持“数据覆盖不足和轨迹记忆”的判断。但它仍是实验证据，不是形式化证明；更严格的验证应做位置网格、多次重复和开环/闭环消融。

### 29.4 为什么只随机化物块位置？随机化范围如何确定？

杯子任务说明物体位置是主要分布偏移来源，因此第二阶段优先随机化黑白物块。范围必须同时满足相机可见、机械臂可达、抓取姿态可行，并避免在 30 条小数据中引入过大的多因素变化。项目尚未随机化抽屉和接触点，这正是关闭阶段泛化不足的可能原因。

### 29.5 30 个 episodes 是否太少？

对大型 VLA 从头训练当然太少，因此使用预训练 PI0.5、LoRA 和已有单物块技能 warm-start。30 条数据足以验证工程方向，但不足以证明强泛化或稳定成功率。更合理的下一步是增加覆盖不同物块、抽屉和失败状态的数据，而不是仅复制相同轨迹。

### 29.6 为什么使用两路相机？

环境相机提供抽屉、物块和机械臂之间的全局关系；腕部相机提供接近、抓取和接触时的局部细节。只有环境相机可能缺少近距离遮挡信息，只有腕部相机又缺少全局定位。两路相机的键、顺序、方向和训练/推理外参必须一致。

### 29.7 LeRobot v3 数据集里具体有什么？

逐帧 state/action/index 存在 parquet 中，图像像素存在按相机组织的视频中；`meta/info.json` 定义 schema 和路径模板，`tasks.parquet` 保存语言任务，episode metadata 保存边界和统计，`stats.json` 保存全局统计。训练时通过时间戳和索引把表格记录与视频帧对应。

### 29.8 为什么不能直接合并两批数据？

即使使用同一台 follower，不同 calibration 也会改变零点、homing offset 和夹爪范围。同一物理姿态会得到不同数值标签，直接合并会让 observation-action 映射互相冲突。本项目 wrist roll 坐标偏移约 83.87°，不能当作小噪声忽略。

### 29.9 calibration 转换为什么要同时处理 state 和 action？

行为克隆学习的是从 observation 到 action 的映射。只转换 state 会使输入在目标坐标系而监督动作仍在源坐标系；只转换 action 也同样不一致。两者必须使用同一物理映射，同时还要重建 episode 和全局统计。

### 29.10 calibration 转换如何验证？

检查转换前后 episode/frame 数完全一致；检查关节范围、分位数和异常值；验证 index 与 episode 边界连续；抽样比较已知姿态在两个 calibration 下是否表示同一物理位置；最后通过 norm 生成、数据加载 smoke test 和真机低速动作验证形成闭环。

### 29.11 为什么保留两条 task prompt？

单物块和双物块行为顺序不同。如果使用同一个默认 prompt，模型只能从视觉猜测要执行哪个任务，监督存在歧义。通过 `prompt_from_task=True` 保留 task 表，推理时发送与训练一致的指令，使语言真正成为任务条件。

### 29.12 sample-range filter 解决什么问题？

旧数据中存在较多低价值 idle frame。直接删除帧会破坏视频索引、episode 边界和 action horizon，因此实现 `EpisodeRangeFilteredDataset`，只限制哪些全局索引可以作为训练窗口起点，底层数据仍保持完整。接触和恢复阶段虽然速度低，却很重要，不能按“动作小”简单删除。

### 29.13 为什么合并数据后要重算 norm？

新旧任务的 state/action 分布以及合法样本集合发生变化。若继续使用旧单物块 norm，模型输入和输出尺度会偏向旧任务，推理反归一化也可能错误。重算时必须复现训练的 filter、10-step horizon、delta mask 和 drop-last 语义，否则“有一个 norm 文件”也不代表与训练一致。

### 29.14 为什么用 parquet 快速计算 norm？

norm 只依赖低维 state/action，没有必要解码 AV1 视频。直接使用 PyArrow 读取 parquet 可以显著减少 I/O 和解码开销，同时更容易精确实现 filter、horizon 和 episode 边界规则。

### 29.15 为什么从 35k checkpoint warm-start？

单物块 checkpoint 已通过真机验证，包含开抽屉、抓取、放置和关闭的基础技能。Warm-start 能保留这些能力，把有限数据和计算用于学习第二个物块、任务顺序和语言条件。它比从 base 完全重新训练更符合小数据场景。

### 29.16 为什么不用旧 optimizer 直接 resume？

新实验的数据 mixture、norm、filter 和目标任务都发生变化。旧 optimizer 的动量、学习率位置和数据加载状态对应旧目标，直接恢复可能导致优化状态不匹配。因此扩展任务时只加载参数并使用 fresh optimizer；只有新混合实验自身中断后，才从 1000/4000 等完整 checkpoint 使用 `--resume`。

### 29.17 为什么不是训练步数越多越好？

小数据行为克隆可能随着训练继续而过拟合。训练 loss 下降只表示训练动作拟合更好，无法保证闭环成功。应通过完整 episode 验证集、不同 checkpoint 真机 A/B 和随机初始状态成功率选择 checkpoint，而不是默认最后一步最好。

### 29.18 为什么前五轴用 delta action、夹爪用 absolute？

身体关节的相对变化通常比绝对目标更容易建模局部运动，并能降低不同姿态的尺度差异；夹爪值直接表达开合状态，使用绝对值更自然。训练变换、norm 和推理逆变换必须使用相同 mask，否则会出现动作尺度或语义错误。

### 29.19 你如何证明同一个 checkpoint 区分了两个任务？

在相近场景、相同服务端 checkpoint 下，仅改变与训练一致的 prompt，分别观察单物块和双物块的任务序列。更严格的实验还应加入 prompt swap、空 prompt 和改写 prompt 对照，统计任务顺序错误率，排除模型只是根据场景中物块数量决定行为。

### 29.20 关闭抽屉多次尝试说明什么？

它说明系统持续接收新观测并生成新动作，而不是一次开环播放；因此可以说观察到闭环调整和一定恢复行为。但重试也可能是策略抖动、接触不确定或偶然成功，不能直接声称模型理解失败。需要统计首次关闭、恢复后关闭和重试次数，并做开环/闭环消融。

### 29.21 为什么关闭抽屉比抓取更难？

关闭属于接触密集操作，视觉很难直接观测接触力和抽屉阻力；微小位置误差会产生完全不同的接触结果；相机遮挡和机械臂回差也更明显。当前系统没有力/触觉反馈，且抽屉位置和接触点没有充分随机化，所以关闭阶段成为主要瓶颈。

### 29.22 如果重新采集，你会怎么设计？

先定义训练分布：随机化两个物块、抽屉左右/前后位置、开合距离、接触高度和推入点；再按子任务平衡数据，增加关闭和失败恢复示范；恢复轨迹应从失败状态直接开始正确修正，避免模型学会“先失败再恢复”；最后清理黑屏、危险推压和无意义 idle 段。

### 29.23 你会如何做正式评测？

固定 90 或 120 秒，每类初始条件至少多次重复，并把位置划分为训练分布内、插值位置和边界位置。记录开抽屉、黑块、白块、首次关闭、恢复后关闭、完整任务成功率，以及完成时间、重试次数、人工干预率和危险碰撞率。报告试验次数和置信区间，而不是只放最好的一次视频。

### 29.24 这个项目最困难的部分是什么？

推荐回答“跨数据与真机链路的一致性”，而不是“环境安装”。真正困难的是让 calibration、schema、prompt、norm、action 语义、checkpoint、相机输入、网络时延和机械臂控制在训练与推理两端全部一致。任何一个接口错位都可能表现为“模型不会抓”，但根因完全不同。

### 29.25 如果只能改一个地方，你会改哪里？

优先改数据采集与评测：增加抽屉位置、接触点和失败恢复覆盖，并建立标准化多轮指标。当前主要瓶颈是分布覆盖和接触阶段，而不是训练 loss 没有继续下降。没有高质量覆盖数据时直接加训练步数或上 RL 很容易优化错误目标。

## 30. VLA 与机器人学习理论问题

### 30.1 什么是 VLA？

VLA 是 Vision-Language-Action 模型：输入视觉观测、语言任务以及通常还包括机器人状态，输出可由机器人执行的动作或动作序列。它与只输出文本的 VLM 不同，必须把语义理解落到连续控制，并处理真实世界中的时延、误差累积、接触和安全问题。

### 30.2 VLA 与传统“感知—规划—控制”流水线有什么区别？

传统系统通常显式拆分目标检测、位姿估计、任务/运动规划和控制器，模块可解释、易加入约束，但接口误差会逐级传播。VLA 倾向于从多模态观测端到端预测动作，能利用大模型先验并减少手工接口，但更依赖数据分布，安全验证和失败解释也更困难。实际系统往往采用混合架构：VLA 负责策略，底层仍保留限位、Hold、碰撞保护等确定性控制。

### 30.3 PI0.5 在这个项目中扮演什么角色？

PI0.5 是预训练的视觉—语言—动作策略。OpenPI 将双相机图像、任务 prompt 和机器人 state 适配到统一 observation，模型的视觉语言主干提取多模态表示，Action Expert 通过 flow matching 生成 action chunk。项目使用 LoRA 在少量 SO101 数据上适配，而不是从头训练整个模型。

### 30.4 PaliGemma 和 Action Expert 分别做什么？

可以把 PaliGemma 视为处理图像与语言上下文的视觉语言主干，Action Expert 负责结合上下文与动作噪声/时间信息，预测用于生成连续动作的向量场。前者提供语义和视觉表征，后者把表征转为机器人动作序列。面试中不应把 Action Expert 说成传统独立运动规划器。

### 30.5 LoRA 的原理是什么？

对冻结权重 `W0` 不直接做全量更新，而是学习低秩增量：

```text
W = W0 + (alpha / r) * B * A
```

其中秩 `r` 远小于原矩阵维度。这样训练参数、梯度和优化器状态更少，适合小数据和有限显存场景，也能降低对预训练能力的破坏。代价是低秩容量有限，效果依赖插入层、rank、学习率和数据质量。

### 30.6 为什么这里适合 LoRA，而不是全量微调？

真实数据只有几十个 episodes，模型规模大，全量微调显存和优化器开销高，也更容易过拟合或遗忘预训练知识。LoRA 可以用较小的可训练参数适配 SO101 动作和视觉分布。它并不能自动解决数据覆盖不足，杯子任务就是反例。

### 30.7 什么是行为克隆？

行为克隆把示范数据视为监督学习，学习策略 `π(a|o, language)`，让预测动作接近专家动作。优点是简单稳定、容易利用遥操作数据；缺点是训练数据只覆盖专家访问过的状态，部署时一个小误差会进入未见状态并继续累积。

### 30.8 什么是 covariate shift 和 compounding error？

训练时 observation 来自专家轨迹，部署时 observation 来自模型自己的动作结果，两者分布不同，这就是 covariate shift。策略一旦出现小误差，后续可能进入训练集中没有的状态，误差随时间累积，长程任务尤其明显。增加扰动和恢复数据、DAgger 类在线聚合、闭环高频重规划都可缓解。

### 30.9 什么是 flow matching？

Flow matching 在噪声动作与真实动作之间构造随时间变化的概率路径，让网络学习对应的速度场。推理时从噪声动作出发，通过数值积分沿学习到的向量场逐步得到动作序列。与直接回归单一动作相比，它更适合表达多模态连续动作分布；与经典扩散模型相关，但训练目标和采样表述通常以连续向量场为中心。

### 30.10 为什么动作预测可能是多模态的？

同一观测下可能存在多条都正确的轨迹，例如从物块左侧或右侧接近。简单 MSE 回归可能平均这些动作，得到物理上并不合理的中间轨迹。生成式动作模型可以表达多个可能模式，但最终稳定性仍取决于数据一致性和推理采样。

### 30.11 什么是 action chunking？

模型一次预测未来 `H` 步动作，而不是每一步只预测一个动作。它能减少推理调用频率，学习短期时间一致性，并缓解单步噪声。但 chunk 越长越可能在执行后半段脱离最新观测；chunk 越短又增加计算和网络压力。

### 30.12 Action horizon 如何选择？

需要权衡控制频率、推理延迟、任务动态性和模型能力。本项目 `H=10`、30 Hz，对应约 333 ms。局域网 RTT 约 90 ms，还能通过跳过过期前缀使用后续动作；公网 1～3 秒已经超过整个 horizon，无法靠队列修复。接触任务通常希望更频繁地利用新观测。

### 30.13 为什么要做动作归一化？

不同关节和夹爪的数值范围差异大，直接训练会让大尺度维度主导 loss。归一化把各维映射到相近尺度，有利于优化稳定。推理输出必须用同一统计反归一化；norm 与 calibration、action 表示或数据 mixture 不一致会直接产生错误动作。

### 30.14 为什么使用 quantile norm 而不是只用 mean/std？

分位数对少量异常值更稳健，常用低/高分位数定义主要数据范围，再映射到统一区间。代价是范围外数据会被外推或裁剪，因此仍要检查新数据是否超出训练分布。具体使用哪种统计必须以实际 OpenPI 配置和 `norm_stats.json` 为准。

### 30.15 Delta action 与 absolute action 的优缺点是什么？

Delta action 表示相对当前状态的变化，局部运动尺度小，对绝对零点变化更不敏感，但长时间执行会积累误差，推理时还必须正确加回当前 state。Absolute action 直接给目标位置，更容易保持全局目标，但对 calibration 和尺度更敏感。本项目采用身体关节 delta、夹爪 absolute 的混合表示。

### 30.16 语言指令如何影响动作？

Prompt 被 tokenize 后与视觉和状态上下文共同进入模型。对于相同场景，不同指令应改变动作分布和任务顺序。要证明模型真正使用语言，可做 prompt swap、同义改写、错误颜色、空 prompt 等对照，而不能只观察两个不同场景下执行了两个任务。

### 30.17 Robot state 为什么重要？

图像不能精确提供所有关节角、夹爪开合和自身运动状态。state 为策略提供本体感觉，使相同视觉下能够根据当前机械臂姿态输出不同动作。State 的维度、顺序、单位、tokenization/连续输入方式必须与训练 checkpoint 一致。

### 30.18 `discrete_state_input` 应该如何回答？

不要脱离具体 checkpoint 给统一答案。项目早期杯子配置与后期抽屉混合配置不是同一个实验，代码中可能采用不同 state 输入开关；当前 `drawer_one_two_blocks` 目标配置明确设置 `discrete_state_input=False`。面试时应展示自己会根据实际 config、模型版本和 checkpoint 核对，而不是背一个固定布尔值。

### 30.19 为什么训练 loss 低不代表真机成功？

训练 loss 只衡量训练分布上的动作拟合。真机还受到视觉分布偏移、闭环误差累积、时延、相机外参、机械回差、接触动力学和安全限幅影响。杯子任务的 loss 很低但位置泛化差，正是最直接的项目证据。

### 30.20 验证集为什么要按 episode 划分？

相邻帧高度相关。如果随机按 frame 划分，同一条轨迹前后相邻画面可能同时进入训练和验证，造成严重泄漏和虚高指标。按完整 episode 划分更能测试未见轨迹，但最终仍需要真实机器人评测。

### 30.21 如何判断过拟合？

观察训练与 episode-level 验证 loss 的差距；比较多个 checkpoint 在固定真机条件下的表现；测试不同物体/抽屉位置、光照和起始姿态；检查模型是否重复示范轨迹而忽略目标变化。只有训练 loss 持续下降不能判断没有过拟合。

### 30.22 什么是闭环控制，项目为什么是闭环？

闭环系统在执行过程中持续读取新观测，并根据当前误差更新后续动作；开环则预先生成整段轨迹后不再看环境。本项目客户端持续发送最新双相机和 state，服务端反复生成短 action chunk，因此总体是视觉闭环。一个 chunk 内执行若干步仍带有局部开环成分。

### 30.23 闭环恢复是否等于“模型会思考”？

不等于。策略根据新观测映射到新动作，可以出现恢复行为，但这可能来自训练分布中的相似状态或策略动力学。除非有对照实验和内部推理证据，否则应描述为“闭环反馈下的行为恢复”，而不是理解失败或自主思考。

### 30.24 如果做 RL 后训练，你会怎么设计？

先补齐行为克隆数据和可靠评测，再定义分阶段 reward：开抽屉、黑块入 drawer、白块入 drawer、抽屉完全关闭、碰撞/超时惩罚。可以从仿真或安全受控的离线/少量在线方案开始，避免真实机械臂无约束探索。还需要完成检测器、reset 机制、安全约束和 reward hacking 检查。项目目前没有实施 RL，因此这里只能作为方案回答。

### 30.25 Offline RL 与 online RL 在这个任务中的取舍是什么？

Offline RL 使用已有数据，真机风险低，但受数据覆盖限制，容易对分布外动作估值错误；online RL 能主动探索并优化成功率，但真实机器人样本昂贵、reset 慢且存在安全风险。实际可先用行为克隆初始化，再使用高质量恢复数据、保守离线方法或小范围安全在线微调。

### 30.26 为什么不直接增加更多模型参数？

当前主要失败来自数据分布、接触反馈和系统时延，而不是明确的模型容量不足。更大模型会增加训练和推理成本，甚至加重延迟，却不能自动补足未见状态。应先通过数据和消融确认瓶颈，再决定是否扩大模型或 LoRA rank。

## 31. 部署、实时性与系统设计问题

### 31.1 为什么 AutoDL 公网推理失败而局域网成功？

公网连接本身没有断，但 1～3 秒返回时间远大于约 333 ms action horizon。动作到达时对应的是旧图像和旧 state，继续执行会产生 stale action。局域网 RTT 降到约 90 ms 后，动作仍处于 horizon 内，可以跳过少量过期前缀后执行。

### 31.2 什么是 stale action？

动作是根据过去某一时刻的 observation 预测的。如果动作返回时机器人和环境已经变化太多，该动作就过期。客户端应记录 observation 时间、服务端/网络耗时和当前时间，超过阈值时丢弃结果并 Hold，而不是为了保持运动而执行旧动作。

### 31.3 为什么要跳过 action chunk 前几步？

模型预测的是从观测时刻开始的未来动作，但网络和推理已经消耗一段时间。按 30 Hz 将这段延迟换算成步数，就可以从 chunk 中选择更接近当前时刻的动作。它只能补偿小于 horizon 的时延，不能修复 1～3 秒公网延迟。

### 31.4 Brain/Body 解耦有什么好处？

Brain 负责异步采集观测和请求模型，频率受 GPU/网络影响；Body 以稳定 30 Hz 执行动作、安全检查和 Hold。解耦可以避免一次推理抖动阻塞底层控制，同时需要线程安全队列、时间戳和明确的旧动作失效策略。

### 31.5 为什么需要低通和平滑？

模型输出和 chunk 切换可能存在抖动，低通能减少高频机械冲击。但平滑过强会压缩有效动作并增加相位滞后，本项目曾出现 `alpha` 和单步限幅过小导致肉眼几乎不动。因此参数要通过低速 A/B 测试确定，不能只追求曲线平滑。

### 31.6 真机侧有哪些安全保护？

动作维度与 NaN/Inf 检查、关节范围、单步最大变化、夹爪范围、stale 拒绝、无新动作时 Hold、异常退出释放扭矩、短时测试和人工急停。VLA 输出不能直接绕过底层安全层发送给舵机。

### 31.7 为什么服务端返回动作但机械臂可能不动？

要分层排查：服务端 target 是否真的变化；客户端是否因 stale/异常拒绝；平滑和限幅后变化是否小于舵机分辨率；扭矩和外部供电是否开启；calibration 是否匹配；运行时间是否太短；相机或 GUI 异常是否让客户端提前退出。

### 31.8 相机一致性为什么重要？

模型没有自动知道“env”和“hand”相机被交换，也无法天然适应镜像、旋转、安装高度变化。相机顺序、方向、视野、曝光和场景背景都是训练分布的一部分。部署前应与训练参考图对齐，并使用稳定设备路径而不是易变化的 `/dev/videoN`。

### 31.9 黑屏帧为什么危险？

如果每条 episode 开头都存在相似黑屏，而任务动作也总在黑屏后开始，模型可能把黑屏当作启动提示。这是与任务无关的视觉捷径。兼容旧 checkpoint 可以临时调整 warmup，但正确做法是清理数据并重新训练/微调。

### 31.10 如果网络突然变慢，系统应如何降级？

停止接受超龄 chunk，清空旧动作队列并 Hold；记录 RTT 和图像年龄；连续超时达到阈值后进入安全停止，而不是重复最后一个运动动作。恢复连接后应使用最新 observation 重新请求，不应继续消费断线前缓存。

## 32. 开放题与改进方案

### 32.1 如何证明位置随机化真的有效？

设计对照实验：固定位置数据模型与随机位置数据模型使用相同训练量和评测协议；在训练位置、插值位置、边界位置分别多次测试；比较完整成功率和抓取位姿误差。只展示随机化模型成功视频不能证明提升。

### 32.2 如何区分视觉闭环恢复和随机抖动？

记录首次失败后的观测、动作方向和误差是否持续减小；统计不同失败状态下的恢复概率；与冻结观测或开环执行完整 chunk 的基线比较；检查恢复动作是否与错误方向相关。稳定、状态相关且显著优于开环，才是更强证据。

### 32.3 如何做 ablation？

优先做能回答项目关键假设的消融：固定位置 vs 位置随机化；单相机 vs 双相机；旧 norm vs mixture norm；默认 prompt vs task prompt；从 base 训练 vs 35k warm-start；固定执行 chunk vs adaptive skip；最后 checkpoint vs 中间 checkpoint。

### 32.4 如何提高关闭抽屉成功率？

数据侧增加不同抽屉位置、开度、接触点和失败恢复；控制侧加入实际关节反馈、接触阶段更短 horizon、速度/力限制和完成检测；硬件允许时加入电流或力觉；训练侧对关闭/恢复片段合理采样，避免被大量移动帧淹没。

### 32.5 如何防止模型学会“先失败再恢复”？

恢复 episode 应从已经失败的状态开始，第一段动作就是正确修正；训练 mixture 中仍以一次成功示范为主，恢复数据只占合理比例；评测同时统计首次成功和最终成功，防止模型通过增加重试换取最终成功率。

### 32.6 如何改成长 horizon？

需要更多覆盖完整任务的连贯示范、明确的任务阶段条件和更强的长期记忆。可以探索层级策略：高层根据语言和视觉选择子任务，低层 VLA 执行短 horizon 动作；也可以增加历史帧或状态记忆。直接把 action horizon 调大不等于获得长期规划。

### 32.7 如果换一台 SO101，checkpoint 能直接用吗？

不能默认可以。需要核对关节顺序、方向、零点、范围、夹爪映射、相机配置和动力学差异。至少要做 calibration 转换、低速动作验证和可能的少量适配数据。项目经验正说明同一 follower 的不同 calibration 都可能造成显著差异。

### 32.8 如果面试官质疑“这只是行为克隆”，怎么回答？

承认训练主体确实是基于示范的行为克隆/flow-matching 动作学习，不包装成通用智能。项目价值在于把预训练 VLA 适配到真实机器人，并系统解决数据分布、calibration、norm、实时闭环和安全问题；同时通过失败案例认识到行为克隆的分布外局限，并提出可验证的后续方案。

### 32.9 如果面试官问项目创新点是什么？

不要声称提出了新的基础模型。可以回答工程与实验创新：将不同 calibration 的真机数据可靠融合；在 OpenPI 数据管线中对齐 task prompt、sample filter、delta action 和 norm；通过第一阶段失败主动设计位置随机化数据；构建考虑 action age 的局域网闭环执行系统，并对恢复后成功建立更严格的解释和评测标准。

### 32.10 如果再给两周，你会做什么？

第一周补采抽屉/接触点随机化和失败恢复数据，清理黑屏并完成 episode-level 数据审计；第二周用选定 checkpoint 参数进行较低学习率微调，完成 20～30 次固定时限评测和关键消融。只有在评测链路稳定后，才启动小规模 RL 后训练可行性实验。

## 33. 面试前检查清单

### 33.1 必须能现场画出的三张图

1. 数据流：LeRobot parquet/video → repack → SO101 input → delta/norm → PI0.5 → action → unnorm/absolute。
2. 训练关系：旧单物块 35k params + 新旧混合数据/norm → fresh optimizer 50k experiment。
3. 部署架构：本地相机/SO101 → WebSocket → GPU policy server → action chunk → adaptive skip/Body。

### 33.2 必须能解释的五个失败

1. 杯子任务位置变化后失败：固定数据导致轨迹记忆。
2. AutoDL 公网推理失败：延迟超过 action horizon。
3. 不同数据不能直接合并：calibration 坐标不一致。
4. 有动作日志但机械臂不动：过强平滑/限幅、短运行时间或硬件链路问题。
5. 抽屉需要多次关闭：接触密集、数据覆盖和无力觉反馈共同影响。

### 33.3 建议准备的展示材料

- 30～60 秒最终双物块任务视频，保留完整任务顺序。
- 一段杯子任务位置变化后的失败视频，用来说明为什么重新设计数据。
- 数据集样例：两路图像、state/action 曲线和 task prompt。
- calibration 转换前后的关节分布图，特别是 wrist roll。
- 训练/W&B 曲线，但主动说明 loss 不等于成功率。
- 客户端 RTT、action age 和 stale 统计截图。
- 一页系统架构图和一页失败案例/下一步方案。

### 33.4 面试时的回答习惯

- 先回答结论，再讲原因，最后给项目证据。
- 主动说清个人贡献与同事已有基础。
- 遇到没有做过的实验直接说“这是我的方案，尚未验证”。
- 不把训练 loss 当作真机指标，不用一次成功代表成功率。
- 不回避失败案例；杯子任务泛化失败和关抽屉重试正是项目最有价值的分析材料。
- 所有数字只使用本节事实表中能够解释来源的数字。

### 33.5 可以反问面试官的问题

- 团队当前更关注 VLA 的数据引擎、基础模型训练，还是下游真机部署与评测？
- 真机数据采集如何做任务分布设计和质量控制？
- 团队如何定义长程操作的成功率、恢复成功和安全失败？
- 当前策略采用单步、action chunk 还是分层控制？实时推理的延迟预算是多少？
- 对 behavior cloning 之外的 DAgger、offline RL 或在线 RL，团队目前采用什么路线？
- 实习生是否有机会参与数据采集、训练和真机评测的完整闭环？

## 34. 第一次 PI0.5 LoRA 微调：参数与训练指标详解

本节整理第一阶段“抓取黑色物块放入白色杯子”任务的实际配置，并解释这些参数和指标在后续双物块抽屉任务中的含义。

### 34.1 `config.py` 默认值与实际运行值要区分

第一阶段新增的配置名为 `pi05_so101_lora`，代码位于：

~~~text
custom_vla/openpi/src/openpi/training/config.py
~~~

配置文件曾保留 `100000 steps`、`warmup=3000`、`peak_lr=3e-4`、`num_workers=12` 等默认值，但最终杯子实验通过训练命令覆盖了其中一部分。面试时应该以实际运行参数为准：

| 参数 | 第一次杯子任务实际值 |
| --- | ---: |
| Config | `pi05_so101_lora` |
| Batch size | 16 |
| DataLoader workers | 4 |
| Train steps | 50,000 |
| Warmup steps | 1,000 |
| Peak learning rate | `1e-4` |
| Decay steps | 50,000 |
| Final learning rate | `1e-6` |
| Gradient clipping | 0.5 |
| Action horizon | 10 |
| Save interval | 1,000 |
| Keep period | 5,000 |

### 34.2 相比官方通用配置，主要适配了什么

#### 模型与 LoRA

- 选择 PI0.5：`pi05=True`。
- 视觉语言主干使用 `gemma_2b_lora`。
- Action Expert 使用 `gemma_300m_lora`。
- 通过 `freeze_filter` 冻结基础参数，只训练允许更新的 LoRA adapter。
- `ema_decay=None`，不额外维护一套完整 EMA 参数。
- `action_horizon=10`，每个样本监督和预测未来 10 步动作。
- 第一阶段杯子配置使用 `discrete_state_input=True`。后期抽屉混合配置是另一个实验，不能把这个布尔值推广到所有 checkpoint。

#### SO101 数据适配

- 数据路径改为 `SO101_DATASET_DIR`，避免绑定旧服务器绝对路径。
- `asset_id="blacknew"`，加载该数据对应的 normalization statistics。
- 将 `observation.images.env`、`observation.images.hand`、`observation.state` 和 `action` 重组为 OpenPI/SO101 期望的字段。
- 使用固定 prompt：`Grab the black cube and place it in the white cup`。
- 原始 SO101 action 是 absolute；前五个身体关节转换为相对当前 state 的 delta，gripper 保持 absolute。
- 正式实验关闭高斯图像噪声。

#### 优化与工程参数

- 使用 AdamW，并将全局梯度裁剪阈值设为 0.5。
- 调整 batch size、训练步数、warmup、peak/final learning rate。
- 配置 W&B、日志间隔和 Orbax checkpoint 保存策略。
- 将数据、权重、缓存、日志和 checkpoint 放到数据盘。

### 34.3 `batch_size=16` 到底表示什么

一个训练样本不是单独一张图，而是一个训练窗口：

~~~text
当前环境相机图像
+ 当前腕部相机图像
+ 当前机械臂 state
+ 语言指令
+ 从当前时刻开始的未来 10 步专家 action
~~~

`batch_size=16` 表示一次梯度更新同时读取 16 个这样的窗口，先对其 loss 求平均，再反向传播。

Batch 较大时：

- 梯度平均后通常更稳定，曲线更平滑。
- GPU 并行利用率通常更高。
- 显存消耗增加。
- 对高度相关的机器人相邻帧而言，样本数增加不一定等于多样性同比增加。

Batch 较小时：

- 显存占用更低。
- 梯度噪声和曲线波动更大。
- 相同 step 数下看到的总训练窗口更少。

第一阶段大致执行：

~~~text
50,000 steps × batch 16 = 800,000 个 sample windows
~~~

相对于约 31,000 个数据起点，相当于反复抽样约 25.8 轮。但 action window 互相重叠、相邻视频帧高度相关，因此不能理解成 25.8 轮完全独立的数据。

### 34.4 Learning rate 控制什么

Learning rate 控制每次优化器更新的总体尺度。直观近似为：

~~~text
新参数 ≈ 旧参数 - LR × 梯度
~~~

项目实际使用 AdamW，还会结合一阶动量、二阶矩估计和 gradient clipping，因此实际更新不完全等于这个简单公式。

学习率太大时可能出现：

- loss 剧烈震荡甚至发散；
- 参数越过较优区域；
- 梯度尖峰、NaN 或 Inf；
- 小数据 LoRA 快速记住训练集。

学习率太小时可能出现：

- loss 下降很慢；
- LoRA 参数几乎没有得到有效更新；
- 固定训练时间内仍然欠拟合。

第一阶段学习率调度为：

~~~text
step 0～1000：从很小的 LR 线性 warmup 到 1e-4
step 1000～50000：cosine decay 到 1e-6
~~~

Warmup 防止训练刚开始时使用过大的更新尺度；后期 decay 让模型用更小的步长收敛。需要注意：warmup 直接控制的是 optimizer update，不是直接控制 `grad_norm`。LR 会通过之前的参数变化间接影响未来的梯度。

Batch size 与 learning rate 有关联：更大的 batch 通常梯度方差更小，有时可以配合更大的 LR，但不能机械地线性放大。LoRA、小数据、强相关视频帧和真实机器人动作分布都会改变最佳组合，所以本项目选择 batch 16 和 peak LR `1e-4`，本质上是显存、吞吐和稳定性的折中。

### 34.5 代入双物块抽屉任务，loss 是什么

训练 loss 不是“是否把两个物块放进抽屉”，也不是任务完成率。它是 PI0.5 的 flow-matching 速度场回归误差。

假设当前训练帧处于“机械臂正在接近黑块”阶段，一个样本包含当前双相机、state、完整任务 prompt，以及示范接下来的 10 步 action。30 Hz 下 10 步只覆盖：

~~~text
10 / 30 ≈ 0.333 秒
~~~

因此模型不是用一个 loss 直接监督整段约 45 秒任务，而是在每个时刻学习“接下来约 0.33 秒怎么动”。部署时持续获取新观测、反复预测短 action chunk，最终串成：

~~~text
开抽屉 → 抓黑块 → 放黑块 → 抓白块 → 放白块 → 关抽屉
~~~

设归一化后的真实未来动作 chunk 为 `a`，高斯噪声为 `epsilon`，随机 flow 时间为 `t`。代码构造：

~~~text
x_t = (1-t) * a + t * epsilon
u_t = epsilon - a
~~~

模型根据图像、语言、state、带噪动作 `x_t` 和时间 `t` 预测速度 `v_theta`，优化：

~~~text
Loss = mean[(v_theta - u_t)^2]
~~~

代码对 batch、10-step horizon 和动作维度求平均。SO101 实际有 6 个动作维度，进入模型后会 padding 到 32 维，因此常见张量是 `[B, 10, 32]`。

| 当前观测所处阶段 | 当前样本的监督内容 |
| --- | --- |
| 打开抽屉 | 示范接下来约 0.33 秒的拉抽屉动作 |
| 接近黑块 | 接下来 10 步的关节接近轨迹 |
| 抓黑块 | 关节微调和 gripper 闭合动作 |
| 放置黑块 | 移动到抽屉并松开 gripper |
| 白块阶段 | 对白块重复对应的局部动作 |
| 关闭抽屉 | 推抽屉的局部接触动作 |

总体 loss 是不同 episode、不同阶段和不同 batch 的平均。因此关闭抽屉即使学得较差，只要它占所有训练窗口的比例较小，总 loss 仍可能很好看。要分析长程任务，需要额外记录分阶段 loss 和真实机器人子任务成功率。

不同实验如果 norm、action horizon、action dimensions 或数据过滤不同，loss 的绝对数值也不能直接横向比较。

### 34.6 `grad_norm` 是什么

代码记录：

~~~python
grad_norm = optax.global_norm(grads)
~~~

它是所有可训练参数梯度的全局 L2 范数：

~~~text
Grad norm = sqrt(sum(g_i^2))
~~~

`grads` 通过 `trainable_filter` 计算，因此主要反映 LoRA 可训练参数的梯度强度。它回答：

> 当前这个 batch 产生的误差，想把可训练参数往多大的方向推动？

| 现象 | 可能含义 |
| --- | --- |
| 前期较大 | 模型还没有适应新场景和动作分布 |
| 总体逐渐下降 | 模型逐渐接近训练数据的监督目标 |
| 偶发尖峰后恢复 | 当前 batch 更难或含有少见状态，通常可以接受 |
| 长期增大 | 训练不稳定、异常数据或学习率过高 |
| NaN/Inf | 数值训练已经失败 |
| 长期接近 0 但 loss 很高 | 可能梯度消失或优化停滞 |

项目的 gradient clipping 阈值是 0.5。日志中的 `grad_norm` 在 optimizer 裁剪前计算，因此偶尔看到大于 0.5 并不代表裁剪失效；Optax 随后会按比例缩小梯度再交给 AdamW。

### 34.7 `param_norm` 是什么

代码记录：

~~~python
param_norm = optax.global_norm(kernel_params)
~~~

即模型 kernel 权重的全局 L2 范数：

~~~text
Param norm = sqrt(sum(theta_i^2))
~~~

当前实现从合并后的完整模型中选取二维及以上 kernel，并排除了 bias、scale、position embedding 和 input embedding，但没有再次使用 LoRA `trainable_filter`。所以第一阶段约 `1803～1810` 的 param norm 很可能包含大量冻结的基础模型权重，不只是 LoRA adapter。

因此不能说“param norm 上升就代表学到更多知识”。更准确的用途是：

- 平滑变化说明整体权重没有突然数值爆炸；
- 突然大幅跳变可能表示异常更新或 checkpoint 恢复问题；
- NaN/Inf 表示训练失败；
- 它不能表示任务成功率，也不能直接衡量 LoRA 学了多少。

如果希望更精确分析 LoRA，可以额外记录：

~~~text
trainable_param_norm
lora_param_norm
update_norm
update_norm / param_norm
~~~

### 34.8 三个指标应该一起怎么说

| 指标 | 实际衡量内容 | 是否等于任务成功率 |
| --- | --- | --- |
| Loss | Flow velocity 的训练拟合误差 | 否 |
| Grad norm | 当前 batch 对可训练参数的梯度强度 | 否 |
| Param norm | 被统计 kernel 权重的整体尺度 | 否 |

第一阶段训练结果可以支持的结论是：

> Loss 从 `0.0609` 下降到约 `0.001～0.002`，grad norm 总体下降且尖峰能够恢复，param norm 平滑变化，没有出现 NaN 或参数爆炸，说明训练过程数值稳定，并且模型对训练示范的 flow-matching 拟合显著改善。

不能得出的结论是“模型已经真正学会任务或具有良好泛化”。杯子任务正好说明：训练 loss 很低，物块或杯子位置稍微变化后仍可能失败。

## 35. 为什么必须对齐 Calibration：它不只是活动范围

### 35.1 先给结论

“把一套数据迁移到另一套校准文件下”这个理解基本正确。更准确地说：

> Calibration 定义了原始电机编码器空间与模型使用的 degrees/percentage 空间之间的映射。校准对齐就是保持物理姿态和动作不变，只把数据中的数值表达从 source calibration 改写为 target calibration 的表达。

它不会改变视频中的机械臂姿态，也不会凭空生成新动作；改变的是 parquet 中 `observation.state` 和 `action` 的数字，以及依赖这些数字的统计量。

### 35.2 校准文件中有什么

SO101 每个电机的 calibration 至少包含：

~~~text
id
drive_mode
homing_offset
range_min
range_max
~~~

这些字段不只是安全活动范围：

| 字段 | 含义 |
| --- | --- |
| `range_min/range_max` | 记录有效编码器范围；身体关节的中点还被当作 0° 参考，gripper 则用整个区间映射到 0～100 |
| `homing_offset` | 定义原始物理编码器零点如何平移到当前电机的 Present Position 坐标 |
| `drive_mode` | 可定义方向翻转；本项目转换脚本只接受两边均为 0 的情况 |

因此 calibration 同时定义原点、方向、tick 到 degrees 的解释、夹爪百分比映射和合法范围。这就是它体现“坐标系”的地方。

### 35.3 LeRobot 如何把编码器值变成角度

对于身体关节，LeRobot 使用：

~~~text
mid = (range_min + range_max) / 2
degree = (present_position - mid) * 360 / 4095
~~~

Feetech 电机中又有：

~~~text
present_position = physical_encoder_position - homing_offset
~~~

合起来就是：

~~~text
degree = (physical_encoder_position - homing_offset - mid) * 360 / 4095
~~~

可以看到，即使机械臂物理上一动不动，只要 `homing_offset` 或 range 中点不同，输出的 degree 就会不同。

对夹爪，LeRobot 使用：

~~~text
percentage = (present_position - range_min)
             / (range_max - range_min) * 100
~~~

因此不同的 range 宽度不仅改变原点，还可能改变夹爪数值的尺度。

### 35.4 身体关节的具体数值例子

假设同一个 shoulder 关节有两套 calibration。

Source calibration：

~~~text
homing_offset = 100 ticks
range_min = 1000
range_max = 3000
mid_source = 2000
~~~

Target calibration：

~~~text
homing_offset = 50 ticks
range_min = 800
range_max = 2800
mid_target = 1800
~~~

现在 source 数据中记录：

~~~text
state_source  = 30°
action_source = 40°
~~~

Source 的 30° 对应：

~~~text
present_source ≈ 2000 + 30/360*4095 = 2341.25 ticks
physical_encoder ≈ present_source + homing_offset
                 ≈ 2441.25 ticks
~~~

同一个物理编码器位置放到 target calibration 下：

~~~text
present_target ≈ 2441.25 - 50 = 2391.25 ticks
degree_target ≈ (2391.25 - 1800)*360/4095
              ≈ 51.98°
~~~

因此同一个物理姿态：

~~~text
Source 表示：30°
Target 表示：约 51.98°
~~~

两套坐标之间的转换偏移是：

~~~text
offset_ticks = mid_source + homing_source
               - homing_target - mid_target
             = 2000 + 100 - 50 - 1800
             = 250 ticks

offset_degree = 250 * 360 / 4095
              ≈ 21.98°
~~~

所以 source 数据迁移到 target 后：

~~~text
state_target  = 30 + 21.98 ≈ 51.98°
action_target = 40 + 21.98 ≈ 61.98°
~~~

物理动作仍然是从当前姿态向正方向移动 10°，只是数值表达换成了 target 坐标系。

### 35.5 为什么 state 和 action 都要改

在数据集中：

- `observation.state` 表示当前物理姿态；
- `action` 表示示范者下一时刻希望 follower 到达的目标姿态。

继续使用上面的例子，正确的物理运动是 +10°。

| 处理方式 | 训练中看到的 state → action | 表面 delta | 是否正确 |
| --- | --- | ---: | --- |
| 两个都不转换 | `30 → 40` | `+10°` | 仍属于 source 表达，不能与 target absolute 数据直接混合 |
| 只转换 state | `51.98 → 40` | `-11.98°` | 错误，运动方向和幅度都被破坏 |
| 只转换 action | `30 → 61.98` | `+31.98°` | 错误，凭空放大动作 |
| state/action 都转换 | `51.98 → 61.98` | `+10°` | 正确，物理运动保持不变 |

行为克隆学习的是从 observation 到 action 的映射。只转换其中一个，相当于告诉模型：“机械臂现在在 target 坐标的 51.98°，专家却要求它去 source 坐标的 40°”，监督关系自然是错的。

### 35.6 如果不转换 action，真机上会发生什么

假设模型或数据仍输出 source 的 `40°`，但客户端使用 target calibration 解释这个值。Target 会把 40° 反算为：

~~~text
present_target = mid_target + 40/360*4095
               ≈ 2255 ticks
physical_encoder = present_target + homing_target
                 ≈ 2305 ticks
~~~

而 source 的 40° 原本对应：

~~~text
physical_encoder ≈ 2000 + 40/360*4095 + 100
                 ≈ 2555 ticks
~~~

两者相差约 250 ticks，也就是约 21.98°。因此完全相同的数字 `40°`，在两套 calibration 下会把机械臂发送到不同物理位置。

### 35.7 夹爪为什么更必须转换

夹爪使用 absolute 0～100，而不是前五轴那样转换为 delta。假设：

Source gripper：

~~~text
homing_offset = 100
range = [1000, 3000]
~~~

Target gripper：

~~~text
homing_offset = 300
range = [700, 2500]
~~~

Source 数据中的 25% 对应：

~~~text
present_source = 1000 + 25% * 2000 = 1500
physical_encoder = 1500 + 100 = 1600
~~~

同一物理夹爪位置在 target 下：

~~~text
present_target = 1600 - 300 = 1300
target_percentage = (1300 - 700)/(2500 - 700)*100
                  ≈ 33.33%
~~~

所以 source 的 25% 迁移到 target 后应写成约 33.33%。如果直接把 25% 发送给 target，它对应的物理夹爪开度会不同，可能导致抓不住或无法松开。

### 35.8 本项目转换前后到底改变了什么

转换前：

~~~text
视频：双物块示范的真实画面
state/action：使用新采集者 calibration 表达
统计信息：基于 source 数值计算
~~~

转换后：

~~~text
视频：完全不变
episode/frame/task：完全不变
前五轴 state/action：增加 source→target 的角度偏移
wrist roll：转换后重新 wrap 到 [-180, 180)
gripper state/action：按两套 range 和 homing offset 重新映射到 0～100
episode/global stats：根据新数值重新计算
~~~

本项目实际计算出的 source→target 身体关节偏移为：

| Joint | Offset |
| --- | ---: |
| shoulder_pan | `+1.318681°` |
| shoulder_lift | `+0.175824°` |
| elbow_flex | `+0.043956°` |
| wrist_flex | `-0.263736°` |
| wrist_roll | `-83.868132°` |

wrist roll 的差异接近 84°，说明它绝不是可以忽略的浮点噪声。

### 35.9 一个重要细节：前五轴 delta 会抵消常数零点偏移

本项目训练时对前五个身体关节计算：

~~~text
delta_action = absolute_action - current_state
~~~

若 calibration 差异对某个身体关节只是同一个常数 `c`：

~~~text
(action + c) - (state + c) = action - state
~~~

因此，在“前五轴只使用 delta、模型完全不消费 absolute state、没有角度 wrap/方向/尺度变化”的理想条件下，身体关节的局部 delta 监督确实可能不受常数零点偏移影响。这也是为什么不能简单地说“不转换就一定导致每个 delta 都错误”。

但项目仍统一转换，原因包括：

1. gripper 保持 absolute，range 和 offset 不一致不会抵消；
2. state 可能进入模型，具体取决于 checkpoint 的 `discrete_state_input` 和数据流；
3. wrist roll 存在 `[-180,180)` 回绕，边界附近不能只按普通常数处理；
4. 数据集中保留 absolute state/action，后续分析、norm、其他配置和真机 inverse transform 都需要统一语义；
5. warm-start checkpoint 和旧单物块数据基于 target calibration，统一坐标能避免隐含的数据域标识和任务冲突；
6. 当前转换成本低且可验证，比依赖“某个配置碰巧只看 delta”更稳健。

这是更严谨的结论：校准转换对 absolute action/state 是必要的；对纯常数偏移下的 delta body action 数学上可能抵消，但完整系统仍然需要统一坐标约定。

### 35.10 为什么转换后还要重新计算 norm

Calibration 转换改变了 parquet 中 state/action 的数值分布，尤其 gripper 可能同时发生平移和缩放，wrist roll 还可能发生回绕。原来的 min/max/mean/std/quantile 已不再描述转换后的数据。

因此需要依次更新：

~~~text
data parquet 中的 observation.state/action
        ↓
episode-level statistics
        ↓
global meta/stats.json
        ↓
按训练 filter、horizon 和 delta mask 重新计算 OpenPI norm_stats.json
~~~

Normalization 与 calibration 是两层不同的变换：

- calibration：物理编码器空间 → 机器人统一的 degree/percentage 坐标；
- normalization：degree/percentage 数据分布 → 模型更容易学习的数值范围。

不能用重算 norm 代替 calibration 对齐。Norm 只能缩放统计分布，无法保证同一物理姿态在两批数据中具有相同语义。

### 35.11 面试简洁回答

> Calibration 不只是记录机械臂活动范围。对 SO101 来说，`homing_offset` 和 range 中点共同定义身体关节的 0°，range 还定义夹爪 0～100 的映射。因此同一物理姿态在两套 calibration 下可能分别表示成 30° 和 52°。我做的转换相当于保持视频和物理轨迹不变，把新数据中的 state/action 从 source 坐标重新表达成旧单物块 checkpoint 使用的 target 坐标。State 和 action 必须一起转换，否则 observation-action 监督会错位。前五轴若只有常数零点差并且训练只使用 delta，偏移在 `action-state` 中可能抵消；但夹爪是 absolute、state 可能进入模型、wrist roll 有回绕，而且数据分析、norm 和真机 inverse transform 都要求统一坐标，所以项目仍对完整 state/action 做了显式、可验证的迁移。

## 36. 项目介绍 PPT 设计方案

### 36.1 PPT 的目标和主线

这份 PPT 面向 VLA、具身智能和机器人学习实习面试，建议正文控制在 8～10 分钟、10 页左右。它不是项目文档的压缩版，也不是安装命令汇总，而应讲清楚一条有转折的工程故事：

```text
为什么做真实机器人 VLA
        ↓
第一次杯子任务固定场景成功
        ↓
改变物体位置后失败，发现轨迹记忆
        ↓
重新设计位置随机化的长程双物块任务
        ↓
解决 calibration、数据融合、norm 和 warm-start
        ↓
解决公网时延、动作过期和真机安全问题
        ↓
完成双物块长程任务实机闭环验证
        ↓
分析关抽屉重试与尚未解决的能力边界
```

这条主线的核心不是“机械臂最终动了”，而是：

- 完成数据、训练、部署和真机验证的全流程；
- 从杯子任务的泛化失败中识别固定轨迹问题；
- 根据失败结论主动改变第二阶段的数据采集；
- 处理真实机器人数据中特有的 calibration、norm 和 action 语义一致性；
- 对闭环恢复、成功率和未实施方案保持严谨表述。

### 36.2 第 1 页：封面

推荐标题：

```text
PI0.5 VLA 在 SO101 上的长程多任务操作与真机部署
```

副标题：

```text
从数据采集、LoRA 微调到视觉闭环执行
```

页面下方放：

- 姓名；
- 应聘方向：VLA / 具身智能 / 机器人学习实习；
- 技术栈：PI0.5、OpenPI、LeRobot、SO101、LoRA、JAX。

推荐视觉：使用最终双物块任务的真机画面，确保能够同时看到机械臂、抽屉和黑白物块。不要使用终端截图作为封面。

开场话术：

> 这个项目不是只完成了一次模型微调，而是完整走通了 VLA 从数据采集、数据处理、训练到真实机械臂部署的闭环，并通过第一次任务的泛化失败重新设计了第二阶段的数据。

### 36.3 第 2 页：项目目标与最终结果

推荐标题：

```text
项目目标：打通真实机器人的完整 VLA 闭环
```

左侧使用流程图：

```text
Leader/Follower 遥操作采集
            ↓
LeRobot v3 双相机数据
            ↓
PI0.5 LoRA 微调
            ↓
GPU Policy Server
            ↓
SO101 视觉闭环执行
```

右侧只放最终结果：

- 支持单物块和双物块两条语言指令；
- 双物块任务包含四个连续子任务；
- 完成端到端真实 SO101 验证；
- 服务端推理中位延迟约 66 ms；
- 局域网 RTT 中位数约 90 ms。

页面底部使用四张连续帧：

```text
开抽屉 → 放黑块 → 放白块 → 关抽屉
```

本页只让面试官先知道“项目做到了哪里”，不要提前展开 config 和 loss。

### 36.4 第 3 页：第一阶段——杯子任务

推荐标题：

```text
第一次成功，却暴露了轨迹记忆问题
```

建议三栏布局：

#### 已完成

- 使用同事采集的 50 episodes、31,000 帧双相机数据；
- 在 AutoDL 完成 50k-step PI0.5 LoRA 微调；
- 迁移到公司局域网服务器后完成固定场景真机推理。

#### 发现的问题

- 物块或杯子位置稍微变化后任务明显失败；
- 机械臂仍执行接近训练示范的一条轨迹。

#### 得出的结论

> 固定场景中的一次成功，不等于模型学会了任务。

推荐视觉：左侧放固定位置成功帧，右侧放位置变化失败帧，中间标出物块/杯子的位置变化。最好准备一段 10～15 秒失败视频。

本页是整场介绍的转折点，需要主动说明训练 loss 已经很低，但位置泛化仍然差，从而说明 loss 不等于真机成功率。

### 36.5 第 4 页：重新设计长程任务和数据

推荐标题：

```text
从固定轨迹复现到位置随机化的长程任务
```

首先展示任务顺序：

```text
打开抽屉
   ↓
抓取黑色物块并放入
   ↓
抓取白色物块并放入
   ↓
关闭抽屉
```

然后展示数据设计：

- 本人采集 30 episodes、40,355 帧；
- 环境相机 + 腕部相机；
- 6 维 state、6 维 action；
- 采集频率 30 Hz；
- 在相机可见、机械臂可达范围内随机放置黑白物块。

推荐视觉：放 3～4 组不同物块摆放位置的顶视图，使用相同颜色的框标出黑块和白块。图片比“进行了数据增强”这句话更有说服力。

讲解重点：

> 数据随机化不是范围越大越好。它必须保持在相机可见、机械臂可达、遥操作能够稳定完成的范围内，否则少量数据会同时引入过多变化。

### 36.6 第 5 页：Calibration 对齐

推荐标题：

```text
两批数据不能直接合并：Calibration 定义了坐标表达
```

推荐结构图：

```text
单物块数据（Target Calibration）
                    \
                     → 坐标统一 → Mixed Dataset
                    /
双物块数据（Source Calibration）
```

页面正文只保留四点：

- 两批数据由不同人员使用不同 calibration 采集；
- calibration 不只是活动范围，还定义关节零点和夹爪 0～100 映射；
- 同时转换 `observation.state` 与 `action`；
- wrist roll 的 source→target 偏移约为 `-83.87°`。

页面右下角放一个简化数值例子：

```text
同一物理姿态：
Source 表示为 30°
Target 表示为 51.98°

Source：state 30 → action 40
Target：state 51.98 → action 61.98
物理运动仍为 +10°
```

讲解重点：只转换 state 或只转换 action 都会破坏 observation-action 对应关系。可以补充一个严谨例外：如果前五轴只有常数零点偏移且训练只看 delta，偏移可能抵消；但 gripper absolute、state 输入、wrist wrap、norm 和完整数据语义仍要求统一 calibration。

### 36.7 第 6 页：数据融合与训练方案

推荐标题：

```text
如何把单物块技能扩展为单/双物块多任务策略
```

先放数据表：

| 数据 | Episodes | Frames | Task prompt |
| --- | ---: | ---: | --- |
| 单物块 | 32 | 33,478 | 单物块指令 |
| 双物块 | 30 | 40,355 | 双物块指令 |
| 合计 | 62 | 73,833 | 2 tasks |

再画训练流程：

```text
单物块数据 + 校准后的双物块数据
                 ↓
保留两条 task prompt
                 ↓
Mixed sample-start filter
                 ↓
按实际 action 语义重新计算 mixture norm
                 ↓
加载单物块 checkpoint 35000/params
                 ↓
Fresh optimizer 训练 50,000 steps
```

右下角放关键参数：

```text
Batch size: 16
Action horizon: 10
Warmup: 1,000
Peak LR: 2e-5
Final checkpoint: 49999
```

必须主动解释：

> 这是参数 warm-start，不是从旧实验带着 optimizer 直接 resume。新旧数据 mixture、filter 和 norm 已经改变，所以新实验从 step 0 使用 fresh optimizer；只有新混合实验自身中断后才使用 resume。

### 36.8 第 7 页：PI0.5 数据与模型流程

推荐标题：

```text
从双相机观测到 10-step Action Chunk
```

推荐横向流程图：

```text
双相机 + Prompt + 6D State
              ↓
Repack / SO101 Inputs
              ↓
前五轴 Delta + Gripper Absolute
              ↓
Normalization
              ↓
PaliGemma 视觉语言主干
              ↓
Action Expert + Flow Matching
              ↓
[10, 32] Action Chunk
              ↓
反归一化 + 恢复 Absolute + 裁成 [10, 6]
```

正文只解释三个理论点：

- LoRA：冻结基础权重，用低秩参数适配少量真机数据；
- Flow matching：从带噪动作学习速度场，推理时从噪声积分得到动作序列；
- Action chunking：一次预测未来 10 步，学习短期时间一致性并减少推理调用。

Loss 可以在图下简写为：

```text
x_t = (1-t)a + tε
u_t = ε-a
Loss = mean[(vθ-u_t)²]
```

不要在正文推导 Transformer、RoPE 或完整 flow matching 数学，放到技术附录即可。

### 36.9 第 8 页：真机部署与实时闭环

推荐标题：

```text
公网可连接，不等于满足实时控制
```

画出部署结构：

```text
本地 Ubuntu
双相机 + SO101 + 30 Hz Body
              │
              │ WebSocket
              ▼
公司 GPU 服务器
PI0.5 Policy Server
```

放一张延迟对比表：

| 部署方式 | 延迟表现 | 结论 |
| --- | ---: | --- |
| AutoDL 公网 | 单轮约 1～3 秒 | 超过整个 action horizon |
| 公司局域网 | RTT 中位数约 90 ms | 可以进行短 action chunk 闭环 |

突出计算：

```text
10 steps / 30 Hz ≈ 333 ms
```

客户端实现：

- Camera、Brain、Body 解耦；
- 根据 observation/action age 跳过已过期的 chunk 前缀；
- stale action 拒绝；
- 动作低通、关节/夹爪限幅；
- 异常、断线或无新动作时安全 Hold。

讲解重点：端口能连通只证明网络可达，不能证明动作到达时仍然有效。

### 36.10 第 9 页：真机结果与失败分析

推荐标题：

```text
完成端到端任务，但关闭阶段仍是主要瓶颈
```

左侧嵌入 20～30 秒最终任务视频或四张连续帧；右侧放结论：

- 同一 checkpoint 可根据 prompt 执行单物块和双物块任务；
- 双物块任务完整执行到最终关闭抽屉；
- 物块位置存在变化时仍能完成抓取；
- 关闭抽屉有时需要多次重新定位。

建议使用结果分级：

| 分类 | 定义 |
| --- | --- |
| 严格成功 | 固定时限内一次完成所有子任务，无明显关闭反复 |
| 恢复后成功 | 首次关闭失败，策略根据新观测自行调整后完成 |
| 部分成功 | 两个物块已放入，但抽屉未完全关闭 |
| 失败 | 超时、人工干预、危险碰撞、掉落或顺序错误 |

推荐表述：

> 关闭失败后，策略会根据后续观测产生不同动作并最终完成，说明系统呈现一定的视觉闭环恢复行为。但这不能直接证明模型理解了失败，也不能用一次最终成功代替标准化成功率。

如果有空间，可以放一次 90 秒运行指标：服务端约 65.9 ms、RTT 约 89.9 ms、接受 776 个动作块、stale 拒绝 1 次、Body missed ticks 为 0。

### 36.11 第 10 页：总结、个人贡献与下一步

推荐标题：

```text
项目总结：真实 VLA 的难点是全链路一致性
```

左侧放个人贡献：

- 设计并采集位置随机化的双物块数据；
- 完成跨 calibration 的 state/action 转换；
- 完成数据合并、task prompt、filter 和 norm；
- 完成 PI0.5 LoRA warm-start、训练恢复和 W&B/Orbax 管理；
- 构建局域网视觉闭环和真机安全控制；
- 从失败现象区分数据、模型、网络、客户端和硬件问题。

右侧放尚未实施的下一步：

- 随机化抽屉位置、开度和推入接触点；
- 增加首次推偏、没有关严等恢复示范；
- 清理 episode 开头黑屏和曝光异常帧；
- 进行 20～30 次固定时限标准化评测；
- 在数据和评测成熟后尝试 RL 后训练。

结束语：

> 这个项目让我认识到，真实机器人 VLA 的核心不只是模型训练，而是数据分布、calibration、动作语义、系统实时性、安全控制和闭环评测的一致性。

### 36.12 技术附录建议

附录用于面试官追问，不计入 10 页正文，也不需要主动逐页讲。

#### 附录 A：LoRA

放公式：

```text
W = W0 + (alpha/r)BA
```

准备回答训练参数量、显存、rank、学习率和小数据过拟合。

#### 附录 B：Flow Matching 与 Loss

使用“真实动作 ↔ 带噪动作”的箭头图，说明 `x_t`、目标速度 `u_t` 和模型预测 `v_theta`。补充 loss 不是完整任务奖励，而是局部 10-step 速度回归误差。

#### 附录 C：Warm-start 与 Resume

| Warm-start | Resume |
| --- | --- |
| 只加载模型参数 | 恢复模型、optimizer、step 和数据状态 |
| 可用于新数据 mixture | 只适合同一实验中断 |
| 新实验从 step 0 开始 | 延续原实验 step |

#### 附录 D：Calibration

展示：

```text
physical encoder
      ↓ homing_offset / range
degrees or gripper percentage
      ↓ dataset norm
model normalized value
```

放 `30°→40°` 迁移成 `51.98°→61.98°` 的例子，并准备解释为什么 state/action 要一起转换。

#### 附录 E：Loss、Grad Norm、Param Norm

| 指标 | 含义 | 不能说明什么 |
| --- | --- | --- |
| Loss | Flow velocity 拟合误差 | 不能直接说明任务成功率 |
| Grad norm | 可训练参数梯度的全局范数 | 不能直接说明模型能力 |
| Param norm | 当前代码统计的 kernel 总体尺度 | 不能说明 LoRA 学到了多少 |

#### 附录 F：正式评测与消融

准备位置分组、子任务成功率、首次关闭/恢复后关闭、完成时间和安全失败指标，以及固定位置 vs 随机位置、单相机 vs 双相机、旧 norm vs mixture norm 等消融方案。

### 36.13 最值得准备的展示素材

正文优先使用以下材料：

1. 20～30 秒双物块最终任务视频，保留完整任务顺序；
2. 杯子位置变化后的失败视频，用来解释为什么重新设计数据；
3. 3～4 张不同黑白物块位置的采集画面；
4. calibration 转换前后的关节分布，重点显示 wrist roll；
5. 数据样例：双相机、state/action 和 task prompt；
6. W&B loss/grad norm/param norm 曲线，并主动标注“训练指标 ≠ 真机成功率”；
7. 客户端 RTT、action age 和 stale 统计截图；
8. 一张完整系统架构图和一张失败分析图。

最有说服力的展示顺序是：

```text
杯子位置变化后的失败
        ↓
第二阶段位置随机化数据
        ↓
最终双物块任务视频
```

它能够直观呈现“发现问题—重新设计—完成验证”的项目闭环。

### 36.14 PPT 视觉规范

- 使用 16:9 页面比例；
- 正文 9～10 页，每页只表达一个核心观点；
- 标题尽量写成结论，例如“公网可连接，不等于满足实时控制”；
- 标题字号至少 30 pt，正文字号至少 20～22 pt；
- 每页正文尽量不超过 6～8 个短句；
- 使用最多三种主色：深色正文、蓝色方案、红色问题/风险；
- 图片和流程图优先于大段文字；
- 不在正文放长命令、完整 config、终端安装日志和大段源码；
- 流程箭头、字体、圆角和颜色风格保持一致；
- 图片标注清楚 `env camera`、`wrist camera`、黑块、白块和抽屉；
- 所有视频保存在本地并提前测试，不依赖现场网络。

### 36.15 8～10 分钟演讲时间分配

| 页面 | 建议时间 |
| --- | ---: |
| 1. 封面 | 20 秒 |
| 2. 目标与结果 | 45 秒 |
| 3. 杯子任务与泛化失败 | 60 秒 |
| 4. 双物块任务与数据设计 | 60 秒 |
| 5. Calibration | 75 秒 |
| 6. 数据融合与训练 | 75 秒 |
| 7. 模型数据流 | 60 秒 |
| 8. 部署与实时性 | 75 秒 |
| 9. 结果与失败分析 | 75 秒 |
| 10. 总结 | 40 秒 |

如果只有 5 分钟，可以合并：

- 第 1、2 页；
- 第 4、5、6 页为一页“数据与训练”；
- 第 7、8 页为一页“模型与部署”；
- 保留第 3 页失败转折和第 9 页最终结果。

### 36.16 演讲时的表达原则

- 先说结论，再解释原因，最后给数字或视频证据；
- 主动区分本人采集的双物块数据与同事已有的杯子/单物块数据；
- 不从 Conda、依赖下载和系统盘清理开始讲项目；
- 不把 loss 下降说成任务成功率提升；
- 不把恢复后成功说成模型“理解失败”或“自主思考”；
- 不把 RL、抽屉随机化和恢复数据写成已经完成；
- 不回避失败，杯子任务泛化失败是推动第二阶段设计的关键证据；
- 当面试官追问理论时再进入附录，正文始终围绕项目决策和实验结果。

### 36.17 PPT 开场与结束完整话术

开场：

> 我介绍的是 PI0.5 在 SO101 上的端到端 VLA 项目。我完整参与了训练部署、双物块任务设计与采集、跨 calibration 数据融合、LoRA 继续训练和局域网真机闭环。项目最重要的转折是：第一次杯子任务虽然在固定位置成功，但位置变化后模型仍重复相似轨迹，所以我没有把一次成功当作任务学会，而是在第二阶段主动随机化物块位置并设计了更长程的双物块抽屉任务。

结束：

> 最终同一个 checkpoint 能够根据语言指令执行单物块和双物块任务，双物块任务完成了端到端实机验证。关闭阶段仍会出现重试，因此我将结果区分为严格成功和恢复后成功，没有宣称未经统计的稳定成功率。这个项目让我认识到，真实 VLA 的核心不仅是模型本身，而是数据分布、calibration、norm、动作语义、实时控制和评测标准的全链路一致性。
