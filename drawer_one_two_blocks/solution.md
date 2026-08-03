# PI0.5 SO101 单物块/双物块抽屉混合训练记录

## 1. 目标与最终方案

本次任务在已经通过真机验证的单物块抽屉 LoRA checkpoint 35000 基础上，引入“依次抓取两个物块放入抽屉并关闭抽屉”的新数据，并继续训练一个同时覆盖两种指令的 PI0.5 LoRA。

最终采用的方案是：

1. 下载双物块 LeRobot 数据集 `ycyc0926/drawer_two_blocks`。
2. 因为两批数据由不同人员使用不同校准文件采集，但 follower 是同一台机械臂，所以把新数据从新采集者校准系转换到旧数据校准系。
3. 将旧单物块数据和转换后的双物块数据合并为一个 LeRobot v3 数据集，并保留两个不同 task prompt。
4. 旧数据继续沿用原有 idle-frame keep ranges；新数据保留全部帧起点。
5. 针对混合后的有效样本重新计算 state/action normalization statistics。
6. 模型参数从已通过真机验证的旧 checkpoint 35000 warm-start；由于任务分布和 norm 都改变，优化器从零重新初始化。
7. 新混合训练先后从完整 checkpoint 1000、4000 恢复模型、优化器和数据加载状态，最终完成 50000-step 训练并保存 checkpoint 49999。
8. 使用独立 W&B 0.28.1 日志旁路进程，回填历史指标并持续实时上传后续曲线，避免 OpenPI 环境中旧 W&B 0.19.11 与服务器其他账号作业发生凭证串线。
9. 使用 WebSocket 构建服务器 GPU 推理、Windows 本地 SO101 控制的远程闭环系统；模型根据最新双目图像和关节状态反复推理，真机表现出抽屉关闭失败后的再次对准与恢复行为。

## 2. 输入与输出

### 输入

- 旧单物块数据：`/hdd/tyf/pilot_open_drawer_place_block_close_drawer_merged_v1_0723`
- 新双物块 Hugging Face 数据集：`ycyc0926/drawer_two_blocks`
- 下载后的新数据：`/hdd/tyf/drawer_two_blocks_raw_0731`
- 新数据 follower 校准：`/home/likunwei/pi0/VLA/openpi/jiaozhun/1/drawer_follower.json`
- 旧数据 follower 校准：`/home/likunwei/pi0/VLA/openpi/jiaozhun/2/my_awesome_follower_arm.json`
- 已通过真机验证的旧参数：

```text
/hdd/tyf/openpi_checkpoints/
pi05_so101_drawer_lora_idle_filtered/
drawer_merged_v1_0723_idle_filtered_lora_100k/
35000/params
```

### 输出

- 校准对齐的新数据：`/hdd/tyf/drawer_two_blocks_calib2_0731`
- 合并数据：`/hdd/tyf/drawer_one_two_blocks_calib2_merged_v1_0731`
- 混合过滤文件：`/hdd/tyf/drawer_one_two_blocks_calib2_merged_v1_0731/meta/openpi_mixed_filter_v1.json`
- 新 norm：

```text
/home/likunwei/pi0/VLA/openpi/assets/
pi05_so101_drawer_one_two_blocks_calib2_lora/
so101_drawer_one_two_blocks_calib2_mixed_filter_v1/
norm_stats.json
```

- 训练配置名：`pi05_so101_drawer_one_two_blocks_calib2_lora`
- 正式实验名：`drawer_one_two_blocks_calib2_from_35k_lora_30k_0731`
- checkpoint 目录：

```text
/hdd/tyf/openpi_checkpoints/
pi05_so101_drawer_one_two_blocks_calib2_lora/
drawer_one_two_blocks_calib2_from_35k_lora_30k_0731
```

- 训练日志：`/hdd/tyf/openpi_training_logs/drawer_one_two_blocks_calib2_from_35k_lora_30k_0731.log`
- W&B 旁路日志：`/hdd/tyf/openpi_training_logs/drawer_one_two_blocks_calib2_from_35k_lora_30k_0731_wandb.log`
- W&B run：<https://wandb.ai/can498987-/openpi/runs/3atuhn9w>

## 3. 数据处理详情

### 3.1 下载新数据

可使用下面的等价命令下载，不需要通过数据可视化 Space 抓取单个 episode：

```bash
huggingface-cli download ycyc0926/drawer_two_blocks \
  --repo-type dataset \
  --local-dir /hdd/tyf/drawer_two_blocks_raw_0731
```

下载结果为 LeRobot v3 数据集：30 episodes、40355 frames、30 FPS、1 个 task。

### 3.2 校准转换

使用新增脚本：

```bash
cd /home/likunwei/pi0/VLA/openpi

.venv/bin/python scripts/convert_so101_calibration.py \
  --source-root /hdd/tyf/drawer_two_blocks_raw_0731 \
  --output-root /hdd/tyf/drawer_two_blocks_calib2_0731 \
  --source-calibration jiaozhun/1/drawer_follower.json \
  --target-calibration jiaozhun/2/my_awesome_follower_arm.json
```

脚本先完整复制原数据，再修改副本，不原地覆盖下载数据。转换规则为：

- 前 5 个身体关节通过 encoder 中点、homing offset 和 4095 tick/revolution 转换到旧校准坐标。
- wrist roll 转换后归一到 `[-180, 180)`。
- gripper 通过源/目标 `range_min`、`range_max` 和 homing offset 映射到目标 0-100 区间，并裁剪到合法范围。
- 同时转换 `action` 和 `observation.state`。
- 重新生成全局 `meta/stats.json` 和 episode 级统计字段。

本次计算出的身体关节偏移为：

| Joint | source -> target offset |
|---|---:|
| shoulder_pan | +1.318681 deg |
| shoulder_lift | +0.175824 deg |
| elbow_flex | +0.043956 deg |
| wrist_flex | -0.263736 deg |
| wrist_roll | -83.868132 deg |

较大的 wrist-roll 偏移来自两份校准对同一 follower 使用了不同零点约定，因此不能直接把两批数值拼接后训练。

### 3.3 合并数据

合并时保留旧数据为 episodes 0-31，将转换后的新数据重编号为 episodes 32-61；连续重建 frame/index、episode metadata、视频引用、全局统计和 task 表。

| 数据部分 | Episodes | Frames | Task index |
|---|---:|---:|---:|
| 单物块旧数据 | 32 | 33478 | 0 |
| 双物块新数据 | 30 | 40355 | 1 |
| 合并结果 | 62 | 73833 | 0/1 |

保留的两个 task prompt 为：

```text
Open the drawer, place the block inside, and close the drawer
```

```text
Open the drawer, put the black block into the drawer, then put the white block into the drawer, and close the drawer.
```

配置中使用 `DataConfig(prompt_from_task=True)`，并在 SO101 repack 中保留 `prompt` 字段，因此模型能够看到不同指令，而不是把两种行为混成同一个默认文本。

### 3.4 构建混合样本过滤文件

使用：

```bash
.venv/bin/python scripts/build_mixed_lerobot_filter.py \
  --merged-root /hdd/tyf/drawer_one_two_blocks_calib2_merged_v1_0731 \
  --old-filter /hdd/tyf/pilot_open_drawer_place_block_close_drawer_merged_v1_0723/meta/openpi_idle_filter_v1.json \
  --old-episode-count 32 \
  --output /hdd/tyf/drawer_one_two_blocks_calib2_merged_v1_0731/meta/openpi_mixed_filter_v1.json
```

结果：

- 原始总 sample starts：73833
- 旧数据过滤后保留：31521
- 新数据全量保留：40355
- 混合后有效 starts：71876
- 过滤文件完整覆盖 62 个 episode。

`EpisodeRangeFilteredDataset` 只改变允许作为训练窗口起点的索引，不修改底层 LeRobot 数据；动作 horizon 仍由原数据集按 episode 边界读取。

### 3.5 重新计算 normalization statistics

使用快速 parquet 版本避免为 state/action norm 无意义地解码相机视频：

```bash
PYTHONPATH=src:/home/likunwei/lerobot/src \
.venv/bin/python scripts/compute_so101_norm_stats_fast.py \
  --config-name pi05_so101_drawer_one_two_blocks_calib2_lora
```

计算与训练配置保持一致：

- 使用 mixed filter。
- action horizon 为 10。
- 前 5 个关节转换为 delta action，gripper 保持绝对值。
- batch size 为 16。
- 71876 个有效 starts 中，按照训练的 `drop_last` 语义使用 71872 个。
- 输出独立 asset ID，避免写入或污染数据集目录。

## 4. 代码改动

### 修改的现有文件

1. `openpi/src/openpi/training/config.py`
   - 为 SO101 数据配置接入可选 sample-range filter。
   - 当 `prompt_from_task=True` 时把 LeRobot task prompt 放入 repack。
   - 新增 `pi05_so101_drawer_one_two_blocks_calib2_lora`。
   - 配置 mixed dataset、mixed norm asset、mixed filter、checkpoint 35000 warm-start、LoRA、50000 steps、batch 16、1000-step warmup、peak LR `2e-5`、AdamW gradient clipping 0.5。

2. `openpi/src/openpi/training/data_loader.py`
   - 新增 `EpisodeRangeFilteredDataset`。
   - 校验 JSON episode/range、episode 长度和边界。
   - 将过滤后的局部索引映射回 LeRobot 全局索引。
   - 保持未配置 filter 的原始行为不变。

3. `openpi/scripts/compute_norm_stats.py`
   - 输出正在使用的数据集、asset ID 和 filter。
   - filter 缺失时立即报错。
   - 优先写入显式 asset ID，防止绝对 repo path 覆盖输出位置。

### 新增代码文件

1. `openpi/scripts/convert_so101_calibration.py`
   - 安全复制并转换 SO101 calibration coordinate system。
   - 更新 parquet、episode stats 和 global stats。

2. `openpi/scripts/build_mixed_lerobot_filter.py`
   - 复用旧数据 keep ranges，为新 episodes 自动加入全范围。

3. `openpi/scripts/compute_so101_norm_stats_fast.py`
   - 直接读取 parquet 计算与训练语义一致的 SO101 norm。

4. `openpi/scripts/upload_openpi_training_log_to_wandb.py`
   - 从 `TRAIN_METRICS` 文本日志回填 loss/grad norm/param norm。
   - 支持 `--max-step`、去重、run ID 文件、`--follow` 实时追踪。
   - 训练结束标记出现后自动 finish W&B run。

### 新增的小型生成文件

1. `openpi/assets/pi05_so101_drawer_one_two_blocks_calib2_lora/so101_drawer_one_two_blocks_calib2_mixed_filter_v1/norm_stats.json`
2. `artifacts/openpi_mixed_filter_v1.json`
3. `solution.md`

## 5. 训练配置与命令

### 5.1 第一次从旧 checkpoint 35000 warm-start

配置中的 `CheckpointWeightLoader(.../35000/params)` 只加载模型参数；优化器为新的任务混合重新初始化，训练 step 从 0 开始。

```bash
cd /home/likunwei/pi0/VLA/openpi

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/likunwei/miniconda3/envs/pi0-zero/lib:$LD_LIBRARY_PATH
export PYTHONPATH=src:/home/likunwei/lerobot/src:$PYTHONPATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=disabled

.venv/bin/python scripts/train.py \
  pi05_so101_drawer_one_two_blocks_calib2_lora \
  --exp-name drawer_one_two_blocks_calib2_from_35k_lora_30k_0731 \
  --checkpoint-base-dir /hdd/tyf/openpi_checkpoints \
  --num-train-steps 50000 \
  --batch-size 16 \
  --num-workers 12 \
  --log-interval 100 \
  --save-interval 1000 \
  --keep-period 5000 \
  --no-wandb-enabled
```

### 5.2 从新混合 checkpoint 1000 恢复

checkpoint 1000 大小约 8.9 GB，包含完整 train state。恢复时增加 `--resume`，日志必须使用 `tee -a`，不能覆盖 step 0-1000 的历史记录：

```bash
.venv/bin/python scripts/train.py \
  pi05_so101_drawer_one_two_blocks_calib2_lora \
  --exp-name drawer_one_two_blocks_calib2_from_35k_lora_30k_0731 \
  --checkpoint-base-dir /hdd/tyf/openpi_checkpoints \
  --num-train-steps 50000 \
  --batch-size 16 \
  --num-workers 12 \
  --log-interval 100 \
  --save-interval 1000 \
  --keep-period 5000 \
  --resume \
  --no-wandb-enabled \
  2>&1 | tee -a /hdd/tyf/openpi_training_logs/drawer_one_two_blocks_calib2_from_35k_lora_30k_0731.log
```

训练 tmux：`drawer_two_blocks_0731`。

2026-07-31 训练曾在约 step 4890 出现 JAX/XLA/CUDA 主线程自旋：GPU 参数仍驻留显存，但利用率为 0%，主进程占用一个 CPU 核且 DataLoader workers 全部休眠。进程无法响应两次 `Ctrl-C`，最终按 `SIGTERM`、必要时 `SIGKILL` 的顺序终止，并从最近完整 checkpoint 4000 恢复。恢复时将训练目标和 cosine decay 一并从 30000 延长为 50000；已有实验目录名中的 `_30k_` 为历史命名，不代表当前实际目标步数。

## 6. W&B 实时曲线

服务器同时有另一个账号的 W&B 作业，且 OpenPI `.venv` 内的 W&B 0.19.11 不可靠地读取到了旧账号。为避免升级训练环境或影响 GPU 1 作业，使用独立凭证文件和隔离 W&B 0.28.1：

```text
/hdd/tyf/wandb_can498987.netrc
```

该文件权限为 600，不进入 Git，也不上传。本仓库中不保存任何 API key。

实时旁路命令的核心形式为：

```bash
NETRC=/hdd/tyf/wandb_can498987.netrc \
WANDB_MODE=online \
WANDB_DIR=/hdd/tyf/wandb \
uv run --isolated --no-project --with wandb==0.28.1 \
  python scripts/upload_openpi_training_log_to_wandb.py \
  --log /hdd/tyf/openpi_training_logs/drawer_one_two_blocks_calib2_from_35k_lora_30k_0731.log \
  --project openpi \
  --entity can498987- \
  --run-name drawer_one_two_blocks_calib2_from_35k_lora_30k_0731 \
  --group so101_drawer_one_two_blocks_calib2 \
  --max-step 1000 \
  --follow \
  --poll-interval 2
```

W&B tmux：`drawer_wandb_0731`。

训练每 100 steps 写一次 `TRAIN_METRICS`，因此网页通常每约 2-3 分钟出现一个新点。W&B 云端 API 已确认 run state 为 `running`，并成功收到恢复后的 step 1100；随后 step 1800、1900、2000 也已实时上传。

## 7. checkpoint 与容量

- 主机 RAM：249 GiB，总可用量检查时约 207 GiB。
- GPU 0：32.6 GiB，训练时约使用 30.3 GiB。
- `/hdd`：检查时剩余约 8.3 TB。
- 每个 checkpoint：约 8.9 GB。
- `save_interval=1000`，`keep_period=5000`，Orbax `max_to_keep=1`。

因此并不会永久保留所有 1000-step checkpoint。50000-step 目标预计主要保留 5000、10000、15000、20000、25000、30000、35000、40000、45000 和最终 checkpoint，总量约 89 GB，外加少量异步保存临时空间。

## 8. 验证结果

- 新旧数据均为同一 SO101 follower、LeRobot v3、30 FPS。
- 校准转换后的 episode/frame 数与原新数据完全一致。
- 合并数据 episode index、frame index 和 dataset bounds 连续。
- 合并结果为 62 episodes、73833 frames、2 tasks。
- mixed filter 覆盖全部 62 episodes，保留 71876 sample starts。
- norm 文件成功生成并由训练配置加载。
- AV1 视频通过 PyAV/libdav1d 解码检查。
- checkpoint 1000 成功恢复，读取约 10.7 GiB model/train state。
- GPU 0 恢复后达到正常 100% utilization。
- 恢复后 step 1100 指标为 `loss=0.0092`、`grad_norm=0.0803`。
- W&B 云端确认 step 1100；后续 step 2000 已同步，loss 为 `0.0067`。
- 训练最终完成到目标 50000 steps，并保存完整 checkpoint 49999；最终目录约 8.9 GB，包含 params、train state 和 norm assets。
- 最终模型已完成服务器/客户端远程真机推理验证；一次关闭未完成时，策略能够依据后续观测继续调整并最终关闭抽屉。
- 实测单次服务端推理约 63-66 ms，局域网 RTT 约 86-125 ms；客户端依据端到端延迟从 10-step action chunk 中自适应跳过约 3 个过期动作。
- 关闭抽屉仍存在一次成功率不稳定的问题；当前结论是模型具备一定闭环恢复能力，但尚需通过针对性恢复数据、位置随机化和更严格的对照实验提高首次关闭成功率。

## 9. 未上传到公开仓库的内容

以下内容有意不上传：

- 原始、校准转换后和合并后的数据集（体积较大）。
- 所有 checkpoint 和 optimizer state。
- 训练/W&B 本地日志和 W&B 二进制 run 文件。
- `~/.netrc`、独立 W&B netrc、API key 或其他凭证。
- 两位采集者的原始 calibration 文件；文档只记录本机输入路径和转换方法。
- 与本实验无关的 websocket、FA16、BridgeVLA、其他 W&B 工具和备份文件。

## 10. 发布目录结构

目标仓库：`ycyc0926/VLA-pi0.5-SO101`，分支：`main`。

```text
drawer_one_two_blocks/
├── solution.md
├── artifacts/
│   └── openpi_mixed_filter_v1.json
└── openpi/
    ├── assets/pi05_so101_drawer_one_two_blocks_calib2_lora/
    │   └── so101_drawer_one_two_blocks_calib2_mixed_filter_v1/
    │       └── norm_stats.json
    ├── scripts/
    │   ├── build_mixed_lerobot_filter.py
    │   ├── compute_norm_stats.py
    │   ├── compute_so101_norm_stats_fast.py
    │   ├── convert_so101_calibration.py
    │   └── upload_openpi_training_log_to_wandb.py
    └── src/openpi/training/
        ├── config.py
        └── data_loader.py
```

发布过程不 clone 或 checkout 目标仓库；只在临时 bare Git object database 中 fetch `main` 的单个浅层 commit/tree，以该 commit 为 parent 添加 `drawer_one_two_blocks/`，然后 fast-forward push。

## 11. 简历版项目总结

### 11.1 一句话介绍

基于 PI0.5 和 LoRA 为 SO101 单臂机器人构建“打开抽屉、抓取一个或两个物块、放入抽屉并关闭”的多任务 VLA 系统，完成了从异构真机数据校准、LeRobot 数据合并、训练恢复与 W&B 监控，到 WebSocket 远程闭环推理和真机问题诊断的完整链路。

### 11.2 简历项目描述示例

**PI0.5 + SO101 多任务抽屉操作 VLA**

- 基于 PI0.5 LoRA 将已验证的单物块策略扩展为单/双物块语言条件多任务策略，处理 62 个真机 episodes、73833 帧、2 条任务指令，并完成 50000-step 微调。
- 解决两位采集者使用不同 follower calibration 导致的坐标系不一致问题，实现 state/action 联合校准转换、episode/global statistics 重建和转换结果校验。
- 扩展 OpenPI 数据管线，支持按 episode range 过滤低价值 idle 样本、保留 task prompt，并实现不解码视频的快速 norm 计算，将数据处理与训练语义对齐。
- 设计 checkpoint warm-start/resume、Orbax 周期保留、训练日志旁路上传 W&B 的容错训练流程；定位并恢复 JAX/XLA/CUDA 主线程自旋导致的 GPU 低利用率故障。
- 搭建 GPU 服务端与 Windows SO101 客户端的 WebSocket 闭环推理系统，实测服务端推理约 63-66 ms、局域网 RTT 约 86-125 ms，并使用动作块时延对齐、限速、Hold、零状态和异常动作检查保障真机安全。
- 真机观察到抽屉未一次关闭时策略可依据新观测继续对准并最终完成，进一步设计失败状态恢复示范、抽屉位置随机化和关闭阶段过采样方案，以提高首次成功率与空间泛化性。

### 11.3 技术栈

```text
PI0.5 / LoRA / JAX / XLA / OpenPI / LeRobot v3
SO101 / Feetech / PyAV / AV1 / WebSocket
Orbax Checkpoint / W&B / tmux / Hugging Face Dataset
Python / NumPy / PyArrow / OpenCV
```

### 11.4 项目中可以强调的个人贡献

1. 不只是调用训练脚本，而是打通了数据、训练、部署和真机评估的完整闭环。
2. 发现“同一台 follower 不等于数据坐标天然一致”，实现了不同 calibration 之间的 state/action 对齐。
3. 修改 OpenPI 数据加载与 norm 计算逻辑，使过滤、prompt、delta action 和 normalization 的训练语义保持一致。
4. 能区分参数 warm-start 和完整 resume，并针对数据分布变化选择新的 optimizer 和实验目录。
5. 对 GPU 利用率、DataLoader、JAX 主线程、checkpoint 完整性和 W&B 上传链路进行系统化故障排查。
6. 将真机失败现象转化为可验证的数据与控制改进方案，而不是只依赖继续增加训练步数。

## 12. 从采集到推理的完整流程

### 12.1 采集与任务设计

1. 使用同一台 SO101 follower 采集单物块和双物块抽屉操作。
2. 为两种任务分别保留精确语言指令，避免多任务数据被同一个默认 prompt 混合。
3. 记录环境相机、腕部相机、六维关节状态和六维绝对动作，数据频率为 30 FPS。
4. 后续补采应同时包含标准一次成功示范和从“没关严、推偏、反弹”等状态直接开始的恢复示范。
5. 通过左右/前后移动和轻微旋转抽屉、改变开度与接触位置提高空间泛化性，但不采集已经关闭后持续大力推压的危险行为。

### 12.2 数据下载与完整性检查

1. 从 Hugging Face 下载完整 LeRobot v3 dataset，而不是从可视化页面逐 episode 抓取。
2. 检查 episode/frame 数、task 表、parquet schema、视频引用和时间戳。
3. 保留原始数据只读副本，所有转换写入新目录，避免不可恢复的原地修改。

### 12.3 校准坐标统一

1. 读取源采集者和目标 follower 的 calibration JSON。
2. 将新数据的前五轴 encoder/角度、wrist roll 和 gripper 映射到旧数据坐标系。
3. 同时转换 `observation.state` 与 `action`；只转换其中一个会破坏监督学习目标。
4. 重建 episode/global stats，并通过范围、分位数、episode 数和 frame 数验证转换结果。

### 12.4 合并、过滤与 prompt

1. 重编号 episode、frame 和 global index，合并 parquet、视频 metadata、episode metadata 和 task 表。
2. 使用 `prompt_from_task=True` 保留单物块和双物块两条指令。
3. 旧数据复用已经验证的 idle keep ranges，新数据先保留全部窗口起点。
4. Filter 只限制训练窗口的合法起点，不破坏底层数据和 10-step action horizon。
5. 对接触、缓慢推抽屉、失败后的细微调整和终止 Hold 等低速度帧应放宽过滤，避免误删关键恢复动作。

### 12.5 normalization

1. 按训练真实语义计算 norm：相同 filter、10-step horizon、前五轴 delta action、gripper 绝对值。
2. 使用 PyArrow 直接读取 parquet，state/action norm 不解码视频。
3. 新任务与旧任务合并后，norm 必须来自完整训练 mixture，不能只根据少量恢复数据计算。
4. 如果后续新增数据仍在现有 q01-q99 内，可评估复用旧 norm 以减少尺度漂移；若大量超出范围，则基于完整新旧 mixture 重算。

### 12.6 配置与训练

1. 在 `config.py` 中定义 PI0.5、LoRA、action horizon、dataset、filter、asset ID、prompt 和优化器。
2. 首次扩展任务时从已验证 checkpoint 的 `params` warm-start，并使用新 optimizer 从 step 0 开始。
3. 只有数据、norm、模型和实验目录完全不变时才使用 `--resume` 恢复完整 train state。
4. 使用 GPU 0、batch size 16、定期日志和 checkpoint；最终完成 50000 steps，保存 checkpoint 49999。
5. W&B 旁路进程解析 `TRAIN_METRICS`，通过固定 run ID 去重并在训练重启后继续上传同一条曲线。

### 12.7 服务端与客户端推理

1. 服务端从 checkpoint 根目录加载 params 与 norm assets，在 GPU 上启动 WebSocket policy server。
2. Windows 客户端连接 SO101、环境/腕部相机，并向服务器发送最新图像、六维状态和精确 prompt。
3. 服务端返回 `(10, 6)` action chunk；客户端根据图像年龄、RTT 和控制频率跳过已经过期的前缀动作。
4. Body 循环执行限速、低通平滑、夹爪裁剪、动作维度/NaN 检查和 Hold，异常时停止使用旧动作。
5. 真机评估分别记录一次成功率、最终成功率、平均恢复次数、完成时间、振荡和持续推压时间。

### 12.8 下一轮恢复数据训练

新增恢复数据后，不应直接在原实验上 `--resume`。推荐流程为：

```text
比较 40000 / 45000 / 49999 真机效果
                  ↓
选择最佳 checkpoint 的 params
                  ↓
旧单物块 + 旧双物块 + 新位置一次成功 + 失败状态恢复数据
                  ↓
重新构建 filter，检查或重算完整 mixture norm
                  ↓
新 config / 新实验名 / fresh optimizer / 较低学习率
                  ↓
训练 10000-20000 steps，并周期性真机评估
```

建议按 sample 数量从以下比例起步：原单物块 30%、原双物块 30%、新位置一次成功 25%、关闭恢复 15%。恢复数据最好从已经失败的状态开始，第一帧之后就是正确修正，避免模型学会“先故意失败再恢复”。

## 13. 主要问题与解决方式

| 问题 | 原因与判断 | 解决方式 |
|---|---|---|
| 两批数据数值分布不一致 | 同一 follower 被两位采集者使用不同 calibration，尤其 wrist roll 零点相差约 83.87° | 将新数据 state/action 统一转换到旧 `calib2`，重建统计后再合并 |
| 两种任务可能被模型混淆 | 默认 prompt 会丢失单/双物块语言区别 | 合并 task 表并开启 `prompt_from_task=True`，推理时传入训练中的精确指令 |
| idle 过滤可能破坏动作窗口 | 直接删除帧会破坏 horizon 和视频索引，接触阶段小动作也容易被误删 | 只过滤合法 sample start；对关闭、接触、恢复和终止 Hold 放宽规则 |
| norm 计算很慢 | 原流程为只计算 state/action 仍解码 AV1 视频 | 实现基于 parquet 的快速 norm，并与 filter、delta action、drop_last 语义对齐 |
| 新数据加入后如何继续训练 | 直接 `--resume` 会恢复旧 optimizer、数据状态和 assets | 新 mixture 使用 checkpoint `params` warm-start、新实验和 fresh optimizer；仅原实验故障恢复使用 `--resume` |
| GPU 显存占满但利用率约 0% | 训练在约 step 4890 出现 JAX/XLA/CUDA 主线程自旋，DataLoader workers 休眠且进程不响应 Ctrl+C | 保存现场信息，依次 SIGTERM/SIGKILL，服务器重启恢复 CUDA，再从最近完整 checkpoint 4000 resume |
| W&B 登录串号或版本冲突 | 共享服务器凭证和旧 W&B 环境相互影响 | 使用权限 600 的独立 netrc、隔离 W&B 0.28.1 和固定 run ID 的旁路上传器；仓库不保存密钥 |
| checkpoint 占用空间大 | 单个完整 checkpoint 约 8.9 GB | 每 1000 步保存用于容错，`keep_period=5000` 只长期保留周期节点和最终节点 |
| 服务端提示 checkpoint 不存在 | shell 变量的长路径中误包含换行和空格 | 用 root/config/exp/step 分段拼接，并用 `test -f _CHECKPOINT_METADATA` 启动前校验 |
| 客户端有角度日志但机械臂不动 | 当次模型 target 与 q_cmd 差值仅 0.01-0.05°，小于或接近舵机一个 encoder tick，本质是策略输出 Hold | 对照校准 hash、prompt、相机顺序和训练起始姿态；使用已验证旧 checkpoint 做软硬件 A/B 测试 |
| 抽屉不能一次关闭但会继续调整 | 闭环策略根据新图像和状态产生恢复动作，但关闭阶段数据、接触控制或位置覆盖仍不足 | 补采失败状态恢复与不同抽屉位置数据，保留小动作；增加实际关节反馈、完成检测和最大重试/推压限制 |
| action chunk 受到网络时延影响 | 30 Hz 下 10 步只有约 333 ms horizon，约 100 ms RTT 会使前几步到达时过期 | 客户端按图像年龄和延迟自适应选择 chunk 起点，拒绝超过 horizon 的过期结果并执行 Hold |

## 14. 面试高频问题与参考回答

### 14.1 为什么选择 PI0.5 和 LoRA，而不是从头训练？

PI0.5 已具备视觉、语言和机器人动作的预训练能力，当前真机数据只有 62 个 episodes，不足以从头训练大型 VLA。LoRA 只训练低秩适配参数，显存和训练成本更低，同时保留基础模型的通用表征。项目中 32.6 GiB GPU 训练占用约 30 GiB，说明全量微调在当前硬件上风险更高。

### 14.2 为什么不能把两批数据直接拼起来？

虽然是同一台 follower，但两位采集者使用了不同 calibration，数值零点和 gripper 范围不同。直接合并会让同一个物理姿态对应两组状态/动作标签，模型会学到互相冲突的监督。项目中 wrist roll 的坐标偏移约 83.87°，因此必须先统一 calibration。

### 14.3 校准转换时为什么 state 和 action 都要转换？

行为克隆学习的是从 observation 到 action 的映射。如果只转换 state，action 仍在旧坐标；只转换 action，输入又不一致，两种情况都会使监督关系错误。因此两者必须使用同一物理映射，并重新计算统计信息。

### 14.4 如何让模型区分一个物块和两个物块？

数据集 task 表保留两条精确 prompt，数据配置启用 `prompt_from_task=True`，repack 也保留 prompt 字段。推理时客户端发送与训练一致的任务文本。这样语言条件真正进入模型，而不是依赖同一个默认 prompt 猜测任务。

### 14.5 为什么前五轴使用 delta action，夹爪使用绝对 action？

身体关节使用 delta 能减小不同绝对姿态带来的尺度变化，更适合学习局部轨迹；夹爪天然是 0-100 的开合状态，绝对值更直接。训练、norm 和推理反变换必须使用完全相同的 mask，否则动作尺度会错误。

### 14.6 warm-start 和 resume 有什么区别？

Warm-start 只加载已有 checkpoint 的模型参数，新的数据 mixture、norm 和任务分布使用 fresh optimizer，从 step 0 建立新实验。Resume 恢复模型、optimizer、step 和数据加载状态，只适合同一实验中断后的继续。本项目扩展任务时使用 warm-start，训练故障后从 checkpoint 4000 使用 resume。

### 14.7 为什么要重新计算 norm？

模型不是直接学习原始角度，而是学习规范化后的 state/action。数据 mixture 和 calibration 改变后，旧分位数可能不再代表新分布。norm 必须与 filter、delta transform、action horizon 保持一致；否则即使 checkpoint 能加载，实际输入输出尺度也可能错误。

### 14.8 idle filter 为什么只过滤窗口起点，不直接删除帧？

模型每个样本需要连续 10 步 action。直接删帧会打断时间连续性、改变视频/状态对齐并破坏 episode 边界。过滤起点可以减少低价值样本，同时让每个合法起点仍从原数据读取完整 horizon。

### 14.9 训练时 GPU 只有 1% 利用率，你如何定位？

先区分是数据加载慢、保存 checkpoint、编译，还是进程假死。故障现场中模型仍占显存、GPU utilization 为 0、主进程占一个 CPU 核、workers 全部休眠、日志不再增长且 Ctrl+C 无响应，因此判断为 JAX/XLA/CUDA 主线程自旋，而不是普通 I/O 瓶颈。终止异常进程、重启恢复 CUDA，并从最近完整 Orbax checkpoint 继续。

### 14.10 远程推理为什么使用 action chunk？

逐步请求会让网络抖动直接造成控制中断。一次返回 10 步可以给客户端短期动作缓冲，但 30 Hz 下 horizon 只有约 333 ms，所以客户端还要根据图像年龄和 RTT 跳过过期前缀，并在结果过旧或缓冲耗尽时 Hold，而不是盲目执行旧轨迹。

### 14.11 模型反复调整后关闭抽屉，是否说明它有自我纠错能力？

更准确的说法是“基于新观测的闭环恢复能力”，不是模型有自我意识。第一次未关闭后图像和关节状态发生变化，模型重新推理并产生修正动作。要证明这不是随机振荡，还需要统计动作是否随状态变化、恢复次数、最终成功率，并与冻结观测或开环回放进行安全对照。

### 14.12 为什么有时终端持续输出动作但机械臂不动？

需要区分通信链路和策略输出。一次日志中 target 与 q_cmd 仅相差约 0.01-0.05°，小于或接近 STS3215 一个 encoder tick，电机收到的整数目标可能没有变化。通过校准文件 hash、实际关节 readback、相机顺序、精确 prompt，以及旧 checkpoint A/B 可以判断是硬件、输入分布还是模型在输出 Hold。

### 14.13 如何提高关闭抽屉的一次成功率？

主要从数据和控制两侧改进：采集不同抽屉位置、开度和接触点的一次成功数据；从已推偏、没关严或反弹状态直接采恢复数据；保留低速度接触帧并适度过采样关闭阶段。控制侧增加实际关节 readback、完成检测、最大重试和持续推压限制。恢复数据不能成为多数，否则模型可能学会先失败再修正。

### 14.14 为什么移动抽屉能够提高泛化？

固定位置采集会让模型把背景、像素位置和固定关节轨迹当作捷径。移动和轻微旋转抽屉会迫使模型利用视觉估计目标与机械臂的相对关系。但变化范围必须分层覆盖且保留未见位置作为测试集，否则只能证明记住了更多离散点。

### 14.15 如何选择最佳 checkpoint？

训练 loss 只能反映离线监督误差，不能直接代表真机接触任务成功率。应在相同场景集合上比较 40000、45000、49999 等 checkpoint，记录一次成功率、最终成功率、恢复次数、时间、振荡和安全事件，再选真机综合表现最好的节点，而不是默认最后一步最好。

### 14.16 项目的主要局限是什么？

数据规模仍小，两个任务的采集者、初始姿态和场景分布存在偏差；SO101 缺少腕部六维力传感器，关闭抽屉主要依赖视觉和关节位置；当前 Body 平滑主要锚定已下发命令而不是连续实测位置；尚未完成多随机种子、大规模抽屉位置和严格 held-out 场景的统计评估。这些都是下一阶段的明确改进方向。

### 14.17 如果再做一轮，你会怎么设计实验？

先固定评估协议并测试多个现有 checkpoint；再按网格改变抽屉位置、旋转和开度，采集多数一次成功示范和少量失败状态恢复示范；为关闭阶段放宽 filter，并保留 held-out 位置。选择最佳 checkpoint 参数 warm-start，以较低学习率训练 10000-20000 步，每 3000-5000 步真机测试，最终比较是否同时提高首次成功率、最终成功率并减少平均重试次数。

### 14.18 如何证明项目是你真正做的，而不是只运行了开源代码？

可以从三个层面回答：第一，数据层实现 calibration 转换、LeRobot 合并、filter 和快速 norm；第二，训练层修改配置和 DataLoader，设计 warm-start/resume、checkpoint 与 W&B 容错流程；第三，部署层完成远程闭环客户端并根据延迟、舵机分辨率、真机姿态分布诊断不动和关闭重试问题。然后结合具体文件、日志数字和故障现场说明个人决策过程。

## 15. 面试表达注意事项

1. 不要把“最终能够关上”直接说成“模型具有推理意识”，应说闭环视觉反馈下表现出恢复行为。
2. 不要只报告训练 loss；没有完成严格统计前，不虚构成功率。可以明确说明当前已验证现象和下一步评估协议。
3. 强调为什么做 calibration、norm、filter 和 warm-start，而不是只罗列命令。
4. 面试时准备画出以下数据流：

```text
双目图像 + 关节状态 + Prompt
             ↓ WebSocket
       GPU PI0.5 Policy
             ↓ 10×6 Action Chunk
延迟对齐 → 安全检查 → 平滑限速 → SO101
             ↑               ↓
             └────新观测闭环───┘
```

5. 准备展示 W&B 曲线、数据可视化、不同 checkpoint 对比视频和一次失败后恢复的视频；这些比只展示最终成功片段更能体现工程完整性。
