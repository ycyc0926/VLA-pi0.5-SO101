# `blacknew` LeRobot v3 数据阅读指南

这份说明只陈述当前仓库能够验证的事实；无法从现有文件证明的采集参数会明确标成“推断”。

## 1. `datasets/` 里到底有什么

当前只有一个可用数据集：`datasets/AlexFeng1/blacknew`。

```text
blacknew/
├── data/chunk-000/file-000.parquet       # 31,000 个时间步的低维 state/action 和索引
├── videos/
│   ├── observation.images.env/...mp4     # 环境相机，640×480、30 FPS、AV1
│   └── observation.images.hand/...mp4    # 手腕相机，640×480、30 FPS、AV1
├── meta/
│   ├── info.json                         # 数据集总说明和 feature schema
│   ├── stats.json                        # 原始 LeRobot 字段的全局统计
│   ├── tasks.parquet                     # task_index 到文本指令的映射
│   └── episodes/...parquet               # 每个 episode 的边界、视频时间段和局部统计
└── norm_stats.json                       # OpenPI 训练预处理后的归一化统计，不属于 LeRobot meta
```

`.cache/huggingface` 是下载缓存和远端文件清单，不是第二份训练数据。缓存清单显示远端仓库曾同时有根目录和 `blacknew/test/` 两组同内容路径；本地真正加载的是上面列出的根目录文件。

数据集的硬指标：50 episodes、31,000 行低维样本、30 FPS、一个 task、两路 RGB 视频。episode 平均 620 帧（约 20.67 秒），最短 459 帧，最长 755 帧。按 parquet 行数计算，总演示时长约 1,033.33 秒（17.22 分钟）。

## 2. 先读懂 `meta/info.json`

`info.json` 是入口，不保存任何一帧真实数据。

- `codebase_version: v3.0`：文件布局遵循 LeRobot v3。
- `robot_type: so_follower`：采集端记录的机器人类别。当前新版注册名通常写作 `so101_follower`，不要仅因名字不同就判断数据损坏。
- `fps: 30`：逻辑采样时钟。第 `i` 帧的 `timestamp` 由 `i / 30` 得到，并不是原始墙上时钟。
- `splits.train: 0:50`：训练 split 使用 episode `[0, 50)`。
- `data_path`、`video_path`：通过 `chunk_index/file_index` 拼出真实文件路径。
- `chunks_size: 1000`：一个 chunk 最多容纳多少个文件，不是 episode 数，也不是帧数。
- `data_files_size_in_mb`、`video_files_size_in_mb`：写入器决定何时切换下一个合并文件的目标阈值。

### Feature schema

每个时刻的核心关系是：

```text
observation.state[t] + env_image[t] + hand_image[t] + task
                              -> 预测 action[t:t+H]
```

- `observation.state: float32[6]`：跟随机械臂此刻读回的 6 个位置。
- `action: float32[6]`：leader 机械臂给出的 6 个目标位置。它不是 `state` 的复制；state 是 follower 实际位置，action 是控制目标。
- 前五维依次是 shoulder pan/lift、elbow flex、wrist flex/roll；第六维是 gripper。
- 当前 SO-101 配置默认前五维使用 degree 模式、夹爪使用约 `[0,100]` 的校准范围。但 `info.json` 没保存当时的 `use_degrees` CLI 参数，因此单位最终应结合采集配置/校准文件确认。
- `timestamp`：episode 内相对时间，理论值为 `frame_index / fps`。
- `frame_index`：episode 内从 0 重新计数。
- `episode_index`：这是第几个演示。
- `index`：整个数据集内不重置的全局行号。
- `task_index`：连接 `tasks.parquet` 的整数外键。
- 图像 feature 写成 `dtype: video`，表示主 parquet 不嵌入像素；加载器根据 episode 的视频时间段解码 MP4。

## 3. `meta` 里另外三个文件

### `tasks.parquet`

它只有一行：`0 -> "Grab the black cube"`。所以主 parquet 的每一行 `task_index=0`。

注意，当前 OpenPI 训练配置实际使用的默认 prompt 是 `Grab the black cube and place it in the white cup`。原因是 `LeRobotSO101DataConfig` 没把 task 字段映射成 prompt，而 `ModelTransformFactory(default_prompt=...)` 补入了更完整的默认文本。这是“数据集任务文本”和“本次训练文本”两个不同层次，不能混为一谈。

### `episodes/chunk-000/file-000.parquet`

一行描述一个 episode。以 episode 0 为例：

- `length=683`，对应主 parquet 全局区间 `[0, 683)`。
- 两路视频都使用物理 MP4 的 `[0.0, 22.7667)` 秒。
- `data/chunk_index=0, data/file_index=0` 表示它的低维行位于第一个合并 parquet。
- `stats/...` 是这个 episode 自己的统计，最后再聚合成全局 `stats.json`。

一个 parquet 可以合并很多 episode；episode 边界来自这里，不能用“一个文件就是一个 episode”的旧格式经验来猜。

本数据还有一个应当知道的边界情况：两路 MP4 各有 31,411 帧，比主 parquet 多 411 帧。差异来自 episode 7（视频片段 641 帧、低维 459 行）和 episode 18（视频片段 797 帧、低维 568 行）。后续 episode 的 `from_timestamp` 会跨过这些多余尾帧，因此不会让后续样本整体偏移；但检查这两个 episode 时，应知道尾部视频没有对应低维训练行。

### `stats.json`

这是原始 LeRobot feature 的统计，包括 `min/max/mean/std/q01/q10/q50/q90/q99/count`。

- state/action 的 `count` 都是 31,000。
- 图像被换算到 `[0,1]` 后统计，图像 `count=6187` 是统计过程使用的图像样本数，不等于数据集总帧数。
- 这些 action 统计仍然是 6 维绝对位置，不能直接替代 `norm_stats.json`。

JSON 标准不允许注释，因此不要直接给 `info.json` 或 `stats.json` 插入 `// 中文注释`，否则加载器会解析失败。本文件就是它们的伴随注释。

## 4. 原始采集如何直接变成 v3

仓库证据支持的流程如下：

```text
SO-101 follower 电机 Present_Position + env/hand 相机图像
                              │
SO-101 leader 电机 Present_Position（作为目标 action）
                              ↓
lerobot-record / record_loop（30 Hz）
                              ↓
build_dataset_frame：按 info feature 顺序组成 float32[6] 与图像字段
                              ↓
LeRobotDataset.add_frame：补 frame_index、timestamp，暂存当前 episode
                              ↓
LeRobotDataset.save_episode
     ├── 低维数组 -> data/*.parquet
     ├── 图像帧 -> 两路 AV1 MP4
     ├── task 文本 -> tasks.parquet + task_index
     ├── episode 边界/视频时间范围 -> meta/episodes/*.parquet
     └── schema/计数/统计 -> info.json + stats.json
```

关键代码阅读顺序：

1. `lerobot/pyproject.toml` 的 `lerobot-record` CLI 映射。
2. `lerobot/src/lerobot/scripts/lerobot_record.py` 的 `record()` 和 `record_loop()`。
3. `lerobot/src/lerobot/robots/so_follower/so_follower.py` 的 `get_observation()` / `send_action()`。
4. `lerobot/src/lerobot/teleoperators/so_leader/so_leader.py` 的 `get_action()`。
5. `lerobot/src/lerobot/utils/feature_utils.py` 的 `build_dataset_frame()`。
6. `lerobot/src/lerobot/datasets/dataset_writer.py` 的 `add_frame()` / `save_episode()` / `_save_episode_data()` / `_save_episode_video()`。
7. `lerobot/src/lerobot/datasets/dataset_metadata.py` 的 `save_episode()`。

`custom_vla/README.md` 明确写着“遥操作采集 Lerobot v3.0 格式数据”，并给出了 `lerobot-record + so101_follower + so101_leader` 命令。因此最有证据的结论是“采集时直接写 v3”，而不是“现有仓库里的某个 converter 把另一种原始文件转换成 v3”。仓库里的 ALOHA/DROID/LIBERO converter 是其他数据源示例，没有任何代码引用 `blacknew`。

无法从当前仓库还原的内容包括：`blacknew` 的完整原始 CLI、两路相机设备编号、采集机上的校准 JSON，以及是否在上传前做过 episode 删除/拼接。示例命令只有一台 `env` 相机，而最终数据有 `env + hand`，所以示例不能冒充当时的精确命令。

## 5. v3 数据进入 OpenPI 训练时又经历什么

这一步才是“数据集格式”到“模型张量”的处理，不会回写原始 v3 文件：

```text
LeRobotDataset(delta_timestamps=未来 10 步)
  -> RepackTransform（env/hand/state/action 改成 SO101 适配键）
  -> SO101Inputs（两路图像、state、action；缺失的第三视角用零图占位）
  -> DeltaActions（前 5 维 action -= 当前 state；gripper 保持绝对值）
  -> Normalize（使用 norm_stats.json 的 q01/q99 映射到约 [-1,1]）
  -> ResizeImages(224×224) + TokenizePrompt
  -> π0.5 / flow-matching loss
```

对应代码：

- `custom_vla/openpi/src/openpi/training/data_loader.py::create_torch_dataset()` 组出未来 action horizon。
- 同文件 `transform_dataset()` 明确规定 Repack → Data transforms → Normalize → Model transforms 的顺序。
- `custom_vla/openpi/src/openpi/training/config.py::LeRobotSO101DataConfig.create()` 配置 SO-101 映射和 mask。
- `custom_vla/openpi/src/openpi/policies/so101_policy.py::SO101Inputs` 组模型字典。
- `custom_vla/openpi/src/openpi/transforms.py::DeltaActions` 的公式是前五维 `action_delta = action_absolute - state_current`。

`norm_stats.json` 是运行 `scripts/compute_norm_stats.py --config-name=...` 后，针对上述“已经转成 delta 的模型输入”生成的统计：state 仍是 6 维位置；actions 的前五维是 delta，第六维仍是 absolute gripper。它与 `meta/stats.json` 的原始绝对 action 统计语义不同。

## 6. 实际查看命令

推荐从只读检查脚本开始：

```bash
/root/autodl-tmp/envs/lerobot/bin/python tools/inspect_lerobot_v3.py
/root/autodl-tmp/envs/lerobot/bin/python tools/inspect_lerobot_v3.py --episode 7 --rows 5
```

在线可视化适合看动作是否合理；本地脚本更适合核对 schema、索引边界和统计。判断一条样本时，至少同时看 `state[t]`、`action[t]`、两路 `image[t]`、task，以及未来 action chunk，单看某一列很容易误解 VLA 的监督信号。
