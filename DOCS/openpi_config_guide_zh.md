# OpenPI `config.py`、LIBERO 与 SO-101 配置导读

本文对应 `custom_vla/openpi/src/openpi/training/config.py`。阅读时始终区分三件事：

- `TrainConfig`：描述“一次实验”，包括模型、数据、优化器、日志和 checkpoint。
- `DataConfigFactory`：还不是数据管线；调用 `create()` 后才得到最终 `DataConfig`。
- transform：真正逐样本执行的纯数据变换，不负责训练循环。

## 1. 文件整体结构

```text
AssetsConfig / DataConfig
        │
        ├── ModelTransformFactory
        │     ├── PI0 transforms
        │     ├── PI0.5 transforms
        │     └── PI0-FAST transforms
        │
        └── DataConfigFactory
              ├── LeRobotAlohaDataConfig
              ├── LeRobotLiberoDataConfig
              ├── DROID configs
              ├── LeRobotSO101DataConfig（项目定制）
              └── LeRobotSYSMO32DataConfig（项目定制）

TrainConfig
   └── _CONFIGS 中实例化多个具名实验
          └── _CONFIGS_DICT / get_config / cli
```

程序入口不是读取 YAML，而是：

```text
python scripts/train.py pi05_so101_lora --exp-name=xxx
  -> config.cli()
  -> 用 name 从 _CONFIGS_DICT 取 TrainConfig
  -> create_data_loader()
  -> config.data.create(config.assets_dirs, config.model)
  -> 构造模型、优化器、数据迭代器和 checkpoint manager
```

## 2. `TrainConfig` 每个参数

| 参数 | 含义 | 常见误区 |
|---|---|---|
| `name` | 配置注册名，也是 assets/checkpoint 路径的一部分 | 不是一次运行的名字；同一配置可跑多个 `exp_name` |
| `project_name` | W&B project 名 | 不控制 checkpoint 路径 |
| `exp_name` | 本次实验名，checkpoint 最后一级目录 | 默认 `tyro.MISSING`，启动训练时必须提供 |
| `model` | `Pi0Config`/`Pi0FASTConfig`，定义网络结构及输入输出 shape | 不包含已训练权重 |
| `weight_loader` | JAX 模型初始化后从哪里加载 base/部分参数 | 与“恢复本实验 optimizer 状态”的 `resume` 不同 |
| `pytorch_weight_path` | `train_pytorch.py` 读取的 `model.safetensors` 目录 | JAX `scripts/train.py` 不使用 |
| `pytorch_training_precision` | PyTorch 路径使用 bf16 或 fp32 | 不直接控制 JAX 参数精度 |
| `lr_schedule` | 学习率随 step 的变化 | `decay_steps` 是 schedule 时间尺度，不等于必须训练这么久 |
| `optimizer` | AdamW/SGD 及梯度裁剪等参数 | `clip_gradient_norm` 在 AdamW 更新前应用 |
| `ema_decay` | 参数指数滑动平均系数 | `None` 表示完全不维护 EMA，LoRA 常这样配置 |
| `freeze_filter` | 哪些参数被冻结 | `trainable_filter` 自动取其补集；LoRA 只留下 adapter 可训练 |
| `data` | `DataConfigFactory` 实例 | 还未读取数据，也还没有加载 norm stats |
| `assets_base_dir` | norm stats 等数据资产根目录 | 实际还会拼上 `name/asset_id` |
| `checkpoint_base_dir` | checkpoint 根目录 | 最终路径为 `base/name/exp_name` |
| `seed` | 初始化、shuffle、训练噪声的随机种子 | 仍需确定性算子等条件才能完全复现 |
| `batch_size` | global batch size | 不是每张 GPU 的 batch |
| `num_workers` | DataLoader 工作进程数 | 越大占用的 CPU/内存越多，0 表示主进程加载 |
| `num_train_steps` | optimizer 更新次数 | 不是 epoch 数；OpenPI 主要按 step 训练 |
| `log_interval` | 每多少 step 聚合/记录指标 | 日志里的值通常是该窗口平均值 |
| `save_interval` | 普通 checkpoint 保存间隔 | 最后一步通常也会保存 |
| `keep_period` | 周期性长期保留 checkpoint | 其他普通 checkpoint 可能被清理；`None` 不设长期保留点 |
| `overwrite` | 删除同名实验目录后从头训练 | 是破坏性操作，且不能与 `resume=True` 同时设置 |
| `resume` | 从同一实验目录的最新 checkpoint 恢复 | 会恢复模型、优化器和数据迭代状态，不只是加载 base 权重 |
| `wandb_enabled` | 是否启用 W&B | 关闭不影响本地 checkpoint |
| `policy_metadata` | 推理服务暴露给客户端的机器人元信息 | 不进入模型，也不参与 loss |
| `fsdp_devices` | 一个 FSDP 参数分片组包含多少设备 | `1` 表示不进行跨设备参数分片 |

三个计算属性：

- `assets_dirs = assets_base_dir / name`。
- `checkpoint_dir = checkpoint_base_dir / name / exp_name`。
- `trainable_filter = 所有参数 - freeze_filter`。

`__post_init__()` 只做一个关键约束：`resume` 与 `overwrite` 不能同时为真。

## 3. 用 `pi05_libero` 解开一份 `TrainConfig`

### 3.1 模型

```python
Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
```

- `pi05=True` 让 `model_type` 成为 `PI05`，并使用 PI0.5 的 adaRMSNorm action expert。
- 未指定 `action_dim`，沿用统一内部维度 32。LIBERO 的真实 7 维 action 后面会补零。
- `action_horizon=10`：一个样本监督 `action[t:t+10]`，最终模型输出 shape 是 `[10, 32]`。
- 未指定 `max_token_len`，PI0.5 的 `__post_init__()` 自动设为 200。
- `discrete_state_input=False`：`TokenizePrompt` 不把 state 数值拼进离散 prompt token。state 仍保留在数据字典，也仍会归一化。
- 没有 LoRA variant，也没有 `freeze_filter`，因此这是全量微调配置。

### 3.2 数据

```python
LeRobotLiberoDataConfig(
    repo_id="physical-intelligence/libero",
    base_config=DataConfig(prompt_from_task=True),
    extra_delta_transform=False,
)
```

- `prompt_from_task=True`：`create_torch_dataset()` 根据 LeRobot `task_index` 查询任务文本，先生成 `prompt`。
- `extra_delta_transform=False`：LIBERO action 本来就是 delta，不再减一次 state。
- 模型类型为 PI0.5，所以 `create_base_config()` 自动设置 `use_quantile_norm=True`。
- `asset_id` 没显式指定，默认使用 `repo_id`，归一化统计位于 config assets 目录对应的 repo 子路径。

### 3.3 优化与权重

- global batch 256。
- warmup 10,000 step 到 `5e-5`。
- `peak_lr == decay_lr == 5e-5`，所以 warmup 之后近似常数学习率；虽然 `decay_steps=1,000,000`，实际只训练 30,000 step。
- AdamW 全局梯度裁剪为 1.0。
- EMA 为 0.999。
- JAX 从 `pi05_base/params` 初始化。
- `pytorch_weight_path` 仍是占位字符串，只在改用 PyTorch 训练脚本时需要替换。

没有显式覆盖的字段使用 `TrainConfig` 默认值，例如 `seed=42`、`num_workers=2`、`log_interval=100`、`save_interval=1000`、`keep_period=5000`、W&B 开启、FSDP devices 为 1。

## 4. 追踪 `LeRobotLiberoDataConfig.create()`

### 4.1 RepackTransform

`RepackTransform` 的映射方向是“新键: 原始样本旧键”，而不是相反：

```text
LeRobot 样本                         Repack 后
image                         -> observation/image
wrist_image                   -> observation/wrist_image
state                         -> observation/state
actions                       -> actions
prompt                        -> prompt
```

它会先 flatten 原字典，再按目标结构创建一个新字典。训练数据需要 Repack，是因为数据集字段与在线环境接口不同。推理时通常不使用这个 data config 内的 Repack；推理客户端应直接发送 policy adapter 约定的键。

### 4.2 LiberoInputs

`LiberoInputs` 是训练和推理共用的“机器人输入适配器”：

```text
observation/image       -> uint8 HWC -> image/base_0_rgb
observation/wrist_image -> uint8 HWC -> image/left_wrist_0_rgb
不存在的右腕图          -> zeros     -> image/right_wrist_0_rgb
observation/state       -> state
actions（仅训练存在）   -> actions
prompt                  -> prompt
```

`image_mask` 表示图像真假。PI0/PI0.5 的右腕占位图为 `False`；PI0-FAST 的兼容逻辑设为 `True`。

注意 `_parse_image()` 只做 float→uint8 和 CHW→HWC；resize 与最终图像 `[-1,1]` 处理发生在更后面。

### 4.3 LiberoOutputs

模型统一输出 `[T,32]`，LIBERO 环境只需要前 7 维：

```python
actions[:, :7]
```

它只用于推理输出。训练标签不需要经过 `LiberoOutputs`。

### 4.4 ModelTransformFactory

对 `PI05` 返回以下 input transforms：

1. `InjectDefaultPrompt(None)`：此配置依赖 dataset task，所以通常已有 prompt。
2. `ResizeImages(224, 224)`：三路图像保持比例并 padding。
3. `TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False)`。
4. `PadStatesAndActions(32)`：7 维 action/state 右侧补零到统一维度。

### 4.5 LIBERO 完整训练/推理顺序

```text
训练：
LeRobot + task prompt
 -> RepackTransform
 -> LiberoInputs
 -> quantile Normalize
 -> ModelTransformFactory inputs
 -> Observation + [10,32] action label

推理：
LIBERO 环境请求
 -> LiberoInputs
 -> Normalize
 -> ModelTransformFactory inputs
 -> model.sample_actions() 得到 [10,32]
 -> ModelTransformFactory outputs（PI0.5 为空）
 -> Unnormalize
 -> LiberoOutputs 裁成 [10,7]
```

`pi05_libero` 的 `extra_delta_transform=False`，所以两条方向都没有 Delta/AbsoluteActions。

## 5. 同样方法理解 SO-101

本项目实际配置是 `pi05_so101_lora`。

### 5.1 模型与训练策略

- `pi05=True`、`action_horizon=10`、内部 `action_dim=32`。
- `paligemma_variant="gemma_2b_lora"` 和 `action_expert_variant="gemma_300m_lora"`：两个 Transformer 都插入 LoRA adapter。
- `discrete_state_input=True`：state 被量化并加入 PI0.5 prompt token。这是它与 `pi05_libero` 的重要区别。
- `freeze_filter=get_freeze_filter()`：冻结 base 参数，只训练 LoRA 参数。
- `ema_decay=None`：不维护整套参数 EMA。
- global batch 16，12 个 DataLoader workers，训练计划 100,000 step。
- 学习率先在 3,000 step 内升到 `3e-4`，随后余弦降到 `1e-7`；梯度范数裁到 0.5。

### 5.2 数据与 assets

```python
repo_id = os.environ["SO101_DATASET_DIR"]
assets = AssetsConfig(asset_id="blacknew")
prompt_from_task = False
extra_delta_transform = True
```

- 数据来自本地 blacknew 根目录。
- `asset_id="blacknew"` 规定统计资产的子目录名。训练查找路径精确为
  `(data.assets.assets_dir 或 assets_base_dir/name) / blacknew / norm_stats.json`；推理则从
  `checkpoint/assets/blacknew/norm_stats.json` 强制读取同一份统计。
- 不采用数据集中的短 task 文本 `Grab the black cube`；ModelTransformFactory 注入完整默认 prompt。
- blacknew 保存 absolute target，所以前五关节需要 delta 化，夹爪保持 absolute。

### 5.3 SO-101 RepackTransform

```text
blacknew 原字段                    Repack 后/客户端接口
observation.images.env       -> observation.images.images_env
observation.images.hand      -> observation.images.images_wrist
observation.state            -> observation.state
action                       -> action
```

看起来重复的 `images.images_env` 不是 LeRobot 标准，而是本项目自定义 policy API。训练靠 Repack 得到这些键；真实客户端直接发送这些键，因此二者进入同一个 `SO101Inputs`。

### 5.4 SO101Inputs

```text
images_env   -> _parse_image -> image/base_0_rgb
images_wrist -> _parse_image -> image/left_wrist_0_rgb
无右腕相机   -> 224x224 零图 -> image/right_wrist_0_rgb，mask=False
6D follower state             -> state
[10,6] action（仅训练）        -> actions
prompt（若客户端提供）         -> prompt
```

`_parse_image()` 支持 float `[0,1]`、其他数值图像、CHW、HWC 和灰度输入，统一输出 uint8 HWC RGB。它不负责 resize。

### 5.5 DeltaActions 与归一化

```python
mask = make_bool_mask(5, -1)
```

可读作五个 `True`，然后其余维度为 `False`：

```text
训练输入：
actions[..., 0:5] -= state[0:5]
actions[..., 5]    保持 absolute gripper

推理输出：
actions[..., 0:5] += state[0:5]
actions[..., 5]    保持 absolute gripper
```

因此 `norm_stats.json` 中 `actions` 的前五维统计的是 delta，第六维统计的是 absolute gripper。Normalize 必须在 DeltaActions 之后。

### 5.6 SO101Outputs

模型先输出 `[10,32]`，推理 pipeline 按下面顺序还原：

```text
[10,32] normalized delta
 -> Unnormalize
 -> AbsoluteActions（前五维加当前 state）
 -> SO101Outputs: actions[:, :6]
 -> [10,6] absolute motor targets
```

客户端随后决定动作块是全部执行、只执行前几步，还是结合异步队列平滑执行。

## 6. LIBERO 与 SO-101 对照

| 项目 | `pi05_libero` | `pi05_so101_lora` |
|---|---|---|
| 真实 action 维度 | 7 | 6 |
| action horizon | 10 | 10 |
| 内部 action dim | 32 | 32 |
| 原始 action 语义 | 已是 delta | absolute target |
| 额外 DeltaActions | 关闭 | 前 5 维开启 |
| prompt 来源 | LeRobot task | 固定完整 default prompt/客户端 prompt |
| state 是否进入离散 token | 否 | 是 |
| 微调方式 | 全量 | LoRA |
| EMA | 0.999 | 关闭 |
| 输出裁剪 | 前 7 维 | 前 6 维 |

## 7. 当前定制代码中必须知道的四个问题

### 7.1 `_CONFIGS` 被覆盖

该文件先定义了包含 `pi05_libero` 的官方 `_CONFIGS`，后面又执行第二次：

```python
_CONFIGS = [pi05_so101, pi05_so101_lora, ...]
```

最终 `_CONFIGS_DICT` 只包含第二份列表。实测当前注册项只有：

```text
pi05_so101
pi05_so101_lora
pi05_sysmo32_env
pi05_sysmo32_fistbump36_20260612
```

所以 `get_config("pi05_libero")` 当前会失败。若要在这个定制文件中重新运行 LIBERO，应先明确是否将第二处改为 `_CONFIGS += [...]`，而不是误以为它已经注册。

### 7.2 SO-101 高斯噪声当前不生效

`AddGaussianNoiseImage` 被放在 `SO101Inputs` 后面，但前者寻找 `observation.images.*` 顶层键，后者已经把图像重组到了 `data["image"]`，所以开启开关也匹配不到图像。当前正式配置为 `False`，不会影响现有训练。若以后做增强，应改用模型图像键或调整 transform 顺序。

### 7.3 `TASK_INDEX_TO_PROMPT` 只是未接线的说明表

SO-101 `create()` 中虽然定义了 task 映射，但没有把它加入 transforms。实际 prompt 来自：

1. 客户端显式发送的 `prompt`；或
2. `ModelTransformFactory(default_prompt=...)` 的默认文本。

理解代码时不要把“定义了字典”误当成“字典已经参与流水线”。

### 7.4 当前 norm stats 资产路径需要显式安排

历史训练日志显示旧配置的 `asset_id` 等于完整 dataset 路径，因此曾直接从
`datasets/AlexFeng1/blacknew/norm_stats.json` 加载。当前代码已经把 `asset_id` 改成 `blacknew`，
checkpoint 中也确实存在 `assets/blacknew/norm_stats.json`，这对可迁移推理是正确方向。

但重新训练时，仅传：

```text
--assets-base-dir=/root/autodl-tmp/VLA/assets
```

会让加载器寻找：

```text
/root/autodl-tmp/VLA/assets/pi05_so101_lora/blacknew/norm_stats.json
```

当前仓库没有这个 staged 文件。重新训练前需要二选一：

1. 把数据集的 `norm_stats.json` 放到上述规范 assets 目录；或
2. 显式设置 `data.assets.assets_dir=/root/autodl-tmp/VLA/datasets/AlexFeng1`，再由 `asset_id=blacknew` 拼出数据集目录。

推理 checkpoint 不受这个训练前 staging 问题影响，因为 `create_trained_policy()` 会覆盖配置加载结果，
从 checkpoint 自己的 `assets/blacknew` 读取统计。
