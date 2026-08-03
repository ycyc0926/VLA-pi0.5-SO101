# OpenPI `data_loader.py` 与 `transforms.py` 中文阅读指南

> 整理时间：2026-07-28（UTC）
>
> 对应源码：
> - `custom_vla/openpi/src/openpi/training/data_loader.py`
> - `custom_vla/openpi/src/openpi/transforms.py`

本文以本项目的 π0.5 + SO-101 + LeRobot v3 数据链为主线，说明训练数据如何从 parquet/视频变成模型真正接收的 `Observation` 和 action tensor，并重点解释 `create_data_loader()`、`create_torch_dataset()`、`transform_dataset()` 以及七个核心 transform。

## 1. 先建立完整数据流

训练侧可以概括为：

```text
TrainConfig
  ↓ create_data_loader
DataConfig
  ↓ create_torch_dataset
LeRobot 原始单帧 + 未来 action 序列
  ↓ transform_dataset（逐样本、惰性执行）
OpenPI 格式单样本
  ↓ TorchDataLoader（collate）
batch 字典
  ↓ DataLoaderImpl
(Observation, actions)
```

SO-101 的具体 transform 顺序是：

```text
blacknew 原始样本
  ↓ RepackTransform
字段名和嵌套结构统一
  ↓ SO101Inputs
转换为 OpenPI 的 image/state/actions 结构
  ↓ DeltaActions
前五个关节 absolute action 转成 delta action
  ↓ Normalize
state/actions 分位数归一化
  ↓ InjectDefaultPrompt + ResizeImages + TokenizePrompt
注入任务文本、图像缩放、文本和 state token 化
  ↓ PadStatesAndActions
6 维 state/action 补到模型 action_dim=32
  ↓ TorchDataLoader
多个单样本堆叠成 batch
  ↓ DataLoaderImpl
返回 (Observation, actions)
```

推理输出走相反的语义方向：

```text
模型动作 [T,32]
  ↓ Unnormalize
恢复真实机器人单位
  ↓ AbsoluteActions
前五维 delta 加回当前 state
  ↓ SO101Outputs
裁剪并返回 SO-101 使用的 [T,6]
```

## 2. `data_loader.py` 的内部结构

建议按下面顺序阅读：

1. `Dataset`、`IterableDataset`、`DataLoader` Protocol：定义最小接口。
2. `TransformedDataset`：给随机访问数据集套上逐样本 transform。
3. `create_torch_dataset()`：读取 LeRobot 数据并建立未来 action 窗口。
4. `transform_dataset()`：安装 OpenPI 的主 transform 链。
5. `create_data_loader()`：从 `TrainConfig` 选择 LeRobot 或 RLDS 路径。
6. `create_torch_data_loader()`：计算 local batch、sampler 和 worker。
7. `TorchDataLoader`：collate、跨 epoch 循环并转换数组框架。
8. `DataLoaderImpl`：构造模型最终使用的 `Observation`。

### 2.1 三个 Protocol

`Dataset` 代表可以按索引随机访问的数据：

```python
dataset[index]
len(dataset)
```

`IterableDataset` 代表只能迭代的数据流：

```python
for sample in dataset:
    ...
```

`DataLoader` 是 OpenPI 对训练 loader 的最小约定：

```python
loader.data_config()
iter(loader)
```

这些是 `typing.Protocol`，重点在于约束“对象需要提供什么方法”，并不要求继承某个具体基类。

### 2.2 `TransformedDataset`：惰性逐样本变换

核心实现：

```python
def __getitem__(self, index):
    return self._transform(self._dataset[index])
```

创建 wrapper 时不会遍历全部数据。只有真正执行：

```python
sample = dataset[index]
```

才会：

1. 从 parquet 读取 state/action/index。
2. 按时间戳解码视频帧。
3. 依次运行 Repack、SO101Inputs、Normalize、Tokenize、Pad 等变换。

因此 `transform_dataset()` 本身很快；昂贵操作发生在 DataLoader worker 实际取样时。

## 3. `create_torch_dataset()`

函数签名：

```python
create_torch_dataset(
    data_config: DataConfig,
    action_horizon: int,
    model_config: BaseModelConfig,
) -> Dataset
```

### 3.1 参数

| 参数 | 功能 |
|---|---|
| `data_config` | 提供 `repo_id`、`action_sequence_keys`、`prompt_from_task` |
| `action_horizon` | 每个当前帧向后取多少步 action |
| `model_config` | 真实数据路径不直接使用；`repo_id="fake"` 时用来生成符合模型 spec 的假数据 |

### 3.2 `repo_id` 的三个分支

```python
repo_id = data_config.repo_id
```

- `None`：直接报错，不能创建数据集。
- `"fake"`：返回 1024 条 `FakeDataset` 样本。
- 其他字符串：构造真实 `LeRobotDatasetMetadata` 和 `LeRobotDataset`。

本项目的 `repo_id` 实际是：

```text
/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew
```

### 3.3 `action_horizon` 如何变成未来动作窗口

关键代码：

```python
delta_timestamps={
    key: [t / dataset_meta.fps for t in range(action_horizon)]
    for key in data_config.action_sequence_keys
}
```

对于 `blacknew`：

```text
fps = 30
action_horizon = 10
action_sequence_keys = ("action",)
```

所以先生成以秒为单位的偏移：

```text
[0/30, 1/30, 2/30, ..., 9/30]
```

LeRobot 再乘以 FPS，变成帧偏移：

```text
[0, 1, 2, ..., 9]
```

最终一条当前帧样本会携带：

```text
observation.state: [6]
action:            [10,6]
```

这里 action 的第一行是当前帧动作，后九行是未来动作。

### 3.4 episode 尾部如何处理

如果未来索引超过当前 episode，LeRobot 不会跨到下一个 episode，而是：

1. 把越界索引夹到本 episode 最后一帧。
2. 重复最后一帧 action。
3. 生成 `action_is_pad` 布尔 mask。

真实读取 episode 0 最后一帧 `index=682` 的结果：

```text
action.shape = (10, 6)
action_is_pad =
[False, True, True, True, True, True, True, True, True, True]
```

当前 SO-101 `RepackTransform` 没有保留 `action_is_pad`，因此后续模型实际看到的是重复的尾帧动作，而不是显式 padding mask。这是当前数据链的一个重要细节。

### 3.5 `prompt_from_task`

如果：

```python
data_config.prompt_from_task is True
```

则额外包装：

```python
PromptFromLeRobotTask(dataset_meta.tasks)
```

它在主 Repack 之前，把样本中的 `task_index` 查表转换成 `prompt`。如果后续 Repack 没有包含：

```python
"prompt": "prompt"
```

刚生成的 prompt 仍会被 Repack 丢弃。

当前 SO-101 配置使用 `prompt_from_task=False`，训练 prompt 来自 `ModelTransformFactory(default_prompt=...)`。

## 4. `transform_dataset()`

函数签名：

```python
transform_dataset(
    dataset: Dataset,
    data_config: DataConfig,
    *,
    skip_norm_stats: bool = False,
) -> Dataset
```

### 4.1 参数

| 参数 | 功能 |
|---|---|
| `dataset` | 通常是 `create_torch_dataset()` 返回的 LeRobot Dataset |
| `data_config` | 提供 repack、机器人、normalization 和模型 transform |
| `skip_norm_stats` | 跳过归一化统计检查，让 Normalize 成为 no-op |

`skip_norm_stats=True` 主要用于：

- fake dataset。
- 调试原始数据值域。
- 不希望加载 norm stats 的结构测试。

正式训练不应随意开启，否则训练输入值域和 checkpoint 推理端使用的值域会不一致。

### 4.2 四段 transform 顺序

```python
[
    *data_config.repack_transforms.inputs,
    *data_config.data_transforms.inputs,
    Normalize(...),
    *data_config.model_transforms.inputs,
]
```

对应：

| 阶段 | 主要职责 | SO-101 示例 |
|---|---|---|
| Repack | 数据集 schema 重组 | `env` 映射为 `images_env` |
| Data transforms | 机器人语义适配 | `SO101Inputs`、`DeltaActions` |
| Normalize | 真实单位变模型值域 | q01/q99 分位数归一化 |
| Model transforms | 模型输入格式 | prompt、224 图像、token、32 维 padding |

顺序不能随意交换：

- `DeltaActions` 必须在 Normalize 之前，因为 action 和 state 要在相同真实单位中相减。
- `PadStatesAndActions` 必须在 Normalize 之后，因为 SO-101 norm stats 是 6 维，不是 32 维。

## 5. `create_data_loader()`

函数签名：

```python
create_data_loader(
    config: TrainConfig,
    *,
    sharding: Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
)
```

### 5.1 参数

| 参数 | 功能 |
|---|---|
| `config` | 完整 `TrainConfig`，包括 model、data、batch、worker、seed、assets |
| `sharding` | JAX batch 分片规则；PyTorch 路径忽略 |
| `shuffle` | 是否打乱样本；训练通常为 `True` |
| `num_batches` | 最多返回多少个 batch；`None` 表示无限循环 |
| `skip_norm_stats` | 跳过 Normalize，仅适合调试或假数据 |
| `framework` | 决定返回 JAX 数组还是 Torch Tensor，以及是否使用 DDP sampler |

### 5.2 第一步：创建 `DataConfig`

```python
data_config = config.data.create(config.assets_dirs, config.model)
```

这里才真正执行之前读过的：

```text
LeRobotSO101DataConfig.create()
  ├── create_base_config / 加载 norm_stats
  ├── RepackTransform
  ├── SO101Inputs / SO101Outputs
  ├── DeltaActions / AbsoluteActions
  └── ModelTransformFactory
```

### 5.3 第二步：选择 RLDS 或 LeRobot

```python
if data_config.rlds_data_dir is not None:
    return create_rlds_data_loader(...)

return create_torch_data_loader(...)
```

- DROID/RLDS 数据走 `create_rlds_data_loader()`。
- SO-101/blacknew 的 `rlds_data_dir=None`，走 `create_torch_data_loader()`。
- PyTorch 框架目前不支持 RLDS loader。

## 6. `create_torch_data_loader()` 与 `TorchDataLoader`

### 6.1 全局 batch 和 local batch

传入的 `batch_size` 是全局 batch。

PyTorch DDP：

```python
local_batch_size = batch_size // torch.distributed.get_world_size()
```

JAX：

```python
local_batch_size = batch_size // jax.process_count()
```

单进程多 GPU 时，`jax.process_count()` 通常还是 1，所以 local batch 等于全局 batch；随后 JAX sharding 再沿 batch 轴把数据切给同一进程中的多张 GPU。

当前 `TorchDataLoader` 明确不支持 `jax.process_count() > 1` 的多进程/多主机数据加载。

### 6.2 sampler 和 shuffle

PyTorch DDP 初始化后会创建：

```python
DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=shuffle,
    drop_last=True,
)
```

一旦存在 sampler，底层 `DataLoader` 自身的 `shuffle` 必须关闭，避免同时出现两套打乱逻辑。

### 6.3 `num_workers`

```text
num_workers = 0：主训练进程读取和 transform
num_workers > 0：spawn 多进程并行读取
```

worker 使用：

```python
persistent_workers=True
```

避免每个 epoch 重建视频解码 worker。`_worker_init_fn` 还会关闭 worker 内 JAX 的 GPU 预分配，避免每个数据 worker 抢占显存。

### 6.4 collate

`_collate_fn` 对 PyTree 每个叶子执行：

```python
np.stack(samples, axis=0)
```

所以：

```text
单样本 actions: [10,32]
batch actions:   [B,10,32]
```

### 6.5 为什么 loader 会无限循环

内部有两层循环：

```python
while True:
    data_iter = iter(torch_loader)
    while True:
        batch = next(data_iter)
```

一个 epoch 耗尽后，重新创建 iterator，从数据集开头继续。

- `num_batches=None`：无限输出，由 `num_train_steps` 停止训练。
- `num_batches=N`：总共输出 N 个 batch，即使 N 大于单个 epoch 的 batch 数。
- `drop_last=True`：每轮不足一个完整 batch 的尾部样本被丢弃。

### 6.6 JAX 与 PyTorch 输出差异

JAX：

```python
jax.make_array_from_process_local_data(sharding, x)
```

PyTorch：

```python
torch.as_tensor(x)
```

两条路径前面的 Dataset、transform 和 numpy collate 基本共用。

### 6.7 `DataLoaderImpl`

底层 batch 仍然是字典。最外层才执行：

```python
yield Observation.from_dict(batch), batch["actions"]
```

因此训练函数最终拿到：

```text
Observation:
  images
  image_masks
  state
  tokenized_prompt
  tokenized_prompt_mask

actions:
  [batch, action_horizon, model.action_dim]
```

## 7. `transforms.py` 的基础抽象

### 7.1 `DataTransformFn`

每个 transform 都是：

```python
data_dict -> data_dict
```

输入通常是未 batch 的单样本 PyTree。叶子最好是 numpy array；在 DataLoader worker 中使用 JAX array 可能带来额外 GPU 内存占用。

### 7.2 `CompositeTransform` 与 `compose`

```python
for transform in transforms:
    data = transform(data)
```

前一个 transform 的输出是后一个 transform 的输入，因此 transform 顺序本身就是数据语义的一部分。

### 7.3 `Group.push()` 为什么输入和输出方向不同

```python
return Group(
    inputs=(*self.inputs, *inputs),
    outputs=(*outputs, *self.outputs),
)
```

- 新 input 追加到末尾：继续沿正向处理。
- 新 output 插到开头：按相反语义撤销 input。

例如已有：

```text
inputs:  SO101Inputs
outputs: SO101Outputs
```

push delta/absolute 后：

```text
inputs:  SO101Inputs -> DeltaActions
outputs: AbsoluteActions -> SO101Outputs
```

这样模型输出会先恢复 absolute action，再裁剪成机器人 6 维动作。

## 8. `RepackTransform`

最容易理解反的是映射方向：

```text
新键 -> 原始数据旧键
```

示例：

```python
RepackTransform({
    "state": "observation.state",
    "actions": "action",
})
```

等价于：

```python
output["state"] = input["observation.state"]
output["actions"] = input["action"]
```

SO-101 配置使用：

```python
{
    "observation.images.images_env": "observation.images.env",
    "observation.images.images_wrist": "observation.images.hand",
    "observation.state": "observation.state",
    "action": "action",
}
```

这里左边是 `SO101Inputs` 期待的键，右边是 `blacknew` 原始键。

### 8.1 flatten 的 `/` 和 LeRobot 键中的 `.`

`flatten_dict()` 用 `/` 表示真正的嵌套层级：

```python
{"a": {"b": 1}} -> {"a/b": 1}
```

但 LeRobot 的：

```text
observation.images.env
```

通常是一个带点号的顶层键。点号不会被 `flatten_dict()` 拆分。

### 8.2 Repack 会丢字段

输出只保留 `structure` 引用到的叶子。没有列出的：

```text
timestamp
frame_index
episode_index
task_index
action_is_pad
```

都会被删除。

## 9. `Normalize` 与 `Unnormalize`

### 9.1 z-score

```text
x_norm = (x - mean) / (std + 1e-6)
```

适用于 `use_quantiles=False`。

### 9.2 quantile normalization

```text
x_norm = (x - q01) / (q99 - q01 + 1e-6) × 2 - 1
```

映射关系：

```text
q01 -> -1
q99 -> +1
```

它不会 clip，因此小于 q01 的值可能小于 -1，大于 q99 的值可能大于 +1。

π0.5 默认使用 quantile normalization；π0 默认使用普通 z-score。

### 9.3 `apply_tree`

Normalize 不是对所有字段盲目执行，而是把：

```text
data 的叶子路径
norm_stats 的叶子路径
```

进行匹配。只有同名叶子才归一化，通常是：

```text
state
actions
```

图像、token 和 mask 没有 norm stats，所以保持原样。

### 9.4 `strict`

`Normalize(strict=False)`：

- 数据缺少某个统计字段时不报错。
- norm stats 中没有的图像/token 不处理。

`Unnormalize` 内部使用 `strict=True`，确保模型动作输出一定能找到对应统计量并恢复真实单位。

### 9.5 为什么 Normalize 在 Pad 前

SO-101 state/action 是 6 维，norm stats 也是 6 维。训练顺序是：

```text
[6] Normalize -> [6] Pad -> [32]
```

而不是：

```text
[6] Pad -> [32] Normalize with 6D stats
```

推理输出时模型给出 32 维 action。`Unnormalize` 会恢复统计量覆盖的前 6 维，并让额外 padding 维保持模型输出值，随后 `SO101Outputs` 只取前 6 维。

## 10. `PromptFromLeRobotTask`

输入：

```python
{"task_index": 0}
```

任务映射：

```python
{0: "Grab the black cube"}
```

输出：

```python
{
    "task_index": 0,
    "prompt": "Grab the black cube",
}
```

它返回新字典并保留 `task_index`。后面的 Repack 是否继续保留 `prompt`，取决于 `structure` 中是否显式列出该键。

## 11. `TokenizePrompt`

关键过程：

```python
prompt = data.pop("prompt")
tokens, masks = tokenizer.tokenize(prompt, state)
```

输出键：

```text
tokenized_prompt
tokenized_prompt_mask
```

原字符串 `prompt` 被 pop 删除，避免 Python 字符串进入 JAX batch。

如果 prompt 是 numpy 的 0 维字符串标量，会先调用：

```python
prompt.item()
```

转成 Python `str`。

### 11.1 `discrete_state_input`

`False`：

```python
tokenizer.tokenize(prompt, None)
```

`True`：

```python
tokenizer.tokenize(prompt, state)
```

π0.5 的 SO-101 配置使用 `True`，因此 state 会被编码进 token 序列。不过原数值 `state` 键仍保留，并继续经过 padding。

## 12. `PadStatesAndActions`

SO-101 到 π0.5 的 shape 变化：

```text
state:   [6]    -> [32]
actions: [10,6] -> [10,32]
```

补零发生在最后一维右侧：

```text
[a,b,c,d,e,f] -> [a,b,c,d,e,f,0,0,...]
```

`pad_to_dim()` 只补短向量，不会截断长向量：

```python
pad_to_dim(np.zeros(40), target_dim=32).shape == (40,)
```

如果机器人维度大于模型 action_dim，必须在 adapter 或配置中显式处理，不能依赖这里自动裁剪。

## 13. `DeltaActions`

作用：把 absolute action 转成相对当前 state 的 delta action。

公式：

```text
delta_action[t,d] = absolute_action[t,d] - current_state[d]
```

这里减的是“当前观测 state”，不是每个未来时间步对应的未来 state。同一份当前 state 会广播到整个 action horizon。

SO-101 使用：

```python
make_bool_mask(5, -1)
```

结果：

```text
[True, True, True, True, True, False]
```

- 前五个关节使用 delta。
- 第六维 gripper 保持 absolute。

数值示例：

```text
state           = [1,2,3,4,5,6]
absolute action = [2,4,6,8,10,9]
delta action    = [1,2,3,4,5,9]
```

实现中的广播：

```python
np.expand_dims(state, axis=-2)
```

把 `[D]` 变成 `[1,D]`，再沿 action horizon 的 `T` 维广播到 `[T,D]`。

当前代码使用 `-=`，会原地修改 `actions` 对应的 numpy 数组。

## 14. `AbsoluteActions`

它是 DeltaActions 的推理侧逆变换：

```text
absolute_action[t,d] = delta_action[t,d] + current_state[d]
```

仍只处理 mask 为 True 的维度，gripper 等 mask=False 的维度保持原值。

顺序必须是：

```text
模型输出
  ↓ Unnormalize
恢复真实动作单位
  ↓ AbsoluteActions
加回真实单位的 current state
```

不能在归一化空间直接把真实 state 加到 action 上。

## 15. 当前 SO-101 的端到端 shape

真实 `blacknew` 原始单样本：

```text
observation.images.env:  [3,480,640] float32
observation.images.hand: [3,480,640] float32
observation.state:       [6] float32
action:                  [10,6] float32
task_index:              scalar int64
```

经过全部 transform 后：

```text
images:                  每路 [3,224,224]
state:                   [32]
actions:                 [10,32]
tokenized_prompt:        [max_token_len]
tokenized_prompt_mask:   [max_token_len]
```

经过 DataLoader collate 后：

```text
images:                  [B,3,224,224]
state:                   [B,32]
actions:                 [B,10,32]
tokenized_prompt:        [B,max_token_len]
tokenized_prompt_mask:   [B,max_token_len]
```

## 16. 本次验证与排查结果

已完成：

- `data_loader.py`、`transforms.py` AST 语法检查通过。
- `git diff --check` 通过。
- Repack 数值测试通过。
- z-score 和 quantile Normalize 测试通过。
- PromptFromLeRobotTask 和 mock TokenizePrompt 测试通过。
- PadStatesAndActions 测试通过。
- DeltaActions/AbsoluteActions 互逆测试通过。
- `transform_dataset` 惰性执行与 transform 顺序测试通过。
- `TorchDataLoader` 跨 epoch 循环和 `num_batches` 限制测试通过。
- `blacknew` 真实样本和 episode 尾部 action padding 读取通过。

项目 pytest 暂时不能直接启动，因为 `openpi/src/openpi/conftest.py` 依赖 `pynvml`，当前训练环境没有安装该可选测试依赖。本次没有为了注释任务重新安装或同步整套环境，而是使用直接 Python 烟雾测试覆盖核心行为。

## 17. `blacknew/meta/info.json` 的 JSON 问题

真实样本验证时发现，`meta/info.json` 曾被加入四行 `//` 中文注释。标准 JSON 不允许注释，因此 LeRobot 在创建 `LeRobotDatasetMetadata` 时直接报：

```text
JSONDecodeError: Expecting property name enclosed in double quotes
```

已只删除四行非法注释，没有修改任何字段和值。现在：

- `python -m json.tool` 校验通过。
- `LeRobotDatasetMetadata` 创建通过。
- 视频和 parquet 真实样本读取通过。

需要记住：JSON、parquet、视频等数据文件不适合直接加入源码注释。解释应写在 `DOCS/`，或者使用额外的 Markdown/JSON Schema 文档。

## 18. 当前环境实际使用哪份 LeRobot

当前训练环境导入的是：

```text
/root/autodl-tmp/envs/pi0_env/lib/python3.11/site-packages/lerobot
```

不是自动使用根目录：

```text
/root/autodl-tmp/VLA/lerobot
```

也不是自动使用：

```text
/root/autodl-tmp/VLA/custom_vla/lerobot
```

因此以后即使修改本地 LeRobot 源码，训练进程也不一定会使用它。需要满足其中一种条件：

1. 对目标 LeRobot 工作树执行 editable install。
2. 正确设置 `PYTHONPATH`，让目标源码目录排在 site-packages 前。
3. 直接修改当前环境实际安装的包，但这种方式不利于版本管理，不推荐。

排查“我明明改了代码但训练没变化”时，先执行：

```bash
python -c "import lerobot; print(lerobot.__file__)"
```

确认 Python 实际加载的是哪一份代码。

## 19. 建议的断点阅读顺序

第一次调试真实 SO-101 batch 时，可以依次在下面位置观察键和 shape：

```text
1. create_torch_dataset 后的 dataset[0]
2. RepackTransform.__call__ 前后
3. SO101Inputs.__call__ 前后
4. DeltaActions.__call__ 前后
5. Normalize.__call__ 前后
6. TokenizePrompt.__call__ 前后
7. PadStatesAndActions.__call__ 前后
8. _collate_fn 后
9. DataLoaderImpl 构造 Observation 后
```

每一步重点打印：

```python
print(data.keys())
print(data["state"].shape)
print(data.get("actions", None).shape)
```

图像重点打印：

```python
for name, image in data["image"].items():
    print(name, image.shape, image.dtype, image.min(), image.max())
```

这样可以快速区分问题发生在：

- LeRobot 原始数据读取。
- 字段 Repack。
- SO101 policy adapter。
- normalization。
- 模型 token/padding。
- DataLoader collate 或 sharding。
