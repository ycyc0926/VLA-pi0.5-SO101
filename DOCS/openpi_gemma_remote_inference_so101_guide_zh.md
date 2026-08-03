# OpenPI Gemma、远程推理与 SO101 自适应客户端代码导读

> 更新时间：2026-07-29（UTC）

本文围绕以下代码展开：

- `custom_vla/openpi/src/openpi/models/gemma.py`
- `custom_vla/openpi/scripts/serve_policy.py`
- `custom_vla/openpi/src/openpi/serving/websocket_policy_server.py`
- `custom_vla/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- `custom_vla/openpi/packages/openpi-client/src/openpi_client/zpf_pi0_so101_client_adaptive_pro_v4.py`

阅读时先抓住一条主线：`gemma.py` 不是完整的 VLA，而是 π0/π0.5 中负责让视觉语言条件与动作序列交互的双专家 Transformer。`serve_policy.py` 把包含数据变换和模型的完整 Policy 包装成 WebSocket 服务；官方客户端同步发送一次观测并等待一段动作；SO101 adaptive v4 再用多个线程，将这种阻塞推理与机器人实时控制解耦。

## 1. `gemma.py` 在 π0 中的位置

核心代码见 [`gemma.py`](../custom_vla/openpi/src/openpi/models/gemma.py)。

对于当前 `pi05_so101_lora` 配置，Gemma 内部包含两个专家：

| 专家 | 配置 | 隐藏宽度 D | 主要作用 |
|---|---:|---:|---|
| 专家 0 | `gemma_2b_lora` | 2048 | 处理图像、Prompt、状态组成的 prefix |
| 专家 1 | `gemma_300m_lora` | 1024 | 处理带噪动作和流匹配时间组成的 suffix |

两者都是 18 层、8 个 Query Head、1 个 KV Head，每个 Head 为 256 维。

Gemma 接收的不是原始图片、字符串或关节角，而是上游 π0 已经构造好的 embedding。可以把接口简化为：

```text
输入：
[
    prefix_embedding: [B, P, 2048],
    suffix_embedding: [B, S, 1024],
]
positions:  [B, P+S]
mask:       [B, P+S, P+S]
kv_cache:   可选

输出：
[
    prefix_output: [B, P, 2048],
    suffix_output: [B, S, 1024],
]
new_kv_cache
```

其中：

- `B`：batch size。
- `P`：prefix token 数量。
- `S`：suffix token 数量。
- `D`：模型隐藏宽度。
- `N`：Query Head 数量。
- `K`：KV Head 数量。
- `H`：每个 Head 的维度。

### 1.1 `Config` 和 `get_config()`

`Config` 定义单个专家的结构：

- `width`：Transformer 隐藏宽度。
- `depth`：Transformer Block 数量。
- `mlp_dim`：FFN 中间层宽度。
- `num_heads`：Query Head 数量。
- `num_kv_heads`：Key/Value Head 数量。
- `head_dim`：每个 Head 的维度。
- `lora_configs`：哪些线性层附加 LoRA。

当前模型中 `N=8、K=1`，属于 GQA（Grouped Query Attention）：

```text
8 个 Query Head
共享 1 个 Key Head 和 1 个 Value Head
```

这能明显减小 K/V 参数和 KV Cache 的大小。

### 1.2 `RMSNorm`

普通模式大致执行：

```python
x_normalized = x / sqrt(mean(x**2) + eps)
output = x_normalized * (1 + learned_scale)
```

参数 `scale` 的初始值是 0，但实际缩放系数是 `1 + scale`，所以初始状态接近恒等变换。

自适应模式还会根据条件产生：

```python
scale, shift, gate = Dense(condition)
output = normalized_x * (1 + scale) + shift
```

同时返回 `gate`，残差连接变为：

```python
x = x + gate * block_output
```

因为该 Dense 层使用零初始化，动作专家刚初始化时 `scale=0、shift=0、gate=0`。这样新动作分支一开始接近恒等映射，不会立刻严重扰动已经预训练好的语言模型。

### 1.3 `Embedder`

`Embedder` 是语言 Token 的 Embedding 表：

```text
token ids [B,T]
    ↓ 查表
embedding [B,T,D]
    ↓ 乘 sqrt(D)
Gemma 输入
```

`decode()` 可以把隐藏向量映射回词表 logits，但 π0 动作生成的主路径主要使用 `encode()`，不会依靠语言解码头生成机器人动作。

### 1.4 `Attention`：两个专家如何交流

这是 `gemma.py` 中最关键的部分。

两个专家的隐藏宽度不同，因此先分别用自己的权重完成 Q/K/V 投影：

```text
专家0 [B,P,2048] → Q0/K0/V0
专家1 [B,S,1024] → Q1/K1/V1
```

投影后，二者具有相同的 Attention Head 结构：

```text
Q:   [B,T,N,H]
K/V: [B,T,K,H]
```

代码随后沿 Token 维拼接，而不是沿特征维硬拼接 2048 和 1024：

```python
q = concatenate([q_prefix, q_suffix], axis=1)
k = concatenate([k_prefix, k_suffix], axis=1)
v = concatenate([v_prefix, v_suffix], axis=1)
```

Attention 计算可简化为：

```text
Q × Kᵀ
   ↓
logits [B,K,G,T,S]
   ↓ attention mask + softmax
注意力概率
   ↓ × V
共同编码结果
```

其中 `N = K × G`，当前为 `8 = 1 × 8`。

得到共同 Attention 结果后，再按照 prefix 和 suffix 的原始长度切开，并用每个专家自己的输出矩阵投影回各自宽度：

```text
prefix → [B,P,2048]
suffix → [B,S,1024]
```

因此，双专家结构的本质是：

> 两个专家具有不同的参数和隐藏宽度，但会先投影到共同的 Attention Head 空间，在该空间中进行跨模态交流。

### 1.5 `Block`

一个 Block 是标准的两段残差结构：

```text
输入
 ├─ RMSNorm → Attention → Dropout → 残差
 └─ RMSNorm → FFN       → Dropout → 残差
```

每个专家有自己的：

- RMSNorm。
- Q/K/V 投影。
- Attention 输出投影。
- FFN。
- LoRA 参数。

但 Attention 权重是在拼接后的共同 Token 序列上计算的。

文件中还定义了一个本地 `FeedForward` 类，不过当前 `Block` 实际调用的是 `lora.FeedForward`。因此本地 `FeedForward` 更像是遗留实现，不是当前主要执行路径。

### 1.6 `Module`、`remat` 和 `scan`

`Module` 负责把多层双专家 Block 组装成完整 Gemma。

`nn.remat(Block)` 的作用是：

- 训练正向传播时不保存所有中间激活。
- 反向传播时重新计算一部分正向结果。
- 用更多计算换取更低显存占用。
- 推理没有反向传播，因此它主要服务训练过程。

`nn.scan(Block, length=18)` 的作用是：

- 用同一段程序执行 18 个 Transformer Block。
- 每层仍然拥有自己独立的参数。
- 参数树在第 0 维按层堆叠。
- 相比 Python 手工写 18 个 Block，更适合 JAX/XLA 编译。

### 1.7 KV Cache

`KVCache` 保存每层的 Key 和 Value：

```text
K,V: [layer, batch, cached_tokens, kv_heads, head_dim]
```

π0 执行动作推理时会：

1. 图像和 Prompt 构成的 prefix 先经过一次 Gemma。
2. 保存 prefix 的 KV Cache。
3. 后续多次 Euler 动作更新只重新计算 suffix。
4. 每一步 suffix 都复用相同的 prefix KV Cache。

这份 Cache 只在一次 `sample_actions()` 中复用。下一次客户端上传新图像和新机器人状态时，prefix 会重新编码。

### 1.8 本地代码与官方版本的一个差异

本地代码在 `nn.remat()` 中使用：

```python
static_argnums=(6,)
```

当前官方主分支对应代码使用 `(5,)`，除此以外，本地 `gemma.py` 与官方代码基本一致。

按照下面的函数签名：

```python
Block.__call__(
    self,
    xs,
    kv_cache,
    positions,
    attn_mask,
    adarms_cond,
    deterministic,
)
```

`deterministic` 是包含 `self` 后的第 6 个位置参数。Flax 0.10.2 的 `remat` 内部还会自动将索引减 1，以去除未传入 lifted function 的 `self`。因此从当前签名和 Flax 实现看，本地 `(6,)` 是有依据的，不建议在没有回归测试的情况下直接改回 `(5,)`。

参考：

- [OpenPI 官方 gemma.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py)
- [Flax 0.10.2 transforms.py](https://raw.githubusercontent.com/google/flax/v0.10.2/flax/linen/transforms.py)

## 2. 推理服务端 `serve_policy.py`

入口见 [`serve_policy.py`](../custom_vla/openpi/scripts/serve_policy.py)。它主要完成四件事：

1. 解析命令行参数。
2. 创建完整 Policy。
3. 创建 WebSocket 服务。
4. 永久监听客户端请求。

### 2.1 两种 Policy 创建方式

`create_policy()` 支持两种方式：

- `Default`：使用官方预设环境和 checkpoint。
- `Checkpoint`：明确指定训练配置名和 checkpoint 路径。

SO101 不在 `EnvMode` 的默认枚举中，因此应使用 checkpoint 模式。例如：

```bash
cd /root/autodl-tmp/VLA/custom_vla/openpi

export SO101_DATASET_DIR=/root/autodl-tmp/VLA/datasets/AlexFeng1/blacknew

uv run scripts/serve_policy.py \
  --port 5000 \
  policy:checkpoint \
  --policy.config pi05_so101_lora \
  --policy.dir /root/autodl-tmp/VLA/checkpoints/pi05_so101_lora/blacknew_lora_50k_v1/49999
```

SO101 v4 客户端默认端口是 `5000`，而 `serve_policy.py` 默认端口是 `8000`，启动时必须保证两端一致。

### 2.2 Policy 加载后包含什么

`create_trained_policy()` 会加载：

- π0/π0.5 模型结构。
- checkpoint 权重。
- checkpoint 对应的归一化统计量。
- SO101 输入变换。
- 模型输入变换。
- 模型输出反变换。

所以 WebSocket Server 调用的不是裸 `gemma.py`，而是完整流程：

```text
Policy.infer()
  ├─ 输入数据变换
  ├─ Pi0.sample_actions()
  └─ 输出数据反变换
```

### 2.3 WebSocket 服务循环

真正的通信循环位于 [`websocket_policy_server.py`](../custom_vla/openpi/src/openpi/serving/websocket_policy_server.py)：

```python
obs = msgpack_numpy.unpackb(await websocket.recv())
action = self._policy.infer(obs)
await websocket.send(packer.pack(action))
```

连接建立后，服务端先发送一次 metadata。之后每个连接严格按照请求—响应方式工作：

```text
接收一个 observation
→ 完整执行一次 Policy.infer()
→ 返回一个 action chunk
→ 再接收下一次 observation
```

## 3. 官方 WebSocket 客户端

客户端见 [`websocket_client_policy.py`](../custom_vla/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py)。

初始化：

```python
client = WebsocketClientPolicy(host, port)
```

内部会：

1. 拼接 `ws://host:port`。
2. 循环尝试连接。
3. 连接失败时每 5 秒重试。
4. 连接成功后首先接收服务端 metadata。

调用推理：

```python
response = client.infer(observation)
```

`infer()` 本质上执行：

```python
data = self._packer.pack(observation)
self._ws.send(data)
response = self._ws.recv()  # 阻塞等待
return msgpack_numpy.unpackb(response)
```

因此该客户端不是异步 API。

当前项目中的普通 WebSocket 客户端与官方版本一致；本地 `serve_policy.py` 相对官方主要增加了中文注释，执行逻辑没有实质变化。

参考：

- [官方 websocket_client_policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/websocket_client_policy.py)
- [官方 serve_policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/serve_policy.py)

## 4. SO101 一次推理的数据流与输入输出

完整流程如下：

```text
SO101 + 两个摄像头
        │
        ▼
客户端 observation 字典
  observation.images.images_env:   [224,224,3] uint8 RGB
  observation.images.images_wrist: [224,224,3] uint8 RGB
  observation.state:                [6] float32，角度制
  prompt:                            str
        │
        ▼ msgpack + WebSocket
服务端 Policy.infer()
        │
        ├─ SO101Inputs：转换为模型使用的字段
        ├─ Normalize：根据训练统计量归一化
        ├─ 图像处理和 Prompt tokenize
        └─ PadStatesAndActions：从6维补到模型 action_dim=32
        │
        ▼
Pi0.sample_actions()
  prefix = 图像 + Prompt + 状态
  suffix = 高斯噪声动作 + time
  prefix KV Cache 只计算一次
  suffix 执行多次 Euler 更新
        │
        ▼
模型动作 [1,10,32]
        │
        ├─ 去除 batch 维
        ├─ Unnormalize
        ├─ AbsoluteActions：增量动作还原为绝对动作
        └─ So101Outputs：截取真实的6个关节维度
        │
        ▼ WebSocket
客户端 response
  actions:       [10,6]
  policy_timing: 模型采样时间
  server_timing: 服务端完整 Policy.infer 时间
        │
        ▼
SO101 每次取一个动作，以约30 Hz执行
```

`[10,6]` 表示一次请求返回未来 10 个控制步、每步 6 个关节动作。客户端不需要在每一个 30 Hz 控制周期都请求一次服务器。

## 5. 为什么推理命令常用 `uv run`

`uv run` 的作用是：

- 找到项目的 `pyproject.toml` 和 `uv.lock`。
- 创建或使用项目的 `.venv`。
- 确保 JAX、Flax、CUDA、LeRobot 等依赖版本匹配。
- 正确安装和导入 workspace 中的 `openpi-client` 等包。
- 最后启动一个普通 Python 子进程。

因此：

> `uv run` 不是推理算法的一部分，也不会使程序自动变成异步程序。

如果已经激活正确环境，可以直接使用对应 Python 解释器：

```bash
/root/autodl-tmp/envs/pi0_env/bin/python \
  scripts/serve_policy.py \
  --port 5000 \
  policy:checkpoint \
  --policy.config pi05_so101_lora \
  --policy.dir /path/to/checkpoint
```

官方远程推理方案也允许只在机器人电脑安装轻量的 `openpi-client`，客户端不一定要通过 uv 启动。

但是项目中的 SO101 adaptive v4 还额外依赖：

- OpenCV。
- LeRobot 的 `SO101Follower`。
- 串口和机器人硬件相关依赖。

这些不属于最小 `openpi-client` 依赖，所以机器人电脑仍需要配置专门的 SO101 Python 环境。

参考：

- [uv：Running commands](https://docs.astral.sh/uv/concepts/projects/run/)
- [OpenPI 官方远程推理说明](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)

## 6. 推理是否异步执行

这个问题需要分层判断：

| 层次 | 是否异步 |
|---|---|
| 官方客户端 `WebsocketClientPolicy.infer()` | 否，阻塞等待响应 |
| 服务端 WebSocket 收发 | 使用 `asyncio` |
| 服务端 `_policy.infer()` | 同步阻塞 |
| JAX GPU 内核提交 | 底层可能异步 dispatch |
| 转换为 NumPy 并返回 | 会等待设备计算完成 |
| SO101 adaptive v4 | 使用线程并发解耦，不是 async API |

服务端虽然定义了：

```python
async def _handler(...):
```

但其中执行：

```python
action = self._policy.infer(obs)
```

这里没有 `await`，也没有把推理放入线程池或独立 worker。因此模型执行期间会阻塞当前 asyncio 事件循环。当前服务不会仅仅因为使用了 asyncio，就同时并行执行多个 GPU 推理请求。

参考：[OpenPI issue #717](https://github.com/Physical-Intelligence/openpi/issues/717)

## 7. SO101 adaptive v4 相对官方新增了什么

官方 OpenPI 没有 `zpf_pi0_so101_client_adaptive_pro_v4.py` 对应文件。官方只提供通用同步 WebSocket 客户端和简单示例；这个 v4 是当前项目针对实体 SO101 新增的一套完整执行器。

### 7.1 功能对照

| 能力 | 官方通用客户端 | SO101 adaptive v4 |
|---|---|---|
| WebSocket 请求 | 有 | 有 |
| SO101 串口连接 | 无 | 有 |
| 读取 6 维关节状态 | 无 | 有 |
| 双摄像头采集 | 无 | 有 |
| 动作缓冲区 | 无 | 有 |
| Brain/Body 分线程 | 无 | 有 |
| 延迟自适应跳步 | 无 | 有 |
| 动作平滑和限速 | 无 | 有 |
| 推理超时 watchdog | 无 | 有 |
| Hold 安全动作 | 无 | 有 |
| RTT/服务端延迟统计 | 无 | 有 |
| 相机实时预览 | 无 | 有 |

### 7.2 三条并发执行链

v4 的核心是三条执行链：

```text
Camera thread
持续读取双摄像头，只保存最新一对图像

Brain thread
读取最新图像和状态
→ 调用阻塞的 infer()
→ 获得10步动作
→ 替换动作缓冲区

Body main loop
以30 Hz从缓冲区取动作
→ 平滑和限速
→ robot.send_action()
```

主要代码模块包括：

- `ActionBuffer`：保存等待执行的动作块，Brain 写入、Body 消费。
- `RobotStateCache`：缓存最后有效状态和最后发送的命令。
- `LatestFramePairBuffer`：只保留最新双摄像头画面。
- `CameraPairCapture`：唯一负责读取 `VideoCapture` 的后台线程。
- `inference_worker()`：Brain 推理线程。
- `main()` 后半部分：30 Hz Body 控制循环。

这不表示同时运行多个模型推理。Brain 中仍然只有一个阻塞的 `infer()` 请求在途，只是 Body 不需要停止等待它。

### 7.3 为什么需要动作缓冲和 Brain/Body 解耦

当前 action horizon 是 10，控制频率是 30 Hz：

```text
10 / 30 ≈ 0.333 秒
```

也就是一块动作只能覆盖大约 333 ms。一次推理请求的延时包括：

- 读取和准备图像。
- 读取机器人状态。
- 序列化两张图像。
- 网络上传和返回。
- 服务端反序列化和输入变换。
- JAX 模型推理。
- 多次 Euler 动作更新。
- 输出反变换。
- 客户端反序列化。

如果同步写成：

```python
response = infer()
执行10步动作
response = infer()
执行10步动作
```

机器人会在每次等待推理时停止。v4 的做法是：Body 消费当前动作块时，Brain 提前请求下一块动作。

### 7.4 延迟自适应跳步

`choose_adaptive_start_idx()` 会综合以下信息：

- 推理结果已经老了多少毫秒。
- 相当于过去了多少个 30 Hz 控制周期。
- 当前机器人命令位置。
- 动作块中哪一步距离当前状态更近。
- 必须保留多少动作供 Body 执行。

随后选择从动作块的第几步开始执行，跳过已经过时的前几步。

其他延迟和安全机制还包括：

- `stitch_actions_to_anchor()`：使用 smoothstep 平滑衔接新旧动作块。
- `dq_limit_deg`：限制单步关节角变化。
- `alpha`：对目标动作进行低通滤波。
- Gripper 范围限制到 `[0,100]`。
- 动作缓冲为空时发送 Hold。
- 推理超时后关闭 WebSocket 并重新连接。
- Body 启动前先进行同步预热，避免第一次 JIT 编译直接造成 underrun。
- 统计准备时间、RTT、服务端推理时间、图像年龄、缓冲区 underrun 和 Body missed ticks。

## 8. v4 当前仍存在的限制

- 两个摄像头顺序读取，不是硬件级同步。
- 图像与机器人状态不是严格相同时间戳。
- 新动作块直接替换旧动作块中尚未执行的部分，不是论文式 temporal ensemble。
- watchdog 关闭连接只能中断客户端等待，不能取消服务端已经开始的 GPU 计算。
- v4 直接访问客户端私有成员 `_ws`，上游内部实现变化后可能失效。
- 当前 10 步、30 Hz 只有约 333 ms 缓冲余量，长时间网络或推理抖动仍会触发 Hold。
- 服务端 `_policy.infer()` 会阻塞 asyncio 事件循环，多客户端连接不代表多个推理可以并发执行。

## 9. 最终应建立的整体认识

把这几份代码串起来，可以得到下面的理解：

1. `gemma.py` 是 π0 内部的双专家 Transformer，不负责机器人通信。
2. 专家 0 编码视觉语言 prefix，专家 1 编码动作 suffix。
3. 两个专家在统一的 Attention Head 空间中交互。
4. π0 使用 prefix KV Cache，避免每次 Euler 更新都重新编码图像和 Prompt。
5. `Policy.infer()` 在模型前后负责 SO101 数据变换、归一化和绝对/增量动作转换。
6. `serve_policy.py` 加载完整 Policy，并通过 WebSocket 提供请求—响应式服务。
7. 官方 `WebsocketClientPolicy.infer()` 是同步阻塞调用。
8. `uv run` 负责依赖和运行环境，不负责异步执行。
9. SO101 adaptive v4 用 Camera、Brain、Body 三条执行链隐藏推理延时，并增加了缓冲、平滑、限速、Hold、超时重连和延迟统计。
10. v4 是并发解耦，不是多个模型请求并行推理。
