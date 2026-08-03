# OpenPI PI0 / PI0.5 模型、训练与 SO-101 推理代码导读

更新时间：2026-07-29（UTC）

本文承接 [openpi_dataloader_transforms_guide_zh.md](./openpi_dataloader_transforms_guide_zh.md)，重点回答三件事：

1. [pi0_config.py](../custom_vla/openpi/src/openpi/models/pi0_config.py) 如何规定模型结构和输入 shape；
2. [pi0.py](../custom_vla/openpi/src/openpi/models/pi0.py) 如何用 flow matching 学习并生成一段动作；
3. [train.py](../custom_vla/openpi/scripts/train.py) 如何让一个 batch 进入模型、求梯度并更新参数。

最后会把“采集数据 → LeRobot v3 → 训练 batch → PI0.5 → SO-101 电机目标”的全链路串起来。

---

## 1. 先建立整体心智模型

这套 PI0.5 不是逐 token 生成动作，也不是直接做普通回归。它把一段未来动作看成一个连续向量，在训练时给真实动作加噪，让模型学习“从当前带噪动作往噪声方向变化的速度”；推理时则从纯高斯噪声出发，沿相反方向积分回真实动作。

本项目 SO-101 配置的核心数据流是：

    两路相机 + 当前 6 维关节状态 + 任务文本
                        │
                        ▼
      SO101Inputs / Normalize / Tokenize / Pad
                        │
                        ▼
      Observation + actions [B, 10, 32]
                        │
          ┌─────────────┴─────────────┐
          │                           │
       训练路径                    推理路径
          │                           │
    真实动作 + 随机噪声           纯随机噪声 x₁
          │                           │
    学速度场 vθ(xₜ,t)          10 次 Euler 反向积分
          │                           │
       更新参数                   归一化动作 x₀
                                      │
                        Unnormalize / AbsoluteActions
                                      │
                                      ▼
                           [10, 6] absolute 电机目标

这里的 10 是 action_horizon，32 是模型统一 action_dim；SO-101 真正有意义的只有前 6 维。

---

## 2. 简单读 pi0_config.py

入口类是 Pi0Config，它继承 BaseModelConfig。这个类只描述“模型应该长什么样”，真正创建神经网络发生在 create()。

### 2.1 配置字段

| 字段 | 默认值 | 含义 |
|---|---:|---|
| dtype | bfloat16 | Gemma、SigLIP 主要矩阵计算的数据类型，节省显存和计算量 |
| paligemma_variant | gemma_2b | 图像/语言前缀所用 Gemma 的结构 |
| action_expert_variant | gemma_300m | 动作后缀所用较小 Gemma 专家 |
| dropout | 0.0 | 双 Gemma Transformer 的 dropout 概率 |
| action_dim | 32 | 模型统一的 state/action 最后一维 |
| action_horizon | 50 | 一次预测未来多少步动作 |
| max_token_len | 自动决定 | PI0 默认 48，PI0.5 默认 200 |
| pi05 | False | False 使用 PI0 路径，True 使用 PI0.5 路径 |
| discrete_state_input | 自动决定 | 是否把归一化 state 离散后写入语言 prompt；默认与 pi05 相同 |

两个容易混淆的点：

- discrete_state_input 不在 Pi0 神经网络类里直接使用，而是由 ModelTransformFactory 创建 TokenizePrompt 时读取。
- PI0.5 仍然保留 Observation.state，但模型前向不再把它投影成连续 state token；它已经在 tokenizer 阶段变成了文本前缀的一部分。

### 2.2 model_type、create() 和 inputs_spec()

model_type 只根据 pi05 返回 PI0 或 PI05，后续数据 transform 会据此选择不同的 tokenization。

create(rng) 相当于：

    Pi0(self, rngs=nnx.Rngs(rng))

inputs_spec(batch_size) 是模型输入契约。对本项目 pi05_so101_lora、batch_size=B：

| 输入 | shape | dtype |
|---|---|---|
| 三路 image | 每路 [B, 224, 224, 3] | float32 |
| 三路 image mask | 每路 [B] | bool |
| state | [B, 32] | float32 |
| tokenized_prompt | [B, 200] | int32 |
| tokenized_prompt_mask | [B, 200] | bool |
| actions | [B, 10, 32] | float32 |

本地轻量验证得到的结果正是：

    state_spec  = (2, 32)
    prompt_spec = (2, 200)
    action_spec = (2, 10, 32)

### 2.3 get_freeze_filter() 到底冻结什么

它返回的是“冻结参数过滤器”，TrainConfig.trainable_filter 再对它取反。

当前 pi05_so101_lora 同时使用 gemma_2b_lora 和 gemma_300m_lora，因此：

- 两个 Gemma 中原有的非 LoRA 参数被冻结；
- 两个 Gemma 中名字含 lora 的 adapter 参数可训练；
- 不在 llm 路径里的参数也不匹配冻结规则，因此仍可训练。

最后一点非常重要：按当前代码，它并非严格意义上的“只训练 LoRA”。SigLIP 图像编码器、action_in_proj、time MLP、action_out_proj 等非 llm 参数也在 trainable_filter 中。若以后想只训练 LoRA，需要另外扩大 freeze_filter，而不能只看配置名里的 lora。

---

## 3. 当前 pi05_so101_lora 的实际模型

训练配置位于 training/config.py 中 name="pi05_so101_lora" 的 TrainConfig：

| 项目 | 当前值 |
|---|---|
| 模型 | PI0.5 |
| PaliGemma 专家 | gemma_2b_lora，width=2048，depth=18 |
| Action Expert | gemma_300m_lora，width=1024，depth=18 |
| 注意力头 | 两个专家均为 8 个 query heads、1 个 KV head、head_dim=256 |
| 图像编码器 | SigLIP So400m/14，width=1152，depth=27 |
| action_horizon | 10 |
| action_dim | 32 |
| max_token_len | 200 |
| discrete_state_input | True |
| batch_size | 16 |
| 归一化 | PI0.5 使用 q01/q99 分位数归一化 |
| 优化器 | AdamW，全局梯度裁剪 0.5 |
| 学习率 | 3000 step warmup 到 3e-4，之后余弦衰减到 1e-7 |
| EMA | None，不维护 EMA 参数 |
| dropout | 未覆盖默认值，所以实际为 0.0 |

SigLIP 用 14×14 patch 处理 224×224 图像，因此每路图像产生：

    (224 / 14) × (224 / 14) = 16 × 16 = 256 tokens

三路相机物理上会产生 768 个 image tokens，再加固定 200 个 prompt slots：

    prefix_len = 3 × 256 + 200 = 968

SO-101 实际只有 base 和 left_wrist 两路相机。right_wrist 是零图且 mask=False：其 256 个 token 仍会被算出来，但 attention mask 会让模型忽略它们。

---

## 4. pi0.py 的模块地图

Pi0 中的模块可以按“前缀、动作后缀、输出头”理解。

| 模块 | 输入 | 输出 | 功能 |
|---|---|---|---|
| PaliGemma.img | [B,224,224,3] | [B,256,2048] | SigLIP 提取每个图像 patch 的视觉特征，并投影到前缀专家宽度 |
| PaliGemma.llm embed | [B,L] token id | [B,L,2048] | 把文本 token id 映射成 embedding |
| 双专家 Gemma | prefix [B,P,2048]、suffix [B,S,1024] | 相同序列长度和各自宽度 | 用共享注意力连接视觉/语言和动作，但两个 token 流保留各自参数 |
| action_in_proj | [B,H,32] | [B,H,1024] | 把带噪动作映射到动作专家宽度 |
| time_mlp_in/out | [B,1024] | [B,1024] | PI0.5 把 flow 时间 t 变成 adaRMSNorm 条件 |
| action_out_proj | [B,H,1024] | [B,H,32] | 把动作专家输出变成速度场 vθ |

经典 PI0 还会创建 state_proj 和 action_time_mlp；PI0.5 不创建它们。

### 4.1 双专家不是简单拼 embedding

前缀宽度是 2048，动作后缀宽度是 1024，所以不能直接在最后一维拼成一个普通张量。

Gemma Attention 的做法是：

1. 两个专家用各自权重分别生成 Q、K、V；
2. Q/K/V 的 head 数和 head_dim 相同，因此可以在 token 序列轴拼接；
3. 在 prefix+suffix 的总序列上计算一次 attention；
4. 再按 token 数切回两个流；
5. 用各自的输出投影、RMSNorm 和 FFN 继续处理。

因此它实现的是“共享注意力上下文、不同专家参数”。动作 token 可以读图像和语言，视觉/语言 token 不会被动作 token 反向污染。

---

## 5. 两个辅助函数

### 5.1 posemb_sincos()

输入：

    pos: [B]，这里是 flow matching 时间 t
    embedding_dim: 1024

输出：

    [B, 1024]

它用一半 sin、一半 cos，在多个频率上编码标量 t。embedding_dim 必须是偶数。本地验证 t=[0,1] 时输出 shape 为 [2,1024]。

注意：这不是图像或 Transformer token 的位置编码，而是“扩散/flow 时间”的编码。

### 5.2 make_attn_mask()

输入：

- input_mask [B,N]：该 token 是否有效，False 表示 padding 或不存在的相机；
- mask_ar [N] 或 [B,N]：True 表示从这里开始一个新的 attention block。

实现先对 mask_ar 做累加，得到每个 token 所在的 block 编号。一个 query 可以看见编号小于等于自己的有效 key。

对于 PI0.5：

    prefix_ar = [False, False, ...]
    suffix_ar = [True, False, False, ...]

因此：

- 所有 prefix tokens 属于 block 0，彼此全注意力；
- 所有 action tokens 属于 block 1；
- prefix 只能看 prefix；
- action 可以看全部有效 prefix，也可以看整个 action chunk。

动作 chunk 内不是逐步 causal attention，而是同一 block 的双向注意力。模型一次联合预测整段 10 步速度。

---

## 6. embed_prefix()：图像和任务进入模型

输入是 Observation：

    images: 三路 [B,224,224,3]
    image_masks: 三路 [B]
    tokenized_prompt: [B,200]
    tokenized_prompt_mask: [B,200]

处理步骤：

1. 依次把每路图像送进 SigLIP；
2. 每路得到 [B,256,2048]；
3. 把该相机的单个 bool mask 重复成 [B,256]；
4. 用 Gemma embedder 把 prompt token 变成 [B,200,2048]；
5. 沿 token 轴拼成 prefix。

输出：

    prefix_tokens: [B,968,2048]
    prefix_mask:   [B,968]
    prefix_ar:     [968]

prompt 的真实长度通常小于 200，多余位置由 prompt mask 屏蔽。right_wrist 的 256 个位置同样被 image mask 屏蔽。

训练时图像增强发生在 embed_prefix 之前的 preprocess_observation()：

- base 相机会随机裁剪、resize、旋转；
- wrist 相机不做几何裁剪旋转；
- 所有有效图像做颜色抖动；
- 输入最终保持 float32 的 [-1,1]。

---

## 7. embed_suffix()：带噪动作和时间进入动作专家

输入：

    noisy_actions x_t: [B,10,32]
    timestep t:        [B]

当前 PI0.5 路径：

1. action_in_proj：

       [B,10,32] -> [B,10,1024]

2. posemb_sincos：

       [B] -> [B,1024]

3. 两层 time MLP 加 swish：

       [B,1024] -> [B,1024]

4. action tokens 本身不与 time embedding 直接相加；time embedding 作为 adarms_cond 送给动作专家每层的 adaptive RMSNorm。

输出：

    suffix_tokens: [B,10,1024]
    suffix_mask:   [B,10]，全 True
    suffix_ar:     [10]，内容为 [True,False,...]
    adarms_cond:   [B,1024]

adaRMSNorm 会由时间条件生成 scale、shift、gate，调制归一化后的动作特征和残差强度。直观上，同一份带噪动作在不同 t 下应该采取不同的去噪方向，时间条件就是告诉动作专家“当前噪声阶段”。

经典 PI0 与此不同：

- state 经 state_proj 形成一个连续 token；
- time embedding 在每个 horizon 位置复制；
- action embedding 与 time embedding 拼接后通过 action_time_mlp；
- 不使用 adaRMSNorm。

---

## 8. compute_loss()：flow matching 训练目标

这是 pi0.py 的训练核心。

### 8.1 输入输出

输入：

    observation
    actions a: [B,H,D]，本项目是 [B,10,32]
    rng
    train=True

输出：

    chunked_loss: [B,H]

每个 batch、每个未来时间步有一个 loss；动作维 D 已在 compute_loss 内求平均。train_step 再把 B 和 H 求平均得到标量。

### 8.2 构造带噪动作

代码先采样：

    ε ~ N(0,I)，shape 与 actions 相同
    t ~ Beta(1.5,1)，再限制到约 [0.001,1]

线性插值路径：

    x_t = t ε + (1-t) a

所以：

- t=0 时 x_t 接近真实动作 a；
- t=1 时 x_t 接近纯噪声 ε。

这条路径关于 t 的真实速度是：

    u_t = d x_t / dt = ε - a

模型接收 observation、x_t、t，预测：

    vθ(observation, x_t, t)

训练损失：

    L = mean_D[(vθ - u_t)²]

### 8.3 一次联合前向

compute_loss() 会：

1. 预处理并增强 observation；
2. embed_prefix(observation)；
3. embed_suffix(observation, x_t, t)；
4. 合并 prefix/suffix mask 并生成 attention mask；
5. 把 [prefix_tokens, suffix_tokens] 一次送进双专家 Gemma；
6. 取 suffix 最后 H 个输出；
7. action_out_proj 得到 [B,H,D] 的 v_t；
8. 与 u_t 做 MSE。

本项目 H=10、D=32，所以大致 shape 为：

    prefix expert input: [B,968,2048]
    action expert input: [B,10,1024]
    logical attention:   [B,978,978]
    predicted velocity:  [B,10,32]
    returned loss:       [B,10]

双专家内部不会真的创建一个最后维统一为 2048/1024 的 embedding 张量；978 只是逻辑总 token 长度。

### 8.4 两个值得留意的训练细节

第一，SO-101 的 6 维动作被补成 32 维后，loss 会计算全部 32 维。后 26 维目标原本为 0，但仍被加入 flow 噪声，因此模型也在学习这些 padding 维的速度场。推理后它们会被 SO101Outputs 丢弃。

第二，LeRobot 在 episode 尾部会把越界未来动作夹到最后一帧，并提供 action_is_pad；当前 SO-101 RepackTransform 没保留该 mask，compute_loss 也没有 padding loss mask。因此这些重复的末帧动作仍参与训练。

---

## 9. sample_actions()：从噪声反向积分成动作

训练已经学到速度场以后，推理不需要真实 actions，只需要 observation 和一个初始噪声。

### 9.1 输入输出

输入：

    observation
    noise: 可选 [B,10,32]
    num_steps: 默认 10

输出：

    normalized actions: [B,10,32]

如果未提供 noise，函数自己采样标准高斯噪声作为 x₁。

### 9.2 为什么 dt 是负数

代码约定：

    t=1：噪声
    t=0：数据

训练学的是随 t 增大时的速度 dx/dt。推理要从 1 走到 0，所以：

    dt = -1 / num_steps
    x_{t+dt} = x_t + dt · vθ(x_t,t)

当前 num_steps=10，会做 10 次 Euler 更新。

### 9.3 KV cache

图像和 prompt 在 10 次积分中不变，因此：

1. 先只计算一次 prefix，保存每层 attention 的 K/V；
2. 每个 Euler step 只重新计算当前 x_t 对应的 suffix；
3. suffix 的 query 读取缓存中的 prefix K/V 和本次 suffix K/V。

这避免 10 次重复跑完整视觉/语言前缀。动作 suffix 自身仍需每个积分步重算，因为 x_t 和 t 一直变化。

sample_actions() 返回的仍是模型空间里的“分位数归一化、32 维、前五关节为 delta”的动作，不能直接下发电机。

---

## 10. train.py 的 main()

main() 可以分成五段。

### 10.1 环境与随机数

- 初始化日志；
- 检查 global batch_size 能被 JAX 设备数整除；
- 设置 JAX compilation cache；
- 用 config.seed 创建 rng，再拆成 train_rng 和 init_rng。

JAX 不使用隐式全局随机数。训练循环虽然反复传同一个 train_rng，但 train_step 内会 fold_in(state.step)，所以每一步得到不同且可复现的随机流。

### 10.2 并行和 checkpoint

- make_mesh(config.fsdp_devices) 创建设备 mesh；
- data_sharding 沿 batch 轴切分；
- replicated_sharding 用于小标量、RNG 和日志指标；
- 初始化 checkpoint manager 和 W&B。

当前配置 fsdp_devices 默认是 1，因此没有把单份模型参数跨多张设备切片；若有多组设备，主要按 batch 做数据并行。

### 10.3 DataLoader 产生第一个 batch

create_data_loader() 最终返回：

    batch = (Observation, actions)

当前 batch 的关键 shape：

    images:
      base_0_rgb       [16,224,224,3]
      left_wrist_0_rgb [16,224,224,3]
      right_wrist_0_rgb[16,224,224,3]
    image_masks: 每路 [16]
    state:             [16,32]
    tokenized_prompt:  [16,200]
    actions:           [16,10,32]

main() 预取第一个 batch 并打印 shape、图像统计、state 均值，还把部分相机图上传 W&B。这一批随后会用于第一次 train_step，不会被丢掉。

### 10.4 初始化 TrainState

init_train_state() 做的事情：

1. 从配置创建学习率 schedule 和 Optax optimizer；
2. config.model.create() 初始化完整模型结构；
3. 从 pi05_base checkpoint 加载能匹配的预训练参数；
4. 把被冻结的参数转成 bfloat16；
5. 只为 trainable_filter 匹配的参数创建 AdamW 优化器状态；
6. 建立 TrainState：

       step
       params
       model_def
       opt_state
       tx
       ema_params

model_def 是网络静态结构，params 是可变化状态。训练时再用 nnx.merge() 合成可调用模型。

### 10.5 JIT 和训练循环

train_step 被 functools.partial 固定 config 后再交给 jax.jit。JIT 配置说明：

- RNG replicated；
- TrainState 按 FSDP 规则放置；
- batch 按 data_sharding 切分；
- donate_argnums=(1,) 允许新 state 复用旧 state 的设备内存。

循环每一步：

    train_state, info = ptrain_step(train_rng, train_state, batch)
    记录 loss / grad_norm / param_norm
    batch = next(data_iter)
    按间隔保存 checkpoint

保存 checkpoint 时不仅保存参数和优化器状态，也把 norm_stats 复制到 checkpoint/assets/blacknew，确保推理使用与训练完全相同的归一化统计。

---

## 11. train_step()：batch、反向传播、更新参数

这是最值得反复读的一段。

### 11.1 恢复可调用模型

    model = nnx.merge(state.model_def, state.params)
    model.train()

TrainState 为了 JAX 变换把结构和状态分开保存；merge 后才重新得到 Pi0 实例。model.train() 把 deterministic 切到训练模式。

不过当前 pi05_so101_lora 的 dropout=0，所以 Gemma dropout 实际没有随机丢弃；train=True 仍会启用图像增强和 flow 随机采样。

### 11.2 loss_fn

    chunked_loss = model.compute_loss(
        rng, observation, actions, train=True
    )                         # [B,10]
    loss = mean(chunked_loss) # scalar

没有单独写 loss.backward()。JAX 的 nnx.value_and_grad() 同时完成“前向算 loss”和“反向自动微分”。

### 11.3 每一步独立随机数

    train_rng = fold_in(rng, state.step)

compute_loss 再把它拆成：

- preprocess_rng：图像增强；
- noise_rng：flow 高斯噪声；
- time_rng：采样 t；
- dropout_rng：Gemma dropout。

同一 seed 和 step 会得到同一随机结果，便于复现。

### 11.4 只对可训练参数求导

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state
    )(...)

argnums=diff_state 告诉 NNX：第 0 个参数 model 只对 trainable_filter 选中的状态叶子微分。

因此 grads 中不包含被冻结的 Gemma base 权重。当前配置中会有：

- 两个 Gemma 的 LoRA adapter 梯度；
- SigLIP 和动作投影/时间 MLP 等未被冻结参数的梯度。

### 11.5 Optax 生成 update

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

当前 tx 是：

    clip_by_global_norm(0.5)
        -> AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10)
        -> 使用当前 cosine learning rate

顺序是先按所有可训练梯度的整体 norm 裁剪，再做 AdamW 动量、自适应缩放和学习率更新。

注意 updates 通常已经带有“下降方向”的负号，apply_updates 做的是：

    new_param = old_param + update

### 11.6 放回完整模型状态

    nnx.update(model, new_params)
    full_new_params = nnx.state(model)

这里只把更新后的可训练叶子写回 model；冻结叶子仍保留原值。随后提取完整 state，保证下一个 step 仍拥有完整网络。

TrainState 更新：

    step += 1
    params = full_new_params
    opt_state = new_opt_state

当前 ema_decay=None，所以 EMA 分支不执行。

### 11.7 日志指标

train_step 返回：

- loss：当前 scalar flow matching loss；
- grad_norm：更新前所有可训练梯度的全局 norm；
- param_norm：模型中 kernel 类参数的全局 norm。

param_norm 过滤了 bias、scale、position embedding、input embedding 等叶子，主要用来观察大矩阵权重是否异常膨胀。

### 11.8 一句话等价伪代码

    observation, actions = batch
    x_t, target_velocity = add_flow_noise(actions)
    predicted_velocity = model(observation, x_t, t)
    loss = mse(predicted_velocity, target_velocity)
    grads = autodiff(loss, trainable_parameters_only)
    grads = clip_global_norm(grads, 0.5)
    new_parameters = adamw_update(parameters, grads, learning_rate)

---

## 12. SO-101：采集数据如何一步步变成训练监督

以下用 blacknew 和当前 pi05_so101_lora 配置说明。

### 12.1 采集/转换后的 LeRobot v3 数据

每个时间点至少有：

- observation.images.env：环境相机；
- observation.images.hand：腕部相机；
- observation.state：[6] 当前 follower 状态；
- action：[6] absolute 目标；
- episode_index、frame_index、timestamp、task_index 等索引字段。

视频保存在 data/video chunk 中，数值字段保存在 Parquet，meta 保存 schema、episode、task、统计等信息。原始数据如何转换成 LeRobot v3，见本项目已有数据说明文档。

### 12.2 为当前帧查询未来 10 步

create_torch_dataset() 根据 fps 给 LeRobotDataset 设置：

    delta_timestamps = [0/fps, 1/fps, ..., 9/fps]

因此一个当前帧不再只有 action [6]，而是得到 action chunk [10,6]。episode 末尾不足 10 步时重复最后一帧。

### 12.3 训练 transform 顺序

顺序必须牢记：

1. RepackTransform

       observation.images.env
           -> observation.images.images_env
       observation.images.hand
           -> observation.images.images_wrist

2. SO101Inputs

       两路图像统一成 HWC uint8
       创建第三路零图 right_wrist
       state 保持 [6]
       action 保持 [10,6]

3. DeltaActions

       前 5 维：action[t,d] -= current_state[d]
       第 6 维 gripper：保持 absolute

4. Normalize

PI0.5 使用 checkpoint 统计中的 q01/q99：

       normalized = (x-q01)/(q99-q01+eps) × 2 - 1

统计量是在 Repack + SO101Inputs + DeltaActions 之后计算的，所以 actions 统计描述的是“前五维 delta + 第六维 absolute gripper”，不是原始六维 absolute action。

5. ModelTransformFactory

       注入默认任务文本
       图像 resize/pad 到 224×224
       把归一化后的 6 维 state 离散进 prompt
       prompt pad/truncate 到 200 tokens
       state [6] -> [32]
       actions [10,6] -> [10,32]

TokenizePrompt 位于 PadStatesAndActions 之前，因此文本中编码的是 6 个真实 state 值，不是补零后的 32 个值。

6. collate 和 Observation.from_dict

       单样本堆叠成 B=16
       uint8 image 转为 float32 [-1,1]
       最终返回 (Observation, actions)

---

## 13. 训练好的权重如何变成 SO-101 推理动作

### 13.1 创建 Policy

create_trained_policy()：

1. 从 checkpoint/params 恢复模型参数；
2. 从同一个 checkpoint/assets/blacknew 加载 norm_stats；
3. 创建训练侧对应的 input transforms；
4. 创建反方向的 output transforms。

使用 checkpoint 自带统计量很重要。即使项目 assets 目录后来变化，推理仍使用训练时的 q01/q99。

### 13.2 客户端 observation 进入服务器

客户端发送与 SO101Inputs 约定相同的字段：

    observation.images.images_env
    observation.images.images_wrist
    observation.state
    可选 prompt

推理 input pipeline：

    SO101Inputs
        -> Normalize
        -> InjectDefaultPrompt
        -> ResizeImages
        -> TokenizePrompt(discrete_state_input=True)
        -> PadStatesAndActions
        -> 添加 batch 维
        -> Observation.from_dict

### 13.3 模型生成 action chunk

Policy.infer() 调用 JIT 后的 model.sample_actions()：

    noise [1,10,32]
        -> 10 次速度场预测和 Euler 更新
        -> normalized delta actions [1,10,32]

去掉 batch 维后得到 [10,32]。

### 13.4 输出逆变换

顺序是：

1. Unnormalize

       用 checkpoint q01/q99 把 state/actions 恢复到机器人单位

2. AbsoluteActions

       前五维 absolute[t,d]
           = predicted_delta[t,d] + current_state[d]
       第六维 gripper 已是 absolute，不相加

3. SO101Outputs

       actions [10,32] -> actions [10,6]

服务器最终返回的是 10 步、6 维 absolute 目标，而不是 delta。

### 13.5 客户端真正下发电机

项目 smooth_preview_v3 客户端还会在模型输出之后做一层执行安全处理：

- 检查 action chunk shape、NaN/Inf 和关节大范围限位；
- 从 action buffer 逐步取 [6] 目标；
- 对前五关节和夹爪做平滑混合；
- 限制单周期最大关节变化；
- 最终调用 robot.send_action() 下发 absolute 目标。

因此要区分三层：

    模型输出：归一化的 32 维 delta/absolute 混合表示
    Policy 输出：[10,6] 机器人单位下的 absolute 目标
    客户端下发：经过平滑和安全限速后的单步 absolute 命令

---

## 14. 建议按这个顺序自己打断点

第一次跟一条真实样本时，推荐依次观察：

1. create_torch_dataset() 返回的原始 sample：

       observation.state [6]
       action [10,6]

2. transform_dataset() 结束后的 sample：

       image 三路
       state [32]
       tokenized_prompt [200]
       actions [10,32]

3. train_step() 入口 batch；
4. compute_loss() 内的 actions、noise、time、x_t、u_t；
5. embed_prefix() 的 prefix_tokens 和 prefix_mask；
6. embed_suffix() 的 suffix_tokens 和 adarms_cond；
7. action_out_proj 后的 v_t；
8. value_and_grad 返回的 loss 和 grads；
9. tx.update 前后的一个 LoRA 参数；
10. Policy.infer() 中 sample_actions 前后的动作；
11. Unnormalize、AbsoluteActions、SO101Outputs 每一步的 shape 和数值单位。

不建议在 JIT 编译后的函数里直接大量 print。先用单 batch、关闭或绕过 JIT 做 shape 调试；需要在 JIT 内观察时使用 jax.debug.print，并只打印少量标量。

---

## 15. 最后用一句话串起来

采集到的相机、当前关节状态和 absolute 示教动作被整理成 LeRobot v3；DataLoader 为每帧取未来 10 步，把前五关节改成相对当前 state 的 delta，按训练统计归一化并补到 32 维；PI0.5 用视觉/语言前缀条件化动作专家，通过 flow matching 学会从噪声预测整段动作的速度场；训练时 JAX 自动微分并由 AdamW 更新可训练参数，推理时从纯噪声反向积分出 [10,32]，再反归一化、加回当前 state、裁成 [10,6]，最后由 SO-101 客户端平滑限速后逐步下发电机。
