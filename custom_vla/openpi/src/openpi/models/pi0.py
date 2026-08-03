"""PI0 / PI0.5 的 JAX/Flax NNX 模型主体。

中文阅读主线：
1. make_attn_mask：把 padding mask 和“分块注意力”规则合成最终 attention mask；
2. posemb_sincos：把 flow matching 的连续时间 t 编码成向量；
3. Pi0.__init__：创建视觉编码器、双 Gemma 专家和动作输入/输出投影；
4. embed_prefix：把图像和任务文本编码成条件前缀；
5. embed_suffix：把带噪动作 x_t 和时间 t 编码成动作后缀；
6. compute_loss：训练速度场 v_theta(x_t, t)；
7. sample_actions：从高斯噪声开始，用 Euler 法反向积分得到动作。

本文中的常用 shape 符号：
    B = batch size
    P = prefix token 数
    H = action_horizon
    D = action_dim
    E_pg = PaliGemma 专家宽度
    E_act = Action Expert 宽度
"""

import logging

# einops 用字符串表达 repeat/reshape，便于明确每个 tensor 轴的含义。
import einops
# nnx 是当前模型外层使用的 Flax 模块系统；bridge 用来包装仍由 flax.linen 实现的 Gemma/SigLIP。
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
# JAX 提供随机数、JIT、while_loop 等；jax.numpy 是可微分、可编译的 NumPy 接口。
import jax
import jax.numpy as jnp
from typing_extensions import override

# BaseModel、Observation、图像预处理等公共模型接口。
from openpi.models import model as _model
# Pi0Config 决定 PI0/PI0.5、动作维数、horizon 和两个 Gemma variant。
from openpi.models import pi0_config
# Gemma 是双专家 Transformer；SigLIP 是图像编码器。
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
# at 主要用于运行时 shape/dtype 类型检查，不改变模型数学逻辑。
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# 目的是设计因果/双向 prefix 分块注意力 mask，供 Gemma Transformer 使用。
def make_attn_mask(input_mask, mask_ar):
    """把有效 token mask 与分块规则合成为 [B,N,N] attention mask。

    核心思想：mask_ar=True 表示“从该 token 开始一个新 attention block”。
    对 mask_ar 累加后，每个 token 会得到一个 block 编号。query 只能读取
    block 编号小于等于自己的有效 key，所以能同时表示双向 prefix、因果序列
    和 PI0 使用的 prefix/action 分块注意力。

    例如：
      [1,1,1,1]：每个 token 都开启新 block，即普通 causal attention；
      [0,0,0,1,1,1]：前三个 token 同属双向 prefix，后三个 token 逐个 causal；
      [0,0,0,1,0,0]：prefix 是 block 0，后三个 token 同属 action block 1。

    Args:
        input_mask: bool[B,N]，True 是有效 token，False 是 padding/不存在的相机。
        mask_ar: bool[N] 或 bool[B,N]，True 表示从当前位置开始一个新 block。

    Returns:
        bool[B,N,N]；第 2 维是 query，第 3 维是 key。
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)  # [N] -> [B,N]，让 batch 内每个样本使用同一分块规则。
    cumsum = jnp.cumsum(mask_ar, axis=1)  # [B,N]；把 True 累加成每个 token 所属的 block 编号。
    # 左侧是 key 的 block：[B,1,N]；右侧是 query 的 block：[B,N,1]。
    # key_block <= query_block 表示 query 可读取本 block 和所有更早 block。
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # [B,N,N]，轴顺序为 batch/query/key。
    # input_mask[:,None,:] 屏蔽无效 key；input_mask[:,:,None] 屏蔽无效 query。
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  # [B,N,N]，同时要求 query/key 有效。
    return jnp.logical_and(attn_mask, valid_mask)  # 分块规则和 padding 规则必须同时满足。


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """把每个样本的标量 flow 时间 t 编码成多频率 sin/cos 向量。

    Args:
        pos: [B]，在本文件中就是 t，约位于 [0,1]。
        embedding_dim: 输出宽度，必须为偶数；PI0.5 SO-101 中为 1024。
        min_period/max_period: 最敏感和最平缓的两个周期边界。

    Returns:
        [B,embedding_dim]；前一半是 sin，后一半是 cos。
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")  # sin/cos 各占一半，奇数维无法平分。

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)  # [E/2]，在 0~1 间均匀选择频率档位。
    period = min_period * (max_period / min_period) ** fraction  # [E/2]，按对数尺度从 min_period 过渡到 max_period。
    sinusoid_input = jnp.einsum(
        "i,j->ij",  # 对每个 batch 时间和每个频率做外积：[B] x [E/2] -> [B,E/2]。
        pos,  # [B]，每个样本自己的 flow 时间。
        1.0 / period * 2 * jnp.pi,  # [E/2]，把周期转换成角频率。
        precision=jax.lax.Precision.HIGHEST,  # 时间编码数值很小，使用较高 einsum 精度。
    )
    # 在最后一维拼接，得到 [B,E]；它编码的是 flow 时间，不是 Transformer token 位置。
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    """PI0/PI0.5：视觉语言前缀条件化的连续动作 flow-matching 模型。"""

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        """按 Pi0Config 创建视觉编码器、双 Transformer 专家和动作投影层。"""
        # BaseModel 只保存三个公共超参数，不会在这里额外创建网络层。
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05  # True 走 PI0.5 的“state 离散进 prompt + time 调制 adaRMS”路径。
        paligemma_config = _gemma.get_config(config.paligemma_variant)  # 前缀专家配置；SO-101 LoRA 配置宽度为 2048。
        action_expert_config = _gemma.get_config(config.action_expert_variant)  # 动作专家配置；SO-101 LoRA 配置宽度为 1024。

        # Gemma 当前仍是 flax.linen.Module，ToNNX 把它桥接为可被 NNX split/merge/filter 的模块。
        # configs 列表的第 0 个专家处理 prefix，第 1 个专家处理 suffix。
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],  # 两套参数共享 attention 上下文，但各自有投影、Norm 和 FFN。
                embed_dtype=config.dtype,  # 大矩阵计算通常使用 bfloat16。
                adarms=config.pi05,  # PI0.5 为动作专家启用 adaptive RMSNorm。
                dropout=config.dropout,  # 中文注释：将顶层 Pi0Config 中的 Dropout 概率透传到底层 Gemma。
            )
        )
        # Linen 桥接模块需要显式 lazy_init；PI0.5 只让第 1 个动作专家接收 adaRMS 条件。
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        # SigLIP So400m/14：14x14 patch；224x224 图像会产生 16x16=256 个 patch token。
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 把 SigLIP 原宽度最终投影到 prefix 专家的 E_pg。
                variant="So400m/14",  # So400m 视觉 Transformer，patch size=14。
                pool_type="none",  # 不做全局池化，保留全部 256 个 patch token。
                scan=True,  # 用 JAX scan 表达重复层，减小编译图规模。
                dtype_mm=config.dtype,  # 视觉 Transformer 的矩阵乘使用配置 dtype。
            )
        )
        # 用一份 fake image 只初始化参数结构；不会把 fake observation 保存为模型输入。
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)  # 沿用上游命名：内部同时保存双 Gemma 和 SigLIP。
        # [B,H,D] -> [B,H,E_act]，把连续带噪动作变成动作专家 token。
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            # PI0.5：两层 MLP 把 sin/cos(t) 变为动作专家每层 adaRMSNorm 的条件向量。
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # 经典 PI0：state 是独立连续 token；time 直接与每个 action token 拼接后过 MLP。
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        # [B,H,E_act] -> [B,H,D]，输出 flow 速度；它不是直接输出 absolute 电机命令。
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # model.train()/model.eval() 会自动改这个字段；主要用于控制 dropout 等训练期随机层。
        self.deterministic = True  # 新建模型默认处于确定性推理状态。

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """把三路图像和任务 prompt 编码成 PaliGemma 条件前缀。

        SO-101 PI0.5 的典型输出：
            tokens:     [B,968,2048] = 3*256 个图像 token + 200 个文本位置
            input_mask: [B,968]，屏蔽右腕占位图和 prompt padding
            ar_mask:    [968]，全 False，表示整个 prefix 属于同一个双向 attention block

        注意：这里的 ar 不是说 prefix 自回归；恰恰因为都是 False，它们共享同一 block，
        图像和文本 token 可以彼此双向读取。
        """
        input_mask = []  # 暂存每一路图像/文本的有效位置 mask，最后沿 token 轴拼接。
        ar_mask = []  # 暂存每个 token 是否开启新 attention block；prefix 中都填 False。
        tokens = []  # 暂存各路 image tokens 和 prompt embeddings。

        # 第一块：编码图像。obs.images 通常按 base、left_wrist、right_wrist 顺序保存。
        for name in obs.images:
            # 输入 [B,224,224,3] -> SigLIP patch tokens [B,256,E_pg]。
            # train=False 关闭 SigLIP 内部 dropout；训练图像增强已在 preprocess_observation 中完成。
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)  # 保存本相机的 [B,256,E_pg]，稍后与其他视角/文本拼接。
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],  # [B]；每个样本只用一个 bool 表示该相机是否存在。
                    "b -> b s",  # 把单个相机 mask 沿该相机的 patch token 轴重复。
                    s=image_tokens.shape[1],  # s=256，输出 [B,256]。
                )
            )
            # False 不开启新 block：所有有效图像 token 以及后面的文本 token 彼此双向可见。
            ar_mask += [False] * image_tokens.shape[1]  # 为本视角追加 256 个 False。

        # 第二块：把任务文本 token id 查 embedding table，加入同一个 prefix block。
        if obs.tokenized_prompt is not None:
            # [B,L] int token id -> [B,L,E_pg] embedding；这里只查表，不执行 Transformer。
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)  # L=200（PI0.5 默认最大长度），含真实 token 和补齐位置。
            input_mask.append(obs.tokenized_prompt_mask)  # [B,L]；只有真实文本/state token 为 True。
            # 文本和图像仍属于 prefix block 0，所以图像与语言之间也是 full attention。
            ar_mask += [False] * tokenized_inputs.shape[1]  # 追加 L 个 False。

        tokens = jnp.concatenate(tokens, axis=1)  # 沿序列轴拼接：[B,P,E_pg]。
        input_mask = jnp.concatenate(input_mask, axis=1)  # 同顺序拼成 [B,P]，必须与 tokens 对齐。
        ar_mask = jnp.array(ar_mask)  # Python bool list -> JAX bool[P]，供 make_attn_mask 使用。
        return tokens, input_mask, ar_mask  # 返回“内容、有效性、分块规则”三件套。

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """把带噪动作 x_t 和时间 t 编成 Action Expert 使用的 suffix。

        Args:
            obs: 当前 observation；经典 PI0 从中读取连续 state，PI0.5 不在这里读取 state。
            noisy_actions: x_t，shape [B,H,D]；不是原始动作，也不是普通“加噪标签”。
            timestep: 每个样本的 flow 时间 t，shape [B]。

        Returns:
            tokens: PI0.5 为 [B,H,E_act]；经典 PI0 多一个 state token。
            input_mask: suffix 中所有位置是否有效。
            ar_mask: 定义 state/action attention block。
            adarms_cond: PI0.5 的 [B,E_act] 时间条件；经典 PI0 返回 None。
        """
        input_mask = []  # suffix 的有效 token mask；state/action 都没有 padding，所以最终为全 True。
        ar_mask = []  # suffix 的 attention block 边界。
        tokens = []  # 经典 PI0 可包含 state token；两种模型都包含 H 个 action tokens。

        if not self.pi05:
            # 经典 PI0 才把连续 state 投影成一个 suffix token；PI0.5 的 state 已离散进 prompt。
            state_token = self.state_proj(obs.state)[:, None, :]  # [B,D] -> [B,E_act] -> [B,1,E_act]。
            tokens.append(state_token)  # state 放在 action tokens 前面。
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))  # [B,1]，state token 始终有效。
            # True 开启 suffix 的 state block；prefix token 因 block 更早，无法反向看到 state/action。
            ar_mask += [True]  # state 是 block 1（prefix 是 block 0）。

        # 不论 PI0/PI0.5，都先把连续 x_t 从 D 维投影到动作专家宽度。
        action_tokens = self.action_in_proj(noisy_actions)  # [B,H,D] -> [B,H,E_act]。
        # 用多频率 sin/cos 编码 t；敏感周期覆盖约 [0.004,4.0]，适合 t∈[0,1]。
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            # PI0.5：时间不直接加到 action token，而是经过 MLP 后调制每层 adaptive RMSNorm。
            time_emb = self.time_mlp_in(time_emb)  # [B,E_act] -> [B,E_act]。
            time_emb = nnx.swish(time_emb)  # 加非线性，使时间条件不局限于线性频率组合。
            time_emb = self.time_mlp_out(time_emb)  # [B,E_act] -> [B,E_act]。
            time_emb = nnx.swish(time_emb)  # 得到最终 adaRMS 条件。
            action_expert_tokens = action_tokens  # [B,H,E_act]；token 内容只由 x_t 投影得到。
            adarms_cond = time_emb  # [B,E_act]；Gemma 会在每层生成 scale/shift/gate。
        else:
            # 经典 PI0：把同一个 t embedding 复制 H 份，与每个动作 token 直接拼接。
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)  # [B,H,2*E_act]。
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)  # [B,H,2E] -> [B,H,E]。
            action_time_tokens = nnx.swish(action_time_tokens)  # 融合动作内容和 flow 时间。
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)  # 保持 [B,H,E_act]。
            action_expert_tokens = action_time_tokens  # 作为动作专家真正输入的 H 个 token。
            adarms_cond = None  # 经典 PI0 的 Gemma 使用普通 RMSNorm。

        tokens.append(action_expert_tokens)  # PI0.5 只有这一项；PI0 则接在 state token 后。
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))  # [B,H]，动作 token 全有效。
        # 第一个 action 的 True 开启一个新 block，后 H-1 个 False 与它共享同一 action block。
        # 因此 H 个动作 token 彼此双向可见，并都能读取 prefix；不是 horizon 内逐步 causal。
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)  # PI0.5: [B,H,E_act]；PI0: [B,1+H,E_act]。
        input_mask = jnp.concatenate(input_mask, axis=1)  # 与 suffix token 顺序严格对齐。
        ar_mask = jnp.array(ar_mask)  # Python list -> JAX bool[H] 或 bool[1+H]。
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """计算 flow matching 速度回归损失。

        设真实归一化动作是 a，高斯噪声是 epsilon，代码采用：
            x_t = (1-t)*a + t*epsilon
            u_t = d(x_t)/dt = epsilon-a

        模型根据 observation、x_t、t 预测 v_t，并最小化 ||v_t-u_t||^2。

        Args:
            rng: 当前训练 step 的 JAX 随机 key。
            observation: 已经过 DataLoader transforms 的批量观测。
            actions: 真实动作 chunk [B,H,D]；SO-101 当前为 [B,10,32]。
            train: True 时启用图像增强和配置中的 dropout。

        Returns:
            [B,H]；动作维 D 已求均值，train_step 会再对 B/H 求均值得到标量。
        """
        # 四种随机过程必须使用互不相同的子 key，否则会产生不必要的随机相关性。
        preprocess_rng, noise_rng, time_rng, dropout_rng = jax.random.split(rng, 4)
        # 将 uint8/不同尺寸图像统一为 [-1,1]、224x224；train=True 时执行随机图像增强。
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]  # 去掉 horizon/action_dim，只保留可能有多层的 batch 轴；常见为 (B,)。
        noise = jax.random.normal(noise_rng, actions.shape)  # epsilon ~ N(0,I)，shape [B,H,D]。
        # Beta(1.5,1) 更常采到靠近 1 的时间；缩放后避开精确的 0 和 1。
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # [B]，约在 [0.001,1]。
        time_expanded = time[..., None, None]  # [B] -> [B,1,1]，以便广播到 H 和 D。
        # 线性概率路径：t=0 是真实动作 a，t=1 是高斯噪声 epsilon。
        x_t = time_expanded * noise + (1 - time_expanded) * actions  # [B,H,D]，模型实际看到的带噪动作。
        u_t = noise - actions  # [B,H,D]，线性路径对 t 的解析导数，也是监督目标速度。

        # 训练时 prefix 和 suffix 都没有 KV cache，所以合在一次双专家 Transformer 前向中计算。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)  # [B,P,E_pg]、[B,P]、[P]。
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )  # PI0.5: [B,H,E_act]、[B,H]、[H]、[B,E_act]。
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)  # [B,P+S]，S=H（PI0.5）。
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)  # [P+S]，定义 prefix/action 两个 block。
        attn_mask = make_attn_mask(input_mask, ar_mask)  # [B,P+S,P+S]，第二轴 query、第三轴 key。
        # 有效 token 的位置从 0 连续递增；padding 位置不会参与 attention，所以其重复位置无影响。
        positions = jnp.cumsum(input_mask, axis=1) - 1  # [B,P+S]，供 Gemma 的 RoPE 使用。

        # 双专家一次前向：第 0 路 prefix 宽 E_pg，第 1 路 suffix 宽 E_act。
        # Gemma 内部会分别投影 Q/K/V，再沿 token 轴拼起来做共享 attention，最后切回两个专家。
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],  # 两路 embedding 最后一维可不同，因此用 list 而不是直接 concatenate。
            mask=attn_mask,  # suffix 可看 prefix，prefix 不可看 suffix；action chunk 内是双向 attention。
            positions=positions,  # RoPE 位置覆盖逻辑总序列 P+S。
            adarms_cond=[None, adarms_cond],  # prefix 用普通 RMSNorm；PI0.5 action expert 用时间条件 adaRMS。
            rngs=nnx.Rngs(dropout=dropout_rng),  # 把当前 step 的随机流交给 Dropout，内部会继续按层拆分。
            deterministic=not train,  # train=True -> False，可启用 dropout；当前 SO-101 配置 dropout 实际为 0。
        )
        # prefix_out 在 flow loss 中没有直接监督；它的作用是通过 attention 为 suffix 提供条件。
        # 经典 PI0 的 suffix 首位还有 state token，所以统一只取最后 H 个 action 位置。
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # [B,H,E_act] -> [B,H,D]。

        # 先得到 [B,H,D] 逐维平方误差，再只沿最后的动作维 D 求均值，返回 [B,H]。
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        #num_steps: int | at.Int[at.Array, ""] = 50,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """从高斯噪声出发，沿模型速度场反向积分得到归一化动作 chunk。

        代码约定 t=1 为纯噪声、t=0 为数据，这与 pi0 论文正文的时间方向相反，
        但只要训练和推理采用同一约定，数学结果不受影响。

        Args:
            rng: 未显式传 noise 时，用它生成初始高斯噪声。
            observation: 批量观测；不需要真实 actions。
            num_steps: Euler 积分步数；当前默认 10。
            noise: 可选的固定 x_1 [B,H,D]，用于复现实验或比较模型。

        Returns:
            x_0 [B,H,D]，仍位于模型归一化空间；Policy 输出 transform 才会
            Unnormalize、把 delta 加回 state，并裁成机器人真实维数。
        """
        # 推理不做随机增强；这里只做尺寸、dtype、[-1,1] 值域等确定性预处理。
        observation = _model.preprocess_observation(None, observation, train=False)
        # 要从 t=1 走向 t=0，所以 dt 为负；10 步时 dt=-0.1。
        dt = -1.0 / num_steps  # 标量；可能是 Python float，也可能是 JAX scalar。
        batch_size = observation.state.shape[0]  # PI0.5 虽不在 suffix 读取 state，仍用它取得 B。
        if noise is None:
            # x_1 ~ N(0,I)，shape [B,H,D]；它是常微分方程积分的初始状态。
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 图像和 prompt 在所有 Euler step 中不变，先单独跑一次 prefix 并缓存每层 K/V。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)  # [B,P,E_pg]、[B,P]、[P]。
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)  # [B,P,P]，prefix 内部双向 attention。
        positions = jnp.cumsum(prefix_mask, axis=1) - 1  # [B,P]，prefix 的有效 RoPE 位置。
        # [prefix_tokens,None] 表示只运行第 0 个专家；返回的 kv_cache 含所有层 prefix K/V。
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            """执行一次 Euler 更新：(x_t,t) -> (x_t + dt*v_t, t+dt)。"""
            x_t, time = carry  # x_t: [B,H,D]；time 是所有 batch 共用的标量。
            # embed_suffix 需要 [B] 时间，所以把标量 time 广播成每个样本一份。
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            # 当前 suffix query 如何读取本次 suffix key；PI0.5 动作 chunk 内彼此双向可见。
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)  # [B,S,S]。
            # 这里只为 suffix query 构造 mask；prefix K/V 已在 cache 中，不再产生 prefix query。
            prefix_attn_mask = einops.repeat(
                prefix_mask,  # [B,P]，False 的右腕/prompt padding 仍不可读。
                "b p -> b s p",  # 为每一个 suffix query 复制一份 prefix key mask。
                s=suffix_tokens.shape[1],  # 输出 [B,S,P]。
            )
            # key 轴按“缓存的 prefix K/V + 本次 suffix K/V”排列，因此也按该顺序拼 mask。
            full_attn_mask = jnp.concatenate(
                [prefix_attn_mask, suffix_attn_mask], axis=-1
            )  # [B,S,P+S]。
            assert full_attn_mask.shape == (
                batch_size,  # B。
                suffix_tokens.shape[1],  # query 数 S。
                prefix_tokens.shape[1] + suffix_tokens.shape[1],  # key 数 P+S。
            )  # 防止 cache 长度、mask 和 token 拼接顺序不一致。
            # suffix 的 RoPE 位置紧接每个样本的“有效 prefix 长度”，而不是物理 padding 长度 P。
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]  # [B,1]，每个样本有效 prefix token 数。
                + jnp.cumsum(suffix_mask, axis=-1)  # [B,S]，suffix 内从 1 开始计数。
                - 1  # 转成从 0 开始的位置编号。
            )  # [B,S]。

            # 第 0 个专家传 None：不重算 prefix；第 1 个专家处理当前 x_t 对应的 suffix。
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],  # 只为 suffix 生成新 Q/K/V。
                mask=full_attn_mask,  # suffix query 可读取 prefix cache 和当前 suffix。
                positions=positions,  # 这里只提供本次新 suffix token 的位置。
                kv_cache=kv_cache,  # 复用循环外捕获的只读 prefix K/V。
                adarms_cond=[None, adarms_cond],  # PI0.5 用当前 time 调制动作专家。
            )
            assert prefix_out is None  # 第 0 路输入为 None，所以不应产生 prefix 输出。
            # 只取最后 H 个 action token；经典 PI0 suffix 前面可能还有一个 state token。
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # [B,H,D] 预测速度。

            # 显式 Euler：负 dt 让状态沿学到的 dx/dt 从噪声端 t=1 走向数据端 t=0。
            return x_t + dt * v_t, time + dt  # 新 carry 仍是 ([B,H,D], scalar)。

        def cond(carry):
            """控制 while_loop 是否继续；允许接近 0 的浮点误差。"""
            x_t, time = carry  # x_t 在判断中不用，但 carry 结构必须完整解包。
            # 因 dt<0，-dt/2 是一个很小正数；到达 t≈0 后停止，避免多走一步。
            return time >= -dt / 2

        # lax.while_loop 可被 JIT 编译；初始 carry 是 (x_1=noise, t=1.0)。
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0  # [B,H,D]，尚未执行 Policy 的 Unnormalize/AbsoluteActions/SO101Outputs。
