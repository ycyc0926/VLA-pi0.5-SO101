# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""供 π0 / π0.5 使用的 Gemma 实现，改编自 big_vision。

这个文件只实现 Transformer 主干，不直接接收摄像头图像、自然语言字符串或
机器人关节角。
上游 ``pi0.py`` 会先把图像、Prompt、状态和带噪动作变成 embedding，再把它们作为不同专家的
token 序列传入这里。当前 π0 通常使用两个专家：

1. 第 0 个专家（PaliGemma/Gemma 2B）处理图像、文本等条件 prefix；
2. 第 1 个专家（Gemma 300M）处理状态、时间和带噪动作组成的 action suffix。

两个专家可以有不同的隐藏宽度和参数，但 Q/K/V 投影后的 head 结构必须相同。Attention 会沿
token 维把两个专家的 Q/K/V 拼接，使视觉语言 token 和动作 token 能在共同注意力空间中交互。

下面是本文件 ``einsum`` 字母使用的张量维度约定：
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    """单个 Gemma 专家的网络结构配置。

    一个 ``Module`` 可以接收多个 ``Config``，每个配置对应一套专家权重。
    不同专家允许使用不同
    的 ``width`` 和 ``mlp_dim``，但为了共同计算 Attention，层数和 head 结构必须兼容。
    """

    width: int  # token 隐藏向量宽度 D，例如 2B 专家为 2048，300M 动作专家为 1024。
    depth: int  # Transformer Block 层数；同一个 Module 中的所有专家必须相同。
    mlp_dim: int  # FFN 中间层宽度，通常远大于 width。1024-4096-1024
    num_heads: int  # Query Head 数量 N。8
    num_kv_heads: int  # Key/Value Head 数量 K；小于 num_heads 时使用 GQA。1
    head_dim: int  # 每个 Attention Head 的维度 H。256
    # 按模块名保存 LoRA 配置；空字典表示该专家使用普通全量线性层。
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """根据模型名称返回专家配置。

    ``dummy`` 只用于单元测试；带 ``_lora`` 的配置与基础模型结构一致，只是在 Attention 和 FFN
    线性层上附加低秩可训练分支。
    """
    if variant == "dummy":
        # 极小模型，用于快速初始化、形状测试和 CI，不用于真实训练或推理。
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 约 311M 参数；在 π0 中通常作为动作专家。
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        # 视觉语言主专家，隐藏宽度与 FFN 都显著大于 300M 专家。
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        # 网络主体仍是 2B，只在 Attention/FFN 上训练 rank=16 的 LoRA 增量。
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 约 311M 参数；动作专家使用更高的 LoRA rank=32。
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")  # 尽早拒绝拼写错误，避免悄悄使用错误模型。


@at.typecheck
class RMSNorm(nn.Module):
    """RMSNorm，并可选支持 AdaRMS（条件调制和门控残差）。

    输入：
        x: ``[B, T, D]``，待归一化 token。
        cond: ``None`` 或 ``[B, D]``。为 ``None`` 时执行普通 RMSNorm；否则根据条件产生
            ``scale``、``shift`` 和 ``gate``。

    输出：
        ``(normalized_x, gate)``。普通模式的 gate 为 ``None``；自适应模式的 gate 形状为
        ``[B, 1, D]``，供后面的残差连接使用。
    """

    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # 记住输入 dtype；模型通常使用 bfloat16/float16 以节省显存。
        # 先升为 float32 再计算均方值，避免半精度平方、求和带来的数值误差。
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        # RMSNorm 不减均值，只除以均方根；1e-6 防止全零向量除零。
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            # 普通 RMSNorm：scale 从 0 开始，但实际乘数为 1+scale，因此初始化时接近恒等缩放。
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # 保持与 Flax RMSNorm 的 ``1 + scale`` 参数化方式一致。
            return normed_inputs.astype(dtype), None  # 计算结束后转回原 dtype；普通模式没有 gate。

        # 自适应 RMSNorm：由样本级条件 cond 同时生成缩放、平移和残差门控，共 3D 个数。
        # kernel 使用零初始化，因此训练开始时 scale=shift=gate=0，
        # 新动作分支不会突然破坏预训练表示。
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        # cond 原本是 [B,D]；先插入 token 维得到 [B,1,3D]，
        # 再切成三个 [B,1,D]，对所有 token 广播。
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # 根据动作/时间条件调制归一化结果。
        return normed_inputs.astype(dtype), gate  # gate 稍后在 ``x + y * gate`` 中控制残差强度。


@at.typecheck
class Embedder(nn.Module):
    """第 0 个专家使用的语言 Token Embedding/词表投影。

    ``encode`` 把整数 token id ``[B,T]`` 变为 embedding ``[B,T,D]``；``decode`` 使用同一张
    权重表把隐藏向量投影回词表 logits。π0 动作生成主要使用 ``encode``，不依赖 ``decode``。
    """

    vocab_size: int
    embed_dim: int

    def setup(self):
        # 参数形状 [V,D]：每个词表 token 对应一个 D 维可训练向量。
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]  # 用 token id 高级索引词表：[B,T] -> [B,T,D]。
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)  # 按 Transformer 惯例乘 sqrt(D)，稳定表示尺度。
        return x  # 返回语言 token embedding。

    def decode(self, x):
        # 权重共享（weight tying）：直接使用输入词表矩阵的转置，
        # 不再创建一套独立输出权重。
        return jnp.dot(x, self.input_embedding_table.T)  # [...,D] -> [...,V]。


@at.typecheck
class Attention(nn.Module):
    """让多个专家在共同 Head 空间中交互的多头自注意力。

    ``xs`` 是一个专家列表，例如 ``[prefix, suffix]``。每个专家先用自己的权重和隐藏宽度产生
    Q/K/V；因为所有专家的 head 数量和 head_dim 相同，随后可以沿 token 维拼接并统一计算
    Attention。最后再按原 token 长度切开，用各自的输出投影恢复到各专家的隐藏宽度。

    输入：
        xs: 每个元素为 ``[B, Ti, Di]`` 或 ``None``；``None`` 表示本次不运行该专家。
        positions: 当前输入 token 的位置编号 ``[B,T]``，用于 RoPE。
        attn_mask: ``[B,1,T,S]``；True 表示 Query 可以关注对应 Key。
        kv_cache: 可选的历史 prefix K/V，用于动作推理时复用视觉语言条件。

    输出：
        out: 与 ``xs`` 一一对应的输出列表，形状仍为 ``[B,Ti,Di]`` 或 ``None``。
        ``(k, v)``: 拼接缓存后的 K/V，可作为下一次 suffix 前向的 KV Cache。
    """

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # 专家的 width 可以不同，但下面会把 Q/K/V 沿 token 维拼接；
        # 因此 head 结构必须完全一致。
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        # 找到本次真正运行的第一个专家，以它的 dtype 作为 Attention 计算输出 dtype。
        dtype = next(x.dtype for x in xs if x is not None)  # 一般是 bfloat16/float16。

        # qkvs 中每项对应一个实际运行的专家，元素统一为 (q,k,v)。
        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                # 推理 prefix 阶段可能只运行专家0，suffix 阶段也可能通过 None 跳过某个专家。
                continue
            if config.num_kv_heads == config.num_heads:
                # 标准 MHA：N=K，可以用一套融合权重一次性投影出 Q、K、V。
                qkv_einsum = lora.Einsum(
                    # 权重 [3,K,D,H]，首维 3 分别代表 Q、K、V。
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                # x [B,S,D] × W [3,K,D,H] -> [3,B,S,K,H]；迭代/解包时首维会成为 q、k、v。
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                # GQA：Query Head 数 N 大于 KV Head 数 K，Q 与 K/V 的权重形状不同，必须分开投影。
                q_einsum = lora.Einsum(
                    # Query 权重 [N,D,H]，例如当前 SO101 为 [8,D,256]。
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)  # [B,T,D] -> [B,T,N,H]。
                kv_einsum = lora.Einsum(
                    # K/V 共用融合权重 [2,K,D,H]；当前 K=1，因此缓存比保存8个 KV Head 小很多。
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)  # 各为 [B,S,K,H]。
                qkvs.append((q, k, v))  # 统一保存为三元组，方便下一步跨专家拼接。

        # zip(*qkvs) 分别收集所有专家的 q、k、v；axis=1 是 token 维，不是特征维。
        # 例如 prefix 长 P、suffix 长 A，拼接后 q/k/v 的 token 长度为 P+A。
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)  # 给 Query 注入相对位置信息，形状不变。
        q *= self.configs[0].head_dim ** -0.5  # 除以 sqrt(H)，避免点积随维度增大而使 softmax 饱和。

        k = _apply_rope(k, positions=positions)  # Key 使用与 Query 一致的位置旋转。

        # RoPE 内部为了精度会临时升到 float32，但返回前必须转回模型 dtype。
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            # 推理时 cache 通常是已经算好的视觉语言 prefix；当前 k/v 是本轮 action suffix。
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)  # Key 序列变为 [cached prefix, current suffix]。
            v = jnp.concatenate([cache_v, v], axis=1)  # Value 必须以完全相同的 token 顺序拼接。

        # 把 N 个 Query Head 拆成 K 组、每组 G=N/K 个 Query Head，显式表示 GQA 的共享关系。
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        # 每个 Query Head 与所属组共享的 Key Head 做点积：输出 [B,K,G,T,S]。
        # 点积累加显式使用 float32，提高长序列/半精度模型的数值稳定性。
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        # mask 的 S 必须包含 cache 和当前 token 的总 Key 长度；
        # 形状错通常表示 prefix/suffix mask 构造错误。
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # 被屏蔽位置放极大负数，softmax 后其概率近似为 0；
        # 该常量与 Gemma 官方实现保持一致。
        # 不直接用 -inf，可减少部分低精度/后端上出现 NaN 的风险。
        big_neg = -2.3819763e38
        # attn_mask [B,1,T,S] 插入 G 维后可广播到 logits [B,K,G,T,S]。
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        # 在 Key 维 S 上归一化，再转回模型 dtype。
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # 注意力概率加权 Value：[B,K,G,T,S] × [B,S,K,H] -> [B,T,K,G,H]。
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        # 合并 K 和 G，恢复普通多头表示 [B,T,N,H]。
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        # 前面把所有专家 token 拼在一起；
        # 这里按各专家原长度切回去并使用各自的输出投影。
        out = []
        start = 0  # 当前专家在 encoded 总 token 序列中的起点。
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]  # 当前专家占据 [start:end] 这段 token。
                out_einsum = lora.Einsum(
                    # 每个专家用自己的 [N,H,D_i] 权重，把公共 Head 表示投回专家宽度 D_i。
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))  # [B,Ti,N,H] -> [B,Ti,Di]。
                start = end  # 下一个实际运行专家从这里继续切片。
            else:
                out.append(None)  # 保持输出列表与原始专家列表同位置对齐。

        return out, (k, v)  # 返回输出以及“cache + 当前 token”的完整 K/V。


@at.typecheck
class FeedForward(nn.Module):
    """Gemma 风格的门控前馈网络（当前主 Block 使用的是 ``lora.FeedForward``）。

    这里保留了不带 LoRA 的基础实现，计算过程近似 ``GELU(xW_gate) * (xW_up)``，再通过
    ``W_down`` 投影回原宽度。它有助于理解 FFN，但当前 ``Block`` 的实际执行路径调用的是
    ``openpi.models.lora.FeedForward``。
    """

    features: int  # 输入和输出隐藏宽度 D。
    hidden_dim: int  # FFN 扩展后的中间宽度 M。

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # 参数取出后转为输入 dtype，避免把半精度激活意外提升为 float32。
        # 将 gate 和 up projection 两套 [D,M] 权重堆在首维，形状为 [2,D,M]。
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])  # [B,T,D] -> [B,T,M]，生成门控分支。
        gate_value = nn.gelu(ff_gate)  # 使用 GELU 把门控分支变成非线性权重。

        ff1 = jnp.dot(x, w_gating[1])  # 另一套上投影，同样得到 [B,T,M]。
        activations = gate_value * ff1  # 逐元素相乘，形成 gated-MLP 中间激活。

        # 下投影把扩展宽度 M 还原为模型宽度 D。
        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)  # [B,T,M] -> [B,T,D]。
        assert outputs.dtype == dtype  # 防止某一步无意中将模型提升到 float32，增加显存占用。
        return outputs  # 这里只返回 FFN 增量，残差相加由外层 Block 完成。


@at.typecheck
class Block(nn.Module):
    """一层双专家 Transformer Block。

    数据依次经过：``RMSNorm -> Attention -> 残差 -> RMSNorm -> FFN -> 残差``。每个专家
    拥有独立的 Norm、Attention 投影和 FFN 参数，但 Attention 会暂时合并不同专家的 token。

    ``xs`` 是 scan 的 carry，输入输出都是专家隐藏状态列表；
    ``kv_cache`` 是本层输入的历史 K/V，
    本层计算出的完整 K/V 会作为 scan 的输出收集起来。
    """

    configs: tuple[Config, ...]  # 每个专家一份结构配置。

    dropout: float = 0.0  # 0 表示关闭 Dropout；推理配置通常为 0。
    dropout_bdims: tuple[int, ...] = ()  # 指定 Dropout mask 需要广播的维度。

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        # 给编译器标注激活的设备分片方式；单卡时通常不会改变数值和形状。
        xs = sharding.activation_sharding_constraint(xs)
        # 关闭 Dropout 时直接使用恒等函数，避免创建无意义的随机数操作。
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        # 本层所有专家共用一个跨专家 Attention 容器。
        attn = Attention(configs=self.configs, name="attn")

        # 第一阶段：分别对各专家做 Attention 前 RMSNorm，同时保存 AdaRMS 产生的残差 gate。
        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                # adarms_cond[i] 为 None 时是普通 RMSNorm，否则使用条件调制并返回 [B,1,D_i] gate。
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)  # 保持与专家列表相同的顺序和 None 占位。
            gates.append(gate if x is not None else None)  # 未运行的专家没有残差 gate。

        # 跨专家 Attention：返回每个专家的 Attention 增量和本层新的 K/V。
        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        # 对专家树中的所有非 None 叶子施加 Dropout。
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        # 普通专家执行 x+y；AdaRMS 专家执行 x+y*gate。zip(strict=True) 可防止专家数量静默错位。
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # 第二阶段：每个专家独立执行 FFN；这里不再跨专家混合 token。
        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # FFN 前再次归一化，并为这一条残差支路单独计算 gate。
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,  # FFN 输入/输出宽度与该专家 width 相同。
                    hidden_dim=config.mlp_dim,  # 中间扩展宽度由该专家配置决定。
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),  # 非 LoRA 配置会得到 None。
                )(x)  # [B,Ti,D_i] -> [B,Ti,D_i]。
            out.append(x)  # x 为 None 时继续保留占位；否则保存 FFN 增量。
            gates.append(gate if x is not None else None)  # 每个专家各自的 FFN 残差 gate。

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)  # FFN 输出也经过 Dropout。
        # 将 FFN 增量接回 Attention 残差后的 xs，得到本 Transformer Block 的最终输出。
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache  # xs 继续作为下一层 carry；kv_cache 由 scan 沿 layer 维收集。


# 每层一份 K/V。l=层数、b=batch、_t=缓存 token 数、_k/_v=KV Head 数、_h=head_dim。
# 前导 layer 维由下面的 ``nn.scan`` 生成；单层 Attention 实际接收的是去掉 l 维后的 K/V。
KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """支持多套专家权重的 Gemma Transformer 主干。

    一个实例通常接收 ``[视觉语言 prefix, 动作 suffix]`` 两组 embedding。第 0 个专家额外拥有
    语言词表 Embedder；所有专家经过相同层数的 Block，并在每层 Attention 中交换信息。

    训练时可同时传入所有专家；推理时也可以传入 ``[prefix, None]`` 先构造 prefix KV Cache，
    再传入 ``[None, suffix]``（或相应调用形式）复用缓存、只计算动作 token。
    """

    configs: Sequence[Config]  # 专家配置列表；列表下标也是专家编号。
    embed_dtype: str  # 激活和 embedding 使用的 dtype 字符串，例如 ``bfloat16``。

    dropout: float = 0.0  # Transformer 残差分支的 Dropout 比例。
    dropout_bdims: tuple[int, ...] = ()  # 空元组表示各浮点元素独立采样 Dropout mask。
    adarms: bool = False  # 保留的模块配置字段；实际是否调制由调用时的 adarms_cond 决定。

    def setup(self):
        """创建词嵌入、带重计算的扫描 Block，以及每个专家的最终 RMSNorm。"""

        # nn.scan 会让所有专家同步经过一层，所以专家深度必须完全一致。
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        # 只有第0个 PaliGemma 专家需要把离散语言 token id 映射成隐藏向量。
        # 动作专家的 embedding 由 pi0.py 中的状态/动作线性层产生，不需要词表。
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )

        # remat（activation checkpointing）：训练反向时重算 Block 内部激活，用计算时间换显存。
        # 推理没有反向传播，因此这个包装不会形成“异步推理”，
        # 也不会改变模型数学结果。
        block_cls = nn.remat(
            Block,
            prevent_cse=False,  # Block 位于 scan 内部，无需额外阻止公共子表达式消除。
            # Block.__call__ 中 0=self、6=deterministic；Python 布尔值要作为静态参数参与 JAX tracing。
            static_argnums=(6,),
            policy=jax.checkpoint_policies.nothing_saveable,  # 不保存中间激活，反向时重新计算。
        )

        # scan 用一段 Block 程序循环 depth 次。
        # 参数在第0维按层堆叠，但每层参数彼此独立，并非权重共享。
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},  # 参数树增加 layer 轴：[L,...]，每次迭代取对应层切片。
            split_rngs={"params": True, "dropout": True},  # 每层初始化和 Dropout 使用不同随机数。
            in_axes=(
                0,  # kv_cache 按 layer 轴扫描：第 i 层只读取第 i 层历史 K/V。
                nn.broadcast,  # positions 对所有层相同。
                nn.broadcast,  # attention mask 对所有层相同。
                nn.broadcast,  # AdaRMS 条件对所有层相同。
                nn.broadcast,  # deterministic 布尔值对所有层相同。
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
            length=self.configs[0].depth,  # 当前真实模型为18层，dummy模型为4层。
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        # 所有 Block 结束后，每个专家再单独做一次 RMSNorm；
        # 名称规则确保第0专家兼容预训练权重。
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        """把语言 token id ``[B,T]`` 编码为第0专家的 embedding ``[B,T,D0]``。"""

        return self.embedder.encode(tokens).astype(self.embed_dtype)  # 统一转换为模型激活 dtype。

    @at.typecheck
    def __call__(
        self,
        # 每个专家一组 token；某项为 None 表示本次跳过该专家，但列表位置不能改变。
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],  # 当前 Query/新 Key token 的 RoPE 位置。
        mask: at.Bool[at.Array, "b t s"],  # 输入是 [B,T,S]，下面会补上 head 广播维。
        # 每个专家的 AdaRMS 条件 [B,D_i]；None 表示该专家使用普通 RMSNorm。
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,  # 可选的各层 prefix K/V；训练完整序列时通常为 None。
        deterministic: bool = True,  # True 关闭 Dropout，用于评估和推理。
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        """运行全部 Transformer 层并返回各专家输出和逐层 KV Cache。

        ``T`` 是当前运行专家 token 长度之和；``S`` 是可见 Key 长度，使用缓存时通常等于
        ``cached_prefix_length + current_suffix_length``。
        """

        # 对专家树中所有数组统一 dtype；JAX tree 会保留用于跳过专家的 None 节点。
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        # [B,T,S] -> [B,1,T,S]，中间的 1 会广播到所有 KV Head 和 Query group。
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            # 调用者完全不提供条件时，所有专家都使用普通 RMSNorm。
            adarms_cond = [None] * len(self.configs)

        # embedded 是 scan carry；输入 kv_cache 沿层读取，输出 kv_cache 沿层重新堆叠。
        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        # 及时检测某个算子意外产生 float32 激活，否则显存和速度问题会很难定位。
        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        # 每个专家分别做最终 RMSNorm，只取返回元组中的第0项；
        # 最终 gate 不再用于残差。
        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """用最小假输入走遍所有分支，使 Flax 创建完整参数树。

        Linen 只会为实际执行到的子模块创建参数，
        所以初始化时必须显式调用词嵌入和所有专家；
        ``use_adarms`` 决定哪些专家还需要创建 AdaRMS 的调制 Dense 参数。
        """

        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))  # 触发创建第0专家的 [V,D0] 词表参数。
        self(
            # 每个专家提供 1 个假 token；各专家最后一维使用自己的 width。
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            # 每专家1个 token，所以拼接后的总 token 数等于专家数。
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            # 最小 [B,T,S] mask；数值不重要，本次调用的目的只是创建参数。
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            # 需要 AdaRMS 的专家得到 [1,D_i] 假条件，其余专家传 None。
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """给 Q/K 应用旋转位置编码 RoPE，形状保持不变。

    输入 ``x`` 通常为 ``[B,L,N,H]`` 或 ``[B,L,K,H]``，``positions`` 为 ``[B,L]``。
    RoPE 不添加一个位置向量，而是把 head_dim 的前后两半看成二维坐标对，
    根据位置旋转；这样
    Q 与 K 的点积会自然包含相对位置信息。
    """

    # 为 H/2 对旋转坐标生成从低频到高频的指数；head_dim 必须能均分成两半。
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents  # [H/2]，不同特征对使用不同旋转周期。
    radians = positions[..., None] / timescale[None, None, :]  # [B,L,H/2]。
    radians = radians[..., None, :]  # [B,L,1,H/2]，新增 head 广播维，所有 Head 使用相同位置角度。
    assert radians.dtype == jnp.float32  # 三角函数保持 float32，避免半精度位置误差。
    sin, cos = jnp.sin(radians), jnp.cos(radians)  # 预先计算每个位置和频率的旋转系数。
    x1, x2 = jnp.split(x, 2, axis=-1)  # 把最后一维 H 切成两组 H/2，作为成对旋转坐标。
    # 对每对坐标应用二维旋转矩阵 [[cos,-sin],[sin,cos]]，再拼回完整 head_dim。
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32  # 与 float32 sin/cos 运算后结果会暂时升为 float32。
    # big_vision 原实现可能让训练时 RoPE 结果留在 float32、
    # 推理缓存时再降精度，造成两条路径不一致。
    # 这里始终转回输入 dtype，与常见 Gemma/Transformers 实现一致，也避免 KV Cache 显存翻倍。
    return res.astype(x.dtype)


def _name(name, i):
    """为专家参数生成兼容 PaliGemma checkpoint 的名字。"""

    # 第0专家沿用原始名字（如 attn），才能直接匹配并加载已有 PaliGemma 权重。
    # 后续专家增加编号后缀（如 attn_1），避免参数重名；
    # π0 实际通常只有 PaliGemma 和动作专家两个。
    if i == 0:
        return name  # 第0专家不加后缀，对齐预训练 checkpoint 参数路径。
    # 动作专家等新增专家使用独立参数名，通常需要新初始化或加载微调权重。
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    """执行普通残差或 AdaRMS 门控残差，并保留被跳过专家的 ``None``。"""

    # 输入和增量必须同时存在或同时为 None；不一致说明专家切片/列表对齐出现程序错误。
    assert (x is None) == (y is None)
    if x is None:
        return None  # 本次未运行该专家，继续传递占位。
    if gate is None:
        return x + y  # 普通 Transformer 残差连接。
    return x + y * gate  # AdaRMS：按样本、按通道控制新分支写入残差流的强度。
