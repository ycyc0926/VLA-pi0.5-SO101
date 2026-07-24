#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224

# 1. config 注册
@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
    """
        第一部分:模型参数
    """
    # 2. 模型声明
    paligemma_variant: str = "gemma_2b" # 声明vlm 模型 2B
    action_expert_variant: str = "gemma_300m" # 声明action expert 模型 300M
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    # 3. 动作chunk
    n_obs_steps: int = 1 # 观测steps
    chunk_size: int = 50  # 模型一次性预测的动作步数 Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # 实际执行多少步后重新规划 Number of action steps to execute

    # 4. 最大状态维度
    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # 5. flowmatching 配置
    """
    num_inference_steps:推理时的步数(速度-质量权衡)
        Flow Matching 是在时间 t ∈ [0,1] 上逐渐将噪声变成动作的过程：
        x_t = (1-t) * noise + t * action
        这些参数就是为了控制这个"时间维度"的行为。
    """
    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10  # Number of denoising steps during inference(从 t=1 到 t=0 走多少步)
    """
    目的：训练时用于给"时间t"采样的参数
    Beta 分布参数：训练时重点学习哪个阶段
        α=1.5, β=1.0 时,分布偏向 t=1(高噪声区域),因为作者发现：多训练高噪声区域能提升性能
        不同参数的效果：
        α=1.0, β=1.0 → 均匀分布(所有时间同等重要)
        α=1.5, β=1.0 → 偏向 t=1(多学去噪初期)
        α=1.0, β=1.5 → 偏向 t=0(多学精细调整)
    """
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    """
    目的: Scale/Offset:避开边界的不稳定区域
        Beta分布采样在 [0,1]，但边界 t=0 和 t=1 有问题：
        - t=1: 纯噪声，梯度可能不稳定
        - t=0: 真实动作，模型容易过拟合
        t_final ∈ [0.001, 0.001+0.999] = [0.001, 1.0]
        效果：
            - 避免 t=0(永远不会看到完美真实动作）
            - 避免 t=1(永远不会看到纯噪声）
            - 给边界留一点安全距离
    """
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    """
    Min/Max Period:时间的位置编码精度
        用于时间 t 的正弦位置编码, 将时间 t 编码成向量，让模型能理解"当前在去噪过程的哪个阶段"
    计算频率范围
        min_period=0.004 → 最高频率（捕捉细微时间变化）
        max_period=4.0   → 最低频率（捕捉整体时间位置）
        
    不同频率的正弦波组合
        freqs = torch.linspace(
            math.log(1/min_period),  # log(1/0.004) ≈ 5.5
            math.log(1/max_period),  # log(1/4) ≈ -1.39
            dim // 2
        )
        
    用多个频率的 sin/cos 编码时间
        这样模型能同时知道"大致时间位置"和"微小时间变化"
    """
    min_period: float = 4e-3
    max_period: float = 4.0

    # 6. Real-Time Chunking (RTC) configuration 实时动作分块
    """
    思想：
        让机器人在执行当前动作的同时，在后台异步地生成下一个动作块，并通过一个巧妙的“引导”机制，让前后两个块能平滑地连接起来
    过程：
        1.边做边想(异步执行):
            当机器人开始执行第1个动作块(步1-50)时, 模型在后台直接开始生成第2个动作块(步51-100)了。当第1块执行完,第2块已经基本就绪,消除了等待时间。
        2.无缝衔接(引导对齐):
            由于机器人在模型思考时已经移动了一部分,第2块的开头必须和第1块的结尾完美“对上”。RTC 通过在去噪过程中加入一个“引导项”来解决这个问题。
        它会把已经执行过的动作(比如步46-50)“冻结”起来,然后强制要求新生成的动作块(步51-...)的开头部分(比如步51-60)与这些已执行的动作平滑地连接。
    """
    rtc_config: RTCConfig | None = None

    # 7. 输入图像分辨率
    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0 # pi0默认的三个视角 左腕、右腕、TOP

    # 8. 归一化
    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY, # 图像不归一化，可能图像已是 [0, 1]
            "STATE": NormalizationMode.MEAN_STD, # 使用均方差MSE标准化
            "ACTION": NormalizationMode.MEAN_STD, # 使用均方差MSE标准化
        }
    )

    """
        第二部分：常规训练配置
            使用 warmup + cosine decay 的学习率策略
    """
    # 9. Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization 以时间换空间的内存优化技术,训练时不保存所有中间激活值，反向传播时重新计算,可节省约30-50%显存，但训练速度降低20-30%
    compile_model: bool = False  # Whether to use torch.compile for model optimization PyTorch 2.0+ 的模型编译优化, 将Python代码编译成优化的内核，融合操作;推理速度提升20-50%，训练速度提升10-20%;首次运行有编译开销，适合长期运行的任务
    compile_mode: str = "max-autotune"  # Torch compile mode "default"：快速编译，适度优化 ; "reduce-overhead"：减少Python开销，适合小模型 ; "max-autotune"：最大优化，编译时间长但运行最快 ; "max-autotune-no-cudagraphs"：同最大优化但不使用CUDA Graphs
    device: str | None = None  # Device to use for the model (None = auto-detect) None：自动检测（有GPU用cuda，否则cpu） ; "cuda"：使用GPU（默认0号卡） ; "cuda:0"：指定使用0号GPU ; "cpu"：强制使用CPU（一般不推荐）

    #10. Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder ; True：冻结视觉编码器（SigLIP）的权重，只训练语言模型和动作专家; 适用：新任务与预训练数据视觉分布相似，只需适应新动作
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections; 冻结整个VLM（视觉+语言），只训练动作专家和投影层; 适用：快速适配新机器人，数据量少时防止过拟合; 效果：参数量从~2.5B降到~300M，显存需求大降 

    #11. Optimizer settings: see openpi `AdamW``
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr` 太小：收敛慢，可能陷入局部最优; 太大：震荡不收敛，Loss爆炸
    optimizer_betas: tuple[float, float] = (0.9, 0.95) # 两个动量参数 β1 = 0.9:梯度的一阶矩（均值）的衰减率 ; β2 = 0.95：梯度的二阶矩（方差）的衰减率; 越大表示考虑更多历史梯度
    optimizer_eps: float = 1e-8 # 防止除零的极小值，数值稳定性
    optimizer_weight_decay: float = 0.01 # 权重衰减（L2正则化），防止过拟合，0.01是常见值，越大正则化越强
    optimizer_grad_clip_norm: float = 1.0 # 梯度裁剪阈值，防止梯度爆炸； 如果梯度范数 > 1.0，等比例缩放回1.0

    #12. Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000 # 预热步数：从0线性增长到峰值学习率；目的:避免一开始就用大学习率导致不稳定
    scheduler_decay_steps: int = 30_000 # 衰减步数：从峰值余弦下降到最终学习率 ; 总训练步数 = warmup + decay（或自定义）
    scheduler_decay_lr: float = 2.5e-6 # 最终学习率：衰减结束时的学习率（峰值的1/10）

    tokenizer_max_length: int = 48  # see openpi `__post_init__` # 语言指令的最大token长度;超过会被截断，不足会padding

    """
        第三部分
    """
    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """
            Validate and set up input/output features.
            动态构建模型需要的输入/输出特征字典
            简单说，它在告诉系统："模型需要接收什么数据，输出什么数据"
        """
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        # 目的：确保状态输入存在，如果配置里没定义，就自动添加这个特征
        # OBS_STATE 通常是字符串 "state"（机器人状态：关节角度、速度等）
        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        # 目的：确保动作输出存在
        # ACTION 通常是字符串 "action"
        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
