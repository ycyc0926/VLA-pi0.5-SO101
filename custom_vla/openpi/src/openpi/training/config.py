"""See _CONFIGS for the list of available configs.

中文阅读地图：
1. AssetsConfig / DataConfig：描述数据管线最终需要的“成品配置”。
2. ModelTransformFactory：把统一的机器人数据变成具体 PI0/PI0.5 模型所需格式。
3. DataConfigFactory 及各 LeRobot*DataConfig：按机器人/数据集组装数据变换链。
4. TrainConfig：把模型、数据、优化器、日志和 checkpoint 合成一次实验。
5. _CONFIGS / get_config()：用字符串 name 找到某个 TrainConfig 实例。

训练输入的固定顺序在 training/data_loader.py::transform_dataset 中：
Repack -> robot-specific inputs -> Normalize -> model inputs。
推理输出的逆向顺序在 policies/policy_config.py 中：
model outputs -> Unnormalize -> robot-specific outputs。
"""

import torch
import numpy as np  # 中文注释：供 SYSMO front-only 高斯噪声插件处理 uint8 图像使用。
import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.policies.sysmo_policy as sysmo_policy
import os  # 中文注释：用于从命令行环境变量读取 SYSMO-32 的任务名和数据集路径。
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

from openpi.transforms import PromptIndexToText
import openpi.policies.so101_policy as so101_policy

class AddGaussianNoiseImage(_transforms.DataTransformFn):
    def __init__(self, std=0.03, keys=("observation.images.env", "observation.images.hand")):
        self.std = std
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        for key in self.keys:
            if key in data:
                img_tensor = data[key]
                # 只有在训练模式下叠加噪声
                noise = torch.randn_like(img_tensor) * self.std
                data[key] = torch.clamp(img_tensor + noise, min=0.0, max=1.0)
        return data


class AddGaussianNoiseToModelImages(_transforms.DataTransformFn):
    """对 policy 已解析出的模型输入图像增加高斯噪声；仅用于显式开启的数据增强。"""

    def __init__(self, std: float = 0.03, keys: tuple[str, ...] = ("base_0_rgb",)):
        self.std = float(std)
        self.keys = tuple(keys)

    def __call__(self, data: dict) -> dict:
        image_dict = data.get("image")
        if not isinstance(image_dict, dict):
            return data  # 中文注释：图像尚未进入模型格式时不处理，避免误改原始样本结构。

        for key in self.keys:
            if key not in image_dict:
                continue
            image = np.asarray(image_dict[key])
            noise = np.random.normal(
                loc=0.0,
                scale=self.std * 255.0,
                size=image.shape,
            ).astype(np.float32)
            image_dict[key] = np.clip(
                image.astype(np.float32) + noise,
                0.0,
                255.0,
            ).astype(np.uint8)  # 中文注释：模型图像是 uint8 [0,255]，std 仍按 [0,1] 比例解释。

        return data

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 中文注释：第一层“字段改名/重组”。例如把 LeRobot 的 observation.images.env
    # 映射为 SO101Inputs 约定的 observation.images.images_env；通常只用于训练数据集读取。
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 中文注释：第二层“机器人语义适配”，训练和推理共用；负责组装 state、三路 image/mask、
    # action，并可在归一化前把 absolute action 转成 delta action。
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 中文注释：第三层“模型格式适配”，在 Normalize 之后执行；负责 224x224 resize、
    # prompt tokenization，以及把短 state/action padding 到模型统一 action_dim。
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("action",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        # 中文注释：Factory 根据 model_type 返回不同的 transform，而不是创建神经网络本身。
        # PI0 与 PI0.5 的主要差别在 TokenizePrompt 是否把 state 一起离散化进 token。
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        # 若上游没有 prompt 才注入默认值；客户端/数据集已有 prompt 时不会覆盖。
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        # 保持纵横比并 padding 到 224x224，三路图像都处理。
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            # True：state 编入离散 token；False：state 不进入 prompt token。
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        # SO101/LIBERO 只有 6/7 维，PI0 默认 action_dim=32；右侧补零统一 shape。
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        # 中文注释：asset_id 决定去哪个子目录读取 norm_stats；默认与 repo_id 相同，
        # 也可像 blacknew 一样显式设成稳定、无斜杠的名字。
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            # PI0.5/FAST 使用 q01/q99 映射到约 [-1,1]；经典 PI0 使用 mean/std z-score。
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    #extra_delta_transform: bool = False#改为绝对位置训练
    extra_delta_transform: bool = True

    # 🚀 新增：实验参数开关
    use_gaussian_noise: bool = False  # 是否开启高斯噪声
    gaussian_std: float = 0.03        # 噪声标准差

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # 中文注释：RepackTransform 的写法是“新键: 数据集中的旧键”。
                        # 左侧是 LiberoInputs 接下来要读取的统一接口，右侧是 LeRobot 样本字段。
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            # inputs 在训练和推理请求进入模型前执行；outputs 只在推理动作离开模型后执行。
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            # make_bool_mask(6, -1) => 前 6 维 True，其余维（第 7 维 gripper）False。
            # push 会把 DeltaActions 追加到输入尾部，同时把 AbsoluteActions 放到输出头部，
            # 从而形成训练 absolute->delta、推理 delta->absolute 的镜像关系。
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        # Libero 的 prompt 来自数据集 task，因此这里不需要 default_prompt。
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """一次训练/推理实验的顶层配置。

    它本身不执行训练；scripts/train.py 和 training/data_loader.py 读取这些字段，
    分别创建模型/优化器/checkpoint 管理器与数据流水线。
    """

    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # 中文注释：W&B project 名；不同 exp_name 的 run 会归到这个 project 下。
    project_name: str = "openpi"
    # 中文注释：本次运行名，也是 checkpoint 路径最后一级；必须由 CLI 或配置提供。
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # 中文注释：JAX 训练初始化权重的策略；可从 base checkpoint 部分加载并与当前模型树对齐。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # 中文注释：仅 train_pytorch.py 使用的 safetensors 目录；JAX scripts/train.py 不读取它。
    pytorch_weight_path: str | None = None

    # 中文注释：仅 PyTorch 训练精度；本项目 JAX LoRA 路径主要由模型 dtype 决定。
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    # 中文注释：学习率随 step 如何变化，以及参数用何种梯度更新规则。
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    # 中文注释：指数滑动平均参数；None 表示不维护 EMA，LoRA 配置通常关闭以节省内存。
    ema_decay: float | None = 0.99

    # 中文注释：匹配“冻结参数”的过滤器；trainable_filter 会取其补集。LoRA 只训练 lora 参数。
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 中文注释：不是最终 DataConfig，而是工厂；创建 DataLoader/Policy 时才调用 data.create()。
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 中文注释：norm_stats 等数据资产根目录；实际目录还会拼接 config name/asset id。
    assets_base_dir: str = "./assets"
    # 中文注释：checkpoint 根目录；最终为 checkpoint_base_dir/name/exp_name。
    checkpoint_base_dir: str = "./checkpoints"

    # 中文注释：模型初始化、shuffle、flow matching 噪声等随机过程的可复现种子。
    seed: int = 42
    # 中文注释：全局 batch；多设备时由 sharding 进一步切分，不是“每张卡 batch”。
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    # 中文注释：0 表示在训练主进程加载；增加它可并行解码视频，但会增加 CPU/内存占用。
    num_workers: int = 2
    # 中文注释：优化器更新次数，不是 epoch 数；一个 step 消耗一个 global batch。
    num_train_steps: int = 30_000

    # 中文注释：每多少 step 聚合并记录 loss/grad_norm 等指标。
    log_interval: int = 100
    # 中文注释：普通 checkpoint 保存间隔。
    save_interval: int = 1000
    # 中文注释：能被该周期整除的 checkpoint 长期保留；其他旧点可能被清理。
    keep_period: int | None = 5000

    # 中文注释：删除同名实验目录后重训；与 resume 互斥，属于破坏性选项。
    overwrite: bool = False
    # 中文注释：从同一 checkpoint_dir 的最新 step 恢复模型、优化器和数据加载状态。
    resume: bool = False

    # 中文注释：是否把训练指标、配置和样例图上传 W&B。
    wandb_enabled: bool = True

    # 中文注释：随 Policy 暴露给客户端的机器人元信息，例如 ALOHA reset_pose；不参与 loss。
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    # 中文注释：1 表示不跨设备切分一份模型参数；必须能整除当前 JAX device 数。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        # 中文注释：选择 PI0.5；每个样本监督未来 10 步动作。这里显式关闭离散 state token，
        # 因而 state 仍可用于 delta 变换/归一化，但不会被 TokenizePrompt 拼进语言 token。
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            # LeRobot Hub 数据源；DataLoader 会按 action_horizon 取 action[t:t+10]。
            repo_id="physical-intelligence/libero",
            # 把 LeRobot tasks.parquet 对应文本注入为样本 prompt。
            base_config=DataConfig(prompt_from_task=True),
            # LIBERO 原始 action 已是 delta，所以不重复减当前 state。
            extra_delta_transform=False,
        ),
        # 256 是 global batch；显存不足时最先调低此值。
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            # 前 10k step 从很小的学习率线性升到 5e-5。
            warmup_steps=10_000,
            peak_lr=5e-5,
            # 计划在 1M step 内衰减；本配置只训练 30k，且首尾 lr 都是 5e-5，
            # 所以 warmup 后实际上近似保持常数 5e-5。
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        # 先把全局梯度范数裁到 1.0，再执行 AdamW 更新。
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # 同时维护参数的 0.999 EMA 副本，推理/保存时可获得更平滑的权重。
        ema_decay=0.999,
        # JAX 路径从官方 PI0.5 base 参数初始化，而不是从零训练。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # 仅 PyTorch 训练脚本读取；占位路径必须在使用 train_pytorch.py 前替换。
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

# 中文注释：注意第一份官方 _CONFIGS 已在上方结束。下面先定义项目自定义 DataConfig，
# 随后文件又用 `_CONFIGS = [...]` 建立第二份列表；这会覆盖包含 pi05_libero 的第一份列表。
@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    # blacknew parquet 保存 absolute joint target；True 表示训练前把前五关节改为相对当前 state 的 delta。
    extra_delta_transform: bool = True

    # 新增：实验参数开关 (默认关闭，不影响原来的配置)
    use_gaussian_noise: bool = False  
    gaussian_std: float = 0.03

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 1. 字段重组。RepackTransform 写法是“新键: blacknew 中的旧键”。
        # 新键刻意与真实 SO101 客户端请求保持一致，使训练样本与推理请求进入同一个 SO101Inputs。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.images_env": "observation.images.env",
                        "observation.images.images_wrist": "observation.images.hand",
                        "observation.state": "observation.state",
                        "action": "action",
                        # "prompt": "Grab the red ball and place it in the white cup",
                    }
                )
            ]
        )
        
        # 中文注释：这张表目前没有接入 transform，所以不会改变 prompt；当前训练文本实际由
        # 下方 ModelTransformFactory(default_prompt=...) 注入。保留它仅作为任务编号说明。
        TASK_INDEX_TO_PROMPT = {
            0: "Grab the black cube and place it in the white cup",
        }
        _ = TASK_INDEX_TO_PROMPT

        # 2. 机器人语义适配。SO101Inputs 只负责图像格式和统一模型键；归一化稍后由
        # data_loader 根据 norm_stats 自动插入，不能把这里误读成已经 Normalize。
        
        # 逻辑注入：把 SO101Inputs 和噪声插件放到一个列表里
        transform_inputs = [so101_policy.SO101Inputs(model_type=model_config.model_type)]
        
        # 如果 TrainConfig 开启加噪，就在 SO101Inputs 后追加插件。
        # 注意：SO101Inputs 已把图像收进 data["image"]，而 AddGaussianNoiseImage 仍查找旧的
        # observation.images.* 顶层键，因此当前实现实际上不会命中；默认 false 不受影响。
        if self.use_gaussian_noise:
            transform_inputs.append(AddGaussianNoiseImage(std=self.gaussian_std))
        
        data_transforms = _transforms.Group(
            inputs=transform_inputs, # 处理图像解析、state padding
            outputs=[so101_policy.SO101Outputs()],
        )

        # 3. blacknew 的 action 是 absolute。输入侧将前 5 个关节执行 action-state，
        # 第 6 维 gripper 保持 absolute；推理输出侧用 AbsoluteActions 把前 5 维加回当前 state。
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(5, -1)  # 5 关节 delta, 1 gripper绝对位置
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        
        # 4. 模型格式适配：三路图像 resize/pad 到 224x224，注入默认任务文本并 tokenize，
        # 再把 6 维 state 和 [T,6] action 补到模型统一的 32 维。
        model_transforms = ModelTransformFactory(default_prompt="Grab the black cube and place it in the white cup")(model_config)  # 标准: resize 224x224, tokenize prompt

        # 4. 返回一个完整配置
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        

@dataclasses.dataclass(frozen=True)
class LeRobotSYSMO32DataConfig(DataConfigFactory):
    use_gaussian_noise: bool = False  # 中文注释：默认关闭，避免推理服务自动给 front 图加噪。
    gaussian_std: float = 0.03  # 中文注释：按 [0,1] 图像比例解释，例如 0.05 表示约 12.75 灰度级标准差。

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image_front": "observation.images.front",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task_index",
                    }
                )
            ]
        )

        task_name = os.environ.get("SYSMO_TASK_NAME", "FistBump")  # 中文注释：每次训练可通过环境变量指定任务文本，默认 FistBump。

        transform_inputs = [
            PromptIndexToText({0: task_name}),  # 中文注释：将数据集 task_index=0 映射为当前任务文本。
            sysmo_policy.SYSMO32Inputs(model_type=model_config.model_type),
        ]
        if self.use_gaussian_noise:
            transform_inputs.append(
                AddGaussianNoiseToModelImages(
                    std=self.gaussian_std,
                    keys=("base_0_rgb",),
                )
            )  # 中文注释：SYSMO-32 仅给真实 front 图加噪，不处理两个 mask=False 的腕部零图。

        data_transforms = _transforms.Group(
            inputs=transform_inputs,
            outputs=[sysmo_policy.SYSMO32Outputs()],
        )

        # 中文注释：SYSMO-32 当前数据为 12 维双臂 state/action；12 维动作全部按 delta 训练，并在推理输出侧还原为 absolute joint target。
        delta_action_mask = _transforms.make_bool_mask(12)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory(default_prompt=task_name)(model_config)  # 中文注释：训练与推理使用同一个任务文本。

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        # 中文注释：SYSMO-32 使用原始 front 单相机 schema，prompt 从 task_index 映射为数据集任务文本，不依赖 env/hand。


# 中文注释（重要）：这里是第二次赋值而不是 `_CONFIGS += [...]`，所以会丢弃上方包含
# pi05_libero/ALOHA/DROID 的官方配置。文件末尾 _CONFIGS_DICT 只会基于下面这些自定义项创建。
_CONFIGS = [
    # 全量微调
    TrainConfig(
        name="pi05_so101",
        # 中文注释：全量微调版本使用普通 Gemma 权重；action chunk 为未来 10 步。
        # discrete_state_input=False 表示 PI0.5 tokenizer 不把 state 离散化进 prompt token。
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotSO101DataConfig(
            # 旧采集机上的硬编码路径；在当前服务器通常不可用，仅作历史配置参考。
            repo_id="/home/likunwei/lerobot/dataset/pickPlaceCube/",
            base_config=DataConfig(prompt_from_task=False),  # 从 task 提取 prompt
            extra_delta_transform=True,  # 绝对动作
            # default_prompt="Pick up the yellow ball and place it in the black storage box.",
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
            #     asset_id="trossen",
            # ),
        ),
        # 全量和 LoRA 都从同一 PI0.5 base 起步，区别在模型 variant/freeze_filter。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"), # 预训练基础模型
        num_train_steps=30000,  # 调整基于数据量
        batch_size=2,  # GPU 内存调整
        num_workers=0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
    ),

    # lora微调
    TrainConfig(
        name="pi05_so101_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            # 中文注释：两个 Transformer 都换成带 LoRA adapter 的结构；不是缩小 base 权重。
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 每次监督/预测 10 个未来控制目标，模型内部 shape 为 [10, 32]。
            action_horizon=10,
            #action_horizon=50,#修改：对标官方50步
            # PI0.5 将归一化后的 state 数值离散化并拼入 prompt token，使状态真正进入语言前缀。
            discrete_state_input=True,
        ),
        data=LeRobotSO101DataConfig(
            # 中文注释：从环境变量读取当前 blacknew 根目录，避免绑定旧机器绝对路径。
            repo_id=os.environ["SO101_DATASET_DIR"],
            # 训练时 norm_stats 路径为 (data.assets.assets_dir 或 assets_base_dir/name)/blacknew；
            # 推理时强制从 checkpoint/assets/blacknew 读取，确保与训练统计一致。
            assets=AssetsConfig(asset_id="blacknew"),
            #repo_id="/home/likunwei/lerobot/dataset/sysmo32_fistbump_36_20260612",
            # 不读取 tasks.parquet 的 "Grab the black cube"；使用 create() 中更完整的 default_prompt。
            base_config=DataConfig(prompt_from_task=False),
            # blacknew 是 absolute action：前五关节转 delta，gripper 保持 absolute。
            extra_delta_transform=True,
            #extra_delta_transform=False,#绝对位置训练
            #use_gaussian_noise=True, #加入高斯噪声
            # 当前噪声插件键不匹配且正式实验关闭；训练/推理均不会加噪。
            use_gaussian_noise=False, #推理时关闭
            gaussian_std=0.05
        ),
        # 中文注释：初始化 base 权重后，freeze_filter 冻结原参数，只更新 LoRA adapter。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100000,
        batch_size=16,
        num_workers=12,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3000,  # 前 3000 step 线性升至 peak_lr。
            peak_lr=3e-4,  # LoRA 可用比全量微调更高的峰值学习率。
            decay_steps=100000,  # 与总步数一致：warmup 后余弦衰减到训练结束。
            decay_lr=1e-7,  # 训练末期的目标学习率。
        ),
        # 0.5 的全局梯度裁剪比默认 1.0 更保守。
        optimizer=_optimizer.AdamW(clip_gradient_norm=0.5),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # LoRA 不维护整套参数 EMA，减少额外显存/存储。
        ema_decay=None,
    ),
    #Sysmo32通用微调
    TrainConfig(
        name="pi05_sysmo32_env",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotSYSMO32DataConfig(
            repo_id=os.environ.get("SYSMO_DATASET_DIR", "/home/likunwei/lerobot/dataset/sysmo32_fistbump_36_20260612"),  # 中文注释：每次训练通过环境变量指定数据集路径。
            base_config=DataConfig(prompt_from_task=False),
            use_gaussian_noise=os.environ.get("SYSMO_USE_GAUSSIAN_NOISE", "0") == "1",  # 中文注释：通过环境变量控制是否启用图像高斯噪声。
            gaussian_std=float(os.environ.get("SYSMO_GAUSSIAN_STD", "0.05")),  # 中文注释：通过环境变量控制噪声强度。
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100000,
        batch_size=16,
        num_workers=12,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3000,
            peak_lr=3e-4,
            decay_steps=100000,
            decay_lr=1e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=0.5),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #Sysmo32特殊一次的微调
    TrainConfig(
        name="pi05_sysmo32_fistbump36_20260612",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotSYSMO32DataConfig(
            repo_id="/home/likunwei/lerobot/dataset/sysmo32_fistbump_36_20260612",
            base_config=DataConfig(prompt_from_task=False),
            use_gaussian_noise=True,  # 正式训练实验可单独打开。
            gaussian_std=0.05,  # 保留此前 SO101 实验中使用过的噪声强度候选值。
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100000,
        batch_size=16,
        num_workers=12,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3000,
            peak_lr=3e-4,
            decay_steps=100000,
            decay_lr=1e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=0.5),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # 中文注释：新增 SYSMO-32 单相机 PI05 LoRA 配置，训练超参沿用 pi05_so101_lora；使用 SYSMO 专属前 12 维增量动作变换。
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
