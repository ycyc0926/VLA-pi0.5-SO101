import dataclasses
import logging

import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 中文注释：这里只统一 dtype/通道顺序；真正 resize 到 224x224 在 ModelTransformFactory 中发生。
def _parse_image(image, *, key: str):
    image = np.asarray(image)
    
    # 核心诊断：如果收到的是占位符 -1，说明 DataLoader 视频解码失败了
    if image.size > 0 and np.max(image) < 0:
        # 中文注释：原代码引用未定义变量 name，会在真正遇到坏视频时二次抛 NameError；
        # 显式传入 key 后，日志能准确指出 env 还是 wrist 解码失败。
        logging.error(f"!!! CRITICAL: _parse_image received placeholder -1 data for {key} !!!")
        logging.error("Check if your video codec (AV1) is supported by the DataLoader backend.")
    
    # 如果是 [0, 1] 的 float，转成 [0, 255] 的 uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
    # Case 1: CHW → HWC
    if image.ndim == 3 and image.shape[0] in [1, 3] and image.shape[-1] not in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    # Case 2: 灰度 → RGB
    if image.ndim == 2:
        image = np.expand_dims(image, -1)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # 最终强校验（必须HWC）
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Invalid image shape after processing: {image.shape}")

    return image.astype(np.uint8)

@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """把训练样本/客户端请求统一成 PI0 的 Observation 字典。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 中文注释：这两个“重复 images”的键是本项目定义的推理接口；训练时由 RepackTransform
        # 从 blacknew 的 observation.images.env/hand 改名得到，客户端则直接发送相同键。
        base_image = _parse_image(data["observation.images.images_env"], key="env")
        wrist_image = _parse_image(data["observation.images.images_wrist"], key="wrist")
        right_image = np.zeros((224, 224, 3), dtype=np.uint8)  # SO101 没有右腕相机，使用占位图。

        inputs = {
            # observation.state 是当前 follower 的 6 维反馈；后续会 Normalize 并 pad 到 action_dim=32。
            "state": data["observation.state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                # 两台真实相机为 True；不存在的右腕图为 False，告知模型忽略该占位视角。
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change
                # "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0,
                "right_wrist_0_rgb": np.False_,
            }
        }
        if "action" in data:
            # 仅训练数据包含 action，形状由 DataLoader 扩成 [action_horizon, 6]；推理请求没有该键。
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        # 若没有 prompt，后面的 ModelTransformFactory 会注入 SO101 default_prompt。

        return inputs

@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """把模型统一动作张量裁回 SO101 的六个电机目标。"""

    def __call__(self, data: dict) -> dict:
        # 中文注释：进入这里之前已经先 Unnormalize，再把前五维 delta 加回当前 state；
        # 所以返回 [T,6] absolute target，可由客户端按 action chunk 执行。
        return {"actions": np.asarray(data["actions"])[:, :6]}
