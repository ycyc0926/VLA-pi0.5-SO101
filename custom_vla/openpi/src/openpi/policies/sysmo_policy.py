import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image.astype(np.uint8)


@dataclasses.dataclass(frozen=True)
class SYSMO32Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/image_front"])
        empty_image = np.zeros_like(front_image)

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": empty_image,
                "right_wrist_0_rgb": empty_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)[..., :12]  # 中文注释：原始 action 有 14 维，但后 2 维当前无物理意义；训练仅保留真实 12 自由度。

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs
        # 中文注释：SYSMO-32 只有 front 单相机，真实图像只进入 base_0_rgb，缺失腕部相机用零图像和 False mask 表示。


@dataclasses.dataclass(frozen=True)
class SYSMO32Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :12]}  # 中文注释：推理只返回真实 12 自由度的绝对目标位置，不输出原始数据中的 2 个无意义占位维度。
