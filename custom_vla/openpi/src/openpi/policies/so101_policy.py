import dataclasses
from openpi.models import model as _model
import numpy as np
from openpi import transforms
import logging

# 处理输入的图像数据，确保其格式符合特定要求（形状为 (224, 224, 3)，数据类型为 uint8）
def _parse_image(image):
    image = np.asarray(image)
    
    # 核心诊断：如果收到的是占位符 -1，说明 DataLoader 视频解码失败了
    if image.size > 0 and np.max(image) < 0:
        logging.error(f"!!! CRITICAL: _parse_image received placeholder -1 data for {name} !!!")
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
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation.images.images_env"])  # front cam
        wrist_image = _parse_image(data["observation.images.images_wrist"])  # pad if no wrist
        right_image = np.zeros((224, 224, 3), dtype=np.uint8) # 零图像占位

        inputs = {
            "state": data["observation.state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change
                # "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0,
                "right_wrist_0_rgb": np.False_,
            }
        }
        if "action" in data:
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :6]}