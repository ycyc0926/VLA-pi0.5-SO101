# import numpy as np
# from openpi.policies import policy_config as _policy_config
# from openpi.shared import download
# from openpi.training import config as _config

# config = _config.get_config("pi0_libero")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")

# # Create a trained policy.
# policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# # 创建符合 Libero 格式的完整示例
# def create_libero_example():
#     """创建一个包含所有必要字段的 Libero 格式示例。"""
#     return {
#         # 图像: 224x224 RGB 图像 (0-255 uint8)
#         "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
#         "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        
#         # 状态: 机器人状态向量 (Libero 通常是 8维: 关节角度 + 夹爪状态)
#         "observation/state": np.random.randn(8).astype(np.float32),
        
#         # 语言指令
#         "prompt": "pick up the fork",
        
#         # 批次掩码 (用于变长批次)
#         "pad_mask": np.array([True], dtype=bool),
#     }

# example = create_libero_example()

# # 打印示例结构以便调试（修复了字符串的 dtype 问题）
# print("Example keys:", example.keys())
# for key, value in example.items():
#     if hasattr(value, 'shape') and hasattr(value, 'dtype'):
#         # 这是 numpy 数组
#         print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
#     else:
#         # 这是其他类型（如字符串、布尔值等）
#         print(f"  {key}: type={type(value).__name__}, value={value}")

# # 运行推理
# result = policy.infer(example)

# print("\nActions shape:", result["actions"].shape)
# print("Actions sample:", result["actions"][0, :10])  # 打印前10个动作值



import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
"""
config = _config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = droid_policy.make_droid_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)
"""

config = _config.get_config("pi0_aloha_sim")

checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_aloha_sim")
key = jax.random.key(0)

# Create a model from the checkpoint.
model = config.model.load(_model.restore_params(checkpoint_dir / "params"))

# We can create fake observations and actions to test the model.
obs, act = config.model.fake_obs(), config.model.fake_act()

# Sample actions from the model.
loss = model.compute_loss(key, obs, act)
print("Loss shape:", loss.shape)

# # Reduce the batch size to reduce memory usage.
# config = dataclasses.replace(config, batch_size=2)

# # Load a single batch of data. This is the same data that will be used during training.
# # NOTE: In order to make this example self-contained, we are skipping the normalization step
# # since it requires the normalization statistics to be generated using `compute_norm_stats`.
# loader = _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=True)
# obs, act = next(iter(loader))

# # Sample actions from the model.
# loss = model.compute_loss(key, obs, act)

# # Delete the model to free up memory.
# del model

# print("Loss shape:", loss.shape)