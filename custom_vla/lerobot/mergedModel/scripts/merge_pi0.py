# merge_pi0.py
import torch
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# 你的训练产出目录
ckpt_path = "/home/likunwei/lerobot/outputs/train/2026-03-25/10-39-46_pi0/checkpoints/last/pretrained_model"
# 合并后存放的新位置
save_path = "/home/likunwei/lerobot/pi0_merged_final"

print(f"正在加载并合并 LoRA 权重，请稍候...")

# LeRobot 的 from_pretrained 会自动识别并加载适配器
policy = PI0Policy.from_pretrained(ckpt_path)

# 确保权重被合并并切换为评估模式
policy.to("cuda")
policy.eval()

# 保存为单体模型文件夹
policy.save_pretrained(save_path)

print(f"✅ 合并成功！完整模型已保存至: {save_path}")
print("现在该目录下应该已经有了 model.safetensors 文件。")