import sys
import os
import shutil
from safetensors.torch import load_file, save_file

# 1. 核心修正：指向 src 目录，这样 Python 才能找到 lerobot 模块
sys.path.append("/home/likunwei/lerobot/src")

import torch
from peft import PeftModel
# 这里的 import 路径根据 LeRobot 源码结构可能需要微调
# 如果报错，请尝试 from lerobot.policies.pi0.modeling_pi0 import PI0Policy
try:
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
except ImportError:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# 路径定义
BASE_MODEL_DIR = "/home/likunwei/.cache/openpi/openpi-assets/checkpoints/pi0_libero_lerobot"
LORA_DIR = "/home/likunwei/lerobot/outputs/train/2026-03-25/10-39-46_pi0/checkpoints/100000/pretrained_model"
SAVE_DIR = "/home/likunwei/lerobot/mergedModel/grabPlaceBall"

def final_precise_merge():
    # 强制创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"📁 已创建保存目录: {SAVE_DIR}")
    
    device = "cpu"
    
    print("1. 读取底座原始权重...")
    base_tensors = load_file(os.path.join(BASE_MODEL_DIR, "model.safetensors"), device=device)
    
    print("2. 读取 LoRA 权重...")
    lora_tensors = load_file(os.path.join(LORA_DIR, "adapter_model.safetensors"), device=device)
    
    scaling = 1.0 
    merged_count = 0

    print("3. 执行精确路径对齐合并...")
    for k_lora_b, B in lora_tensors.items():
        if "lora_B" not in k_lora_b: continue
        k_lora_a = k_lora_b.replace("lora_B", "lora_A")
        
        # 精准匹配底座 Key
        matched_base_key = k_lora_b.replace("base_model.model.model.", "model.").replace(".lora_B.weight", ".weight")
        
        if matched_base_key in base_tensors:
            A = lora_tensors[k_lora_a]
            delta_w = (B @ A) * scaling
            
            # 自动处理转置
            if base_tensors[matched_base_key].shape != delta_w.shape:
                delta_w = delta_w.T
            
            base_tensors[matched_base_key] += delta_w
            merged_count += 1
        else:
            # 兼容另一种可能的路径格式
            alt_key = matched_base_key.replace("model.", "", 1)
            if alt_key in base_tensors:
                A = lora_tensors[k_lora_a]
                delta_w = (B @ A) * scaling
                if base_tensors[alt_key].shape != delta_w.shape: delta_w = delta_w.T
                base_tensors[alt_key] += delta_w
                merged_count += 1

    print(f"✅ 成功精确合并 {merged_count} 层权重！")

    # --- 注入 Stats ---
    print("4. 注入动作统计量 (解决抓取精度问题)...")
    lora_bin = os.path.join(LORA_DIR, "pytorch_model.bin")
    if os.path.exists(lora_bin):
        extra_data = torch.load(lora_bin, map_location="cpu")
        for k, v in extra_data.items():
            if "normalize" in k:
                # 统一前缀为 model.
                clean_k = "model." + k.split("model.")[-1]
                base_tensors[clean_k] = v
                print(f"   - 注入统计量: {clean_k}")

    print(f"5. 正在写入: {os.path.join(SAVE_DIR, 'model.safetensors')} ...")
    save_file(base_tensors, os.path.join(SAVE_DIR, "model.safetensors"))

    print("6. 正在同步配置文件...")
    for f in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        src = os.path.join(LORA_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SAVE_DIR, f))
            print(f"✅ 已同步: {f}")

    print(f"\n🎉 合并大功告成！")
    print(f"🚀 模型路径: {SAVE_DIR}")
    print("现在请部署到 SO101 机械臂，抓取红球的动作应该会变得非常精准了！")

if __name__ == "__main__":
    final_precise_merge()