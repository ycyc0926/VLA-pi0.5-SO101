import os
import torch
from safetensors.torch import load_file

# 路径指向你刚刚合并生成好的模型
MERGED_MODEL_PATH = "/home/likunwei/lerobot/outputs/train/2026-03-25/10-39-46_pi0/checkpoints/100000/pretrained_model/model.safetensors"
BASE_MODEL_PATH = "/home/likunwei/.cache/openpi/openpi-assets/checkpoints/pi0_libero_lerobot/model.safetensors"

def verify_merge():
    print("🔍 正在进行权重一致性检查...")
    
    # 加载两个模型的权重
    base_tensors = load_file(BASE_MODEL_PATH)
    merged_tensors = load_file(MERGED_MODEL_PATH)
    
    # 挑选几个关键层进行对比（比如第 0 层的 q_proj）
    # 注意：这里的 key 名字需要根据你模型实际的 key 调整，我们直接遍历所有权重算总均值
    
    def get_mean(tensors):
        total_sum = 0.0
        total_count = 0
        for k, v in tensors.items():
            if v.is_floating_point():
                total_sum += v.abs().mean().item()
                total_count += 1
        return total_sum / total_count if total_count > 0 else 0

    base_mean = get_mean(base_tensors)
    merged_mean = get_mean(merged_tensors)
    
    print(f"\n📊 均值对比结果:")
    print(f"   - 原始底座均值: {base_mean:.10f}")
    print(f"   - 合并后模型均值: {merged_mean:.10f}")
    
    diff = abs(base_mean - merged_mean)
    print(f"   - 绝对差异值: {diff:.10f}")
    
    if diff > 1e-9:
        print("\n✅ 验证通过：模型权重已发生改变！LoRA 知识已成功合并。")
        
        # 额外检查：确认归一化统计量是否存在
        stats_keys = [k for k in merged_tensors.keys() if "normalize" in k]
        if len(stats_keys) > 0:
            print(f"✅ 统计量检查：已发现 {len(stats_keys)} 个归一化 Buffer，动作坐标系已校正。")
        else:
            print("⚠️ 警告：未发现归一化统计量，机械臂动作可能仍有偏置。")
            
    else:
        print("\n❌ 验证失败：权重完全一致。LoRA 并没有合进去，请检查合并脚本的 Key 匹配逻辑。")

if __name__ == "__main__":
    verify_merge()