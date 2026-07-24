import os
import shutil
import torch
import sys

# 1. 核心修正：指向 src 目录，这样 Python 才能找到 lerobot 模块
sys.path.append("/home/likunwei/lerobot/src")

from peft import PeftModel
try:
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
except ImportError:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# ========= 配置区 =========
BASE_MODEL = "/home/likunwei/.cache/openpi/openpi-assets/checkpoints/pi0_libero_lerobot"
LORA_PATH = "/home/likunwei/lerobot/outputs/train/2026-03-25/10-39-46_pi0/checkpoints/100000/pretrained_model"
SAVE_PATH = "/home/likunwei/lerobot/mergedModel/grabPlaceBall_debug"
# =========================


def merge_lora():
    print("\n[1] 加载 base model...")
    base = PI0Policy.from_pretrained(BASE_MODEL)

    base_mean = next(base.parameters()).abs().mean().item()
    print(f"Base weight mean: {base_mean:.6f}")

    print("[2] 加载 LoRA...")
    model = PeftModel.from_pretrained(base, LORA_PATH)

    print("[3] merge LoRA...")
    model = model.merge_and_unload()

    merged_mean = next(model.parameters()).abs().mean().item()
    print(f"Merged weight mean: {merged_mean:.6f}")

    if abs(base_mean - merged_mean) < 1e-6:
        print("❌ WARNING: LoRA 很可能没生效")
    else:
        print("✅ LoRA 已生效")

    print("[4] 保存模型...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)

    return model


def copy_processor():
    print("\n[5] 复制 processor...")
    for f in ["policy_preprocessor.json", "policy_postprocessor.json"]:
        src = os.path.join(LORA_PATH, f)
        dst = os.path.join(SAVE_PATH, f)

        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"✅ 已复制 {f}")
        else:
            print(f"❌ 缺少 {f}（严重问题）")


def check_files():
    print("\n[6] 检查目录结构...")
    required = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json"
    ]

    for f in required:
        path = os.path.join(SAVE_PATH, f)
        if os.path.exists(path):
            print(f"✅ {f}")
        else:
            print(f"❌ 缺失 {f}")


def dry_run_with_policy():
    print("\n[7] Dry-run（正确方式：走 policy）...")

    policy = PI0Policy.from_pretrained(SAVE_PATH)
    policy.eval()

    fake_obs = {
        "observation": {
            "images": {
                "images_env": torch.zeros(3, 480, 640),
                "images_wrist": torch.zeros(3, 480, 640),
            },
            "state": torch.zeros(6),
        },
        "task": "pick up the ball"
    }

    try:
        with torch.no_grad():
            action = policy.act(fake_obs)

        print("✅ policy.act 成功")
        print("action shape:", action.shape)

    except Exception as e:
        print("❌ policy.act 失败:", e)


def main():
    model = merge_lora()
    copy_processor()
    check_files()
    dry_run(model)

    print("\n=== DONE ===")
    print(f"输出目录: {SAVE_PATH}")


if __name__ == "__main__":
    main()