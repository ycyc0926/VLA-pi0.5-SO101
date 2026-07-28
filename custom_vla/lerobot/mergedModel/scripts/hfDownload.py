import os
from huggingface_hub import hf_hub_download

# 配置路径
REPO_ID = "google/paligemma-3b-pt-224"
SAVE_DIR = "/home/likunwei/lerobot/mergedModel/grabPlaceBall"
# 不在源码中保存访问令牌；如需私有仓库权限，请先 export HF_TOKEN。
HF_TOKEN = os.environ.get("HF_TOKEN")

# 需要补齐的 4 个关键配置文件
files_to_download = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json"
]

print(f"🚀 开始从 Hugging Face 补全配置文件至: {SAVE_DIR}")

for file in files_to_download:
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            local_dir=SAVE_DIR,
            token=HF_TOKEN,
            local_dir_use_symlinks=False
        )
        print(f"✅ 已成功下载: {file}")
    except Exception as e:
        print(f"❌ 下载 {file} 失败: {e}")

print("\n🎉 补全完成！现在你的模型文件夹已经是一个完美的“独立包”了。")