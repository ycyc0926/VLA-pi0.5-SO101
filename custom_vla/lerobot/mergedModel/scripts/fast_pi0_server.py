import torch
from flask import Flask, request, jsonify
import logging
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

app = Flask(__name__)

# 1. 定义运行设备 (CPU 或 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [Server] 使用设备: {device}")

# 2. 加载模型
MODEL_PATH = "/home/likunwei/lerobot/mergedModel/grabPlaceBall"
print(f"📦 [Server] 正在从 {MODEL_PATH} 加载模型...")

try:
    policy = PI0Policy.from_pretrained(MODEL_PATH)
    policy.to(device)
    policy.eval()
    print("✅ [Server] 模型就绪，开始监听请求...")
except Exception as e:
    print(f"❌ [Server] 模型加载失败: {e}")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # 1. 基础数据转换
        img_env = torch.tensor(data['img_env'], dtype=torch.float32).unsqueeze(0).to(device)
        img_wrist = torch.tensor(data['img_wrist'], dtype=torch.float32).unsqueeze(0).to(device)
        state = torch.tensor(data['state'], dtype=torch.float32).unsqueeze(0).to(device)

        # 2. 【核心修复】构造缺失的 Language Tokens
        # PI0 默认通常需要一个长度为 1 或 64 的 token 序列
        # 我们这里生成一个全零的占位符 (1, 1)，类型必须是 long
        tokens = torch.zeros((1, 1), dtype=torch.long).to(device)

        # 3. 构造完整 Batch
        batch = {
            "observation.images.images_env": img_env,
            "observation.images.images_wrist": img_wrist,
            "observation.state": state,
            "observation.language.tokens": tokens,  # <--- 补上缺失的字段
        }

        # 4. 推理
        with torch.no_grad():
            output = policy.predict_action_chunk(batch)
            
            if isinstance(output, dict):
                action_chunk = output["action"]
            else:
                action_chunk = output

        # 5. 返回
        actions = action_chunk.squeeze(0).cpu().numpy().tolist()
        return jsonify({"actions": actions})

    except Exception as e:
        # 如果依然报同样的错，可能是因为 PI0 需要的是 (1, 64) 维度的 tokens
        # 可以在下面尝试增加维度
        app.logger.error(f"❌ 推理异常: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    # 允许局域网访问
    app.run(host='0.0.0.0', port=8080, debug=False)