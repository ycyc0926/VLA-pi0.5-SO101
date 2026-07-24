import numpy as np
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# 根据你刚才启动服务的日志，IP 是 192.168.1.110
HOST = "192.168.1.110" 
PORT = 5000

# 创建一个指定大小的图像（默认大小为 224x224 像素），并返回一个变量 x
def mk_img(sz=224):
    x = np.zeros((480, 640, 3), dtype=np.uint8)
    x = image_tools.resize_with_pad(x, sz, sz)
    x = image_tools.convert_to_uint8(x)
    return x

# 初始化客户端
cli = WebsocketClientPolicy(host=HOST, port=PORT)
print("Connected. Server metadata:", cli.get_server_metadata())

# 构造观察值字典 (Observation)
obs = {
    "observation/image": mk_img(224),
    "observation/wrist_image": mk_img(224),
    # "observation/right_image": mk_img(224),
    "observation/state": np.zeros((6,), dtype=np.float32),  # ★ 长度 6、float32
    "prompt": "Grab the red ball and place it in the white cup",  
}

# 执行推理
resp = cli.infer(obs)

# 打印结果
print("SUCCESS. Keys:", resp.keys())
if "actions" in resp:
    a = resp["actions"]
    print("actions shape:", getattr(a, "shape", None), "dtype:", getattr(a, "dtype", None))