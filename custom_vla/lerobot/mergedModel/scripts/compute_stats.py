from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 注意：请根据你的实际路径修改 repo_id 和 root
# 按照你之前的路径，应该是：
# repo_id = "grasp_place_ball"
# root = "/home/likunwei/lerobot/dataset"

ds = LeRobotDataset(repo_id="grasp_place_ball", root="/home/likunwei/lerobot/dataset/grasp_place_ball")

# 获取统计信息中的第一个特征键名
any_feat = next(iter(ds.meta.stats.keys()))

# 检查该特征的统计项中是否有以 "q" 开头的键（即分位数）
has_quantiles = any(k.startswith("q") for k in ds.meta.stats[any_feat].keys())

print(f"has_quantiles: {has_quantiles}")