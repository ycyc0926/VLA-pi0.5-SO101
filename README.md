# PI0.5 + SO101：从数据采集到真机部署的 VLA 项目

基于 OpenPI、LeRobot v3 和 SO101 机械臂完成的端到端 VLA 工程实践，覆盖遥操作采集、数据校准与融合、PI0.5 LoRA 微调、checkpoint 管理、局域网策略服务和真实机械臂闭环执行。

本项目用于 VLA/具身智能实习求职展示。完整工程记录、简历表述、面试介绍和面试题库见 [`DOCS/solution.md`](DOCS/solution.md)。

> 当前状态：训练和真机部署闭环已经完成；尚未进行随机初始条件下的标准化多轮成功率统计，因此不宣称未经验证的稳定成功率。

## 项目主线

项目包含两个阶段。

### 阶段一：抓取物块放入杯子

- 使用同事已经采集的 50 个 episode、31,000 帧双相机数据。
- 在 AutoDL 上完成 PI0.5 LoRA 微调和推理服务部署。
- 发现公网单轮推理约 1～3 秒，超过 10-step action chunk 在 30 Hz 下约 333 ms 的有效时间窗。
- 将训练和推理迁移到公司局域网 GPU 服务器后，固定场景真机推理成功。
- 稍微改变物块或杯子位置后任务明显失败，说明模型主要记住了训练轨迹和场景捷径，没有获得可靠的空间泛化。

### 阶段二：单/双物块长程抽屉任务

同事已有单物块任务：打开抽屉、放入一个物块、关闭抽屉。在此基础上，自行设计并采集了更长程的双物块任务：

```text
打开抽屉 → 抓取黑色物块并放入 → 抓取白色物块并放入 → 关闭抽屉
```

采集了 30 个 episode、40,355 帧双物块数据，并在相机可见、机械臂可达范围内随机摆放黑白物块。随后完成：

- 对齐不同采集者的 SO101 follower calibration。
- 同时转换 `observation.state` 和 `action`。
- 合并 32 条单物块与 30 条双物块数据，共 62 episodes、73,833 帧和 2 条任务指令。
- 构建 mixed sample-start filter，并按真实训练语义重新计算 normalization statistics。
- 从已通过真机验证的单物块 checkpoint `35000/params` 参数 warm-start。
- 使用 fresh optimizer 完成新的 50,000-step LoRA 训练，保存 checkpoint `49999`。
- 在真实 SO101 上通过不同语言指令执行单物块和双物块任务。

双物块长程任务已经完整执行到关闭抽屉。关闭阶段有时需要多次重新定位后才能成功，说明策略呈现一定的视觉闭环反馈和恢复行为，但首次关闭成功率和整体稳定性仍需提高。

## 本人负责内容

- AutoDL 与公司 GPU 服务器的训练、推理环境部署。
- SO101 leader/follower 遥操作、双相机接入和双物块数据采集。
- 物块位置随机化与数据完整性检查。
- 跨 calibration 的 state/action 坐标转换。
- LeRobot v3 单/双物块数据合并、task prompt 保留和样本过滤。
- 混合数据 normalization statistics 重算。
- PI0.5 LoRA warm-start、训练恢复、Orbax checkpoint 和 W&B 监控。
- WebSocket 远程策略服务、action chunk 延迟对齐、stale action 拒绝、动作限幅和平滑、安全 Hold。
- 公网延迟、GPU 卡死、相机黑屏/顺序、串口权限、舵机扭矩和动作幅度等问题排查。

## 关键训练配置

| 参数 | 数值 |
| --- | --- |
| Model | PI0.5，PaliGemma 2B LoRA + Action Expert 300M LoRA |
| Config | `pi05_so101_drawer_one_two_blocks_calib2_lora` |
| Warm-start | 单物块 checkpoint `35000/params` |
| 新实验 | 50,000 steps，最终 checkpoint `49999` |
| Batch size | 16 |
| Action horizon | 10 |
| Learning rate | 1,000-step warmup，peak `2e-5`，cosine decay 到 `1e-7` |
| Action 表示 | 前 5 轴 delta，gripper absolute |
| Mixed data | 62 episodes、73,833 帧、71,876 个有效 sample starts |

“从 35k 继续训练”指参数 warm-start，而不是恢复旧 optimizer 后从 step 35000 接着计数。由于任务 mixture、filter 和 norm 已改变，新实验使用 fresh optimizer 并从 step 0 训练 50,000 steps。按模型承接的训练量可称为约 8 万步量级，严格 checkpoint 口径是“旧 35k 参数 + 新实验 50k steps”。

## 真机系统与性能

```text
SO101 leader ──遥操作──> SO101 follower ──采集──> LeRobot v3
                                                   │
                         校准 / 合并 / norm / LoRA
                                                   │
                                                   v
本地 Ubuntu + 双 USB 相机 + SO101 follower
                    │
                    │ 局域网 WebSocket
                    v
公司 GPU 服务器 + PI0.5 LoRA Policy Server
```

一次 90 秒双物块运行的记录：

| 指标 | 结果 |
| --- | ---: |
| 服务端推理中位延迟 | 约 65.9 ms |
| 端到端 RTT 中位数 | 约 89.9 ms |
| 动作结果年龄中位数 | 约 131.4 ms |
| 接受动作块 | 776 |
| stale 拒绝 | 1 |
| Body missed ticks | 0 |

## 关键结论

1. 固定初始状态容易让 VLA 学到视觉捷径和轨迹记忆；数据多样性比单纯增加训练步数更重要。
2. 同一台 follower 的不同采集批次也可能因为 calibration 不同而数值不一致，合并前必须同时对齐 state/action。
3. 新任务 mixture 需要重新核对 prompt、filter、action 语义和 norm；warm-start 与 resume 不能混用。
4. 网络可连接不代表实时闭环可用，RTT、图像年龄和 action horizon 必须共同考虑。
5. 关闭失败后最终完成可以作为闭环反馈的现象证据，但不能夸大为模型理解失败或已经获得稳定自纠错能力。

## 尚未实施的改进

- 随机化抽屉位置、开合距离和推入接触点。
- 补采首次推偏、未完全关严等失败状态下的恢复示范。
- 清除 episode 开头的黑屏和曝光异常帧。
- 在固定时限和随机初始条件下进行 20～30 次标准化评测。
- 在更丰富的行为克隆数据基础上尝试 RL 后训练。

以上均为后续方案，不属于当前已完成结果。

## 仓库导航

```text
.
├── README.md
├── DOCS/
│   ├── solution.md                   # 完整工程记录、简历材料和面试题库
│   ├── solution_local.md             # 本地/局域网真机部署记录
│   ├── blacknew_dataset_guide_zh.md
│   ├── openpi_config_guide_zh.md
│   ├── openpi_dataloader_transforms_guide_zh.md
│   ├── openpi_pi0_training_inference_guide_zh.md
│   └── openpi_gemma_remote_inference_so101_guide_zh.md
├── drawer_one_two_blocks/
│   ├── solution.md                   # 单/双物块混合训练复现记录
│   ├── artifacts/
│   └── openpi/                       # 本任务新增训练代码和 norm stats
├── custom_vla/                       # 实际修改的 OpenPI/LeRobot 代码快照
├── openpi/ 与 lerobot/               # 上游源码阅读快照
└── tools/                            # 数据检查工具
```

推荐阅读：

1. [`DOCS/solution.md`](DOCS/solution.md)：项目全流程，以及简历和面试准备。
2. [`drawer_one_two_blocks/solution.md`](drawer_one_two_blocks/solution.md)：校准、合并、norm、训练与恢复细节。
3. [`DOCS/solution_local.md`](DOCS/solution_local.md)：局域网推理、SO101 接入和真机排错。

## 关键增量代码

```text
drawer_one_two_blocks/openpi/src/openpi/training/config.py
drawer_one_two_blocks/openpi/src/openpi/training/data_loader.py
drawer_one_two_blocks/openpi/scripts/convert_so101_calibration.py
drawer_one_two_blocks/openpi/scripts/build_mixed_lerobot_filter.py
drawer_one_two_blocks/openpi/scripts/compute_so101_norm_stats_fast.py
drawer_one_two_blocks/openpi/scripts/upload_openpi_training_log_to_wandb.py
```

训练数据、视频、模型权重、checkpoint、optimizer state、本地日志、校准原文件和 API 凭据不进入公开仓库。

## 致谢

- [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- [ROS-LiKunwei/VLA](https://github.com/ROS-LiKunwei/VLA)
