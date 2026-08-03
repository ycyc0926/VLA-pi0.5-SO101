# PI0.5 部署到 SO101 的本地推理记录

## 0. 项目主线：从轨迹记忆到长程闭环操作

### 0.1 一句话概述

在 SO101 平台上完成 PI0.5 从数据处理、LoRA 微调、跨数据集 warm-start 到局域网实机部署的完整闭环：先复现“抓取物块放入杯子”任务并定位其泛化失败原因，再自采随机化双物块数据，将同事的单物块抽屉策略扩展为“开抽屉--依次放入黑白物块--关抽屉”的长程语言条件任务。

### 0.2 第一阶段：物块放入杯子任务

#### 数据与训练来源

第一阶段使用同事已经采集好的“抓取物块放入杯子”数据。本人负责将数据和 PI0.5 训练流程部署到 AutoDL，完成 LoRA 微调、checkpoint 服务和 SO101 本地客户端接入。

任务指令示例：

```text
Grab the black cube and place it in the white cup
```

#### AutoDL 公网推理问题

AutoDL 上的模型能够正常加载，但公网 TCP 映射下单次推理结果约需要 1--3 秒。当前 checkpoint 每次只返回 10 步动作，在 30 Hz 控制频率下 action horizon 约为：

```text
10 / 30 = 0.333 秒
```

公网返回的动作远远超过 333 ms 有效窗口，客户端只能拒绝 stale action，或者冒险执行已经过时的动作。因此问题不在模型是否成功加载，而在公网链路无法满足短 action chunk 的实时闭环要求。

#### 迁移到公司局域网服务器

随后将训练和推理迁移到公司双 RTX 5090 服务器，通过局域网连接本地 SO101。局域网实测端到端 RTT 约 90 ms，服务端推理约 66 ms，模型最终能够在固定场景中完成物块放入杯子的任务。

#### 泛化失败与结论

虽然固定场景推理成功，但只要稍微改变物块或杯子的位置，成功率就会明显下降，机械臂仍倾向执行接近训练示范的一条固定轨迹。这说明当时的数据覆盖不足，策略更接近记忆视觉条件下的轨迹，而没有形成可靠的目标定位和空间泛化能力。

该阶段得到的核心结论是：

1. 实机成功一次不等于模型学会任务。
2. 训练数据如果初始状态过于固定，VLA 很容易学习场景捷径和轨迹记忆。
3. 数据采集必须主动随机化物块、目标容器和机械臂初始状态，而不是每次严格复现同一摆放。
4. 泛化能力必须通过改变初始条件后的多轮测试验证，不能只展示训练分布内的一次成功视频。

### 0.3 第二阶段：长程双物块抽屉任务

#### 任务来源与扩展

同事已有一个基础任务：打开抽屉，将一个物块放入抽屉，再关闭抽屉。本人在此基础上设计并采集了更长程的双物块任务：

```text
Open the drawer, put the black block into the drawer, then put the white block into the drawer, and close the drawer.
```

完整子任务顺序为：

1. 拉开抽屉。
2. 抓取桌面上的黑色物块并放入抽屉。
3. 抓取桌面上的白色物块并放入抽屉。
4. 关闭抽屉。

该任务比单物块任务具有更长的时间跨度、更严格的任务顺序和更多接触操作，能够检验策略的语言条件控制、长程动作衔接和视觉闭环能力。

#### 数据归属与本人工作

| 数据/模块 | 来源 | 本人负责内容 |
| --- | --- | --- |
| 物块放杯子数据 | 同事已有数据 | AutoDL/公司服务器 LoRA 训练、服务部署、实机推理和泛化分析 |
| 单物块抽屉数据与 35k checkpoint | 同事已有任务 | 校准坐标核对、数据兼容、作为继续训练起点 |
| 双物块抽屉数据 | 本人采集 | 任务设计、30 条 episode 遥操作采集、物块位置随机化、数据检查 |
| 融合训练 | 本人完成 | 校准对齐、数据合并、归一化统计、从 35000 checkpoint warm-start |
| 实机部署 | 本人完成 | 双相机对齐、局域网推理客户端、动作队列、stale 拒绝、安全限幅和任务验证 |

#### 数据随机化设计

双物块数据采集时，没有将黑色和白色物块始终放在完全相同的位置，而是在机械臂可达范围和相机视野内进行随机摆放。这样做的目标是让模型学习：

- 根据视觉定位不同颜色物块，而不是复现固定关节轨迹。
- 在不同抓取起点下仍保持“先黑后白”的语言任务顺序。
- 根据当前抽屉和物块状态持续产生后续动作。

当前双物块数据集包含 30 条 LeRobot v3 episode、40,355 帧、两路 RGB 图像、六维关节状态和六维动作，采集频率为 30 FPS。

#### 跨数据集继续训练

为了保留同事单物块策略已经学到的开关抽屉能力，同时扩展双物块任务，训练流程不是从基础模型完全重新开始，而是：

1. 对齐主从机械臂校准文件，确认两个数据集的关节含义、顺序和数值空间一致。
2. 检查两路相机键、图像尺寸、FPS、state/action 六维特征和语言任务字段。
3. 合并单物块和双物块数据，同时保留不同 task prompt。
4. 对融合后的数据重新计算 normalization statistics，避免沿用单一数据集统计量造成尺度偏差。
5. 从同事单物块任务的 `35000` checkpoint warm-start，继续进行 LoRA 训练。
6. 使用同一服务端 checkpoint，根据客户端 prompt 分别触发单物块和双物块任务。

这部分工作的重点不只是“接着训练”，而是保证 calibration、feature schema、normalization 和 prompt conditioning 在两个数据集之间一致，否则 checkpoint warm-start 很容易产生动作空间错位或任务混淆。

### 0.4 最终系统架构与性能

```text
SO101 leader --遥操作--> SO101 follower --采集--> LeRobot v3 数据集
                                           |
                                           v
                       单物块数据 + 双物块数据
                                           |
                      对齐校准/合并/归一化/35k warm-start
                                           |
                                           v
本地 Ubuntu + 双 USB 相机 + SO101 follower
                  |
                  | 局域网 WebSocket
                  v
公司 RTX 5090 服务器 + PI0.5 LoRA checkpoint
```

本地客户端负责双相机采集、关节状态读取和 30 Hz Body 控制；服务器负责 PI0.5 推理。客户端实现独立相机线程、Brain/Body 解耦、10-step action chunk 队列、自适应延迟对齐、stale action 拒绝、动作平滑、关节限幅和安全 Hold。

一次 90 秒双物块运行的实测数据：

| 指标 | 结果 |
| --- | --- |
| 服务端推理中位延迟 | 约 65.9 ms |
| 端到端 RTT 中位数 | 约 89.9 ms |
| 动作结果年龄中位数 | 约 131.4 ms |
| 接受动作块 | 776 |
| stale 拒绝 | 1 |
| Body missed ticks | 0 |

### 0.5 实机结果与合理解释

融合训练后的同一个 checkpoint 基本可以通过不同语言指令完成：

- 单物块任务：打开抽屉，放入一个物块并关闭抽屉。
- 双物块任务：打开抽屉，依次放入黑色和白色物块并关闭抽屉。

双物块任务已经完成端到端实机执行。关闭抽屉阶段偶尔会先推到抽屉上方，经过数次位置调整后才最终关闭。

这一现象提供了两个方面的信息：

1. **积极证据**：物块初始位置存在随机变化，策略仍能完成抓取；关闭失败后又能基于新图像继续调整。这些现象说明策略并非只做一次固定开环轨迹回放，并表现出一定的闭环视觉反馈和恢复能力。
2. **当前不足**：多次重试降低了一次性成功率和执行效率，说明关闭抽屉这一接触密集子任务的数据覆盖、视觉定位或控制精度仍然不足。

面试时不应将其描述为“模型理解失败并自主思考”。更准确的表述是：

> 策略根据连续视觉观测重新规划后续 action chunk，在首次接触失败后表现出一定的闭环恢复行为；但该能力仍需要通过固定时限和多轮统计验证。

### 0.6 下一轮数据改进

针对关闭抽屉反复尝试的问题，下一轮采集应重点随机化：

1. 抽屉相对机械臂的左右位置和开合距离。
2. 夹爪接触抽屉正面的高度、横向位置和推入方向。
3. 机械臂完成第二次放置后的过渡姿态。
4. 黑色、白色物块的初始位置和相互关系。
5. 首次推偏、接触抽屉上缘后的恢复示范。

同时应清理 episode 开头的相机黑屏帧。当前 checkpoint 可能把启动黑屏当作动作触发特征，这属于数据捷径，不应作为正式模型能力保留。

### 0.7 正式评测方案

为了证明模型学到任务而不是单条轨迹，建议至少进行 20--30 次固定时限测试，并将初始条件划分为三类：

- 训练分布内位置：与采集范围接近。
- 插值位置：位于训练样本之间但没有完全出现过。
- 边界位置：接近可达范围和相机视野边缘。

每次记录：

- 开抽屉成功率。
- 黑块抓取/放置成功率。
- 白块抓取/放置成功率。
- 首次关抽屉成功率。
- 恢复后关抽屉成功率。
- 完整任务成功率。
- 平均完成时间、关闭重试次数和人工干预率。

只有在随机初始条件下获得稳定结果，才能在简历中写“实现稳定自主操作”。当前可以准确写成“完成长程任务实机闭环验证，并观察到失败后的视觉反馈恢复行为”。

### 0.8 可直接用于简历的版本

**项目名称：PI0.5 VLA 在 SO101 上的长程双物块抽屉操作**

- 基于 SO101 和 PI0.5 LoRA 构建从遥操作采集、LeRobot v3 数据处理、跨任务继续训练到局域网实机部署的完整 VLA 流程；自采 30 条双物块 episode、40,355 帧，并在采集阶段随机化黑白物块位置以减少固定轨迹过拟合。
- 将同事已有的单物块抽屉策略扩展为“开抽屉--依次放入黑白物块--关抽屉”的长程任务；完成机械臂校准对齐、单/双物块数据合并、融合数据 normalization statistics 重算，并从 35k checkpoint warm-start 继续训练。
- 构建本地 SO101/双相机与公司 RTX 5090 服务器的 30 Hz 视觉闭环推理链路，实现 action chunk 队列、自适应延迟对齐、stale action 拒绝、动作限幅和安全 Hold；实测服务端推理中位延迟约 66 ms、端到端 RTT 约 90 ms。
- 实现同一 checkpoint 通过语言指令执行单物块和双物块抽屉任务；双物块任务完成端到端实机验证，并针对关闭抽屉多次重试问题提出抽屉位置、接触点和恢复轨迹随机化的数据改进方案。
- 通过早期物块放杯子任务识别固定场景数据导致的轨迹记忆和泛化失败，完成 AutoDL 公网高延迟、GPU 显存、机械臂校准、相机对齐、Headless OpenCV 和动作执行链路的系统排查。

如果简历篇幅只能保留三条，优先使用前四条中的第 1、2、3 条，并在面试中补充第 4 条的实机结果和不足。

### 0.9 90 秒面试介绍

> 这个项目分成两个阶段。第一阶段我使用同事采集的物块放杯子数据，在 AutoDL 上对 PI0.5 做 LoRA 微调并部署推理。模型能在固定位置完成任务，但 AutoDL 公网延迟达到 1 到 3 秒，而模型的 10-step action horizon 只有约 333 毫秒，所以无法稳定闭环。迁移到公司的 RTX 5090 局域网服务器后，RTT 降到约 90 毫秒并成功推理，但只要改变物块或杯子位置就很容易失败。这让我确认问题不仅是部署，数据初始状态太固定也会让模型记住轨迹。
>
> 第二阶段我在同事已有的单物块抽屉任务上扩展了一个长程任务：打开抽屉，先放黑块，再放白块，最后关抽屉。我自己采集了 30 条双物块数据，并主动随机化两个物块的位置。训练时我对齐了两个任务的机械臂校准和 feature schema，合并数据后重新计算 normalization statistics，再从单物块任务的 35k checkpoint warm-start。
>
> 最后同一个 checkpoint 可以根据不同指令执行单物块和双物块任务。局域网服务端推理中位延迟约 66 毫秒，RTT 约 90 毫秒。双物块任务可以完整完成，关闭抽屉时偶尔需要多次调整，这说明策略有一定闭环恢复表现，但一次性成功率还需要提升。下一步我会随机化抽屉位置和推入接触点，增加推偏后的恢复示范，并用固定时限、多轮实验量化严格成功率和恢复后成功率。

### 0.10 面试追问准备

**为什么从 AutoDL 换到公司服务器？**

不是因为 AutoDL GPU 无法运行模型，而是公网推理延迟大于 action horizon。TCP 映射能建立连接，但不能解决每轮请求 1--3 秒的延迟；局域网将 RTT 降到约 90 ms 后才满足闭环控制要求。

**为什么认为杯子任务是轨迹记忆？**

模型只在训练时的固定物块和杯子位置附近成功，稍微改变相对位置就失败，而且机械臂仍执行相似轨迹。该现象表明数据没有提供足够空间变化，不能证明模型获得目标级泛化。

**为什么采用 35k warm-start，而不是从头训练？**

单物块 checkpoint 已经学到打开抽屉、操作物块和关闭抽屉的基础能力。warm-start 可以保留已有技能，将训练资源集中在双物块任务顺序和更长时间依赖上。但前提是校准坐标、feature schema 和 normalization 兼容。

**为什么要重新计算 normalization statistics？**

融合数据后 state/action 分布发生变化。如果继续使用原单物块数据统计量，输入输出缩放会偏向旧任务，影响训练稳定性和动作反归一化。重新统计可以让融合数据共享一致的数值尺度。

**多次尝试后关闭抽屉能否证明模型会自我纠错？**

它是闭环恢复能力的证据，但不是充分证明。策略每轮都接收新图像和状态，所以能够在接触失败后产生不同动作；也可能存在策略抖动或偶然成功。需要多轮测试、重试次数统计和开环/闭环消融才能形成更强结论。

**这个项目最重要的工程工作是什么？**

不是单纯运行训练脚本，而是打通数据、校准、归一化、checkpoint、网络实时性、双相机输入和机械臂控制之间的接口，并用分层诊断区分模型失败、网络过期动作和硬件执行问题。

## 1. 目标与最终架构

目标是在 Ubuntu 本地电脑连接 SO101 机械臂和两路 USB 相机，将图像和机械臂状态发送到局域网 GPU 服务器上的 PI0.5 策略服务，并接收 6 自由度动作执行抓取任务。

最终可用链路：

```text
本地 Ubuntu + SO101 + 顶视/腕部相机
  -> 局域网 WebSocket
公司 GPU 服务器 + OpenPI PI0.5 checkpoint
```

当前局域网方案已经接入单物块和双物块两个训练任务，并完成双物块抽屉任务的实机闭环验证。一次 90 秒双物块运行中，客户端接受 776 个动作块，端到端 RTT 中位数约 89.9 ms，服务端推理约 65.9 ms，动作结果年龄约 131.4 ms，stale 拒绝 1 次，满足 10-step action horizon 的实时执行要求。

## 2. 本地硬件映射

已确认的稳定设备路径：

| 设备 | 路径/索引 | 用途 |
| --- | --- | --- |
| SO101 follower 串口 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C121031-if00` | 从机械臂控制 |
| SO101 leader 串口 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C121452-if00` | 主机械臂遥操作 |
| 顶视相机 | `/dev/v4l/by-id/usb-DC474C08_P090101_SN0002_1080P_USB_Camera_DC474C08_P090101_SN0002-video-index0` | 工作区视觉，对应数据键 `env` |
| 腕部相机 | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0` | 夹爪视觉，对应数据键 `hand`/推理键 `wrist` |

不要依赖 `/dev/ttyACM*` 和 `/dev/video*` 的临时编号。USB 重插或重启后编号可能改变，机械臂和相机均应使用 `/dev/serial/by-id/...`、`/dev/v4l/by-id/...` 稳定路径。

## 3. 本地环境与权限

### 3.1 Conda 环境

```bash
source ~/miniconda3/bin/activate
conda activate so101_local
cd /home/yc/working_base/VLA/custom_vla
```

`uv` 未安装时，`uv venv ...` 不会创建虚拟环境，因此后续 `source .venv-so101/bin/activate` 会报文件不存在。此次本地执行使用已有的 Conda 环境，不需要依赖 `uv venv`。

### 3.2 串口和相机权限

SO101 串口属于 `dialout` 组，相机属于 `video` 组。执行：

```bash
sudo usermod -aG dialout,video $USER
```

随后必须注销并重新登录，或重启电脑，再确认：

```bash
groups
```

输出中应包含 `dialout` 和 `video`。未重登时，新组权限不会生效。

### 3.3 LeRobot 本地源码缺包

本地 `lerobot` 曾出现：

```text
ModuleNotFoundError: No module named 'lerobot.datasets'
```

原因是当前仓库中的 `lerobot/src/lerobot/datasets` 未被完整检出，但代码已引用 `lerobot.datasets.lerobot_dataset`。修复后应先验证导入：

```bash
python -c "import cv2, numpy; from lerobot.robots.so_follower import SO101Follower; from openpi_client.websocket_client_policy import WebsocketClientPolicy; print('all imports OK', numpy.__version__, cv2.__version__)"
```

## 4. SO101 识别与校准

### 4.1 找到机械臂端口

```bash
lerobot-find-port
```

该工具会要求拔掉 MotorsBus USB 线并回车，通过前后差异识别端口。最终控制命令使用 `/dev/serial/by-id/...` 的稳定符号链接。

### 4.2 校准命令

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C121031-if00 \
  --robot.id=my_awesome_follower_arm \
  --robot.calibration_dir=/home/yc/working_base/VLA/jiaozhun_xunlian
```

校准文件生成位置：

```text
/home/yc/working_base/VLA/jiaozhun_xunlian/my_awesome_follower_arm.json
/home/yc/working_base/VLA/jiaozhun_xunlian/my_awesome_leader_arm.json
```

### 4.3 校准中的人工操作

1. 出现 `Move ... to the middle of its range of motion and press ENTER` 时，将每个关节放在自身机械行程的约中点，不是把末端放到桌面几何中心。
2. 回车后，除 `wrist_roll` 外，逐个关节手动缓慢经过可用行程的两端；不要猛拉、不要撞击机械限位。
3. 再次回车结束记录。输出的 MIN/POS/MAX 应有合理范围，且应显示 calibration saved。

首次错误：

```text
Missing motor IDs: 1 ... 6
```

表示该串口上没有发现六个电机，通常是选错了 `/dev/ttyACM*`、机械臂未通电、数据线/供电有问题，或权限尚未生效；不是“需要先移动机械臂”的报错。

## 5. 相机确认与预览

列出 OpenCV 可访问的相机：

```bash
lerobot-find-cameras opencv
```

已确认 DC474C08 相机为顶视相机，Sonix CAM1 为腕部相机。`v4l2-ctl` 用于查询设备参数，不负责显示预览。实际程序统一使用 `/dev/v4l/by-id/...` 稳定路径。

```bash
ffplay -f video4linux2 -input_format mjpeg -video_size 640x480 -framerate 30 \
  /dev/v4l/by-id/usb-DC474C08_P090101_SN0002_1080P_USB_Camera_DC474C08_P090101_SN0002-video-index0
```

```bash
ffplay -f video4linux2 -input_format mjpeg -video_size 640x480 -framerate 30 \
  /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0
```

首次使用时安装：

```bash
sudo apt update
sudo apt install -y ffmpeg v4l-utils
```

`so101_local` 环境安装的是 `opencv-python-headless 4.13.0.92`，构建信息为 `GUI: NONE`。因此本地推理客户端传入 `--show_camera` 会在 `cv2.imshow()` 报 GTK/HighGUI 未实现并触发安全退出；推理时不要传该参数。遥操作/采集通过 Rerun 显示双路画面，正式推理前则使用 `scene_alignment_check.py` 的 Tkinter 窗口核对参考图和实时画面。

## 6. 云端方案为何不稳定

AutoDL 的公网 HTTP/TCP 映射虽然可以建立 WebSocket，但实测整次推理约 1--3 秒，而当前模型每个 chunk 仅 10 步、控制频率 30 Hz，对应时间窗口约 333 ms。

因此 V4 客户端会正确拒绝过期结果：

```text
Reject stale policy result ... age=1100ms, horizon=333ms
```

V3 客户端可以绕过部分 stale 检查并让机械臂轻微移动，但它会执行明显过时的图像动作，不适合正常抓取部署。公网端口改 TCP 只能改善连接/握手，不足以消除平台代理带来的每次请求延迟。

本地 MX450 显存不足以运行 PI0.5：项目文档要求推理 GPU 显存大于约 8 GB。因此下载权重到本地电脑不能成为可行的 PI0.5 推理方案。

## 7. 公司局域网服务器部署

局域网服务器 IP 使用 `192.168.1.110`。服务器有两张 RTX 5090。早期部署时 GPU 0 被其他 Python 进程占用约 29 GB，加载模型曾出现：

```text
XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory
```

不要终止其他人的进程。早期验证选择空闲 GPU 1 和独立端口 `5001`；当前融合单物块、双物块数据训练后的策略服务使用局域网端口 `5000`，客户端连接 `192.168.1.110:5000`。

早期 `blacknew_43k_may07` 抓取 checkpoint 的历史验证命令：

```bash
cd /home/likunwei/pi0/VLA/openpi
conda activate lerobot

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/likunwei/miniconda3/envs/pi0-zero/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=src:/home/likunwei/lerobot/src

uv run --no-sync scripts/serve_policy.py \
  --port 5001 \
  policy:checkpoint \
  --policy.config pi05_so101_lora \
  --policy.dir /home/likunwei/pi0/stable_checkpoints/blacknew_43k_may07
```

检查 GPU 和端口时，公司服务器没有 `rg`，使用 `grep`：

```bash
nvidia-smi
ss -ltnp | grep ':5001'
ps -ef | grep '[s]erve_policy'
```

注意：`CUDA_VISIBLE_DEVICES=1` 选择物理 GPU 1；程序内部可能仍称它为 `GPU_0`，这是可见设备重编号的正常行为。

当前融合抽屉任务服务使用训练脚本设置的 `CONFIG_NAME`、`CHECKPOINT` 和 `SERVER_LOG`：

```bash
.venv/bin/python scripts/serve_policy.py \
  --port 5000 \
  --default-prompt "Open the drawer, place the requested block or blocks inside, and close the drawer" \
  policy:checkpoint \
  --policy.config "$CONFIG_NAME" \
  --policy.dir "$CHECKPOINT" \
  2>&1 | tee "$SERVER_LOG"
```

## 8. 当前抽屉任务本地推理

当前使用 V4 客户端。它通过独立相机采集线程、Brain 推理线程和 30 Hz Body 控制循环运行，根据图像年龄对 action chunk 做自适应对齐并拒绝过时结果。

双物块训练指令必须与数据集文本一致：

```text
Open the drawer, put the black block into the drawer, then put the white block into the drawer, and close the drawer.
```

当前可用的无限时双物块命令如下；正式量化评测时应将 `--max_run_sec` 改为固定的 90 或 120 秒：

```bash
source ~/miniconda3/bin/activate
conda activate so101_local
cd /home/yc/working_base/VLA/custom_vla

python openpi/packages/openpi-client/src/openpi_client/zpf_pi0_so101_client_adaptive_pro_v4.py \
  --host 192.168.1.110 \
  --port 5000 \
  --serial /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C121031-if00 \
  --robot_id my_awesome_follower_arm \
  --calibration_dir /home/yc/working_base/VLA/jiaozhun_xunlian \
  --use_degrees \
  --cam_top /dev/v4l/by-id/usb-DC474C08_P090101_SN0002_1080P_USB_Camera_DC474C08_P090101_SN0002-video-index0 \
  --cam_wrist /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
  --cam_width 640 \
  --cam_height 480 \
  --cam_fps 30 \
  --cam_fourcc MJPG \
  --camera_warmup_sec 0 \
  --capture_hz 30 \
  --hz 30 \
  --prompt "Open the drawer, put the black block into the drawer, then put the white block into the drawer, and close the drawer." \
  --expected_chunk_size 10 \
  --enqueue_steps 10 \
  --fallback_refill_threshold 4 \
  --min_keep_steps 5 \
  --max_adaptive_skip 5 \
  --stitch_steps 1 \
  --dq_limit_deg 1.0 \
  --max_dq_gripper 5.0 \
  --alpha 0.5 \
  --startup_attempts 10 \
  --startup_infer_timeout 60 \
  --infer_timeout 5 \
  --max_run_sec 0 \
  --log_every_n_steps 10
```

单物块任务使用相同参数，只将 prompt 改为：

```text
Open the drawer, place the block inside, and close the drawer
```

`--max_run_sec 0` 表示持续运行直到手动 `Ctrl+C`，只适合有人看守的演示和调试。模型完成任务、长时间堵转、异响、抖动或碰撞时必须立即停止。

## 9. Checkpoint 与 action horizon 注意事项

当前 `pi05_so101_lora` 配置和 V4 客户端期望 10-step action chunk，日志应显示：

```text
raw=(10, 6)
```

因此可优先测试与 10-horizon 匹配的 checkpoint，例如：

```text
blacknew_10horizon_DeltaPosition_GaussianNoiseDropout/90000
```

不要将 `blacknew_50horizon` 或 `blacknew_50horizon_AbsolutePosition` 直接替换到 `--policy.config pi05_so101_lora` 下。50-horizon checkpoint 需要匹配的训练/服务配置，本地客户端也需要将 `--expected_chunk_size` 改为 50；AbsolutePosition 版本还必须确认动作表示一致。仅修改推理参数或把采样步数调为 50，不能把 10-horizon 模型变成 50-horizon 模型。

## 10. 抓取失败的排查结论

一次正常的局域网日志示例：

```text
accepted=90 | rejected_stale=0 | underrun=0 | missed_ticks=0
RTT_ms(median=93.9) | server_ms(median=78.2)
result_age_ms(median=142.1)
```

这说明网络、推理服务、动作队列和机械臂通信正常。若机械臂向物块移动却上抬而不下抓，优先怀疑策略输入/泛化，而不是延迟：

1. 机械臂起始姿态与训练数据不同。策略会模仿训练轨迹，不是通用几何规划器；应在客户端退出且扭矩释放后，将各关节调整到接近训练采集起始姿态。
2. 相机安装位姿、方向、顺序或画面内容不同。确认 DC474C08 是完整工作区顶视图，Sonix CAM1 是腕部图；不能倒置、镜像或互换。物块、抽屉、桌面背景和相对位置应尽量复现训练场景。
3. 当前 checkpoint 在该摆放下泛化不足。服务正常输出动作不等于模型能完成任务；需要在相同初始条件下对 10-horizon checkpoint 做 A/B 实测。
4. 早期过小的 `dq_limit_deg=0.2`、`alpha=0.1` 会显著压缩动作；当前实测参数已调整为 `dq_limit_deg=1.0`、`alpha=0.5`。即便如此，也不能为了修正错误轨迹继续盲目提高速度限制。

出现上抬异常时，先用 `--max_run_sec 8 --log_every_n_steps 1` 做短时观察。只有确认策略后续会下降接近物块时，才逐步增加运行时间；始终清空机械臂活动范围，并保持可随时断电。

## 11. 可用于简历的表述

**PI0.5--SO101 视觉抓取部署**

- 在 Ubuntu 上完成 SO101 六自由度机械臂、双 USB 相机与 LeRobot/OpenPI 客户端的集成，完成串口权限配置、端口识别、关节标定和稳定设备路径管理。
- 将 PI0.5 LoRA checkpoint 部署至局域网 RTX 5090 推理服务器，通过 WebSocket 将双目视觉与机械臂状态接入 10-step action chunk 闭环控制；双物块实测服务端推理中位延迟约 66 ms、端到端 RTT 约 90 ms。
- 定位并解决公网推理高延迟导致的 stale action、GPU 显存不足、OpenCV headless 预览失败、设备组权限和本地源码依赖缺失等问题，建立可复现的端到端实机推理流程。
- 针对实物抓取失败，完成从网络时延、动作队列、checkpoint horizon、相机外参/画面一致性、机械臂初始姿态到策略泛化能力的分层诊断。

简历中可如实描述为“完成双物块长程任务的端到端实机闭环验证”。在完成固定时限、多轮成功率测试前，不应写成“实现稳定自主抓取”。

## 12. 遥操作、数据采集与训练数据

### 12.1 当前训练校准文件

训练和推理统一使用：

```text
/home/yc/working_base/VLA/jiaozhun_xunlian/my_awesome_follower_arm.json
/home/yc/working_base/VLA/jiaozhun_xunlian/my_awesome_leader_arm.json
```

遥操作已经实测可以驱动从机械臂，因此主从串口、校准映射、舵机供电和 LeRobot 动作发送链路均正常。遥操作时可将两路相机配置为 `env` 和 `hand`，并传入 `--display_data=true`，使用 Rerun 同时查看画面。`lerobot-teleoperate` 只做控制和预览，不会保存数据；正式采集必须使用 `lerobot-record`。

### 12.2 双物块数据集

本地数据集：

```text
/home/yc/working_base/VLA/data/drawer_two_blocks_v1
```

数据集关键信息：

| 项目 | 数值 |
| --- | --- |
| LeRobot 格式 | v3.0 |
| episode 数 | 30 |
| 总帧数 | 40,355 |
| 采集频率 | 30 FPS |
| 平均 episode 时长 | 约 44.8 秒 |
| 图像 | `observation.images.env`、`observation.images.hand` |
| 低维数据 | `observation.state`、`action`，均为 6 维 |

双物块任务文本最终统一为：

```text
Open the drawer, put the black block into the drawer, then put the white block into the drawer, and close the drawer.
```

该提示词远低于 200 token，不存在提示词过长问题。更重要的是训练和推理必须使用完全一致或语义高度一致的任务文本。

采集 episode 时，只记录主机械臂遥操作从机械臂完成任务的过程。一个 episode 完成后进入 reset 时间，再由人工取出物块、恢复抽屉和机械臂初始状态；人工复位过程不应被写入训练 episode。不同 episode 的物块位置允许存在小范围变化，这有利于泛化，但变化不能超出相机可见范围和机械臂可达范围。

### 12.3 融合训练任务

最终 checkpoint 融合了单物块和双物块两个数据集。单物块任务文本为：

```text
Open the drawer, place the block inside, and close the drawer
```

双任务融合使同一个策略服务可以通过 prompt 区分单物块和双物块任务。服务端使用 `pi05_so101_lora` 对应配置和训练 checkpoint，通过 `scripts/serve_policy.py` 在局域网 `5000` 端口提供 WebSocket 策略服务。

## 13. 场景与相机对齐

正式推理前必须先退出遥操作、采集或其他占用相机的进程，再运行：

```text
/home/yc/working_base/VLA/scene_alignment_check.py
```

双物块训练参考图：

```text
/home/yc/working_base/VLA/jiaozhun_xunlian/inference_reference/top_training_reference.jpg
/home/yc/working_base/VLA/jiaozhun_xunlian/inference_reference/wrist_training_reference.jpg
```

单物块训练参考图：

```text
/home/yc/working_base/VLA/jiaozhun_xunlian/tupian_one_task/20260803-124540.png   # 顶视
/home/yc/working_base/VLA/jiaozhun_xunlian/tupian_one_task/20260803-124612.png   # 腕部
```

对齐时检查抽屉位置和方向、物块位置、机械臂初始姿态、相机俯仰角和画面方向。核对完成后按 `Q` 退出，确保两路相机被释放，再启动推理客户端。不要传 `--new-reference`，否则会覆盖训练参考图。

### 13.1 相机启动黑屏问题

客户端原先使用：

```text
--camera_warmup_sec 3
```

这会让采集线程先运行 3 秒，再发送模型输入，等价于跳过相机启动阶段的黑屏/曝光不稳定帧。实测当前 checkpoint 可能把 episode 开头的黑屏学成了任务启动特征，因此当前复现训练分布时改为：

```text
--camera_warmup_sec 0
```

这只是当前 checkpoint 的兼容措施，不是理想的数据设计。后续重新训练应裁掉每个 episode 开头的黑屏帧，避免模型依赖与任务无关的视觉捷径。

## 14. “推理有日志但机械臂不动”的完整诊断

曾出现服务端持续返回 `(10, 6)` 动作、Brain/Body 日志正常，但机械臂肉眼不动。分层诊断结果如下：

1. 六个舵机都能被串口发现，位置、目标、模式和状态寄存器均可读取。
2. `Operating_Mode=0`，位置模式正确；`Max_Torque_Limit`、`Torque_Limit`、PID、加速度和保护电流均正常。
3. `Torque_Enable=1`、`Lock=1` 能成功写入。客户端退出后读到 `Torque_Enable=0` 是 `disable_torque_on_disconnect=true` 的正常安全行为。
4. 安全夹爪微动测试可以完成约 5% 的打开和复位，证明外部供电、舵机动力、校准和底层写入链路正常。
5. 遥操作能够稳定驱动从机械臂，进一步排除机械硬件故障。

最终发现推理测试同时受以下因素影响：

- `--max_run_sec 5` 太短，程序尚未形成明显动作便退出。
- `--dq_limit_deg 0.2` 和 `--alpha 0.1` 叠加后，部分关节每步变化低于舵机可见分辨率。
- `--show_camera` 在 headless OpenCV 上触发 `cv2.imshow()` 异常，客户端只接受两个动作块便安全退出。
- 10-step chunk 在约 130 ms 图像年龄下通常自适应跳过前 4--5 步，需要合理设置队列、拼接和限幅参数。

当前实测可用参数为 `dq_limit_deg=1.0`、`max_dq_gripper=5.0`、`alpha=0.5`、`stitch_steps=1`，并移除 `--show_camera`。推理时主机械臂可以关闭，但从机械臂必须同时连接 USB 和其匹配电压版本的外部电源；不能仅凭舵机 LED 判断动力供电是否正常。

## 15. 实机任务结果与成功标准

融合训练后的策略已经在真实 SO101 上完整执行过双物块任务：打开抽屉、放入黑色物块、放入白色物块并最终关闭抽屉。系统不是单次开环播放轨迹，而是持续根据两路图像和关节状态重新请求动作块。

关闭抽屉仍是主要薄弱环节：夹爪有时先在抽屉上方推，经过多次重新定位后才成功关闭。该结果应定义为“恢复后成功”，不能等同于一次准确完成。

建议采用以下标准：

| 分类 | 标准 |
| --- | --- |
| 严格成功 | 固定时间内完成全部子任务，抽屉完全关闭，无人工干预，关闭阶段没有明显反复 |
| 恢复后成功 | 首次关闭失败，但策略根据后续视觉观测自行调整并在时限内完成 |
| 部分成功 | 两个物块已放入，但抽屉没有完全关闭 |
| 失败 | 超时、需要人工干预、物块掉落、危险碰撞或任务顺序错误 |

`--max_run_sec 0` 的无限运行只用于有人看守的调试和演示，不能用于正式成功率统计。正式评测建议固定为 90 或 120 秒，至少重复 20 次，并记录完整任务成功率、首次关闭成功率、恢复后成功率、平均完成时间、关闭重试次数和人工干预率。

关闭抽屉反复调整可能来自：关闭阶段示范数量不足或轨迹不一致；抽屉位置、相机视角和初始姿态偏差；无力觉条件下的接触状态不确定；约 90 ms RTT 和短 action horizon 对精细接触动作的影响；以及融合任务后的数据分布混淆。

这种现象可以表述为“策略表现出一定的闭环视觉反馈和失败恢复能力”，但不能直接声称模型理解了失败或具有自主推理能力。必须通过多轮标准化实验确认恢复行为的稳定性。

## 16. 后续改进方向

1. 清理所有 episode 开头的黑屏和曝光异常帧，消除启动视觉捷径。
2. 增加抽屉关闭阶段、首次推偏后的恢复轨迹和不同接触位置的专项示范。
3. 随机化抽屉、物块和机械臂初始位置，但控制在相机可见和机械臂可达范围内。
4. 分别统计开抽屉、抓黑块、抓白块、关抽屉四个子任务的成功率，而不是只记录最终二值结果。
5. 对 action horizon、adaptive skip、拼接步数和低通参数做固定场景 A/B 测试，不能只凭单次观感修改。
6. 如果硬件允许，引入力/电流反馈或更柔顺的接触控制，提高关闭抽屉阶段的可靠性。

## 17. 面试介绍建议

可以按“任务--系统--难点--结果--局限”介绍：

> 我在 SO101 双机械臂平台上完成了 PI0.5 的数据采集、LoRA 微调和真实机械臂部署。数据采用 LeRobot v3 格式，包含双路 RGB 图像、六维关节状态、动作和语言指令，并融合了单物块和双物块放入抽屉两个任务。
>
> 部署采用局域网客户端/服务器架构，RTX 5090 服务器负责策略推理，本地 Ubuntu 电脑负责双相机采集和 30 Hz 机械臂控制。我完成了主从机械臂校准、稳定设备路径、相机对齐、动作单位映射、过期动作拒绝、动作队列、平滑限幅和安全 Hold。实测服务端推理中位延迟约 66 ms，端到端 RTT 约 90 ms。
>
> 最终模型能够完成打开抽屉、依次放入两个物块并关闭抽屉的长程任务。在关闭抽屉这种接触密集阶段，模型偶尔首次定位不准，但能够基于后续视觉观测重新调整并完成。我将结果区分为严格成功和恢复后成功，并计划通过固定时限、多轮测试和子任务指标量化稳定性。
>
> 当前限制包括数据量较少、episode 开头存在黑屏视觉捷径，以及系统缺少力觉反馈。下一步会清理数据、增加失败恢复和关闭抽屉示范，并做场景随机化和标准化评测。

简历可使用以下表述：

- 完成 PI0.5 LoRA 在 SO101 上的数据采集、训练与端到端实机部署，构建 LeRobot v3 双相机、关节状态、动作和语言指令数据管线，支持单物块和双物块抽屉任务。
- 构建局域网视觉闭环推理链路，服务端推理延迟中位数约 66 ms、端到端 RTT 约 90 ms，并实现 action chunk 对齐、stale action 拒绝、动作队列、限幅和安全退出。
- 定位并解决串口权限、机械臂校准、舵机扭矩、相机稳定路径、Headless OpenCV、启动黑屏特征和公网高延迟等问题，完成双物块长程任务实机验证。
- 针对抽屉关闭阶段的重复接触问题，建立严格成功、恢复后成功和子任务成功率评测标准，并提出专项恢复数据和接触控制改进方案。
