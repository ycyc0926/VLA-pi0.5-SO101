一、背景
  - 在具身智能领域，Physical Intelligence 团队推出的 π0（pi0）是一个具有分水岭意义的视觉-语言-动作（VLA）基础模型。与过去依赖离散动作字典或传统扩散模型（Diffusion）的策略不同，π0 创新性地采用了流匹配（Flow Matching）算法，突破了以往策略在轨迹生成上的局限。这种架构最大的优势在于能够生成极其平滑、连续的动作轨迹。对于实际的物理部署而言，平滑的轨迹是连接高级语义推理与底层执行的核心。
  - 过去，复现顶尖的泛化机器人策略往往需要极其昂贵的双臂平台（如 ALOHA 或 DROID 集群）。而 Hugging Face 开源的 LeRobot 框架，提供了一套纯 PyTorch 的原生接口，极大地标准化了数据采集格式（LeRobotDataset）、模型训练和硬件控制流程。 SO101 作为一款机械结构优秀的开源轻量化6-DoF机械臂，将硬件成本降到了极致。
  - 该说明文档旨在为从0复现PI0模型提供参考。
二、操作步骤
2.1 环境配置
操作系统
  - 操作系统：Ubuntu 22.04
更新系统并安装基础依赖
  - 打开终端（快捷键 Ctrl + Alt + T）
sudo apt update && sudo apt upgrade -y

安装代码编辑器VS Code
  - 官网下载.deb文件
  - 使用 cd 命令进入.deb文件所在的目录，将 package_name.deb 替换为实际文件名
sudo dpkg -i package_name.deb
安装显卡驱动
# 查看推荐的驱动版本
ubuntu-drivers devices
# 自动安装推荐的驱动（通常是 nvidia-driver-535 或类似版本）
sudo ubuntu-drivers autoinstall
# 安装完成后重启电脑
sudo reboot
  - 重启后，终端输入 nvidia-smi，如果能看到显卡状态表格，说明安装成功
Python 环境管理 (Miniconda)
  - 安装 Miniconda，适配OpenPI
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
# 激活 conda
source ~/miniconda3/bin/activate
conda init bash
  - 安装完成后，关闭并重新打开终端，会看到命令提示符前面多了一个 (base)
配置 pi0 专属环境
  - 创建虚拟环境
conda create -n pi0_env python=3.11 -y
conda activate pi0_env
  - 如果出现报错：(base) joyce@joyce:~$ conda create -n pi0_env python=3.11 -y CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels.
  -  原因：Anaconda（也就是提供 Conda 的公司）最近更新了他们的服务协议（Terms of Service, 简称 ToS）。当试图从他们的官方频道（channels）下载 Python 3.11 时，系统发现你还是新用户，还没有签署过同意协议，所以拦截了你的下载请求。
  - 解决方法：依次执行以下命令
# 第一步：同意 main 频道的协议
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# 第二步：同意 r 频道的协议
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# 第三步：重新执行你的环境创建命令
conda create -n pi0_env python=3.11 -y
# 等系统显示创建完成后，就可以继续按照之前的进度，激活这个专属环境并安装 PyTorch 的核心依赖了：
# 激活环境
conda activate pi0_env
一键安装 PyTorch + CUDA
  - 确保当前在pi0_env 环境下（终端命令行最左边显示 (pi0_env)）
  - 配置清华大学镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
  - 安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -y
  - 在最后问Proceed ([y]/n)?时，输入 y 并回车
  - 安装完成后，可以在终端里输入下面这行命令来验证：
python -c "import torch; print(torch.cuda.is_available())"
# 如果屏幕上输出了 True，PyTorch和显卡环境就彻底打通
  - 安装复现pi0的核心依赖
pip install transformers datasets accelerate diffusers einops
安装openpi
  - 在pi0_env 环境下操作
  - 克隆代码仓库
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
  - 命令跑完，当前目录下会多出一个叫openpi的文件夹，进入刚下载好的代码文件夹：
cd openpi
  - 把 uv 安装在当前的虚拟环境里：
pip install uv
  - 用 uv 的超强解析器安装代码环境：
# 忽略几十G的模型权重，只安装代码环境
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 验证是否安装成功
python -c "import openpi; print('太棒了，openpi 安装成功！')"
配置LeRobot环境
  - 创建虚拟环境
conda create -n lerobot python=3.12 -y
  - 进入虚拟环境
conda activate lerobot
  - 安装ffmpeg
# 1. 换成国内清华源（为了下载不超时）
conda config --remove-key channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

# 2. 安装 ffmpeg
conda install ffmpeg=7.1.1 -c conda-forge -y
  - 验证安装成功
ffmpeg
[图片]
  - 下载LeRobot官方代码库
git clone https://github.com/huggingface/lerobot.git
  - 安装代码仓库
cd lerobot

pip install -e ".[feetech]"
  - 验证安装成功
lerobot-info

python

import lerobot
lerobot.__version__

import torch
torch.cuda.is_available()
import scservo_sdk
[图片]
[图片]
2.2 SO101机械臂配置
克隆定制代码仓库
  - 将GitHub仓库（同步好的 LeRobot、OpenPI 脚本和实机测试代码）拉取到本地电脑
# 放在用户主目录下
cd ~

# 执行克隆命令，--depth 1 参数表示只下载最新的一份代码，不要历史记录
git clone --depth 1 https://github.com/ROS-LiKunwei/VLA.git
物理连接
  - 将两个摄像头和两个机械臂连接到电脑
端口绑定
  - 打开配置文件，在终端输入以下命令并回车，会打开一个叫 nano 的极简文本编辑器
sudo nano /etc/udev/rules.d/99-usb-serial.rules
[图片]
  - 复制粘贴规则，将下面这两行完整的代码复制，然后粘贴到终端 nano 编辑器里
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5B14029829", SYMLINK+="ttyACM16"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5B14115151", SYMLINK+="ttyACM17"
  - 保存并退出 nano
  在终端里按下以下快捷键组合：
  按 Ctrl + O （字母 O，这代表写入保存）。
  按 Enter （回车确认文件名）。
  按 Ctrl + X （退出编辑器，回到正常的终端命令行）。
  - 配置写好了，需要刷新一下系统。在终端运行：
sudo udevadm control --reload-rules && sudo udevadm trigger
  - 通过以下指令，列出所有的端口；再分别拔下主臂与从臂，确认主臂与从臂的端口信息
ls /dev/ttyACM*
  - 赋予权限，如果主臂是0，从臂是1，则：
sudo chmod 777 /dev/ttyACM0
sudo chmod 777 /dev/ttyACM1
校准
  - 切换到（lerobot）环境，进入 ~/VLA/lerobot（下载的定制代码仓库）文件夹
conda activate lerobot
cd ~/VLA/lerobot
  - 从臂ttyACM1，每次开机会变化
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm
  - 运行后先把机械臂移动到中间位置并按下回车，然后转动所有关节到极限角度