import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """支持的运行环境枚举。"""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """从训练好的checkpoint加载Policy的配置类。"""

    # 训练配置名称 (例如: "pi0_aloha_sim")
    config: str
    # Checkpoint目录路径 (例如: "checkpoints/pi0_aloha_sim/exp/10000")
    dir: str


@dataclasses.dataclass
class Default:
    """使用指定环境的默认Policy。"""


@dataclasses.dataclass
class Args:
    """serve_policy脚本的命令行参数。"""

    # 环境类型，仅在使用默认Policy时生效
    env: EnvMode = EnvMode.ALOHA_SIM

    # 默认prompt，当数据中没有"prompt"字段或模型没有默认prompt时使用
    default_prompt: str | None = None

    # WebSocket服务端口
    port: int = 8000
    
    # 是否记录Policy的推理行为（用于调试）
    record: bool = False

    # Policy加载方式：指定Checkpoint路径或使用环境默认Policy
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# 各环境对应的默认Checkpoint配置
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """为指定环境创建默认Policy。
    
    Args:
        env: 运行环境枚举
        default_prompt: 可选的默认prompt
        
    Returns:
        基于预训练checkpoint创建的Policy实例
        
    Raises:
        ValueError: 当环境不支持时抛出异常
    """
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """根据命令行参数创建Policy实例。
    
    根据args.policy的类型决定如何创建Policy：
    - Checkpoint类型：从指定路径加载checkpoint创建Policy
    - Default类型：为指定环境创建默认Policy
    
    Args:
        args: 命令行参数，包含环境类型和Policy配置
        
    Returns:
        初始化好的Policy实例
    """
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    """主函数：启动Policy推理服务。
    
    流程：
    1. 根据参数创建Policy实例
    2.（如启用）包装Policy为PolicyRecorder以记录推理数据
    3. 创建WebSocket服务并开始监听推理请求
    
    推理接口调用位置：
    - WebsocketPolicyServer在接收到客户端请求后，
      通过 self._policy.infer(obs) 调用Policy的推理方法
    - 参见 websocket_policy_server.py 中的 _handler 方法
    """
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # 如果启用record，则包装Policy以记录所有推理数据
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # 创建WebSocket推理服务
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
