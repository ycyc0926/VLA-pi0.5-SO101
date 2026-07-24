import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

import os
os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    """ 
        第一阶段：初始化环境与硬件
    """
    init_logging()
    logging.info(f"Running on: {platform.node()}") # 打印当前运行这台机器的名字

    # 检查 Batch Size 是否能被显卡数量整除。
    # 比如你有 8 张卡，Batch Size 是 32，那每张卡分 4 个数据。如果不能整除，JAX 的并行会报错。
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 设置 JAX 的编译缓存。
    # JAX 第一次运行会把 Python 代码编译成 GPU 高效指令，这很慢。
    # 存到缓存里，下次运行同一段代码就秒开了。
    jax.config.update("jax_compilation_cache_dir", os.environ.get("JAX_COMPILATION_CACHE_DIR", "/root/autodl-tmp/cache/jax"))

    # 随机数种子。JAX 的随机性是显式管理的，为了保证训练可复现。
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng) # 拆成两个，一个给初始化模型，一个给训练采样。

    """
        第二阶段：并行策略与模型准备
    """
    # 创建计算网格 (Mesh)。
    # FSDP (Fully Sharded Data Parallel) 是关键。
    # 它把巨大的模型参数切碎，分布在不同的 GPU 上，这样单张显存放不下的模型也能练。
    mesh = sharding.make_mesh(config.fsdp_devices)
    
    # 定义数据和权重的分布规则：
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)) # DATA_AXIS 表示数据是在不同设备间“切分”的（每张卡看不同的图片）
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()) # replicated_sharding 表示有些参数要在所有卡上“完全复制”一份。

    # 初始化检查点管理器。
    # 它负责：1. 如果训练断了，从哪恢复；2. 每隔多久存一次模型。
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    """
        第三阶段：数据加载(输入接口的核心)
    """
    # 创建数据加载器。这里会去读 LeRobot 格式的数据集。
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding, # 告诉加载器如何把数据分发到多张卡上。
        shuffle=True, # 是否打乱数据
    )
    data_iter = iter(data_loader) # 创建一个迭代器Iterator
    batch = next(data_iter) # 预取一个 batch 来看看数据长啥样
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    
    # === 调试开始：检查输入数据到底有没有内容 ===
    # batch[0] 是 Observation 对象
    obs = batch[0]
    logging.info("=" * 50)
    logging.info("DEBUG: Checking Observation Data Content")
    
    if hasattr(obs, 'images') and obs.images:
        for cam_name, cam_tensor in obs.images.items():
            # 将 JAX 数组转为 Numpy 方便计算
            img_np = np.array(cam_tensor) 
            
            stats = {
                "shape": img_np.shape,
                "dtype": img_np.dtype,
                "min": img_np.min(),
                "max": img_np.max(),
                "mean": img_np.mean(),
                "std": img_np.std(),
            }
            logging.info(f"Camera [{cam_name}]: {stats}")
            
            # 如果 max 为 0，说明真的是全黑
            if stats["max"] == 0:
                logging.error(f"!!! CRITICAL: Camera {cam_name} is DATA-ZERO (Pure Black) !!!")
            elif stats["max"] <= 1.05:
                logging.info(f"--- Info: Camera {cam_name} is normalized [0, 1] ---")
            else:
                logging.info(f"--- Info: Camera {cam_name} is raw [0, 255] ---")
    else:
        logging.error("!!! CRITICAL: No images found in observation! Check your RepackTransform. !!!")

    # 顺便检查一下机器人状态 state
    if hasattr(obs, 'state') and obs.state is not None:
        state_np = np.array(obs.state)
        logging.info(f"Robot State: shape={state_np.shape}, mean={state_np.mean():.4f}")
    
    logging.info("=" * 50)
    # === 调试结束 ===
    
    # “完整性检查”：把 batch 里的第一组图片拼在一起发给 WandB。
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    """
        第四阶段：状态初始化与编译
    """
    # 初始化训练状态（TrainState）
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)    # TrainState 包含：1. 模型权重 2. 优化器状态（Adam 的动量等）3. 当前跑到了第几步
    jax.block_until_ready(train_state) # 强制等待初始化完成。JAX 是异步执行的，这一行确保显存里的模型真的创建好了。
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 如果是中途恢复，就把存好的权重加载回来
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 重点：JIT 编译训练步。
    # 使用 jax.jit 把普通的 Python 函数转换成高度优化的 GPU 核函数。
    # ptrain_step 就是那个“超级加速版”的单步训练函数。
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,), # 优化内存：把旧的 state 内存直接给新的 state 用，不用重复申请。
    )

    """
        第五阶段：正式训练循环
    """
    start_step = int(train_state.step)
    pbar = tqdm.tqdm( # 进度条
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = [] # 存每一步的 Loss，用来算平均值
    for step in pbar:
        # 在定义的计算网格内执行一步训练
        with sharding.set_mesh(mesh):
            # 运行编译好的训练逻辑：计算梯度 -> 更新权重 -> 返回新状态和 Loss
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        
        # 每隔固定步数（比如 100 步）打印并记录一次
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos) # 把这一堆 Loss 压在一起
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos)) # 算平均值
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            logging.info(f"TRAIN_METRICS Step {step}: {info_str}")  # 中文注释：让 tmux 与 tee 日志稳定显示当前 step、loss、grad_norm 和 param_norm。
            wandb.log(reduced_info, step=step) # 发到网页
            infos = [] # 清空缓存
        
        # 获取下一个 batch，为下一步做准备
        batch = next(data_iter)

        # 定期保存模型（Save Checkpoint）
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            
    # 训练结束，确保所有异步的保存任务都写进硬盘
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    """
        这个 config 是通过 Python 的动态配置系统 生成的。在 OpenPI 中，它并不是一个简单的 .yaml 文件，而是一个经过实例化的 Python 对象。
    1. 命令行入口:_config.cli()
        1) 解析命令行参数：它会抓取你输入的第一个参数 pi05_so101_lora。
        2) 动态查找配置文件：它去 config.py 的 _CONFIGS 列表里找name="pi05_so101_lora" 的对象
        3) 对象提取：找到了你刚才定义的那个含有 LoRA、CosineDecay、SO101 数据路径的 TrainConfig 实例
        4) 传递给 main: 这个完整的对象被作为 config 参数传进了 def main(config)
        5) 数据触发: main 函数调用 _data_loader.create_data_loader(config, ...),
        此时会触发你写的 LeRobotSO101DataConfig.create()，从而构建出完整的数据流水线
    2. 配置文件内部
        1) 返回TrainConfig实例
    """
    main(_config.cli())
