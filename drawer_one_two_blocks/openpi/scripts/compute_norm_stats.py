#!/usr/bin/env python3
"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import sys
import os
#强行把当前脚本的上一级目录（即 openpi 根目录）加入搜索路径, 这样 Python 才能找到名为 'openpi' 的包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


# def main(config_name: str, max_frames: int | None = None):
#     config = _config.get_config(config_name)
#     data_config = config.data.create(config.assets_dirs, config.model)

#     if data_config.rlds_data_dir is not None:
#         data_loader, num_batches = create_rlds_dataloader(
#             data_config, config.model.action_horizon, config.batch_size, max_frames
#         )
#     else:
#         data_loader, num_batches = create_torch_dataloader(
#             data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
#         )

#     keys = ["state", "actions"]
#     stats = {key: normalize.RunningStats() for key in keys}

#     for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
#         for key in keys:
#             stats[key].update(np.asarray(batch[key]))

#     norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

#     output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
#     print(f"Writing stats to: {output_path}")
#     normalize.save(output_path, norm_stats)
def main(
    config_name: str,
    max_frames: int | None = None,
):
    config = _config.get_config(config_name)

    data_config = config.data.create(
        config.assets_dirs,
        config.model,
    )

    print("===== Normalization config =====")
    print(f"Config name: {config.name}")
    print(f"Assets base: {config.assets_dirs}")
    print(f"Dataset repo/root: {data_config.repo_id}")
    print(f"Asset ID: {data_config.asset_id}")

    filter_path = getattr(
        data_config,
        "lerobot_filter_dict_path",
        None,
    )
    print(f"LeRobot filter: {filter_path or 'disabled'}")

    if filter_path is not None and not os.path.isfile(filter_path):
        raise FileNotFoundError(
            f"Configured LeRobot filter JSON does not exist: "
            f"{filter_path}"
        )

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            max_frames,
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {
        key: normalize.RunningStats()
        for key in keys
    }

    for batch in tqdm.tqdm(
        data_loader,
        total=num_batches,
        desc="Computing stats",
    ):
        for key in keys:
            stats[key].update(
                np.asarray(batch[key])
            )

    norm_stats = {
        key: stats[key].get_statistics()
        for key in keys
    }

    # Prefer the explicit asset_id. This keeps the normalization statistics
    # separate from the source dataset directory.
    stats_asset_id = (
        data_config.asset_id
        if data_config.asset_id is not None
        else data_config.repo_id
    )

    if stats_asset_id is None:
        raise ValueError(
            "Cannot determine normalization asset ID: "
            "both data_config.asset_id and repo_id are None"
        )

    # Preserve the previous fallback behavior, but warn if an absolute repo
    # path would override config.assets_dirs.
    if (
        data_config.asset_id is None
        and os.path.isabs(str(stats_asset_id))
    ):
        print(
            "WARNING: asset_id is not configured and repo_id is an "
            "absolute path. The output path may resolve inside the "
            "dataset directory. Configure AssetsConfig(asset_id=...) "
            "before continuing."
        )

    output_path = (
        config.assets_dirs
        / str(stats_asset_id)
    )

    print("===== Normalization output =====")
    print(f"Writing stats to: {output_path}")

    normalize.save(
        output_path,
        norm_stats,
    )

    print(
        f"Normalization statistics saved successfully: "
        f"{output_path / 'norm_stats.json'}"
    )

if __name__ == "__main__":
    tyro.cli(main)
