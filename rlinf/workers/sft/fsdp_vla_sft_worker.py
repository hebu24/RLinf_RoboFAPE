# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.data.utils import forward_set_epoch
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class _WeightedRandomSampler(torch.utils.data.Sampler):
    """Weighted sampler with per-epoch re-seeding.

    torch.utils.data.WeightedRandomSampler uses a fixed-seed generator, so every
    epoch would re-draw the SAME frame indices. This variant re-seeds on each
    ``__iter__`` (and honors ``set_epoch`` if ``forward_set_epoch`` reaches it),
    giving epoch-to-epoch variation without on-disk duplication. Sampling is
    with replacement, so per-rank overlap (DDP) is fine -- only the AGGREGATE
    new-data fraction matters, and each rank's batches draw ~fraction new.
    """

    def __init__(self, weights, num_samples: int, seed: int = 0):
        self._weights = torch.as_tensor(weights, dtype=torch.double)
        self._num_samples = int(num_samples)
        self._seed = int(seed)
        self._epoch = 0
        self._iter_count = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch + self._iter_count * 7919)
        self._iter_count += 1
        idx = torch.multinomial(
            self._weights, self._num_samples, replacement=True, generator=g
        )
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self._num_samples


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._sft_diag_interval = int(
            getattr(self.cfg.actor, "sft_diagnostics_interval", 100)
        )

    @staticmethod
    def _to_numpy(value: Any):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy()
        try:
            import numpy as np

            return np.asarray(value)
        except Exception:
            return None

    @staticmethod
    def _add_dim_stats(metrics: dict[str, float], prefix: str, value: Any, names):
        array = FSDPVlaSftWorker._to_numpy(value)
        if array is None or array.size == 0:
            return
        import numpy as np

        array = array.reshape(-1, array.shape[-1])
        finite = np.isfinite(array)
        metrics[f"{prefix}/finite_frac"] = float(finite.mean())
        for dim, name in enumerate(names[: array.shape[-1]]):
            dim_values = array[:, dim]
            metrics[f"{prefix}/{name}_mean"] = float(np.mean(dim_values))
            metrics[f"{prefix}/{name}_std"] = float(np.std(dim_values))
            metrics[f"{prefix}/{name}_min"] = float(np.min(dim_values))
            metrics[f"{prefix}/{name}_max"] = float(np.max(dim_values))
            metrics[f"{prefix}/{name}_p01"] = float(np.percentile(dim_values, 1))
            metrics[f"{prefix}/{name}_p99"] = float(np.percentile(dim_values, 99))

    @staticmethod
    def _collect_openpi_batch_diagnostics(batch: Any) -> dict[str, float]:
        state_names = [
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "roll",
            "pitch",
            "yaw",
            "finger0",
            "finger1",
        ]
        action_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
        metrics: dict[str, float] = {}
        if isinstance(batch, tuple):
            observation, actions = batch
        else:
            observation = batch.get("observation") if isinstance(batch, dict) else None
            actions = batch.get("actions") if isinstance(batch, dict) else None

        state = getattr(observation, "state", None)
        FSDPVlaSftWorker._add_dim_stats(metrics, "sft_diag/state", state, state_names)
        FSDPVlaSftWorker._add_dim_stats(
            metrics, "sft_diag/actions", actions, action_names
        )

        action_array = FSDPVlaSftWorker._to_numpy(actions)
        if action_array is not None and action_array.shape[-1] >= 7:
            gripper = action_array[..., 6].reshape(-1)
            metrics["sft_diag/gripper_mean"] = float(gripper.mean())
            metrics["sft_diag/gripper_min"] = float(gripper.min())
            metrics["sft_diag/gripper_max"] = float(gripper.max())
            metrics["sft_diag/gripper_open_frac_gt_0_5"] = float((gripper > 0.5).mean())
            metrics["sft_diag/gripper_close_frac_lt_0"] = float((gripper < 0.0).mean())

        image_mask = getattr(observation, "image_mask", None)
        if isinstance(image_mask, dict):
            for name, value in image_mask.items():
                mask = FSDPVlaSftWorker._to_numpy(value)
                if mask is not None:
                    metrics[f"sft_diag/image_mask/{name}"] = float(mask.mean())
        return metrics

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(data_paths)
            if repo_id is None:
                raise ValueError(
                    "OpenPI SFT requires data.train_data_paths to be set to a local "
                    "dataset path or LeRobot repo id."
                )

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            # HG-DAgger: controllable per-batch new-data fraction. When
            # data.weighted_new_data_fraction is set and the merged dataset has a
            # per-frame `source` column (0=historical, 1=HG-new, written by
            # merge_datasets.py), swap the DataLoader's sampler for a
            # WeightedRandomSampler so each batch draws ~fraction of its frames
            # from the new HG-DAgger data -- no on-disk duplication. Default
            # (key absent or no `source` column) is the unchanged sampler, so
            # non-HG-DAgger SFT runs are unaffected. Use getattr (not OmegaConf.select)
            # because this module only imports DictConfig.
            _data_cfg = getattr(self.cfg, "data", None)
            fraction = (
                getattr(_data_cfg, "weighted_new_data_fraction", None)
                if _data_cfg is not None
                else None
            )
            if fraction is not None:
                try:
                    self._apply_weighted_new_data_sampler(
                        data_loader, repo_id, float(fraction)
                    )
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "WeightedRandomSampler disabled (using default sampler): %s",
                        exc,
                    )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO
        ]:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _apply_weighted_new_data_sampler(
        self, data_loader: Any, repo_id: str, fraction: float
    ) -> None:
        """Swap the OpenPI DataLoader's sampler for a per-frame WeightedRandomSampler.

        Reads the ``source`` column (0=historical, 1=HG-new) straight from the
        merged dataset's parquet -- ``len(LeRobotDataset) == total_frames`` so
        the column is exactly aligned with the dataset index. Weights make each
        sampled frame's marginal probability of being new-data equal to
        ``fraction``: ``w_i = fraction/n_new`` if ``source_i==1`` else
        ``(1-fraction)/n_orig``. The inner torch DataLoader is rebuilt with the
        new sampler (its sampler is fixed at construction), copying the
        original's collate/worker/drop_last kwargs.
        """
        import glob  # noqa: F401  (kept for parity with save_checkpoint)
        from pathlib import Path

        import numpy as np
        import pandas as pd

        torch_dl = self._openpi_pytorch_dataloader(data_loader)
        dataset = torch_dl.dataset
        src_dir = str(repo_id)
        parquets = sorted(Path(src_dir).rglob("data/chunk-*/episode_*.parquet"))
        if not parquets:
            raise RuntimeError(
                f"No parquet under {src_dir}; cannot read `source` column for "
                "WeightedRandomSampler."
            )
        try:
            source = pd.concat(
                [pd.read_parquet(p, columns=["source"])["source"] for p in parquets],
                ignore_index=True,
            ).to_numpy()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Dataset {src_dir} has no per-frame `source` column "
                f"(read_parquet failed: {exc}). Re-run merge_datasets.py to add "
                "it, or unset data.weighted_new_data_fraction."
            ) from exc
        if len(source) != len(dataset):
            raise RuntimeError(
                f"source length {len(source)} != dataset length {len(dataset)}; "
                "cannot align WeightedRandomSampler to the dataset index."
            )
        source = source.astype(np.int64)
        n_new = int((source == 1).sum())
        n_orig = int((source == 0).sum())
        if n_new == 0 or n_orig == 0:
            logging.info(
                "WeightedRandomSampler skipped: only one source present "
                "(new=%d, orig=%d).", n_new, n_orig,
            )
            return
        w_new = fraction / n_new
        w_orig = (1.0 - fraction) / n_orig
        weights = np.where(source == 1, w_new, w_orig).astype(np.float64)
        sampler = _WeightedRandomSampler(
            weights, num_samples=len(dataset), seed=self._rank
        )
        new_torch_dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=torch_dl.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=torch_dl.num_workers,
            collate_fn=torch_dl.collate_fn,
            worker_init_fn=torch_dl.worker_init_fn,
            drop_last=True,
            persistent_workers=bool(torch_dl.persistent_workers),
            multiprocessing_context=torch_dl.multiprocessing_context,
        )
        # DataLoaderImpl -> TorchDataLoader -> torch DataLoader
        data_loader._data_loader._data_loader = new_torch_dl
        logging.info(
            "WeightedRandomSampler ON: fraction=%.3f new=%d orig=%d "
            "per-rank samples=%d batch_size=%d",
            fraction, n_new, n_orig, len(dataset), torch_dl.batch_size,
        )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if (
            self._sft_diag_interval > 0
            and self.global_step % self._sft_diag_interval == 0
        ):
            step_metrics.update(self._collect_openpi_batch_diagnostics(batch))
        if isinstance(output, dict) and output.get("dynamics_loss", None) is not None:
            step_metrics.update(
                {
                    "dynamics_loss": output["dynamics_loss"].detach().item(),
                    "action_loss": output["action_loss"].detach().item(),
                }
            )
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        # OpenPI checkpoints need norm_stats.json colocated with model weights so
        # downstream rollout/eval can load it via load_norm_stats(ckpt_dir, asset_id).
        # FSDP only shards model/optimizer state, so copy norm_stats.json from the
        # base model directory into the checkpoint.
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            if self._rank == 0:
                import glob
                import shutil

                asset_id = getattr(self.data_config, "asset_id", None)
                model_path = self.cfg.actor.model.model_path
                src_norm_stats = None
                if asset_id is not None:
                    candidate = os.path.join(model_path, asset_id, "norm_stats.json")
                    if os.path.exists(candidate):
                        src_norm_stats = candidate
                if src_norm_stats is None:
                    matches = sorted(glob.glob(os.path.join(model_path, "**", "norm_stats.json"), recursive=True))
                    if matches:
                        src_norm_stats = matches[0]
                if src_norm_stats is not None and asset_id is not None:
                    dst_dir = os.path.join(save_path, asset_id)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_norm_stats = os.path.join(dst_dir, "norm_stats.json")
                    shutil.copy2(src_norm_stats, dst_norm_stats)
                    logging.info(f"Copied norm_stats.json from {src_norm_stats} to {dst_norm_stats}")
                else:
                    logging.warning(f"Could not find norm_stats.json under model_path={model_path} with asset_id={asset_id}; rollout/eval will fail to load norm stats.")
            torch.distributed.barrier()

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()
        elif SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            # OpenPI wraps a regular torch DataLoader rather than a
            # StatefulDataLoader. Persist its logical cursor so resumed runs do
            # not restart sampling at the beginning of the epoch.
            if self._rank == 0:
                torch.save(
                    {
                        "data_epoch": self._data_epoch,
                        "data_iter_offset": self._data_iter_offset,
                    },
                    os.path.join(save_path, "data_progress.pt"),
                )
            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()
        elif SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            progress_path = os.path.join(load_path, "data_progress.pt")
            batches_per_epoch = len(self._openpi_pytorch_dataloader(self.data_loader))
            if os.path.exists(progress_path):
                progress = torch.load(progress_path, weights_only=False)
                self._data_epoch = int(progress["data_epoch"])
                self._data_iter_offset = int(progress["data_iter_offset"])
            else:
                # Backward compatibility for checkpoints created before the
                # OpenPI cursor was persisted. One runner step consumes one
                # batch per gradient-accumulation iteration.
                step_dir = os.path.basename(os.path.dirname(load_path.rstrip(os.sep)))
                if not step_dir.startswith("global_step_"):
                    raise ValueError(
                        f"Cannot infer OpenPI data progress from checkpoint path: {load_path}"
                    )
                checkpoint_step = int(step_dir.removeprefix("global_step_"))
                consumed_batches = checkpoint_step * self.gradient_accumulation
                self._data_epoch, self._data_iter_offset = divmod(
                    consumed_batches, batches_per_epoch
                )
                logging.warning(
                    "Checkpoint has no data_progress.pt; inferred OpenPI data "
                    "cursor from global step: epoch=%d, offset=%d",
                    self._data_epoch,
                    self._data_iter_offset,
                )

            forward_set_epoch(self.data_loader, self._data_epoch)
            self.data_iter = iter(self.data_loader)
            for _ in range(self._data_iter_offset):
                next(self.data_iter)
            logging.info(
                "Restored OpenPI data cursor: epoch=%d, offset=%d/%d",
                self._data_epoch,
                self._data_iter_offset,
                batches_per_epoch,
            )
            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
