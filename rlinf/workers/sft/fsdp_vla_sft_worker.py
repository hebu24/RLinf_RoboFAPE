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
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


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
