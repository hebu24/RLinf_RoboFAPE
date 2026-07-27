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

"""Robometer progress-estimator reward model (HTTP client to the robometer server).

Plugs into RLinf's ``history_buffer`` reward path: the env worker accumulates
the scene's human render-camera frames (``render_images`` -- one frame per
low-level env step) and, at trajectory done, sends them to this model (which
runs in the reward-worker Ray group). This model POSTs the per-env frame stacks
to a running robometer eval server (``POST /evaluate_batch_npy``), parses the
per-frame progress curve in ``[0, 1]``, and returns a per-frame reward tensor::

    reward[env, frame_i] = progress[i]        if the trajectory succeeded
                           progress[i] - 1.0  if it did not

``success`` must come from the env/eval side success signals; if it is missing,
we raise instead of falling back to robometer ``success_probs``. The env worker
later interpolates these down-sampled frame rewards back onto low-level
insertion steps, then aggregates them back to chunk rewards for chunk-level PPO.
"""

from __future__ import annotations

import io
import json
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

import os
import threading

# --- Robometer smoke capture (env-var gated; no-op when RLINF_ROBOMETER_SMOKE_DIR unset) ---
_SMOKE_DIR = os.environ.get("RLINF_ROBOMETER_SMOKE_DIR", "").strip() or None
_SMOKE_CAP = int(os.environ.get("RLINF_ROBOMETER_SMOKE_CAP", "10")) if _SMOKE_DIR else 0
_SMOKE_LOCAL_LOCK = threading.Lock()
_SMOKE_PREV = {}  # per-process: env_id -> {arr,prog,success,shift,len} for reset-detection


def _robometer_downsample_indices(total: int, cap: int = 60):
    """Deterministic uniform down-sample indices over [0, total-1] to <= cap.

    Shared by ``RobometerHistoryRewardModel.compute_reward`` (down-samples the
    POSTed frames) AND ``env_worker.assign_history_reward`` (recomputes the SAME
    indices to map progress -> chunks + set loss_mask). MUST stay in sync: both
    call this with the SAME total (= history buffer length per env).
    """
    if total <= cap:
        return list(range(total))
    import numpy as np

    idx = np.linspace(0, total - 1, cap).astype(int)
    seen: set[int] = set()
    out: list[int] = []
    for i in idx:
        if int(i) not in seen:
            seen.add(int(i))
            out.append(int(i))
    return out


def _extract_env_value(value: Any, env_id: int) -> Any:
    """Extract one env's entry from a possibly-batched success container."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, np.bool_, np.number)):
        return value
    try:
        env_value = value[env_id]
    except Exception:
        return None
    if isinstance(env_value, np.ndarray) and env_value.shape == ():
        return env_value.item()
    if hasattr(env_value, "item"):
        try:
            return env_value.item()
        except Exception:
            return env_value
    return env_value


def _resolve_env_success_from_infos(
    env_infos: dict[str, Any] | None, env_id: int
) -> bool:
    """Resolve success using the same priority as peg-insertion eval.

    Priority:
    1. final_info.episode.success_once
    2. episode.success_once
    3. root success

    Raises:
        ValueError: When no reliable env-side success signal is available.
    """
    if not isinstance(env_infos, dict):
        raise ValueError(
            f"Missing env_infos for env_id={env_id}; cannot resolve success_once."
        )

    final_info = env_infos.get("final_info")
    if isinstance(final_info, dict):
        final_episode = final_info.get("episode")
        if isinstance(final_episode, dict):
            value = _extract_env_value(final_episode.get("success_once"), env_id)
            if value is not None:
                return bool(value)

    episode = env_infos.get("episode")
    if isinstance(episode, dict):
        value = _extract_env_value(episode.get("success_once"), env_id)
        if value is not None:
            return bool(value)

    value = _extract_env_value(env_infos.get("success"), env_id)
    if value is not None:
        return bool(value)

    raise ValueError(
        "Robometer reward requires an env-side success signal but none was found. "
        f"env_id={env_id}, available env_info keys={sorted(env_infos.keys())}."
    )


def _interpolate_insert_progress_from_downsampled_frames(
    progress, history_len: int, pickup_count: int, max_frames: int = 60
):
    """Map down-sampled robometer progress -> per-step insertion reward + mask.

    ``history_len`` is the original full video length = pickup + insertion low-level
    steps. The history is uniformly downsampled before the Robometer POST; this
    function reprojects the returned frame rewards back onto insertion low-level
    steps and linearly interpolates between labeled insertion frames.

    Returns:
        ``(per_step_reward, per_step_has_reward)`` where both have length
        ``max(0, history_len - pickup_count)``.
    """
    n_insert = max(0, history_len - pickup_count)
    per_step_reward = np.zeros(n_insert, dtype=np.float32)
    per_step_has_reward = np.zeros(n_insert, dtype=bool)
    if n_insert == 0:
        return per_step_reward, per_step_has_reward

    ds = _robometer_downsample_indices(history_len, max_frames)
    prog = np.asarray(progress, dtype=np.float32) if progress is not None else np.zeros(0)

    labeled_steps: list[int] = []
    labeled_values: list[float] = []
    for j, orig_idx in enumerate(ds):
        if j >= prog.shape[0]:
            break
        if orig_idx < pickup_count:
            continue
        step_idx = int(orig_idx) - pickup_count
        if 0 <= step_idx < n_insert:
            labeled_steps.append(step_idx)
            labeled_values.append(float(prog[j]))

    if not labeled_steps:
        return per_step_reward, per_step_has_reward

    unique_steps: list[int] = []
    unique_values: list[float] = []
    for step_idx, value in zip(labeled_steps, labeled_values, strict=True):
        if unique_steps and step_idx == unique_steps[-1]:
            unique_values[-1] = value
            continue
        unique_steps.append(step_idx)
        unique_values.append(value)

    if len(unique_steps) == 1:
        per_step_reward[:] = unique_values[0]
        per_step_has_reward[:] = True
        return per_step_reward, per_step_has_reward

    for seg_idx in range(len(unique_steps) - 1):
        start_step = unique_steps[seg_idx]
        end_step = unique_steps[seg_idx + 1]
        start_value = unique_values[seg_idx]
        end_value = unique_values[seg_idx + 1]
        if end_step <= start_step:
            per_step_reward[start_step] = end_value
            continue
        steps = np.arange(start_step, end_step + 1, dtype=np.float32)
        alpha = (steps - float(start_step)) / float(end_step - start_step)
        per_step_reward[start_step : end_step + 1] = (
            (1.0 - alpha) * start_value + alpha * end_value
        )

    first_step = unique_steps[0]
    last_step = unique_steps[-1]
    per_step_reward[:first_step] = unique_values[0]
    per_step_reward[last_step + 1 :] = unique_values[-1]
    per_step_has_reward[:] = True
    return per_step_reward, per_step_has_reward


def _apply_stepwise_success_shift(
    per_step_progress: np.ndarray,
    per_step_success: np.ndarray,
    fail_shift: float = 1.0,
) -> np.ndarray:
    """Convert interpolated progress into reward using per-step success labels."""
    progress = np.asarray(per_step_progress, dtype=np.float32)
    success = np.asarray(per_step_success, dtype=bool)
    if progress.shape != success.shape:
        raise ValueError(
            "per_step_progress and per_step_success must have identical shape: "
            f"{progress.shape=} vs {success.shape=}."
        )
    reward = progress.copy()
    reward[~success] -= float(fail_shift)
    return reward


def _smoke_acquire_slot(smoke_dir, cap):
    """Atomically acquire a dump slot index < cap across processes (3 reward shards)."""
    import fcntl

    os.makedirs(smoke_dir, exist_ok=True)
    lock_path = os.path.join(smoke_dir, ".counter.lock")
    ctr_path = os.path.join(smoke_dir, ".counter")
    with open(lock_path, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            n = 0
            if os.path.exists(ctr_path):
                try:
                    with open(ctr_path) as f:
                        n = int(f.read().strip() or "0")
                except Exception:
                    n = 0
            if n >= cap:
                return None
            with open(ctr_path, "w") as f:
                f.write(str(n + 1))
            return n
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _smoke_save_video(frames, path, fps=30):
    """Save [T, H, W, 3] uint8 as mp4 (cv2 -> imageio -> PNG fallback)."""
    h, w = frames.shape[1], frames.shape[2]
    try:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if wr.isOpened():
            for f in frames:
                wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            wr.release()
            return
    except Exception:
        pass
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(path), [np.asarray(f) for f in frames], fps=fps)
        return
    except Exception:
        pass
    import matplotlib.pyplot as plt

    stem = os.path.splitext(os.path.basename(path))[0]
    for i, f in enumerate(frames):
        plt.imsave(os.path.join(os.path.dirname(path), f"{stem}_{i:03d}.png"), np.asarray(f))


def _smoke_dump(smoke_dir, slot, env_id, arr, prog, env_success, shift, task, dones_present):
    tdir = os.path.join(smoke_dir, f"traj_{slot:03d}")
    os.makedirs(tdir, exist_ok=True)
    _smoke_save_video(np.asarray(arr), os.path.join(tdir, "robometer_input.mp4"), fps=30)
    np.save(os.path.join(tdir, "progress.npy"), np.asarray(prog, dtype=np.float32))
    np.save(os.path.join(tdir, "reward.npy"), np.asarray(prog, dtype=np.float32) - float(shift))
    with open(os.path.join(tdir, "success.txt"), "w") as f:
        f.write("1" if env_success else "0")
    with open(os.path.join(tdir, "meta.json"), "w") as f:
        json.dump(
            {
                "slot": int(slot),
                "env_id": int(env_id),
                "task": task,
                "n_chunks": int(arr.shape[0]),
                "success": bool(env_success),
                "dones_present": bool(dones_present),
            },
            f,
            indent=2,
        )


def _np_to_npy_file_tuple(arr: np.ndarray, filename: str):
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return (filename, buf, "application/octet-stream")


def _build_multipart_payload(samples: list[dict[str, Any]]):
    """Mirror robometer ``scripts/inference/example_inference.build_multipart_payload``.

    Vendored here because the robometer package lives in a separate uv venv and
    is not importable from the RLinf conda env.
    """
    files: dict[str, Any] = {}
    data: dict[str, str] = {}
    numpy_fields = ["frames", "lang_vector", "video_embeddings"]
    for i, sample in enumerate(samples):
        sample_copy = json.loads(json.dumps(sample, default=str))
        traj = sample.get("trajectory", {})
        traj_copy = sample_copy.get("trajectory", {})
        for field in numpy_fields:
            val = traj.get(field, None)
            if val is None:
                continue
            if hasattr(val, "detach") and hasattr(val, "cpu"):
                val = val.detach().cpu().numpy()
            if isinstance(val, np.ndarray):
                key = f"sample_{i}_trajectory_{field}"
                files[key] = _np_to_npy_file_tuple(val, f"{key}.npy")
                traj_copy[field] = {"__numpy_file__": key}
            else:
                traj_copy[field] = val
        if "frames_shape" in traj_copy and isinstance(
            traj_copy["frames_shape"], (tuple, list)
        ):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]
        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)
    return files, data


def _post_evaluate_batch_npy(
    server_url: str,
    samples: list[dict[str, Any]],
    timeout_s: float,
    use_frame_steps: bool,
) -> dict[str, Any]:
    import requests

    files, data = _build_multipart_payload(samples)
    data["use_frame_steps"] = "true" if use_frame_steps else "false"
    url = server_url.rstrip("/") + "/evaluate_batch_npy"
    resp = requests.post(url, files=files, data=data, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


class RobometerHistoryRewardModel(BaseRewardModel):
    """Per-frame progress from a robometer eval server.

    Config (under ``reward.model``): ``server_url``, ``task`` (the robometer
    task string), ``timeout_s``, ``use_frame_steps``, ``fail_shift`` (default
    1.0), ``min_history_size`` (return None below this), and
    ``max_robometer_frames``. ``success_threshold`` is still parsed for backward
    compatibility but peg-insertion RL success now comes only from env infos.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.server_url = cfg.get("server_url", "http://127.0.0.1:8000")
        self.task = cfg.get(
            "task", "Insert the peg vertically into the target hole."
        )
        self.timeout_s = float(cfg.get("timeout_s", 120.0))
        self.use_frame_steps = bool(cfg.get("use_frame_steps", True))
        self.success_threshold = float(cfg.get("success_threshold", 0.5))
        self.fail_shift = float(cfg.get("fail_shift", 1.0))
        self.min_history_size = int(cfg.get("min_history_size", 2))
        # Cap the number of frames POSTed to the robometer per env (uniform
        # down-sample of the full pick-up+insert history buffer). Keeps the
        # single-POST sequence length bounded (calibration + server memory) and
        # matches the frame count the model was probed at (60f -> success 0.84).
        # The env_worker recomputes the SAME down-sample indices (deterministic
        # linspace) to map progress -> chunks + set loss_mask.
        self.max_robometer_frames = int(cfg.get("max_robometer_frames", 60))
        buffers = cfg.get("history_buffers", {}) or {}
        self.render_buffer_name = cfg.get(
            "render_buffer_name",
            next(iter(buffers)) if buffers else None,
        )

    def forward(self, input_data, labels=None):
        raise NotImplementedError(
            "RobometerHistoryRewardModel is an inference-only HTTP reward client."
        )

    @torch.no_grad()
    def compute_reward(self, observations: Any) -> Optional[torch.Tensor]:
        history_input = observations.get("history_input", {}) or {}
        buf = history_input.get(self.render_buffer_name, {})
        frame_lists = buf.get("render_images", [])  # list[env] of list[frame]
        n_envs = len(frame_lists)
        if n_envs == 0:
            return None

        env_infos = observations.get("env_infos")

        ready: list[tuple[int, np.ndarray, int]] = []  # (env_id, real arr, real_t)
        for env_id, frames in enumerate(frame_lists):
            if not frames:
                continue
            arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
            if arr.shape[0] < self.min_history_size:
                continue
            # Uniform down-sample the full pick-up+insert buffer to <= max frames
            # before the single POST (use_frame_steps=False). env_worker recomputes
            # the SAME indices to map progress -> chunks + set loss_mask.
            ds = _robometer_downsample_indices(
                arr.shape[0], self.max_robometer_frames
            )
            if len(ds) < arr.shape[0]:
                arr = arr[ds]
            ready.append((env_id, arr, arr.shape[0]))
        if not ready:
            # Return a real (zero) tensor, not None: the reward worker always
            # sends this back to the env_worker, whose blocking recv_from would
            # hang on a None payload (no channel message) at non-done chunk steps.
            # Zeros are a harmless no-op (env_reward_weight=0 blends nothing).
            return torch.zeros((n_envs, 1), dtype=torch.float32)

        # Pad each env's frames to the batch max with repeat-last-frame so the
        # server's per-batch torch.stack(progress_list) sees EQUAL sequence
        # lengths. Without this, envs with different down-sampled lengths (e.g.
        # 60 vs 43) make the server stack fail ([60,10] vs [43,10]) -> RuntimeError
        # -> 500. real_t is kept so the reward mapping below uses only the real
        # frames' progress (the padded tail's progress is ignored).
        max_t = max(r[2] for r in ready)
        samples = []
        for env_id, arr, real_t in ready:
            if real_t < max_t:
                arr_pad = np.concatenate(
                    [arr, np.repeat(arr[-1:], max_t - real_t, axis=0)], axis=0
                )
            else:
                arr_pad = arr
            samples.append(
                {
                    "sample_type": "progress",
                    "trajectory": {
                        "frames": arr_pad,
                        "frames_shape": list(arr_pad.shape),
                        "task": self.task,
                        "id": str(env_id),
                        "metadata": {"subsequence_length": int(arr_pad.shape[0])},
                        "video_embeddings": None,
                    },
                }
            )
        outputs = _post_evaluate_batch_npy(
            self.server_url, samples, self.timeout_s, self.use_frame_steps
        )
        prog_lists = outputs.get("outputs_progress", {}).get("progress_pred", [])

        out = torch.zeros((n_envs, max_t), dtype=torch.float32)
        for idx, (env_id, arr, real_t) in enumerate(ready):
            t = real_t
            if idx < len(prog_lists) and prog_lists[idx]:
                prog = np.asarray(prog_lists[idx], dtype=np.float32)
                t = min(prog.shape[0], real_t)
                prog = prog[:t]
            else:
                prog = np.zeros(t, dtype=np.float32)
            env_success = _resolve_env_success_from_infos(env_infos, env_id)
            shift = 0.0 if env_success else self.fail_shift
            out[env_id, :t] = torch.from_numpy(prog)
            if _SMOKE_DIR:
                try:
                    _prev = _SMOKE_PREV.get(env_id)
                    _cur_len = int(arr.shape[0])
                    if _prev is not None and _cur_len < _prev["len"]:
                        _slot = None
                        with _SMOKE_LOCAL_LOCK:
                            _slot = _smoke_acquire_slot(_SMOKE_DIR, _SMOKE_CAP)
                        if _slot is not None:
                            _smoke_dump(
                                _SMOKE_DIR,
                                _slot,
                                env_id,
                                _prev["arr"],
                                _prev["prog"],
                                _prev["success"],
                                _prev["shift"],
                                self.task,
                                False,
                            )
                    _SMOKE_PREV[env_id] = {
                        "arr": arr,
                        "prog": prog,
                        "success": env_success,
                        "shift": shift,
                        "len": _cur_len,
                    }
                except Exception:
                    pass
        return out
