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
chunk step, since ``HistoryManager.append_to_history_entries`` is called once
per ``get_reward_model_output``) and, at trajectory done, sends them to this
model (which runs in the reward-worker Ray group). This model POSTs the per-env
frame stacks to a running robometer eval server
(``POST /evaluate_batch_npy``), parses the per-frame progress curve in
``[0, 1]``, and returns a per-chunk reward tensor::

    reward[env, chunk_i] = progress[i]        if the trajectory succeeded
                           progress[i] - 1.0  if it did not

``success`` is the env's test-time criterion (``has_peg_inserted()``) forwarded
in ``env_infos["success"]``; if that is unavailable we fall back to robometer's
own ``success_probs[-1] > success_threshold``. Because the history holds one
frame per chunk step, the robometer per-frame curve maps 1:1 to per-chunk
rewards (no env-step<->chunk resampling needed). Output shape
``[n_envs, max_T]`` (per-env trajectories padded with 0 to the batch max; the
env worker's ``assign_history_reward`` scatters only ``history_lengths[env]``
entries per env, so padding is never read). Returns ``None`` when no env has a
ready trajectory, so the per-step reward path is a no-op and only the
done-step scatter fires (no double-count).
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


def _assign_downsampled_progress_to_chunks(
    progress, history_len: int, pickup_count: int, max_frames: int = 60
):
    """Map down-sampled robometer progress -> per-chunk reward + has_reward mask.

    Pure (no I/O). ``progress`` is the 1-D array of len = len(ds) (the down-sampled
    frame progress values, produced by ``compute_reward`` after down-sampling the
    full pick-up+insert buffer). ``history_len`` is the ORIGINAL (pre-down-sample)
    buffer length per env (= pickup_count + n_insert_chunks). ``pickup_count`` =
    number of prepended pick-up frames (insert chunks start at index pickup_count
    in the original buffer).

    Returns ``(per_chunk_reward[n_insert], per_chunk_has_reward[n_insert])`` where
    n_insert = max(0, history_len - pickup_count). For each down-sampled frame j
    whose original index is in the insert portion (>= pickup_count), the
    corresponding chunk (orig_idx - pickup_count) gets progress[j] + has_reward=True;
    chunks not in the down-sampled set get reward 0 + has_reward=False. Pick-up
    down-sampled frames (< pickup_count) are dropped (not in the rollout trajectory).

    Used by env_worker.assign_history_reward (which recomputes the SAME ds indices
    via _robometer_downsample_indices) so progress[j] <-> chunk alignment is exact.
    """
    import numpy as np

    n_insert = max(0, history_len - pickup_count)
    per_chunk_reward = np.zeros(n_insert, dtype=np.float32)
    per_chunk_has_reward = np.zeros(n_insert, dtype=bool)
    ds = _robometer_downsample_indices(history_len, max_frames)
    prog = np.asarray(progress, dtype=np.float32) if progress is not None else np.zeros(0)
    for j, orig_idx in enumerate(ds):
        if orig_idx < pickup_count:
            continue  # pick-up frame, not in rollout trajectory
        if j >= prog.shape[0]:
            break
        chunk = int(orig_idx) - pickup_count
        if 0 <= chunk < n_insert:
            per_chunk_reward[chunk] = float(prog[j])
            per_chunk_has_reward[chunk] = True
    return per_chunk_reward, per_chunk_has_reward


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
    """Per-chunk progress reward from a robometer eval server.

    Config (under ``reward.model``): ``server_url``, ``task`` (the robometer
    task string), ``timeout_s``, ``use_frame_steps``, ``success_threshold``,
    ``fail_shift`` (default 1.0), ``min_history_size`` (return None below this).
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

    def _env_success(
        self,
        env_id: int,
        success: Any,
        succ_probs_lists: list,
    ) -> Optional[bool]:
        """Test-time success (env_infos['success']) with robometer fallback."""
        if success is not None:
            try:
                s = success[env_id]
                return bool(s.item()) if hasattr(s, "item") else bool(s)
            except Exception:
                pass
        if env_id < len(succ_probs_lists) and succ_probs_lists[env_id]:
            sp = np.asarray(succ_probs_lists[env_id], dtype=np.float32)
            if sp.size:
                return float(sp[-1]) > self.success_threshold
        return None

    @torch.no_grad()
    def compute_reward(self, observations: Any) -> Optional[torch.Tensor]:
        history_input = observations.get("history_input", {}) or {}
        buf = history_input.get(self.render_buffer_name, {})
        frame_lists = buf.get("render_images", [])  # list[env] of list[frame]
        n_envs = len(frame_lists)
        if n_envs == 0:
            return None

        success = None
        env_infos = observations.get("env_infos")
        if env_infos is not None:
            success = env_infos.get("success", None)

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
        succ_section = outputs.get("outputs_success") or {}
        succ_probs_lists = succ_section.get("success_probs", []) if succ_section else []

        out = torch.zeros((n_envs, max_t), dtype=torch.float32)
        for idx, (env_id, arr, real_t) in enumerate(ready):
            t = real_t
            if idx < len(prog_lists) and prog_lists[idx]:
                prog = np.asarray(prog_lists[idx], dtype=np.float32)
                t = min(prog.shape[0], real_t)
                prog = prog[:t]
            else:
                prog = np.zeros(t, dtype=np.float32)
            env_success = self._env_success(env_id, success, succ_probs_lists)
            shift = 0.0 if env_success else self.fail_shift
            out[env_id, :t] = torch.from_numpy(prog - shift)
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
