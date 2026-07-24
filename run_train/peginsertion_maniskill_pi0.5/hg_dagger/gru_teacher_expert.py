#!/usr/bin/env python3
"""GRU state-policy teacher for HG-DAgger.

A thin adapter around the state-based GRU teacher
(``state_policy/checkpoints/gru_hard_p95_3041_e200/best.pt``) that mirrors
``eval_state_policy.py``'s ``PolicyRunner``: load the GRU + normalization from
the checkpoint payload, then predict a 3-D xyz position delta per step,
carrying the recurrent hidden state across an episode.

The collector builds the 7-D DAgger label
(``teacher_xyz + student_executed_rot + student_executed_gripper``); this class
returns only the teacher's xyz.
"""

from __future__ import annotations

import importlib.util as ilu
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]  # .../RLinf_RoboFAPE
_DEFAULT_STATE_POLICY_PY = _REPO / "run_train/peginsertion_maniskill_pi0.5/state_policy/state_policy.py"
_DEFAULT_TEACHER_CKPT = _REPO / "run_train/peginsertion_maniskill_pi0.5/state_policy/checkpoints/gru_hard_p95_3041_e200/best.pt"


def _load_state_policy_module(path: Path):
    spec = ilu.spec_from_file_location("_state_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load state_policy module from {path}")
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detach_hidden(hidden: Any) -> Any:
    if hidden is None:
        return None
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    if isinstance(hidden, tuple):
        return tuple(_detach_hidden(h) for h in hidden)
    if isinstance(hidden, list):
        return [_detach_hidden(h) for h in hidden]
    return hidden


class GRUTeacherExpert:
    """Recurrent per-step xyz-delta teacher (GRU, 9->128->3)."""

    def __init__(
        self,
        ckpt_path: str | Path = _DEFAULT_TEACHER_CKPT,
        state_policy_py: str | Path = _DEFAULT_STATE_POLICY_PY,
        device: str = "cpu",
        action_scale: float = 1.0,
    ) -> None:
        ckpt_path = Path(ckpt_path)
        state_policy_py = Path(state_policy_py)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing teacher checkpoint: {ckpt_path}")
        if not state_policy_py.exists():
            raise FileNotFoundError(f"Missing state_policy.py: {state_policy_py}")

        self.device = torch.device(device)
        self.action_scale = float(action_scale)

        self.state_policy = _load_state_policy_module(state_policy_py)
        if not hasattr(self.state_policy, "load_policy_checkpoint"):
            raise RuntimeError("state_policy.py missing load_policy_checkpoint")
        if not hasattr(self.state_policy, "build_features"):
            raise RuntimeError("state_policy.py missing build_features")

        model, payload = self.state_policy.load_policy_checkpoint(str(ckpt_path), device=str(self.device))
        self.model = model
        self.payload = payload
        self.hidden: Any = None

        model_type = str(payload.get("model_type", "")).lower()
        if model_type not in {"mlp", "gru"}:
            raise RuntimeError(f"Unsupported model_type in checkpoint payload: {model_type}")
        self.model_type = model_type

        normalization = payload.get("normalization")
        if not isinstance(normalization, dict):
            raise RuntimeError("Checkpoint payload missing normalization dict")
        for key in ("x_mean", "x_std", "y_mean", "y_std"):
            if key not in normalization:
                raise RuntimeError(f"Checkpoint normalization missing key: {key}")

        self.x_mean = torch.as_tensor(normalization["x_mean"], dtype=torch.float32, device=self.device).reshape(1, -1)
        self.x_std = torch.as_tensor(normalization["x_std"], dtype=torch.float32, device=self.device).reshape(1, -1)
        self.y_mean = torch.as_tensor(normalization["y_mean"], dtype=torch.float32, device=self.device).reshape(1, -1)
        self.y_std = torch.as_tensor(normalization["y_std"], dtype=torch.float32, device=self.device).reshape(1, -1)

        if self.x_mean.shape[-1] != 9 or self.x_std.shape[-1] != 9:
            raise RuntimeError("x_mean/x_std must be 9-D")
        if self.y_mean.shape[-1] != 3 or self.y_std.shape[-1] != 3:
            raise RuntimeError("y_mean/y_std must be 3-D")

        self.model.to(self.device)
        self.model.eval()

    def reset_episode_hidden(self) -> None:
        """Call at every env.reset()."""
        self.hidden = None

    def _build_input(self, peg_head_xyz: Any, hole_xyz: Any) -> torch.Tensor:
        peg = torch.as_tensor(peg_head_xyz, dtype=torch.float32, device=self.device).reshape(1, -1)
        hole = torch.as_tensor(hole_xyz, dtype=torch.float32, device=self.device).reshape(1, -1)
        if peg.shape[-1] != 3 or hole.shape[-1] != 3:
            raise RuntimeError(f"peg/hole xyz must be 3-D, got {peg.shape}, {hole.shape}")
        features = self.state_policy.build_features(peg, hole)
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device).reshape(1, -1)
        if x.shape[-1] != 9:
            raise RuntimeError(f"build_features output must be [B,9], got {x.shape}")
        return (x - self.x_mean) / self.x_std

    def predict_xyz(self, peg_head_xyz: Any, hole_xyz: Any) -> np.ndarray:
        """Return the 3-D xyz position delta (denormalized, *action_scale)."""
        x_norm = self._build_input(peg_head_xyz, hole_xyz)
        with torch.no_grad():
            if self.model_type == "gru":
                x_seq = x_norm.unsqueeze(1)  # [1,1,9]
                pred_norm, next_hidden = self.model(x_seq, hidden=self.hidden, return_hidden=True)
                self.hidden = _detach_hidden(next_hidden)
                pred_norm = pred_norm[:, -1, :]
            else:
                pred_norm = self.model(x_norm)
            pred = pred_norm * self.y_std + self.y_mean
            pred = pred.reshape(-1)
            if pred.shape[0] < 3:
                raise RuntimeError(f"Policy output dim < 3, got {pred.shape}")
            return (pred[:3] * self.action_scale).detach().cpu().numpy().astype(np.float32)
