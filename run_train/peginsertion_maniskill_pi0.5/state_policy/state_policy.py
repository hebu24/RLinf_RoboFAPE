#!/usr/bin/env python3
"""Shared modules for PegInsertion state-only behavior cloning."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


def build_features(peg_head_xyz: Any, hole_xyz: Any) -> torch.Tensor:
    """Build 9-D state features from peg/hole xyz coordinates.

    The feature layout is ``[peg_xyz(3), hole_xyz(3), hole_minus_peg(3)]``.
    Inputs can be numpy arrays or torch tensors with trailing shape ``(..., 3)``.
    """

    peg = _to_tensor(peg_head_xyz)
    hole = _to_tensor(hole_xyz)
    if peg.shape[-1] != 3 or hole.shape[-1] != 3:
        raise ValueError("peg_head_xyz and hole_xyz must have last-dim = 3")
    rel = hole - peg
    return torch.cat([peg, hole, rel], dim=-1)


class MLPPolicy(nn.Module):
    """Frame-wise MLP policy for predicting action xyz delta."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_size: int = 256,
        output_dim: int = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class GRUPolicy(nn.Module):
    """Sequence policy using GRU over frame features."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_size: int = 256,
        output_dim: int = 3,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del mask
        out, next_hidden = self.gru(features, hidden)
        actions = self.head(out)
        if return_hidden:
            return actions, next_hidden
        return actions


def build_policy(
    model_type: str,
    input_dim: int = 9,
    hidden_size: int = 256,
    output_dim: int = 3,
    num_layers: int = 1,
) -> nn.Module:
    """Build a policy instance from model type and shape config."""

    kind = model_type.lower()
    if kind == "mlp":
        return MLPPolicy(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=max(1, num_layers),
        )
    if kind == "gru":
        return GRUPolicy(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=max(1, num_layers),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_checkpoint_payload(
    model: nn.Module,
    model_type: str,
    config: dict[str, Any],
    normalization: dict[str, Any],
    splits: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Package checkpoint metadata and model state for saving."""

    payload: dict[str, Any] = {
        "model_type": model_type,
        "config": config,
        "normalization": normalization,
        "splits": splits,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    return payload


def load_policy_checkpoint(path: str, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load model and payload from a saved checkpoint path."""

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(payload.get("config", {}))
    model = build_policy(
        model_type=payload["model_type"],
        input_dim=int(cfg.get("input_dim", 9)),
        hidden_size=int(cfg.get("hidden_size", 256)),
        output_dim=int(cfg.get("output_dim", 3)),
        num_layers=int(cfg.get("num_layers", 1)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, payload
