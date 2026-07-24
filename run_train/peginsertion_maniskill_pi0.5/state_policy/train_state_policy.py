#!/usr/bin/env python3
"""Train state-only policy from exported replay-state parquets."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import os.path as osp
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from state_policy import build_checkpoint_payload, build_features, build_policy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_np(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _load_metadata(metadata_path: str) -> dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _accepted_episode_ids(metadata: dict[str, Any]) -> list[int]:
    manifest = list(metadata.get("manifest", []))
    ids = [int(item["compact_episode_index"]) for item in manifest if bool(item.get("accepted", False))]
    return sorted(set(ids))

def _episode_id_from_path(path: str) -> int | None:
    match = re.search(r"episode_(\d+)\.parquet$", osp.basename(path))
    if not match:
        return None
    return int(match.group(1))


def _episode_path_map(data_parquet_dir: str) -> dict[int, str]:
    patterns = [
        osp.join(data_parquet_dir, "episode_*.parquet"),
        osp.join(data_parquet_dir, "chunk-*", "episode_*.parquet"),
    ]
    out: dict[int, str] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            episode_id = _episode_id_from_path(path)
            if episode_id is not None and episode_id not in out:
                out[episode_id] = path
    return out


def _discover_dataset_layout(data_dir: str) -> tuple[str, str, list[int]]:
    metadata_path = osp.join(data_dir, "metadata.json")
    if osp.exists(metadata_path):
        metadata = _load_metadata(metadata_path)
        episode_ids = _accepted_episode_ids(metadata)
        data_parquet_dir = osp.join(data_dir, "data")
        path_map = _episode_path_map(data_parquet_dir)
        if episode_ids and path_map:
            sample_path = path_map.get(episode_ids[0])
            if sample_path is not None:
                sample_cols = set(pd.read_parquet(sample_path).columns.tolist())
                raw_required = {"observation.peg_head_pose", "observation.hole_pose", "actions"}
                if raw_required.issubset(sample_cols):
                    return "lerobot_raw_state", data_parquet_dir, episode_ids
        return "exported_state", data_parquet_dir, episode_ids

    info_path = osp.join(data_dir, "meta", "info.json")
    if osp.exists(info_path):
        data_parquet_dir = osp.join(data_dir, "data")
        path_map = _episode_path_map(data_parquet_dir)
        if not path_map:
            raise RuntimeError(f"No episode parquet files found under {data_parquet_dir}")
        sample_path = path_map[min(path_map.keys())]
        sample_cols = set(pd.read_parquet(sample_path).columns.tolist())
        required_cols = {"observation.peg_head_pose", "observation.hole_pose", "actions"}
        if required_cols.issubset(sample_cols):
            return "lerobot_raw_state", data_parquet_dir, sorted(path_map.keys())

    raise RuntimeError(
        "Unsupported dataset layout: expected metadata.json output, or meta/info.json with "
        "observation.peg_head_pose/observation.hole_pose/actions columns."
    )

def split_episodes(episode_ids: list[int], seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    ids = list(episode_ids)
    rng.shuffle(ids)
    total = len(ids)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _resolve_episode_path(data_dir: str, episode_id: int, episode_paths: dict[int, str]) -> str:
    if episode_id in episode_paths:
        return episode_paths[episode_id]
    return osp.join(data_dir, f"episode_{episode_id:06d}.parquet")


def load_episode_arrays(
    data_dir: str,
    episode_id: int,
    dataset_kind: str,
    episode_paths: dict[int, str],
) -> tuple[np.ndarray, np.ndarray]:
    path = _resolve_episode_path(data_dir, episode_id, episode_paths)
    if dataset_kind == "exported_state":
        df = pd.read_parquet(path, columns=["peg_head_pose", "box_hole_pose", "actions"])
        peg_xyz = np.stack([_to_np(value)[:3] for value in df["peg_head_pose"].to_numpy()]).astype(np.float32)
        hole_xyz = np.stack([_to_np(value)[:3] for value in df["box_hole_pose"].to_numpy()]).astype(np.float32)
    else:
        df = pd.read_parquet(path, columns=["observation.peg_head_pose", "observation.hole_pose", "actions"])
        peg_xyz = np.stack([_to_np(value)[:3] for value in df["observation.peg_head_pose"].to_numpy()]).astype(np.float32)
        hole_xyz = np.stack([_to_np(value)[:3] for value in df["observation.hole_pose"].to_numpy()]).astype(np.float32)
    features = build_features(peg_xyz, hole_xyz).detach().cpu().numpy().astype(np.float32)
    targets = np.stack([_to_np(value)[:3] for value in df["actions"].to_numpy()]).astype(np.float32)
    return features, targets
def compute_norm(train_episodes: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    feats = np.concatenate([pair[0] for pair in train_episodes], axis=0)
    tgts = np.concatenate([pair[1] for pair in train_episodes], axis=0)
    return {
        "x_mean": feats.mean(axis=0),
        "x_std": feats.std(axis=0) + 1e-6,
        "y_mean": tgts.mean(axis=0),
        "y_std": tgts.std(axis=0) + 1e-6,
    }


def normalize_episode(pair: tuple[np.ndarray, np.ndarray], norm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x, y = pair
    x_norm = (x - norm["x_mean"]) / norm["x_std"]
    y_norm = (y - norm["y_mean"]) / norm["y_std"]
    return x_norm.astype(np.float32), y_norm.astype(np.float32)

class FrameDataset(Dataset):
    def __init__(self, episodes: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.x = np.concatenate([ep[0] for ep in episodes], axis=0)
        self.y = np.concatenate([ep[1] for ep in episodes], axis=0)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


class SequenceDataset(Dataset):
    def __init__(self, episodes: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.episodes[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def collate_sequence(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    lengths = torch.tensor([item.shape[0] for item in xs], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True)
    y_pad = pad_sequence(ys, batch_first=True)
    max_len = int(x_pad.shape[1])
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return x_pad, y_pad, mask


def _huber_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    elem = nn.functional.huber_loss(pred, target, reduction="none").mean(dim=-1)
    if mask is None:
        return elem.mean()
    weights = mask.float()
    return (elem * weights).sum() / weights.sum().clamp_min(1.0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_gru: bool) -> float:
    model.eval()
    total = 0.0
    count = 0.0
    with torch.no_grad():
        for batch in loader:
            if use_gru:
                x, y, mask = batch
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                pred = model(x, mask=mask)
                loss = _huber_masked(pred, y, mask)
                total += float(loss.item()) * float(mask.sum().item())
                count += float(mask.sum().item())
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = _huber_masked(pred, y, None)
                total += float(loss.item()) * float(x.shape[0])
                count += float(x.shape[0])
    return total / max(count, 1.0)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="dataset dir (exported_state or raw LeRobot)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", choices=["mlp", "gru"], default="mlp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    dataset_kind, data_parquet_dir, episode_ids = _discover_dataset_layout(args.data_dir)
    if len(episode_ids) < 3:
        raise RuntimeError("Need at least 3 accepted episodes for 80/10/10 split.")

    splits = split_episodes(episode_ids, args.seed)
    episode_paths = _episode_path_map(data_parquet_dir)
    raw_episodes = {
        eid: load_episode_arrays(data_parquet_dir, eid, dataset_kind=dataset_kind, episode_paths=episode_paths)
        for eid in episode_ids
    }

    train_raw = [raw_episodes[eid] for eid in splits["train"]]
    norm = compute_norm(train_raw)
    norm_episodes = {eid: normalize_episode(raw_episodes[eid], norm) for eid in episode_ids}

    train_eps = [norm_episodes[eid] for eid in splits["train"]]
    val_eps = [norm_episodes[eid] for eid in splits["val"]]
    test_eps = [norm_episodes[eid] for eid in splits["test"]]

    use_gru = args.model_type == "gru"
    if use_gru:
        train_loader = DataLoader(SequenceDataset(train_eps), batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequence)
        val_loader = DataLoader(SequenceDataset(val_eps), batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequence)
        test_loader = DataLoader(SequenceDataset(test_eps), batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequence)
    else:
        train_loader = DataLoader(FrameDataset(train_eps), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(FrameDataset(val_eps), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(FrameDataset(test_eps), batch_size=args.batch_size, shuffle=False)

    model = build_policy(model_type=args.model_type, input_dim=9, hidden_size=args.hidden_size, output_dim=3, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_payload: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_count = 0.0
        for batch in train_loader:
            if use_gru:
                x, y, mask = batch
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                pred = model(x, mask=mask)
                loss = _huber_masked(pred, y, mask)
                weight = float(mask.sum().item())
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = _huber_masked(pred, y, None)
                weight = float(x.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_total += float(loss.item()) * weight
            epoch_count += weight

        train_loss = epoch_total / max(epoch_count, 1.0)
        val_loss = evaluate(model, val_loader, device, use_gru)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_payload = build_checkpoint_payload(
                model=model,
                model_type=args.model_type,
                config={
                    "input_dim": 9,
                    "hidden_size": args.hidden_size,
                    "output_dim": 3,
                    "num_layers": 1,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "seed": args.seed,
                },
                normalization={key: value.tolist() for key, value in norm.items()},
                splits={key: list(value) for key, value in splits.items()},
                extra={"best_val_loss": best_val},
            )

    if best_payload is None or best_state is None:
        raise RuntimeError("Training did not produce a valid best checkpoint.")

    model.load_state_dict(best_state)
    test_loss = evaluate(model, test_loader, device, use_gru)
    best_payload["state_dict"] = model.state_dict()

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = osp.join(args.output_dir, "best.pt")
    torch.save(best_payload, best_path)

    metrics = {
        "model_type": args.model_type,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "history": history,
        "num_episodes": {k: len(v) for k, v in splits.items()},
        "checkpoint": best_path,
    }
    metrics_path = osp.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({"best_val_loss": best_val, "test_loss": test_loss}, indent=2))
    print(f"Wrote checkpoint: {best_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()

