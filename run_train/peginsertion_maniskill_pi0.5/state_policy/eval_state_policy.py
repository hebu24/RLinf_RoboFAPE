#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_NVIDIA_VK_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
if "VK_ICD_FILENAMES" not in os.environ and os.path.exists(_NVIDIA_VK_ICD):
    os.environ["VK_ICD_FILENAMES"] = _NVIDIA_VK_ICD
_SYSTEM_VULKAN_LOADER = "/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
if "SAPIEN_VULKAN_LIBRARY_PATH" not in os.environ and os.path.exists(_SYSTEM_VULKAN_LOADER):
    os.environ["SAPIEN_VULKAN_LIBRARY_PATH"] = _SYSTEM_VULKAN_LOADER

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.wrappers import RecordEpisode  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PATH = SCRIPT_DIR.parents[2]
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

TASK_PATH = REPO_PATH / "rlinf" / "envs" / "maniskill" / "tasks" / "peg_insertion_vertical.py"
task_spec = ilu.spec_from_file_location("_peg_insertion_vertical", TASK_PATH)
if task_spec is None or task_spec.loader is None:
    raise RuntimeError(f"Failed to load env task file: {TASK_PATH}")
task_module = ilu.module_from_spec(task_spec)
task_spec.loader.exec_module(task_module)
_ = task_module

from rlinf.envs.maniskill.peg_insertion_lift_planner import (  # noqa: E402
    PegInsertionLiftPlanner,
)
from rlinf.envs.maniskill.peg_insertion_pi05 import (  # noqa: E402
    model_action_to_panda_env_action,
)


@dataclass
class EpisodeResult:
    episode_index: int
    seed: int
    success: bool
    steps: int
    final_relative_pose: dict[str, Any]


def _as_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _arm_controller(env):
    controller = env.unwrapped.agent.controller
    if hasattr(controller, "controllers"):
        return controller.controllers.get("arm", controller)
    return controller


def _sync_target_delta_pose_controller(env) -> None:
    arm_controller = _arm_controller(env)
    config = getattr(arm_controller, "config", None)
    if bool(getattr(config, "use_target", False)) and hasattr(arm_controller, "_target_pose"):
        arm_controller._target_pose = arm_controller.ee_pose_at_base


def _make_env(
    video_dir: Path | None = None,
    video_fps: int = 20,
    render_backend: str | None = None,
):
    capture_video = video_dir is not None
    env_kwargs = {
        "num_envs": 1,
        "obs_mode": "rgb" if capture_video else "state",
        "render_mode": "rgb_array" if capture_video else None,
        "robot_uids": "panda_wristcam",
        "control_mode": "pd_ee_target_delta_pose",
        "sim_backend": "cpu",
        "reward_mode": "normalized_dense",
        "max_episode_steps": 600,
    }
    if capture_video:
        env_kwargs.update(
            sensor_configs={"shader_pack": "default"},
            human_render_camera_configs={"shader_pack": "default"},
        )
        if render_backend:
            env_kwargs["render_backend"] = render_backend
    env = gym.make("PegInsertionVertical-v1", **env_kwargs)
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=str(video_dir),
            save_trajectory=False,
            save_video=True,
            save_on_reset=False,
            clean_on_close=False,
            video_fps=video_fps,
        )
    return env


def _load_policy_module(path: Path):
    spec = ilu.spec_from_file_location("_state_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detach_hidden(hidden: Any) -> Any:
    if hidden is None:
        return None
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    if isinstance(hidden, tuple):
        return tuple(_detach_hidden(item) for item in hidden)
    if isinstance(hidden, list):
        return [_detach_hidden(item) for item in hidden]
    return hidden


def _to_xyz_tensor(value: Any, device: torch.device, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, -1)
    if tensor.shape[-1] != 3:
        raise RuntimeError(f"{name} must be 3-D xyz, got shape={tuple(tensor.shape)}")
    return tensor


class PolicyRunner:
    def __init__(
        self,
        state_policy: Any,
        model: torch.nn.Module,
        payload: dict[str, Any],
        device: torch.device,
        action_scale: float,
    ) -> None:
        self.state_policy = state_policy
        self.model = model
        self.payload = payload
        self.device = device
        self.action_scale = float(action_scale)
        self.hidden: Any = None

        model_type = str(payload.get("model_type", "")).lower()
        if model_type not in {"mlp", "gru"}:
            raise RuntimeError(f"Unsupported model_type in checkpoint payload: {model_type}")
        self.model_type = model_type

        normalization = payload.get("normalization")
        if not isinstance(normalization, dict):
            raise RuntimeError("Checkpoint payload missing normalization dict")

        required_keys = ("x_mean", "x_std", "y_mean", "y_std")
        for key in required_keys:
            if key not in normalization:
                raise RuntimeError(f"Checkpoint payload normalization missing key: {key}")

        self.x_mean = torch.as_tensor(normalization["x_mean"], dtype=torch.float32, device=device).reshape(1, -1)
        self.x_std = torch.as_tensor(normalization["x_std"], dtype=torch.float32, device=device).reshape(1, -1)
        self.y_mean = torch.as_tensor(normalization["y_mean"], dtype=torch.float32, device=device).reshape(1, -1)
        self.y_std = torch.as_tensor(normalization["y_std"], dtype=torch.float32, device=device).reshape(1, -1)

        if self.x_mean.shape[-1] != 9 or self.x_std.shape[-1] != 9:
            raise RuntimeError("x_mean/x_std must be 9-D to match build_features output")
        if self.y_mean.shape[-1] != 3 or self.y_std.shape[-1] != 3:
            raise RuntimeError("y_mean/y_std must be 3-D")

        self.model.to(device)
        self.model.eval()

    def reset_episode_hidden(self) -> None:
        self.hidden = None

    def _build_input(self, peg_head_xyz: Any, hole_xyz: Any) -> torch.Tensor:
        peg = _to_xyz_tensor(peg_head_xyz, self.device, "peg_head_xyz")
        hole = _to_xyz_tensor(hole_xyz, self.device, "hole_xyz")
        features = self.state_policy.build_features(peg, hole)
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device).reshape(1, -1)
        if x.shape[-1] != 9:
            raise RuntimeError(f"build_features output must be [B,9], got shape={tuple(x.shape)}")
        return (x - self.x_mean) / self.x_std

    def predict_xyz_delta(self, peg_head_xyz: Any, hole_xyz: Any) -> np.ndarray:
        x_norm = self._build_input(peg_head_xyz, hole_xyz)
        with torch.no_grad():
            if self.model_type == "gru":
                x_seq = x_norm.unsqueeze(1)
                pred_norm, next_hidden = self.model(x_seq, hidden=self.hidden, return_hidden=True)
                self.hidden = _detach_hidden(next_hidden)
                pred_norm = pred_norm[:, -1, :]
            else:
                pred_norm = self.model(x_norm)

            pred = pred_norm * self.y_std + self.y_mean
            pred = pred.reshape(-1)
            if pred.shape[0] < 3:
                raise RuntimeError(f"Policy output dim < 3, got shape={tuple(pred.shape)}")
            return (pred[:3] * self.action_scale).detach().cpu().numpy().astype(np.float32)


def _load_runner(policy_py: Path, ckpt: Path, device: torch.device, action_scale: float) -> PolicyRunner:
    if not policy_py.exists():
        raise FileNotFoundError(f"Missing state policy definition: {policy_py}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    state_policy = _load_policy_module(policy_py)
    if not hasattr(state_policy, "load_policy_checkpoint"):
        raise RuntimeError("state_policy.py missing load_policy_checkpoint")
    if not hasattr(state_policy, "build_features"):
        raise RuntimeError("state_policy.py missing build_features")

    model, payload = state_policy.load_policy_checkpoint(str(ckpt), device=str(device))
    if not isinstance(payload, dict):
        raise RuntimeError("load_policy_checkpoint must return (model, payload_dict)")

    return PolicyRunner(
        state_policy=state_policy,
        model=model,
        payload=payload,
        device=device,
        action_scale=action_scale,
    )


def _get_peg_hole_xyz(env) -> tuple[np.ndarray, np.ndarray]:
    peg_head_xyz = _as_np(env.unwrapped.peg_head_pose.p).reshape(-1, 3)[0].astype(np.float32)
    hole_xyz = _as_np(env.unwrapped.box_hole_pose.p).reshape(-1, 3)[0].astype(np.float32)
    return peg_head_xyz, hole_xyz


def _evaluate_episode(env, runner: PolicyRunner, episode_index: int, seed: int, max_steps: int) -> EpisodeResult:
    runner.reset_episode_hidden()
    eval_info = env.unwrapped.evaluate()
    success = bool(_as_np(eval_info["success"]).reshape(-1)[0])
    steps = 0

    for step in range(max_steps):
        if success:
            break
        peg_head_xyz, hole_xyz = _get_peg_hole_xyz(env)
        xyz_delta = runner.predict_xyz_delta(peg_head_xyz=peg_head_xyz, hole_xyz=hole_xyz)

        raw_action = np.zeros(7, dtype=np.float32)
        raw_action[:3] = xyz_delta
        raw_action[6] = -1.0
        env_action = model_action_to_panda_env_action(raw_action)
        env.step(env_action.reshape(1, -1))

        eval_info = env.unwrapped.evaluate()
        success = bool(_as_np(eval_info["success"]).reshape(-1)[0])
        steps = step + 1

    peg_head_xyz, hole_xyz = _get_peg_hole_xyz(env)
    rel_world = (peg_head_xyz - hole_xyz).astype(np.float32)
    eval_info = env.unwrapped.evaluate()
    rel_hole = None
    if "peg_head_pos_at_hole" in eval_info:
        rel_hole = _as_np(eval_info["peg_head_pos_at_hole"]).reshape(-1, 3)[0].astype(np.float32)

    final_relative_pose = {
        "peg_head_minus_hole_xyz": rel_world.astype(float).tolist(),
        "peg_head_xyz": peg_head_xyz.astype(float).tolist(),
        "hole_xyz": hole_xyz.astype(float).tolist(),
        "peg_head_pos_at_hole": rel_hole.astype(float).tolist() if rel_hole is not None else None,
    }

    return EpisodeResult(
        episode_index=episode_index,
        seed=seed,
        success=success,
        steps=steps,
        final_relative_pose=final_relative_pose,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str((SCRIPT_DIR / "best.pt").resolve()))
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--render-backend", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA unavailable, fallback to CPU from {args.device}.")
        device = torch.device("cpu")

    policy_py = SCRIPT_DIR / "state_policy.py"
    ckpt = Path(args.checkpoint).expanduser().resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else (SCRIPT_DIR / "eval_state_policy.json").resolve()
    video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else None

    runner = _load_runner(policy_py=policy_py, ckpt=ckpt, device=device, action_scale=args.action_scale)
    env = _make_env(
        video_dir=video_dir,
        video_fps=args.video_fps,
        render_backend=args.render_backend,
    )
    planner = PegInsertionLiftPlanner(base_seed=args.seed)
    results: list[EpisodeResult] = []
    try:
        for ep in range(args.num_episodes):
            episode_seed = int(args.seed + ep)
            planned = planner.plan_lifted_states([0])
            env.reset(
                seed=episode_seed,
                options={
                    "pre_grasped": True,
                    "randomize_initial_poses": False,
                    "robot_qpos": planned["robot_qpos"],
                    "peg_pose": planned["peg_pose"],
                    "hole_pose": planned["hole_pose"],
                },
            )
            _sync_target_delta_pose_controller(env)
            res = _evaluate_episode(
                env=env,
                runner=runner,
                episode_index=ep,
                seed=episode_seed,
                max_steps=args.max_steps,
            )
            results.append(res)
            if video_dir is not None:
                status = "success" if res.success else "failure"
                env.flush_video(name=f"episode_{ep:03d}_seed_{episode_seed}_{status}")
            rel = res.final_relative_pose["peg_head_minus_hole_xyz"]
            print(f"episode={ep:03d} success={int(res.success)} steps={res.steps} rel={rel}")
    finally:
        planner.close()
        env.close()

    success_rate = float(np.mean([r.success for r in results])) if results else 0.0
    report = {
        "checkpoint": str(ckpt),
        "state_policy_py": str(policy_py),
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "device": str(device),
        "action_scale": float(args.action_scale),
        "video_dir": str(video_dir) if video_dir is not None else None,
        "video_fps": int(args.video_fps) if video_dir is not None else None,
        "render_backend": args.render_backend if video_dir is not None else None,
        "success_rate": success_rate,
        "episodes": [asdict(r) for r in results],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"success_rate": success_rate, "episodes": len(results)}, indent=2))
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
