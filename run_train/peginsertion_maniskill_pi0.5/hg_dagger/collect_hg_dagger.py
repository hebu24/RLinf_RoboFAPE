#!/usr/bin/env python3
"""HG-DAgger collector for peg-insertion (pi0.5 wrist student <- GRU state teacher).

Classic DAgger: the student (pi0.5 wrist VLA) **executes its own deployed
policy** in-env to visit student-distribution states; the GRU state teacher
labels the xyz correction **every step**. The 7-D DAgger label is
``teacher_xyz[:3] + student_executed_rot[3:6] + student_executed_gripper[6]``.
No gate -- every visited frame is labeled (pure DAgger).

This is OFFLINE collection (a later merge + SFT resume step does the retraining).
It reuses the LeRobot-v2 writer + env / capture / action helpers from
``collect_hg_dagger_data`` (the controller-collector fork) and adds only:

  * student load via ``rlinf.models.get_model`` + ``predict_action_batch``
  * an ``env_obs`` builder matching ``maniskill_env._wrap_obs`` (simple + rgb)
  * the student-executes / teacher-labels loop
  * the eval-style insert-only reset (``PegInsertionLiftPlanner``)

The student queries one 10-step action chunk then executes all 10 steps
(``execute_action_chunks == action_horizon == 10``), exactly as the insert-only
eval does -- so the collector visits the same states the student visits at eval
time. There is deliberately NO solver reference, NO dry-run, NO success filter,
NO lift-end crop: DAgger aggregates ALL student-visited frames (success and
failure), labeled by the teacher.

Multiprocessing: one worker process per ``--gpu-id`` (spawn), each owns its own
env + student + teacher and writes a LeRobot-v2 shard under
``<outdir>/shard_gpu<id>``. ``merge_datasets.py`` later concatenates the shards
with the original insert-only dataset (adding the ``source`` column).

Requires ``RLINF_ROBOFPE_PATH`` (lift planner imports the solver) and a Vulkan
ICD for SAPIEN rendering. Defaults target the xulab layout.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

# Lightweight top-level imports only: the worker sets CUDA_VISIBLE_DEVICES BEFORE
# importing torch / openpi / mani_skill, so each spawned process pins exactly one
# GPU. Heavy imports happen inside ``_worker``.
_HERE = Path(__file__).resolve().parent  # .../run_train/peginsertion_maniskill_pi0.5/hg_dagger
_REPO = _HERE.parents[2]  # .../RLinf_RoboFAPE
# Sibling imports (collect_hg_dagger_data fork, gru_teacher_expert) live in this
# dir; spawned workers do not inherit cwd, so put it on sys.path explicitly.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# xulab solver path (lift planner imports solutions.solve_PegInsertionVertical).
# Code default in the planner is the H100 path, which is wrong on xulab.
os.environ.setdefault("RLINF_ROBOFPE_PATH", "/home/yingxi/RoboFAC/mani_envs")

mp.set_start_method("spawn", force=True)

_DEFAULT_STUDENT_CKPT = (
    _REPO
    / "logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor"
)
_DEFAULT_TEACHER_CKPT = (
    _REPO
    / "run_train/peginsertion_maniskill_pi0.5/state_policy/checkpoints/gru_hard_p95_3041_e200/best.pt"
)
_DEFAULT_CONFIG_NAME = "pi05_maniskill_peg_insertion_wrist"


def _load_student(student_ckpt: str, config_name: str, device: str):
    """Construct the pi0.5 student standalone (no Hydra).

    ``get_model`` reads ``cfg.model_path`` + ``cfg.openpi.config_name`` (+ optional
    ``cfg.openapi_data``). With ``model_path = <ckpt>/actor`` it finds
    ``actor/model_state_dict/full_weights.pt`` (direct-checkpoint branch) and
    ``actor/physical-intelligence/maniskill/norm_stats.json`` (via
    ``load_norm_stats``), so the student loads with the SAME norm stats it was
    trained with -- matching eval.
    """
    from omegaconf import OmegaConf

    # Bypass the top-level dispatcher (needs model_type/precision/is_lora +
    # Worker.torch_platform); call the OpenPI embodiment get_model directly.
    from rlinf.models.embodiment.openpi import get_model
    import torch

    if not Path(student_ckpt).exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_ckpt}")
    cfg = OmegaConf.create(
        {
            "model_path": str(student_ckpt),
            "openpi": {"config_name": config_name},
            "openpi_data": None,
        }
    )
    model = get_model(cfg)
    model = model.to(device)
    model.eval()
    return model


def _build_env_obs(obs: dict[str, Any], state_tcp, task: str, device: str) -> dict[str, Any]:
    """Build the env_obs dict ``predict_action_batch`` expects (B=1).

    Mirrors ``maniskill_env._wrap_obs`` (wrap_obs_mode="simple", obs_mode="rgb"):
    ``main_images`` / ``wrist_images`` are ``[1, H, W, 3]`` uint8 tensors,
    ``states`` is the aligned 8-D pi0.5 proprio (== ``observation.state_tcp``,
    the SFT state input), ``task_descriptions`` is a length-B list of strings.
    The wrist config uses base + wrist only (no wrist_back / extra_view), so
    those channels are ``None``.
    """
    import torch

    sensor = obs["sensor_data"]
    main_images = torch.as_tensor(sensor["base_camera"]["rgb"], device=device)
    wrist_images = torch.as_tensor(sensor["hand_camera"]["rgb"], device=device)
    states = (
        torch.as_tensor(state_tcp, dtype=torch.float32, device=device)
        .reshape(1, -1)
    )
    return {
        "main_images": main_images,
        "wrist_images": wrist_images,
        "wrist_back_images": None,
        "extra_view_images": None,
        "states": states,
        "task_descriptions": [task],
    }


def _insert_only_reset(env, planner, episode_seed: int):
    """Eval-style insert-only reset: pre-grasped peg at a planner-lifted pose.

    ``plan_lifted_states([0])`` returns a DIFFERENT lifted state each call
    (the planner increments its internal episode counter -> per-episode reset
    variety), matching the insert-only eval loop.
    """
    import collect_hg_dagger_data as ctl  # local import (fork helper lib)

    planned = planner.plan_lifted_states([0])
    reset_result = env.reset(
        seed=episode_seed,
        options={
            "pre_grasped": True,
            "randomize_initial_poses": False,
            "robot_qpos": planned["robot_qpos"],
            "peg_pose": planned["peg_pose"],
            "hole_pose": planned["hole_pose"],
        },
    )
    obs = reset_result[0]
    ctl._sync_target_delta_pose_controller(env)
    reset_state = ctl._as_numpy(env.unwrapped.get_state())[0].astype(
        "float32"
    )
    return obs, reset_state


def _collect_hg_dagger_episode(
    env,
    student,
    teacher,
    *,
    episode_seed: int,
    task: str,
    max_steps: int,
    action_scale: float,
    device: str,
    planner,
) -> tuple[bool, list[dict[str, Any]], "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Run one student-executes / teacher-labels episode.

    Returns ``(success, rows, base_frames, wrist_frames, wrist_back_frames,
    render_frames)`` matching ``LeRobotControllerWriter.add_episode``'s schema.
    """
    import numpy as np
    import torch
    import collect_hg_dagger_data as ctl

    obs, reset_state = _insert_only_reset(env, planner, episode_seed)
    teacher.reset_episode_hidden()

    rows: list[dict[str, Any]] = []
    base_frames, wrist_frames, wrist_back_frames, render_frames = [], [], [], []
    steps = 0
    episode_done = False
    while steps < max_steps and not episode_done:
        # Query the student from the current (chunk-start) observation. The
        # student returns a full [1, action_horizon=10, 7] chunk; we execute all
        # 10 steps (execute_action_chunks == action_horizon == 10, as in eval).
        state_tcp = ctl._capture_state(env)["state_tcp"]
        env_obs = _build_env_obs(obs, state_tcp, task, device)
        with torch.no_grad():
            actions_chunk, _ = student.predict_action_batch(
                env_obs=env_obs, mode="eval"
            )
        chunk = actions_chunk[0].detach().cpu().numpy().astype("float32")  # [10, 7]

        for t in range(chunk.shape[0]):
            if steps >= max_steps:
                break
            pre = ctl._capture_state(env)
            images = ctl._images_from_obs(env, obs)
            student_action_t = chunk[t]  # physical target-delta [dx,dy,dz,droll,dpitch,dyaw,gripper]

            # Teacher labels the xyz correction from the pre-step peg/hole (the
            # SAME world-frame positions the GRU was trained on via eval_state_policy).
            peg_head_xyz = pre["peg_head_pose"][:3]
            hole_xyz = pre["hole_pose"][:3]
            teacher_xyz = teacher.predict_xyz(peg_head_xyz, hole_xyz)  # 3-D

            # 7-D DAgger label: teacher xyz + student's EXECUTED rot + gripper.
            label_action = np.concatenate(
                [teacher_xyz, student_action_t[3:7]]
            ).astype("float32")

            # Execute the STUDENT's full action (visits student states).
            env_action = ctl._raw_action_to_env_action(student_action_t)
            step_obs, _reward, terminated, truncated, _info = env.step(
                env_action.reshape(1, -1)
            )
            post = ctl._capture_state(env)

            rows.append(
                {
                    "actions": label_action,
                    "debug.env_action": env_action,
                    # DAgger has no planned reference target; record the achieved
                    # next TCP (diagnostic only -- not a model input).
                    "debug.ref_next_tcp": post["tcp_matrix_root"]
                    .reshape(-1)
                    .astype("float32"),
                    "debug.tcp_before": pre["tcp_matrix_root"]
                    .reshape(-1)
                    .astype("float32"),
                    "debug.tcp_after": post["tcp_matrix_root"]
                    .reshape(-1)
                    .astype("float32"),
                    "observation.state": pre["state"],
                    "observation.state_tcp": pre["state_tcp"],
                    "observation.peg_pose": pre["peg_pose"],
                    "observation.peg_head_pose": pre["peg_head_pose"],
                    "observation.hole_pose": pre["hole_pose"],
                    "episode_reset_state": reset_state,
                }
            )
            base_frames.append(images["base_camera_rgb"])
            wrist_frames.append(images["hand_camera_rgb"])
            wrist_back_frames.append(images["hand_camera_back_rgb"])
            render_frames.append(images["render_rgb"])

            obs = step_obs
            steps += 1
            if bool(terminated) or bool(truncated):
                episode_done = True
                break

    success = False
    try:
        success = bool(env.unwrapped.evaluate()["success"].detach().cpu().numpy()[0])
    except Exception:
        pass
    if not rows:
        return success, rows, None, None, None, None  # type: ignore[return-value]
    return (
        success,
        rows,
        np.stack(base_frames),
        np.stack(wrist_frames),
        np.stack(wrist_back_frames),
        np.stack(render_frames),
    )


def _worker(
    gpu_id: int,
    *,
    student_ckpt: str,
    teacher_ckpt: str,
    config_name: str,
    outdir: str,
    num_traj: int,
    base_seed: int,
    max_steps: int,
    action_scale: float,
    render_backend_prefix: str,
) -> None:
    """Per-GPU collection worker (spawned). Pins one GPU, owns env+student+teacher."""
    # Pin this GPU BEFORE importing torch / mani_skill / openpi.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import numpy as np  # noqa: F401  (ensure importable in spawned process)
    import torch
    import torch.multiprocessing as tmp  # noqa: F401

    # The fork is a sibling module; import it after CUDA is pinned.
    import collect_hg_dagger_data as ctl
    from rlinf.envs.maniskill.peg_insertion_lift_planner import (
        PegInsertionLiftPlanner,
    )

    from gru_teacher_expert import GRUTeacherExpert

    device = "cuda:0"  # == physical gpu_id after CUDA_VISIBLE_DEVICES pin
    gpu_label = f"gpu{gpu_id}"
    shard_dir = os.path.join(outdir, f"shard_gpu{gpu_id}")
    os.makedirs(shard_dir, exist_ok=True)

    print(f"[{gpu_label}] loading student ({student_ckpt}) on {device} ...", flush=True)
    student = _load_student(student_ckpt, config_name, device)
    print(f"[{gpu_label}] loading teacher ({teacher_ckpt}) ...", flush=True)
    teacher = GRUTeacherExpert(
        ckpt_path=teacher_ckpt, action_scale=action_scale, device=device
    )
    print(f"[{gpu_label}] making capture env (pd_ee_target_delta_pose, rgb) ...", flush=True)
    # CUDA_VISIBLE_DEVICES=<gpu_id> pins one physical GPU as local cuda:0. SAPIEN's
    # render device must be the LOCAL id (cuda:0 == physical gpu_id), not the
    # physical id -- Device("cuda:<physical>") is not found under pinning. This
    # mirrors the fork's _runtime_gpu_ids local-id remap (local 0, prefix "cuda").
    env = ctl._make_env_with_retry(
        "pd_ee_target_delta_pose",
        0,
        render_backend_prefix,
        capture_images=True,
    )
    planner = PegInsertionLiftPlanner(base_seed=base_seed)
    writer = ctl.LeRobotControllerWriter(shard_dir, resume=False)
    writer.collect_mode = "insert_only"

    # Insert-only prompt pool (matches the original insert-only dataset's wording
    # so the SFT prompt distribution stays consistent).
    task_pool = ctl.generate_insert_only_prompts(
        ctl.DEFAULT_NUM_PROMPTS, ctl.DEFAULT_PROMPT_SEED
    )

    passed = 0
    try:
        for ep in range(num_traj):
            episode_seed = int(base_seed + ep)
            task = task_pool[ep % len(task_pool)]
            t0 = time.time()
            try:
                (
                    success,
                    rows,
                    base,
                    wrist,
                    wrist_back,
                    render,
                ) = _collect_hg_dagger_episode(
                    env=env,
                    student=student,
                    teacher=teacher,
                    episode_seed=episode_seed,
                    task=task,
                    max_steps=max_steps,
                    action_scale=action_scale,
                    device=device,
                    planner=planner,
                )
            except Exception as exc:  # noqa: BLE001
                writer.log_attempt(
                    {
                        "gpu_id": gpu_id,
                        "seed": episode_seed,
                        "status": "episode_error",
                        "error": str(exc),
                    }
                )
                print(f"[{gpu_label}] ep={ep} seed={episode_seed} ERROR: {exc}", flush=True)
                continue
            if not rows:
                writer.log_attempt(
                    {"gpu_id": gpu_id, "seed": episode_seed, "status": "empty_episode"}
                )
                print(f"[{gpu_label}] ep={ep} seed={episode_seed} EMPTY", flush=True)
                continue
            writer.add_episode(rows, base, wrist, wrist_back, render, task)
            passed += 1
            dur = time.time() - t0
            print(
                f"[{gpu_label}] ep={ep} seed={episode_seed} frames={len(rows)} "
                f"success={int(success)} {dur:.1f}s task={task!r}",
                flush=True,
            )
            writer.log_attempt(
                {
                    "gpu_id": gpu_id,
                    "seed": episode_seed,
                    "status": "ok",
                    "frames": len(rows),
                    "success": bool(success),
                    "time_sec": dur,
                    "task": task,
                }
            )
    finally:
        try:
            writer.finalize()
        except Exception as exc:  # noqa: BLE001
            print(f"[{gpu_label}] writer.finalize() failed: {exc}", flush=True)
        try:
            planner.close()
        except Exception:
            pass
        print(f"[{gpu_label}] done: passed={passed}/{num_traj} shard={shard_dir}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HG-DAgger collector (pi0.5 wrist student <- GRU state teacher).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--student-ckpt", default=str(_DEFAULT_STUDENT_CKPT))
    parser.add_argument("--teacher-ckpt", default=str(_DEFAULT_TEACHER_CKPT))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--num-traj", type=int, default=10, help="TOTAL episodes across all GPUs")
    parser.add_argument("--gpu-ids", default="2", help="comma-separated physical GPU ids")
    parser.add_argument("--config-name", default=_DEFAULT_CONFIG_NAME)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--render-backend-prefix",
        default="cuda",
        help="passed to the fork's _render_backend_from_gpu_id; 'cuda' renders on "
        "local cuda:0 (== the pinned physical gpu_id). Use 'cpu' for CPU render.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    num_gpus = len(gpu_ids)
    # Split total num_traj across workers; remainder goes to the last worker.
    base = args.num_traj // num_gpus
    rem = args.num_traj - base * num_gpus

    os.makedirs(args.outdir, exist_ok=True)
    print(
        f"[main] student={args.student_ckpt}\n[main] teacher={args.teacher_ckpt}\n"
        f"[main] outdir={args.outdir} gpus={gpu_ids} total_traj={args.num_traj}",
        flush=True,
    )

    procs: list[mp.Process] = []
    for idx, gpu_id in enumerate(gpu_ids):
        worker_traj = base + (rem if idx == num_gpus - 1 else 0)
        if worker_traj <= 0:
            continue
        p = mp.Process(
            target=_worker,
            args=(gpu_id,),
            kwargs={
                "student_ckpt": args.student_ckpt,
                "teacher_ckpt": args.teacher_ckpt,
                "config_name": args.config_name,
                "outdir": args.outdir,
                "num_traj": worker_traj,
                # Disjoint per-worker seeds. PegInsertionLiftPlanner computes
                # base_seed * 1_000_003 internally, so base_seed MUST stay < ~4294
                # (else the solver seed overflows 2**32 -> "Seed must be between 0
                # and 2**32-1"). Use a small offset + mod 4000 as a hard cap.
                "base_seed": (args.base_seed + idx * 100) % 4000,
                "max_steps": args.max_episode_steps,
                "action_scale": args.action_scale,
                "render_backend_prefix": args.render_backend_prefix,
            },
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    print("[main] all workers finished", flush=True)


if __name__ == "__main__":
    main()
