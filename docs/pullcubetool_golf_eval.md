# PullCubeTool-golf Eval

Use the RoboFPE-aligned ManiSkill scene and policy interface: render_camera plus hand_camera, panda_wristcam, pd_joint_pos, 8D actions, and 10-step chunks.

Checkpoint path:

/data/yingxi/RLinf_RoboFAPE/logs/pullcubetool_golf_sft_pi05_wrist_20260921/pullcubetool_golf_sft_pi05_wrist/checkpoints/global_step_<N>/actor

Stable two-GPU command (GPU 0 rollout, GPU 1 Vulkan env, NUM_ENVS=1):

CHECKPOINT_PATH=<...>/global_step_25000/actor LOG_DIR=<...> GPU_IDS=0,1 NUM_EVAL_EPISODES=10 NUM_ENVS=1 MAX_EPISODE_STEPS=350 SEEDS=0-7 SAVE_VIDEO=false EVAL_RAY_PORT=6403 RAY_TMP_DIR=/tmp/rg6403 EVAL_RAY_INCLUDE_DASHBOARD=false EVAL_RAY_NUM_CPUS=32 FORCE_RESTART_RAY=true bash run_train/eval_checkpoint/run_pullcubetool_golf_wrist.sh --save-episode-metrics

RLinf FlexiblePlacementStrategy isolates rollout and env GPUs. Do not overlap eval jobs or set CUDA_VISIBLE_DEVICES manually. Each seed writes trajectory_metrics.json and evaluation_summary.json. If ErrorDeviceLost occurs or create_camera_group hangs, retry on a clean host with idle GPUs, a fresh Ray port, and a short Ray temp directory.
