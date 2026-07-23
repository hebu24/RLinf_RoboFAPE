# AGENTS.md

Brief for AI coding agents working on **RLinf embodied RL**. This file is scoped to robot/VLA reinforcement learning, especially ManiSkill + OpenPI pi0.5. For general contribution rules, code style, and PR process see [CONTRIBUTING.md](CONTRIBUTING.md).

**Quick orientation:** Focus on `runner.task_type: embodied`. RLinf uses **Hydra** to compose YAML configs and **Ray** to launch distributed workers. A run creates a `Cluster`, parses `cluster.component_placement`, starts actor/rollout/env/reward worker groups, then a runner drives the loop: env rollout -> reward/advantage -> actor update -> checkpoint/eval.

---

## Embodied Code Map

- **`examples/embodiment/`** - Main embodied entrypoints and configs.
  - `run_embodiment.sh` sets env vars, config path, log path, then launches training.
  - `train_embodied_agent.py` is the synchronous embodied runner entry.
  - `train_async.py` is the async embodied runner entry.
  - `config/` contains model, env, backend, and experiment YAMLs.
- **`rlinf/config.py`** - Hydra config validation. `validate_cfg(cfg)` dispatches to embodied validation when `cfg.runner.task_type == "embodied"`.
- **`rlinf/envs/`** - Robot environments and action formatting.
  - `envs/__init__.py` maps `SupportedEnvType` to env classes.
  - `envs/maniskill/` contains ManiSkill wrappers and tasks.
  - `envs/action_utils.py` converts model action chunks into env actions.
- **`rlinf/models/embodiment/`** - Embodied policies.
  - `openpi/` wraps PI/OpenPI pi0 and pi0.5 models.
  - `openvla*`, `gr00t*`, `mlp_policy`, `cnn_policy`, `flow_policy` are other embodied models.
- **`rlinf/workers/`** - Ray worker actors.
  - `actor/` trains the policy.
  - `rollout/hf/` runs HuggingFace/OpenPI rollout.
  - `env/` steps vectorized environments.
  - `reward/` computes embodied rewards when configured.
- **`rlinf/runners/`** - Training loops.
  - `embodied_runner.py` for sync embodied RL.
  - `async_embodied_runner.py` and `async_ppo_embodied_runner.py` for async variants.
- **`rlinf/scheduler/` and `rlinf/utils/placement.py`** - Cluster, worker group, and component placement.
- **`requirements/`** - Install logic.
  - `install.sh` installs model/env stacks.
  - `embodied/envs/common.txt` has shared embodied deps.
  - `embodied/models/openpi.txt` has OpenPI runtime pins.
  - `embodied/download_assets.sh` downloads ManiSkill/OpenPI assets.

---

## ManiSkill + OpenPI pi0.5 Setup

For RL training with ManiSkill and pi0.5, install the embodied target with OpenPI model support and the ManiSkill/LIBERO env bundle:

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
source .venv/bin/activate
```

Use `--use-mirror` when downloads are slow:

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero --use-mirror
```

This installs:

- common embodied deps from `requirements/embodied/envs/common.txt`
- OpenPI runtime pins from `requirements/embodied/models/openpi.txt`
- `RLinf/openpi`
- LIBERO, because the installer target is `maniskill_libero`
- ManiSkill `v3.0.0b22`
- ManiSkill assets under `$HOME/.maniskill`
- SAPIEN PhysX assets under `$HOME/.sapien/physx/...`
- OpenPI tokenizer under `$HOME/.cache/openpi`

The virtual environment defaults to `.venv/`. Override with `--venv <dir>` if needed.

---

## Main pi0.5 Configs

Use these configs for ManiSkill pi0.5:

- Sync PPO: `examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`
- Async PPO: `examples/embodiment/config/maniskill_async_ppo_openpi_pi05.yaml`
- Base policy: `examples/embodiment/config/model/pi0_5.yaml`
- ManiSkill env: `examples/embodiment/config/env/maniskill_put_on_plate_in_scene_25_main.yaml`

Important fields:

```yaml
runner:
  task_type: embodied

actor:
  model:
    model_type: openpi
    model_path: /path/to/model/RLinf-Pi05-ManiSkill-25Main-SFT
    openpi:
      config_name: pi05_maniskill

rollout:
  model:
    model_path: /path/to/model/RLinf-Pi05-ManiSkill-25Main-SFT

env:
  train:
    env_type: maniskill
```

Before running, replace both `actor.model.model_path` and `rollout.model.model_path` with the local checkpoint path, for example `RLinf-Pi05-ManiSkill-25Main-SFT`.

---

## Running Embodied RL

Single-machine flow:

```bash
ray start --head
bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05
```

The script sets:

- `EMBODIED_PATH=examples/embodiment`
- `REPO_PATH=<repo root>`
- `MUJOCO_GL=egl`
- `PYOPENGL_PLATFORM=egl`
- `PYTHONPATH=${REPO_PATH}:...`
- `ROBOT_PLATFORM`, defaulting to `LIBERO`

For ManiSkill OpenPI configs, the model-level `policy_setup: widowx_bridge` controls action formatting in `rlinf/envs/action_utils.py`.

---

## How an Embodied Run Works

1. Hydra loads the experiment YAML and all `defaults`.
2. `validate_cfg(cfg)` checks and fills the composed config.
3. `Cluster(cluster_cfg=cfg.cluster, ...)` connects to Ray and records available resources.
4. `HybridComponentPlacement(cfg, cluster)` parses `cluster.component_placement`.
5. Worker groups are created for actor, rollout, env, and optional reward.
6. The runner executes the training loop:
   - rollout worker predicts action chunks
   - env worker steps vectorized ManiSkill envs
   - rewards and advantages are computed
   - actor worker updates the policy
   - metrics/checkpoints/eval are emitted by interval

For ManiSkill pi0.5 PPO, the config typically uses:

```yaml
algorithm:
  adv_type: gae
  loss_type: actor_critic
```

Async PPO variants use an async runner and often set `loss_type: decoupled_actor_critic`.

---

## Ray, Cluster, and Placement

**Ray** is the process/runtime layer. RLinf workers are Ray remote actors.

**Cluster** is RLinf's resource abstraction over Ray. It describes node count, node groups, and available devices.

**Component placement** maps embodied components to devices:

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all
```

Examples:

```yaml
component_placement:
  actor: 0-3
  rollout: 4-7
  env: 0-7
```

Use `HybridComponentPlacement` for normal embodied training. `ModelParallelComponentPlacement` is mainly for model-parallel LLM/Megatron cases and is usually not needed for ManiSkill + pi0.5.

For multi-node embodied runs, set `RLINF_NODE_RANK` before starting Ray on every node, start Ray on head and workers, set `cluster.num_nodes`, then run the training script only on the head.

---

## Metrics, Checkpoints, and Eval

- Metrics use `runner.logger.logger_backends`, such as `tensorboard`, `wandb`, or `swanlab`.
- Common namespaces include `train/`, `eval/`, `env/`, `rollout/`, and `time/`.
- Checkpoints are saved under `runner.logger.log_path/.../checkpoints/global_step_<N>/`.
- Resume by setting `runner.resume_dir` to a checkpoint directory.
- Validation runs when `runner.val_check_interval` is reached.

For ManiSkill pi0.5, watch success metrics such as `env/success_once` or eval success fields in TensorBoard/logs.

---

## Embodied Extension Points

### Add or Modify an Embodied Model

- Register the model in `SupportedModel` in `rlinf/config.py`.
- Implement it under `rlinf/models/embodiment/<name>/`.
- Wire model construction in the relevant model factory/import path.
- Make sure actor and rollout workers can consume `cfg.actor.model` and `cfg.rollout.model`.
- Add install support in `requirements/install.sh` if extra dependencies are needed.
- Add a focused embodied config under `examples/embodiment/config/`.

### Add or Modify an Environment

- Register the env in `SupportedEnvType`.
- Add a lazy import branch in `rlinf/envs/__init__.py::get_env_cls`.
- Implement the env wrapper under `rlinf/envs/<name>/`.
- Add action conversion in `rlinf/envs/action_utils.py` if model outputs need reshaping or remapping.
- Add env YAML under `examples/embodiment/config/env/`.
- Add install/Docker/e2e coverage when dependencies or CI behavior change.

### Add or Modify an Embodied Algorithm

- Advantage functions live in `rlinf/algorithms/advantages.py`.
- Loss functions live in `rlinf/algorithms/losses.py`.
- Register via `rlinf/algorithms/registry.py`.
- Select through:

```yaml
algorithm:
  adv_type: ...
  loss_type: ...
```

Keep tests proportional to risk. For user-facing embodied behavior, add or update focused configs, docs, and unit/e2e checks when practical.

---

## Debugging Notes

- Rendering/EGL issues: check `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and GPU visibility.
- Placement issues: inspect `cluster.component_placement`, Ray node resources, and `cluster.num_nodes`.
- OpenPI path issues: confirm both actor and rollout `model_path` point to the same local checkpoint.
- Action mismatch: check `actor.model.policy_setup`, `action_dim`, `num_action_chunks`, and `prepare_actions_for_maniskill`.
- OOM: reduce `env.train.total_num_envs`, `actor.micro_batch_size`, or split actor/rollout/env placement.

---

## Style and Contribution Rules

Use Google-style Python docstrings and type hints for public APIs. Use project logging (`rlinf.utils.logging.get_logger()` or worker `self.log_*`) instead of `print`. Keep config YAML static and avoid silently overwriting user-facing config fields in code. New user-facing behavior needs tests and docs. If behavior is unclear, add `TODO(agent)` and note the limitation.
