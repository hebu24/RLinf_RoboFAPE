# Ray Isolation — concurrent SFT + eval on one host

**Audience: all agents (and humans) launching SFT training or eval on this box.**
Host: `xulab` (`/data/yingxi/RLinf_RoboFAPE`), multi-GPU H100, shared by multiple jobs.

## TL;DR — the 5 rules

1. **Never bare `ray stop` / `ray stop --force`.** `ray stop` has **no** `--port`/`--address` flag — it kills **all** Ray on the host, including the other guy's training. To tear down one cluster, use the **scoped pkill by GCS port** (below).
2. **Each Ray cluster needs 3 DISTINCT things**: GCS port (`--port`), dashboard-agent port (`--dashboard-agent-listen-port`), and temp-dir (`--temp-dir`). These are the only fixed (non-random) ports Ray uses; collide on any one and a raylet crashes (`boost::beast::http` stack in `raylet.out`). All other Ray ports default to 0 (random) and are safe.
3. **Pin driver + workers with `RAY_ADDRESS=127.0.0.1:<port>`.** RLinf hardcodes `ray.init(address="auto")`; with `RAY_ADDRESS` unset, Ray's `find_gcs_addresses()` greps `ps` for *any* GCS and connects to whichever — so an SFT worker can attach to the eval cluster. `RAY_ADDRESS` overrides that.
4. **Disjoint GPUs.** Ray isolation ≠ GPU memory isolation. Two clusters on the same GPU share HBM and OOM each other. SFT on one half, eval/other-SFT on the other half.
5. **No bare `ray stop` in any script that may run concurrently.** The fixed scripts below are safe; the legacy `run_train/test_maniskill_pi0.5/run.sh` (PutOnPlate PPO entry) still does bare `ray stop` — don't run it while training is up.

## Default port assignments (this box)

| job | script | GCS port | dashboard-agent | temp-dir | GPUs |
|---|---|---|---|---|---|
| wrist SFT | `sft_finetune_pi05base.sh` (CONFIG=..._wrist) | `6379` (`SFT_RAY_PORT`) | `52366` (`SFT_DASHBOARD_AGENT_PORT`) | `/tmp/ray_sft_6379` | 4-7 |
| eval sweep | `sweep_peginsertion_wrist.py` | `6380` (`--ray-port`) | `52365` (Ray default) | `/tmp/ray_eval_wrist_sweep_<pid>` | 0-3 |
| single ckpt eval | `run_peginsertion*.sh` (`MANAGE_RAY=true`) | `6380` (`EVAL_RAY_PORT`) | `52365` | `/tmp/ray_eval_wrist` | per `GPU_IDS` |

Both GCS ports (6379/6380) and the non-default dashboard-agent port (52366) are distinct → these can run at once. **Two eval sweeps at once would collide on 6380 + 52365** — don't; give the second `--ray-port 6390` and a distinct dashboard-agent port.

## The fixed scripts (what makes them safe)

- **`sft_finetune*.sh`** (all variants): sets `SFT_RAY_PORT` (default 6379), `RAY_TMPDIR=/tmp/ray_sft_${SFT_RAY_PORT}` (per-port), `SFT_DASHBOARD_AGENT_PORT` (default 52366), `RAY_ADDRESS=127.0.0.1:${SFT_RAY_PORT}`; `ray start --head --port --temp-dir --dashboard-agent-listen-port`; `_sft_scoped_ray_kill()` pkill by **port** (gcs_server `--gcs_server_port=P`, raylet/dashboard `--gcs-address=...:P`) + `sleep 2`; EXIT trap scoped to own port. No bare `ray stop`.
- **`run_train/eval_checkpoint/sweep_peginsertion_wrist.py`**: `--ray-port` (default 6380), `--object-store-memory`; `_scoped_ray_kill(port)`; `start_shared_ray` pops `RAY_ADDRESS`, starts head with `--include-dashboard=false`, sets `RAY_ADDRESS`; per-checkpoint subprocesses get `RAY_ADDRESS` in env + `MANAGE_RAY=false`. No bare `ray stop`.
- **`run_train/eval_checkpoint/run_peginsertion*.sh`** (single-eval): `EVAL_RAY_PORT` (default 6380), `RAY_ADDRESS` pin, `MANAGE_RAY=true` starts a head with `--port --include-dashboard=false` + scoped EXIT trap. No bare `ray stop`.

## How to run concurrent jobs (copy-paste)

Run each in a detached tmux session so it survives SSH disconnect. Launch order does not matter (ports are distinct).

```bash
# wrist SFT — 4-7, port 6379
tmux new-session -d -s sft_wrist "cd /data/yingxi/RLinf_RoboFAPE && \
  PYTHONUNBUFFERED=1 bash run_sft_insert_wrist_v2.sh 2>&1 | tee logs/sft_insert_wrist_v2_tmux.log"

# eval sweep — 0-3, port 6380 (run when GPUs 0-3 are free)
tmux new-session -d -s eval_sweep "cd /data/yingxi/RLinf_RoboFAPE && \
  MPLCONFIGDIR=/tmp/matplotlib /data/yingxi/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
    --ray-port 6380 --run-script run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh \
    --checkpoint-dir <.../checkpoints> --output-dir <...> \
    --num-eval-episodes 10 --num-envs 1 --gpu-ids 0,1,2,3 --action-scale 1.0 \
    --resume --continue-on-error 2>&1 | tee logs/eval_sweep_insert_only_tmux.log"
```

Attach/monitor:
```bash
tmux ls
tmux attach -t sft_wrist      # detach: Ctrl-b d
tail -f logs/sft_insert_wrist_v2_tmux.log
```

## How to test / debug

**List running clusters + ports:**
```bash
pgrep -fa gcs_server | grep -oE 'gcs_server_port=[0-9]+' | sort -u     # one line per cluster
ss -tlnp | grep -E '6379|6380|52365|52366'                  # who binds what
```

**Confirm two jobs aren't killing each other** (both gcs stay up across an eval launch):
```bash
watch -n5 'pgrep -fa gcs_server | grep -oE "gcs_server_port=[0-9]+" | sort -u'
```

**Scoped teardown of ONE cluster (never bare `ray stop`):** replace `P` with the cluster's GCS port.
```bash
P=6379
pkill -9 -f "gcs_server.*--gcs_server_port=${P}"  || true
pkill -9 -f "raylet.*--gcs-address=[^ ]*:${P}"     || true
pkill -9 -f "dashboard.*--gcs-address=[^ ]*:${P}" || true   # dashboard server + agent both carry --gcs-address=:P
sleep 2
```
The `[^ ]*` anchors the port to the gcs-address value, so a different port can never match.

**Why did a raylet crash?** `boost::beast::http` / "node marked dead" in `raylet.out` almost always = a fixed-port collision (dashboard-agent 52365 is the usual one) or OOM. Check:
```bash
ls -t /tmp/ray_sft_<P>/ray/session_*/logs/raylet.err | head -1 | xargs tail -30
dmesg 2>/dev/null | grep -iE 'out of memory|killed process|oom' | tail
```

**"Failed to connect to GCS at <ip>:<port>"**: the head on that port is dead (killed by a bare `ray stop`, or crashed). Find who did bare `ray stop` (`history`/logs) and stop doing that; the fixed scripts never do.

## Common pitfalls (read before touching Ray)

1. **Bare `ray stop`** — kills every cluster on the host. The #1 cause of "eval killed my training". Forbidden while >1 job is up.
2. **`--dashboard-agent-listen-port` defaults to fixed 52365** — two heads both want 52365 → the second raylet crashes in its HTTP loop. Give each cluster a distinct one (SFT scripts: `SFT_DASHBOARD_AGENT_PORT`; the eval sweep uses 52365 so an SFT must use 52366+).
3. **Shared `RAY_TMPDIR`** — two heads writing the same temp-dir collide on the session dir. `sft_finetune_pi05base.sh` derives it from `SFT_RAY_PORT` (`/tmp/ray_sft_${SFT_RAY_PORT}`); override `SFT_RAY_TMPDIR` only if needed.
4. **Legacy scripts still do bare `ray stop`**: `run_train/test_maniskill_pi0.5/run.sh` (the PutOnPlate PPO entry) still does bare `ray stop`. Don't run it while SFT/eval is up; if you must, port-isolate it first (same `--port` + `--dashboard-agent-listen-port` + scoped pattern).
5. **`RAY_ADDRESS` unset** → `address="auto"` ps-scans to *any* cluster. Always set it (the fixed scripts do).
6. **GPU overlap ≠ process-kill.** Ray isolation (distinct port + scoped + no bare `ray stop`) holds **regardless of GPU overlap** — an eval on GPUs the SFT also uses will NOT ray-kill the SFT. But the shared GPUs time-slice compute (both slower) and HBM adds up; if SFT+eval > 80GB/GPU → CUDA OOM (crashes the later-started one). Cleanest: disjoint GPUs; if you must overlap, keep combined HBM < ~70GB/GPU and `watch nvidia-smi`.
7. **Hard-restart of an SFT** while another SFT is up: the new one's `_sft_scoped_ray_kill` only touches its OWN port, so it's safe — but two SFTs on the same GPUs still OOM. Use disjoint GPUs + distinct `SFT_RAY_PORT`/`SFT_DASHBOARD_AGENT_PORT`.
8. **Disk full → checkpoint save crash.** Each pi0.5 checkpoint is ~16-32GB; a 30k run at `save_interval=2000` = 15 ckpts ≈ 240GB/run (×N concurrent runs). `/opt` hit 100% mid-run → `torch.distributed.checkpoint CheckpointException` (`all_reduce`/`write_data` fail) → SFT crashes at save; the failed ckpt is also corrupt (don't resume from it). **Before launching: `df -h /opt`; keep ≥ ~250GB free per SFT run.** Free space by deleting old `logs/*` runs (the big consumer — `du -sh logs/*/ | sort -h`), NOT Ray temp dirs (those are MB-scale). Symptom: `CheckpointException` + `raylet ... Too many open files` in the tee log.
9. **fd limit (Too many open files).** The SFT raylet + `torch.distributed.checkpoint` shard saves open many fds; the default soft limit 1024 → `socket ... returned -1 ... Too many open files` → grpc errors / save failure (compounds pitfall 8). The SFT scripts now `ulimit -n 1048576` (the eval scripts already did). Any new launcher must set `ulimit -n 1048576` too. Check: `ulimit -n` (should be 1048576, not 1024).

## Config knobs (when max_steps / save_interval change)

When you change `runner.max_steps`, you MUST also change `actor.optim.total_training_steps` to match (the cosine LR schedule decays over `total_training_steps`). Example — both set to 30000, save every 2000:
```yaml
runner:
  max_steps: 30000
  save_interval: 2000
actor:
  optim:
    total_training_steps: 30000   # == max_steps, else LR won't reach min_lr
    lr_warmup_steps: 500
    lr_scheduler: "cosine"
```
Files: `examples/sft/config/peg_insertion_sft_openpi_pi05_wrist.yaml`.

## Scaling GPUs (does 8 GPUs = 2x speed?)

With FSDP `sharding_strategy: "no_shard"` (pure data-parallel; the 3B model fits one 80GB GPU), throughput scales with **per-GPU utilization × #GPUs**. Measured on this box:

- 4 GPUs, `micro_batch=8`, `global_batch=32` → **1.09 it/s** (baseline).
- 8 GPUs, `micro_batch=4`, `global_batch=32` (same experiment) → **1.39 it/s** = only **1.28x**, GPU util 29-48%.

Why not 1.7-2x: the VLA worker uses **OpenPI's data pipeline** (`openpi_data_loader.create_data_loader`), which is CPU-bound and **can't feed 8 GPUs** → GPUs starve (low util). Halving `micro_batch` (8→4) barely helps because the forward has fixed overhead at small batch, and 8-rank all-reduce adds cost. So:
- **Same experiment + more GPUs** (keep `global_batch`, halve `micro_batch`): only ~1.28x here — data-loader-bound. To fix, raise OpenPI dataloader `num_workers`/prefetch (inside `create_data_loader`, not a simple RLinf knob).
- **2x throughput, different experiment** (keep `micro_batch=8`, double `global_batch` to 64): ~2x samples/step at ~same step time (since the GPU was waiting on data anyway) → real 2x throughput, but `global_batch 64≠32` → may need LR retune, and 30k optimizer-steps still take the same wall-clock (you just see 2x data).
- **True 2x wall-clock for the same 30k steps**: not achievable for this small-model DDP unless the data loader is fixed.

Rule of thumb: before assuming N GPUs = Nx speed, check `nvidia-smi` util — if it's <60% with CPU high, you're data-loader-bound, not compute-bound; adding GPUs won't help until you fix the loader.

## Files touched for isolation (reference)

- `sft_finetune_pi05base.sh`, `run_sft_insert_wrist_v2.sh` — SFT detached-head + scoped.
- `run_train/eval_checkpoint/sweep_peginsertion_wrist.py` — `--ray-port`, `_scoped_ray_kill`.
- `run_train/eval_checkpoint/run_peginsertion_wrist.sh`, `run_peginsertion_wrist_insert_only.sh`, `run.sh` — `EVAL_RAY_PORT`, scoped trap.
- `README.md §2` has the Ray-isolation note + commands.
