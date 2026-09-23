import os, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = "/data/yingxi/robometer/failure_detection_env/bin/python"
ROOT = "/data/yingxi/RLinf_RoboFAPE"
COL = f"{ROOT}/run_train/robofpe_sft_data/collect_sft_data.py"
BASE = Path("/data/yingxi/datasets/robofpe_sft/smoke_3tasks_20260920")
TASKS = {
    "PlugCharger-v1": 18,
    "UprightStack-v1": 0,
    "PullCubeTool-golf": 1,
}
TARGET = 4
for task in TASKS:
    (BASE / task / "raw" / task).mkdir(parents=True, exist_ok=True)

def run_one(slot, task, seed, attempt):
    out = BASE / task / f"attempt_{slot}_{seed}_{attempt}"
    cmd = [PY, COL, "collect", "--task-id", task, "--num-traj", "1",
           "--output-dir", str(out), "--robot-uids", "panda_wristcam",
           "--success-only", "--randomize-wrist-camera", "--randomize-render-camera",
           "--randomize-lighting", "--save-video", "--sim-backend", "gpu",
           "--num-workers", "1", "--gpu-ids", str(slot), "--max-attempts-per-traj", "1",
           "--seed", str(seed), "--solver-timeout", "60", "--no-convert"]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(slot)
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, timeout=90,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        files = list((out / "raw" / task).glob("*.h5"))
        ok = p.returncode == 0 and bool(files)
        if ok:
            for f in files:
                shutil.copy2(f, BASE / task / "raw" / task / f.name)
        return task, seed, ok, p.stdout[-300:]
    except subprocess.TimeoutExpired:
        return task, seed, False, "HARD_TIMEOUT"

counts = {t: 0 for t in TASKS}
next_seed = dict(TASKS)
attempts = {t: 0 for t in TASKS}
while min(counts.values()) < TARGET:
    jobs = []
    for slot in range(4):
        available = [t for t in TASKS if counts[t] < TARGET]
        if not available: break
        task = available[slot % len(available)]
        seed = next_seed[task]; next_seed[task] += 1; attempts[task] += 1
        jobs.append((slot, task, seed, attempts[task]))
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        results = list(ex.map(lambda x: run_one(*x), jobs))
    for task, seed, ok, msg in results:
        counts[task] += int(ok)
        print(f"task={task} seed={seed} ok={ok} count={counts[task]}/{TARGET} {msg}", flush=True)
    if sum(attempts.values()) > 60:
        raise SystemExit(f"too many attempts counts={counts}")

for task in TASKS:
    src = BASE / task / "raw" / task
    dst = BASE / task / "lerobot"
    subprocess.run([PY, COL, "convert", "--input", str(src), "--dataset-dir", str(dst),
                    "--overwrite", "--num-convert-workers", "48"], cwd=ROOT, check=True)
    print(f"CONVERTED {task} {dst}", flush=True)
print(f"DONE {counts}", flush=True)
