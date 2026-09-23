import os, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = "/data/yingxi/robometer/failure_detection_env/bin/python"
ROOT = "/data/yingxi/RLinf_RoboFAPE"
COL = f"{ROOT}/run_train/robofpe_sft_data/collect_sft_data.py"
BASE = Path("/data/yingxi/datasets/robofpe_sft/smoke_5tasks_20260920")
TASKS = [
    ("PegInsertionVertical-v1", 0),
    ("PlugCharger-v1", 22),
    ("UprightStack-v1", 0),
    ("PegInsertionSide-v1", 2),
    ("PullCubeTool-golf", 1),
]
TARGET = 4

def run_one(task, seed, slot, attempt, out):
    d = out / f"attempt_gpu{slot}_{seed}_{attempt}"
    cmd = [PY, COL, "collect", "--task-id", task, "--num-traj", "1",
           "--output-dir", str(d), "--robot-uids", "panda_wristcam", "--success-only",
           "--randomize-wrist-camera", "--randomize-render-camera", "--randomize-lighting",
           "--save-video", "--sim-backend", "gpu", "--num-workers", "1",
           "--gpu-ids", str(slot), "--max-attempts-per-traj", "1", "--seed", str(seed),
           "--solver-timeout", "120", "--no-convert"]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(slot)
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, timeout=180,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        files = list((d / "raw" / task).glob("*.h5"))
        ok = p.returncode == 0 and bool(files)
        if ok:
            for f in files:
                for suffix in (".h5", ".json", ".mp4"):
                    src = f.with_suffix(suffix)
                    if src.exists():
                        shutil.copy2(src, out / "raw" / task / src.name)
        return task, seed, ok, time.monotonic() - start, p.stdout[-500:]
    except subprocess.TimeoutExpired:
        return task, seed, False, time.monotonic() - start, "HARD_TIMEOUT"

def main():
    BASE.mkdir(parents=True, exist_ok=True)
    for task, initial_seed in TASKS:
        out = BASE / task
        (out / "raw" / task).mkdir(parents=True, exist_ok=True)
        count = len(list((out / "raw" / task).glob("*.h5")))
        seed = initial_seed + count; attempt = count
        print(f"START {task}", flush=True)
        while count < TARGET:
            jobs = []
            for slot in range(2):
                attempt += 1
                jobs.append((task, seed, slot, attempt, out))
                seed += 1
            with ThreadPoolExecutor(max_workers=2) as ex:
                rows = list(ex.map(lambda x: run_one(*x), jobs))
            for task_id, used_seed, ok, elapsed, tail in rows:
                count += int(ok)
                print(f"RESULT task={task_id} seed={used_seed} ok={ok} count={count}/{TARGET} elapsed={elapsed:.1f} tail={tail}", flush=True)
            if attempt >= 40:
                raise RuntimeError(f"too many attempts task={task} count={count}")
        src = out / "raw" / task
        dst = out / "lerobot"
        subprocess.run([PY, COL, "convert", "--input", str(src), "--dataset-dir", str(dst),
                        "--overwrite", "--num-convert-workers", "48"], cwd=ROOT, check=True)
        videos = list(out.rglob("*.mp4"))
        print(f"DONE {task} successes={count} videos={len(videos)} converted={dst}", flush=True)

if __name__ == "__main__":
    main()
