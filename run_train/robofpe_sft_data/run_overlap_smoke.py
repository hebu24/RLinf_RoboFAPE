import os, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = "/data/yingxi/robometer/failure_detection_env/bin/python"
ROOT = "/data/yingxi/RLinf_RoboFAPE"
COL = f"{ROOT}/run_train/robofpe_sft_data/collect_sft_data.py"
BASE = Path("/data/yingxi/datasets/robofpe_sft/smoke_5tasks_20260920")
TASKS = [("PegInsertionVertical-v1", 0), ("PlugCharger-v1", 22),
         ("UprightStack-v1", 0), ("PegInsertionSide-v1", 2), ("PullCubeTool-golf", 1)]
TARGET = 4
converters = []

def env():
    e = os.environ.copy(); e["MS_ASSET_DIR"] = "/data/yingxi/robofac"; e["CUDA_VISIBLE_DEVICES"] = "0,1"
    e["CUDA_PYTHON_LIB_ROOT"] = "/data/yingxi/robofac/lib/python3.10/site-packages/nvidia"
    return e

def collect_one(task, seed, slot, attempt, out):
    d = out / f"attempt_gpu{slot}_{seed}_{attempt}"
    cmd = [PY, COL, "collect", "--task-id", task, "--num-traj", "1", "--output-dir", str(d),
           "--robot-uids", "panda_wristcam", "--success-only", "--randomize-wrist-camera",
           "--randomize-render-camera", "--randomize-lighting", "--save-video", "--sim-backend", "gpu",
           "--num-workers", "1", "--gpu-ids", str(slot), "--max-attempts-per-traj", "1", "--seed", str(seed),
           "--solver-timeout", "120", "--no-convert"]
    e = env(); e["CUDA_VISIBLE_DEVICES"] = str(slot)
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=e, timeout=180, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        files = list((d / "raw" / task).glob("*.h5")); ok = p.returncode == 0 and bool(files)
        if ok:
            for f in files:
                for suffix in (".h5", ".json", ".mp4"):
                    src = f.with_suffix(suffix)
                    if src.exists(): shutil.copy2(src, out / "raw" / task / src.name)
        return ok, seed, time.monotonic() - start, p.stdout[-400:]
    except subprocess.TimeoutExpired:
        return False, seed, time.monotonic() - start, "HARD_TIMEOUT"

def start_convert(task):
    out = BASE / task; dst = out / "lerobot"
    if dst.exists(): shutil.rmtree(dst)
    log = open(BASE / f"{task}.convert.log", "w")
    p = subprocess.Popen([PY, COL, "convert", "--input", str(out / "raw" / task),
                          "--dataset-dir", str(dst), "--overwrite", "--num-convert-workers", "48"],
                         cwd=ROOT, env=env(), stdout=log, stderr=subprocess.STDOUT)
    converters.append((task, p, log)); print(f"CONVERT_START {task} pid={p.pid}", flush=True)

def main():
    for task, initial in TASKS:
        out = BASE / task; (out / "raw" / task).mkdir(parents=True, exist_ok=True)
        count = len(list((out / "raw" / task).glob("*.h5"))); seed = initial + count; attempt = count
        print(f"COLLECT_START {task} existing={count}", flush=True)
        while count < TARGET:
            jobs = []
            for slot in range(2):
                attempt += 1; jobs.append((task, seed, slot, attempt, out)); seed += 1
            with ThreadPoolExecutor(max_workers=2) as ex:
                rows = list(ex.map(lambda x: collect_one(*x), jobs))
            for ok, used, elapsed, tail in rows:
                count += int(ok); print(f"RESULT {task} seed={used} ok={ok} count={count}/{TARGET} elapsed={elapsed:.1f} {tail}", flush=True)
            if attempt >= 40 and count < TARGET:
                # Do not abort the ordered pipeline because one task has a
                # low motion-planning success rate. Convert what exists and
                # continue collecting the next task; the shortfall is reported
                # in the task directory/log.
                print(f"TASK_SHORTFALL {task} successes={count}/{TARGET} attempts={attempt}", flush=True)
                break
        start_convert(task)
    for task, p, log in converters:
        rc = p.wait(); log.close(); print(f"CONVERT_DONE {task} rc={rc}", flush=True)

if __name__ == "__main__": main()
