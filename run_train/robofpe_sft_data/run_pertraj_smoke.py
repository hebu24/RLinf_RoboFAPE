import os, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PY = "/data/yingxi/robometer/failure_detection_env/bin/python"
ROOT = "/data/yingxi/RLinf_RoboFAPE"
COL = f"{ROOT}/run_train/robofpe_sft_data/collect_sft_data.py"
OUT = Path("/data/yingxi/datasets/robofpe_sft/smoke_pertraj_side_20260920")
TASK = "PegInsertionSide-v1"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "raw" / TASK).mkdir(parents=True, exist_ok=True)

def one(slot, seed):
    d = OUT / f"attempt_{slot}_{seed}"
    cmd = [PY, COL, "collect", "--task-id", TASK, "--num-traj", "1",
           "--output-dir", str(d), "--robot-uids", "panda_wristcam",
           "--success-only", "--randomize-wrist-camera", "--randomize-render-camera",
           "--randomize-lighting", "--save-video", "--sim-backend", "gpu",
           "--num-workers", "1", "--gpu-ids", str(slot), "--max-attempts-per-traj", "1",
           "--seed", str(seed), "--solver-timeout", "60", "--no-convert"]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(slot)
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, timeout=90, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        ok = p.returncode == 0 and list((d / "raw" / TASK).glob("*.h5"))
        if ok:
            for f in (d / "raw" / TASK).glob("*.h5"):
                shutil.copy2(f, OUT / "raw" / TASK / f.name)
            return slot, seed, True, p.stdout[-500:]
        return slot, seed, False, p.stdout[-500:]
    except subprocess.TimeoutExpired:
        return slot, seed, False, "HARD_TIMEOUT"

next_seed = 2
done = 0
while done < 4:
    jobs = []
    for slot in range(4):
        jobs.append((slot, next_seed)); next_seed += 1
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda x: one(*x), jobs))
    for slot, seed, ok, msg in results:
        print(f"slot={slot} seed={seed} ok={ok} {msg}", flush=True)
        done += int(ok)
    if not any(r[2] for r in results):
        raise SystemExit("no successful trajectory in batch")

subprocess.run([PY, COL, "convert", "--input", str(OUT / "raw" / TASK),
                "--dataset-dir", str(OUT / "lerobot"), "--overwrite",
                "--num-convert-workers", "48"], cwd=ROOT, check=True)
print(f"DONE successful={done} output={OUT}")
