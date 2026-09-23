import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = "/data/yingxi/robometer/failure_detection_env/bin/python"
ROOT = "/data/yingxi/RLinf_RoboFAPE"
COL = f"{ROOT}/run_train/robofpe_sft_data/collect_sft_data.py"
BASE = Path("/data/yingxi/datasets/robofpe_sft/strict_compare_20260920")
TASKS = [("PlugCharger-v1", 18, 200), ("UprightStack-v1", 0, 200)]
results = []

def run_one(task, seed, steps, gpu, timeout, tag):
    out = BASE / tag / f"gpu{gpu}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, COL, "collect", "--task-id", task, "--num-traj", "1",
           "--output-dir", str(out), "--robot-uids", "panda_wristcam",
           "--success-only", "--sim-backend", "gpu", "--num-workers", "1",
           "--gpu-ids", str(gpu), "--max-attempts-per-traj", "1", "--seed", str(seed),
           "--solver-timeout", str(timeout), "--max-episode-steps", str(steps), "--no-convert"]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout + 30,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        elapsed = time.monotonic() - start
        files = list((out / "raw" / task).glob("*.h5"))
        return {"task": task, "seed": seed, "gpu": gpu, "timeout": timeout,
                "mode": tag, "status": "success" if p.returncode == 0 and files else "failed",
                "returncode": p.returncode, "elapsed": elapsed, "tail": p.stdout[-1200:]}
    except subprocess.TimeoutExpired as exc:
        return {"task": task, "seed": seed, "gpu": gpu, "timeout": timeout,
                "mode": tag, "status": "hard_timeout", "elapsed": time.monotonic() - start,
                "tail": str(exc)}

for task, seed, steps in TASKS:
    for timeout in (60, 120):
        r = run_one(task, seed, steps, 0, timeout, f"{task}_single_t{timeout}")
        results.append(r); print(r, flush=True)
        jobs = [(i, seed + i) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            rs = list(ex.map(lambda x: run_one(task, x[1], steps, x[0], timeout, f"{task}_4gpu_t{timeout}"), jobs))
        results.extend(rs)
        for r in rs: print(r, flush=True)
(BASE / "results.json").parent.mkdir(parents=True, exist_ok=True)
(BASE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
