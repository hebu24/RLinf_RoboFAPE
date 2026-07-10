#!/usr/bin/env python3
import argparse,json,os,os.path as osp,sys,time,traceback,shutil,multiprocessing as mp
from pathlib import Path
from typing import List
import cv2,gymnasium as gym,numpy as np,pandas as pd,torch
from tqdm import tqdm
from transforms3d import quaternions
from transforms3d.axangles import mat2axangle

# Register RLinf env FIRST (before RoboFPE solution import overrides it)
import importlib.util as _ilu
_s=_ilu.spec_from_file_location("_peg","/opt/yingxi/RLinf_RoboFAPE/rlinf/envs/maniskill/tasks/peg_insertion_vertical.py")
_m=_ilu.module_from_spec(_s); _s.loader.exec_module(_m)

# Now import RoboFPE solution (its task import will be skipped since env is already registered)
ROBOFPE="/home/gpu4/yingxi/RoboFPE/mani_envs"
sys.path.insert(0,ROBOFPE)
sys.path.insert(0,osp.join(ROBOFPE,"data_collection"))
import types as _types; _ts=_types.ModuleType("tasks"); _ts.task_PegInsertionVertical=_types.ModuleType("tasks.task_PegInsertionVertical"); _ts.task_PegInsertionVertical.PegInsertionVerticalEnv=type("PegInsertionVerticalEnv",(),{}); sys.modules["tasks"]=_ts; sys.modules["tasks.task_PegInsertionVertical"]=_ts.task_PegInsertionVertical
from solutions.solve_PegInsertionVertical import solve_peginsertionvertical

TASK="insert the blue peg vertically into the orange hole"
TASK_DESCRIPTIONS = json.load(
    open("/home/gpu4/yingxi/RoboFPE/mani_envs/data_collection/task_descriptions/peg_insertion_vertical.json")
)["PegInsertionVertical-v1"]
FPS=20;IW=224;IH=224;RW=640;RH=480
SDIM=8;ADIM=7;CSIZE=1000
def _compute_stats(df):
    s={}; a=np.stack(df["actions"].values if "actions" in df.columns else df["action"].values)
    s["actions"]={"mean":a.mean(0).tolist(),"std":a.std(0).tolist(),"max":a.max(0).tolist(),"min":a.min(0).tolist(),"count":[len(a)]}
    st=np.stack(df["observation.state"].values)
    s["observation.state"]={"mean":st.mean(0).tolist(),"std":st.std(0).tolist(),"max":st.max(0).tolist(),"min":st.min(0).tolist(),"count":[len(st)]}
    for fld in ["timestamp","frame_index","episode_index","index","task_index"]:
        v=df[fld].values
        s[fld]={"mean":[float(v.mean())],"std":[float(v.std())],"max":[int(v.max())] if fld!="timestamp" else [float(v.max())],"min":[int(v.min())] if fld!="timestamp" else [float(v.min())],"count":[len(v)]}
    return s

def _build_features():
    feat={"actions":{"dtype":"float32","shape":[ADIM],"names":[f"action_{i}" for i in range(ADIM)],"fps":float(FPS)},"observation.state":{"dtype":"float32","shape":[SDIM],"names":[f"joint_{i}" for i in range(SDIM)],"fps":float(FPS)},"timestamp":{"dtype":"float32","shape":[1],"names":None,"fps":float(FPS)},"frame_index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"episode_index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"task_index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"task":{"dtype":"string","shape":[1],"names":None,"fps":float(FPS)}}
    for c,h,w in [("top",IH,IW),("wrist",IH,IW),("render",RH,RW)]:
        feat[f"observation.images.{c}"]={"dtype":"video","shape":[h,w,3],"names":["height","width","channels"],"info":{"video.fps":float(FPS),"video.height":h,"video.width":w,"video.channels":3,"video.codec":"mp4v","video.pix_fmt":"yuv420p","video.is_depth_map":False,"has_audio":False}}
    return feat

class ObservationRecorder:
    def __init__(self, env):
        self.env=env; self.uw=env.unwrapped
        self.records=[]; self._orig=env.step
    def start(self):
        self.records=[]; env=self.env; rec=self
        def _step(action, **kw):
            pre=rec._cap()
            obs,r,term,trunc,info=rec._orig(action, **kw)
            post=rec._cap()
            rec.records.append({"pre":pre,"post":post,"act":np.asarray(action,dtype=np.float32).copy()})
            return obs,r,term,trunc,info
        env.step=_step
    def stop(self):
        self.env.step=self._orig
    def _cap(self):
        env=self.uw; obs={}
        sd=env.get_obs()["sensor_data"]
        for cn in ["base_camera","hand_camera"]:
            if cn in sd:
                rgb=sd[cn]["rgb"]
                if hasattr(rgb,"detach"): rgb=rgb.detach().cpu().numpy()
                obs[cn+"_rgb"]=np.asarray(rgb[0],dtype=np.uint8).copy()
        rc=env._sensors.get("render_camera")
        if rc is not None:
            rgb=rc.get_obs()["rgb"]
            if hasattr(rgb,"detach"): rgb=rgb.detach().cpu().numpy()
            obs["render_rgb"]=np.asarray(rgb[0],dtype=np.uint8).copy()
        else:
            obs["render_rgb"]=obs.get("base_camera_rgb",np.zeros((RH,RW,3),dtype=np.uint8)).copy()
        qp=env.agent.robot.get_qpos()
        if hasattr(qp,"detach"): qp=qp.detach().cpu().numpy()
        qp=np.asarray(qp[0],dtype=np.float32)
        st=np.zeros(SDIM,dtype=np.float32); st[:7]=qp[:7]; st[7]=qp[-1]
        obs["state"]=st
        tp=env.agent.tcp.pose; p=tp.p; q=tp.q
        if hasattr(p,"detach"): p=p.detach().cpu().numpy(); q=q.detach().cpu().numpy()
        obs["tcp_p"]=np.asarray(p[0],dtype=np.float64).copy()
        obs["tcp_q"]=np.asarray(q[0],dtype=np.float64).copy()
        return obs

def compute_ee_delta_actions(records):
    T=len(records); acts=np.zeros((T,ADIM),dtype=np.float32)
    for t in range(T):
        pre=records[t]["pre"]; post=records[t]["post"]
        dp=post["tcp_p"]-pre["tcp_p"]
        Ro=quaternions.quat2mat(pre["tcp_q"]); Rn=quaternions.quat2mat(post["tcp_q"])
        Rd=Rn@Ro.T; ax,ang=mat2axangle(Rd); dr=ax*ang
        g=float(records[t]["act"][-1])
        acts[t]=np.concatenate([dp,dr,[g]])
    if T>1: acts[-1]=acts[-2]
    return acts

class LeRobotWriter:
    def __init__(self, outdir, task=TASK):
        self.od=osp.abspath(outdir)
        self.eps=[]; self.dfs=[]; self.gi=0
        # 多 task 支持
        self.task_to_idx={}; self.all_tasks=[]
        os.makedirs(self.od,exist_ok=True)
        os.makedirs(osp.join(self.od,"data","chunk-000"),exist_ok=True)
        os.makedirs(osp.join(self.od,"meta","episodes","chunk-000"),exist_ok=True)
        for c in ["top","wrist","render"]:
            os.makedirs(osp.join(self.od,"videos",f"observation.images.{c}","chunk-000"),exist_ok=True)

    def _get_task_idx(self, task):
        if task not in self.task_to_idx:
            self.task_to_idx[task]=len(self.all_tasks)
            self.all_tasks.append(task)
        return self.task_to_idx[task]

    def add_episode(self,acts,states,bf,wf,rf):
        import random as _r
        task=_r.choice(TASK_DESCRIPTIONS)
        ti=self._get_task_idx(task)
        ei=len(self.eps); T=len(acts); ts=np.arange(T,dtype=np.float32)/FPS
        df=pd.DataFrame({"action":[r.tolist() for r in acts],"observation.state":[r.tolist() for r in states],"timestamp":ts,"frame_index":np.arange(T,dtype=np.int64),"episode_index":np.full(T,ei,dtype=np.int64),"index":np.arange(self.gi,self.gi+T,dtype=np.int64),"task_index":np.full(T,ti,dtype=np.int64),"task":[task]*T})
        self.dfs.append(df)
        self.eps.append({"episode_index":ei,"data/chunk_index":0,"data/file_index":0,"dataset_from_index":self.gi,"dataset_to_index":self.gi+T,"tasks":[task],"length":T})
        self.gi+=T; self._wv(bf,ei,"top"); self._wv(wf,ei,"wrist"); self._wv(rf,ei,"render"); return ei

    def add_episode(self,acts,states,bf,wf,rf):
        ei=len(self.eps); T=len(acts); ts=np.arange(T,dtype=np.float32)/FPS
        df=pd.DataFrame({"actions":[r.tolist() for r in acts],"observation.state":[r.tolist() for r in states],"timestamp":ts,"frame_index":np.arange(T,dtype=np.int64),"episode_index":np.full(T,ei,dtype=np.int64),"index":np.arange(self.gi,self.gi+T,dtype=np.int64),"task_index":np.zeros(T,dtype=np.int64),"task":[self.task]*T})
        self.dfs.append(df)
        self.eps.append({"episode_index":ei,"data/chunk_index":0,"data/file_index":0,"dataset_from_index":self.gi,"dataset_to_index":self.gi+T,"tasks":[self.task],"length":T})
        self.gi+=T; self._wv(bf,ei,"top"); self._wv(wf,ei,"wrist"); self._wv(rf,ei,"render"); return ei

    def _wv(self,frames,ei,cn):
        vp=osp.join(self.od,"videos",f"observation.images.{cn}","chunk-000",f"file-{ei:03d}.mp4")
        h,w=frames.shape[1],frames.shape[2]; fc=cv2.VideoWriter_fourcc(*"mp4v")
        out=cv2.VideoWriter(str(vp),fc,FPS,(w,h))
        if not out.isOpened(): raise RuntimeError(f"video fail {vp}")
        for f in frames: out.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
        out.release()
    def finalize(self):
        if not self.eps: print("WARNING: no episodes"); return
        import pyarrow as pa, pyarrow.parquet as pq
        cdf=pd.concat(self.dfs,ignore_index=True); cdf["task"]=cdf["task"].astype("string")
        sch=pa.schema([pa.field("actions",pa.list_(pa.float32())),pa.field("observation.state",pa.list_(pa.float32())),pa.field("timestamp",pa.float32()),pa.field("frame_index",pa.int64()),pa.field("episode_index",pa.int64()),pa.field("index",pa.int64()),pa.field("task_index",pa.int64()),pa.field("task",pa.string())])
        pq.write_table(pa.Table.from_pandas(cdf,schema=sch),osp.join(self.od,"data","chunk-000","file-000.parquet"))
        edf=pd.DataFrame(self.eps)
        pq.write_table(pa.Table.from_pandas(edf),osp.join(self.od,"meta","episodes","chunk-000","file-000.parquet"))
        tdf=pd.DataFrame({"task_index":[0]},index=[self.task]); tdf.index.name=None
        tdf.to_parquet(osp.join(self.od,"meta","tasks.parquet"),index=True)
        stats=_compute_stats(cdf)
        with open(osp.join(self.od,"meta","stats.json"),"w") as f: json.dump(stats,f,indent=2)
        TF=len(cdf); TE=len(self.eps)
        feat={"actions":{"dtype":"float32","shape":[ADIM],"names":[f"action_{i}" for i in range(ADIM)],"fps":float(FPS)},"observation.state":{"dtype":"float32","shape":[SDIM],"names":[f"joint_{i}" for i in range(SDIM)],"fps":float(FPS)},"timestamp":{"dtype":"float32","shape":[1],"names":None,"fps":float(FPS)},"frame_index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"episode_index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"task_index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"task":{"dtype":"string","shape":[1],"names":None,"fps":float(FPS)}}
        for c,h,w in [("top",IH,IW),("wrist",IH,IW),("render",RH,RW)]:
            feat[f"observation.images.{c}"]={"dtype":"video","shape":[h,w,3],"names":["height","width","channels"],"info":{"video.fps":float(FPS),"video.height":h,"video.width":w,"video.channels":3,"video.codec":"mp4v","video.pix_fmt":"yuv420p","video.is_depth_map":False,"has_audio":False}}
        dsz=sum(f.stat().st_size for f in Path(self.od).rglob("data/*.parquet"))
        info={"codebase_version":"v3.0","robot_type":"panda_wristcam","total_episodes":TE,"total_frames":TF,"total_tasks":1,"total_videos":TE*3,"total_chunks":1,"chunks_size":CSIZE,"fps":FPS,"data_files_size_in_mb":int(dsz/(1024*1024)),"splits":{"train":f"0:{TE}"},"data_path":"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet","video_path":"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4","features":feat}
        with open(osp.join(self.od,"meta","info.json"),"w") as f: json.dump(info,f,indent=2)
        print(f"Dataset: {TE} eps, {TF} frames -> {self.od}")
    def _stats(self,df):
        s={}; a=np.stack(df["actions"].values if "actions" in df.columns else df["action"].values)
        s["actions"]={"mean":a.mean(0).tolist(),"std":a.std(0).tolist(),"max":a.max(0).tolist(),"min":a.min(0).tolist(),"count":[len(a)]}
        st=np.stack(df["observation.state"].values)
        s["observation.state"]={"mean":st.mean(0).tolist(),"std":st.std(0).tolist(),"max":st.max(0).tolist(),"min":st.min(0).tolist(),"count":[len(st)]}
        for fld in ["timestamp","frame_index","episode_index","index","task_index"]:
            v=df[fld].values
            s[fld]={"mean":[float(v.mean())],"std":[float(v.std())],"max":[int(v.max())] if fld!="timestamp" else [float(v.max())],"min":[int(v.min())] if fld!="timestamp" else [float(v.min())],"count":[len(v)]}
        return s

def _make_env(gpu_id=0, max_retries=5, retry_delay=10.0):
    """Create env bound to a specific Vulkan render GPU, with retry on ErrorDeviceLost."""
    render_backend=f"gpu:{gpu_id}"
    last_err=None
    for attempt in range(max_retries):
        try:
            env=gym.make("PegInsertionVertical-v1",num_envs=1,obs_mode="rgb",robot_uids="panda_wristcam",control_mode="pd_joint_pos",sim_backend="cpu",render_backend=render_backend,render_mode="all",reward_mode="normalized_dense",max_episode_steps=600,sensor_configs=dict(shader_pack="default"),human_render_camera_configs=dict(shader_pack="default"))
            return env
        except RuntimeError as e:
            last_err=e
            if "ErrorDeviceLost" in str(e) or "vk" in str(e).lower():
                print(f"  [gpu{gpu_id}] Vulkan error attempt {attempt+1}/{max_retries}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            raise
    raise last_err

def _collect_episodes(num_traj, base_seed, writer, progress_prefix="", gpu_id=0):
    env=_make_env(gpu_id=gpu_id); recorder=ObservationRecorder(env)
    seed=base_seed; passed=0; attempts=0; fmp=0; fsucc=0; t0=time.time()
    pbar=tqdm(total=num_traj,desc=progress_prefix or "Collecting",position=0,leave=True)
    while passed<num_traj:
        attempts+=1
        recorder.start()
        try:
            res=solve_peginsertionvertical(env,seed=seed,debug=False,vis=False,reset_options={"randomize_initial_poses":True})
            if isinstance(res,tuple) and len(res)==2 and isinstance(res[1],list): rv,sr=res
            else: rv,sr=res,[]
            if rv==-1: success=False; fmp+=1
            else:
                er=env.unwrapped.evaluate(); success=bool(er["success"].item())
            if not success:
                if rv!=-1: fsucc+=1
                seed+=1; recorder.stop(); continue
        except Exception as e:
            errstr=str(e)
            print(f"{progress_prefix}Error seed={seed}: {e}")
            if "ErrorDeviceLost" in errstr or "vk" in errstr.lower():
                # Vulkan GPU device lost (runtime contention): recreate env and retry same seed
                traceback.print_exc()
                try: env.close()
                except: pass
                print(f"{progress_prefix}Recreating env on gpu{gpu_id} after Vulkan error...")
                time.sleep(5.0)
                env=_make_env(gpu_id=gpu_id); recorder=ObservationRecorder(env)
                recorder.stop()
                continue
            traceback.print_exc(); fmp+=1; seed+=1; recorder.stop()
            continue
        records=recorder.records
        recorder.stop()
        if not records:
            print(f"{progress_prefix}No records seed={seed}"); seed+=1; continue
        acts=compute_ee_delta_actions(records)
        states=np.stack([r["pre"]["state"] for r in records])
        bf=np.stack([r["pre"]["base_camera_rgb"] for r in records])
        wf=np.stack([r["pre"]["hand_camera_rgb"] for r in records])
        rf=np.stack([r["pre"]["render_rgb"] for r in records])
        writer.add_episode(acts,states,bf,wf,rf)
        passed+=1; pbar.update(1)
        el=time.time()-t0; rate=passed/el if el>0 else 0
        pbar.set_postfix(dict(succ=f"{passed/attempts:.2%}" if attempts else "n/a",fmp=fmp,fs=fsucc,epl=len(records),r=f"{rate:.1f}e/s"))
        seed+=1
    pbar.close()
    try: env.close()
    except: pass
    return passed, attempts
def _collect_worker(worker_id, num_traj, base_seed, shard_dir, gpu_id=0, startup_delay=0.0):
    if startup_delay>0: time.sleep(startup_delay)
    writer=LeRobotWriter(shard_dir)
    pfx=f"[w{worker_id}/gpu{gpu_id}] "
    passed, attempts=_collect_episodes(num_traj, base_seed, writer, progress_prefix=pfx, gpu_id=gpu_id)
    writer.finalize()
    return worker_id, shard_dir, passed, attempts

def merge_shards(final_dir, shard_dirs):
    import pyarrow as pa, pyarrow.parquet as pq
    final_dir=osp.abspath(final_dir)
    os.makedirs(final_dir,exist_ok=True)
    for sub in ["data","meta","videos"]:
        p=osp.join(final_dir,sub)
        if osp.exists(p): shutil.rmtree(p)
    os.makedirs(osp.join(final_dir,"meta"),exist_ok=True)
    all_dfs=[]; global_epi=0; global_idx=0
    global_task_to_idx={}; global_all_tasks=[]
    for sd in shard_dirs:
        dp=osp.join(sd,"data","chunk-000","file-000.parquet")
        if not osp.exists(dp):
            print(f"merge: skip empty shard {sd}"); continue
        df=pd.read_parquet(dp)
        if "action" in df.columns and "actions" not in df.columns:
            df=df.rename(columns={"action":"actions"})
        n_eps=int(df["episode_index"].max())+1 if len(df) else 0
        if n_eps==0: continue
        for li in range(n_eps):
            gi=global_epi+li; chunk=gi//CSIZE
            for c in ["top","wrist","render"]:
                src=osp.join(sd,"videos",f"observation.images.{c}","chunk-000",f"file-{li:03d}.mp4")
                vdir=osp.join(final_dir,"videos",f"chunk-{chunk:03d}",f"observation.images.{c}")
                os.makedirs(vdir,exist_ok=True)
                dst=osp.join(vdir,f"episode_{gi:06d}.mp4")
                if osp.exists(src): shutil.copy2(src,dst)
        sub=df.copy()
        sub["episode_index"]=sub["episode_index"]-int(sub["episode_index"].min())+global_epi
        sub["index"]=np.arange(global_idx,global_idx+len(sub),dtype=np.int64)
        # 重映射 task_index 到全局
        for old_ti in sub["task_index"].unique():
            mask=sub["task_index"]==old_ti
            task_str=sub.loc[mask,"task"].iloc[0]
            if task_str not in global_task_to_idx:
                global_task_to_idx[task_str]=len(global_all_tasks)
                global_all_tasks.append(task_str)
            sub.loc[mask,"task_index"]=global_task_to_idx[task_str]
        all_dfs.append(sub)
        global_epi+=n_eps; global_idx+=len(sub)
    if not all_dfs:
        print("merge: no episodes found in any shard"); return
    cdf=pd.concat(all_dfs,ignore_index=True); cdf["task"]=cdf["task"].astype("string")
    n_tasks=len(global_all_tasks)
    sch=pa.schema([pa.field("actions",pa.list_(pa.float32())),pa.field("observation.state",pa.list_(pa.float32())),pa.field("observation.state_tcp",pa.list_(pa.float32())),pa.field("timestamp",pa.float32()),pa.field("frame_index",pa.int64()),pa.field("episode_index",pa.int64()),pa.field("index",pa.int64()),pa.field("task_index",pa.int64()),pa.field("task",pa.string())])
    ep_ids=sorted(cdf["episode_index"].unique()); total_chunks=0
    for eid in ep_ids:
        eid=int(eid); chunk=eid//CSIZE
        if chunk+1>total_chunks: total_chunks=chunk+1
        cdir=osp.join(final_dir,"data",f"chunk-{chunk:03d}"); os.makedirs(cdir,exist_ok=True)
        ep_df=cdf[cdf["episode_index"]==eid].reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(ep_df,schema=sch,preserve_index=False),osp.join(cdir,f"episode_{eid:06d}.parquet"))

    nl=chr(10)
    with open(osp.join(self.od,"meta","tasks.jsonl"),"w") as f:
        for i,task in enumerate(self.all_tasks):
            f.write(json.dumps({"task_index":i,"task":task})+nl)
    with open(osp.join(final_dir,"meta","episodes.jsonl"),"w") as f:
        for eid in ep_ids:
            ep_data=cdf[cdf["episode_index"]==eid]
            T=len(ep_data); task=ep_data["task"].iloc[0]
            f.write(json.dumps({"episode_index":int(eid),"tasks":[task],"length":T})+nl)
    stats=_compute_stats(cdf)
    with open(osp.join(final_dir,"meta","stats.json"),"w") as f: json.dump(stats,f,indent=2)
    TF=len(cdf); TE=int(global_epi); total_chunks=int(total_chunks); n_tasks=int(n_tasks)
    feat=_build_features()
    dsz=sum(f.stat().st_size for f in Path(final_dir).rglob("data/**/*.parquet"))
    info={"codebase_version":"v2.0","robot_type":"panda_wristcam","total_episodes":TE,"total_frames":TF,"total_tasks":n_tasks,"total_videos":TE*3,"total_chunks":total_chunks,"chunks_size":CSIZE,"fps":FPS,"data_files_size_in_mb":int(dsz/(1024*1024)),"splits":{"train":f"0:{TE}"},"data_path":"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet","video_path":"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4","features":feat}
    with open(osp.join(final_dir,"meta","info.json"),"w") as f: json.dump(info,f,indent=2)
    print(f"Merged dataset: {TE} eps, {TF} frames, {n_tasks} tasks -> {final_dir}")
    
def collect_data(args):
    outdir=osp.abspath(args.output_dir)
    if args.num_workers<=1:
        os.makedirs(outdir,exist_ok=True)
        writer=LeRobotWriter(outdir)
        passed, attempts=_collect_episodes(args.num_traj, args.seed, writer)
        writer.finalize()
        sr=f"{passed/attempts:.1%}" if attempts else "n/a"
        print(f"Done! {passed}/{attempts} ({sr}), out={outdir}")
        return
    nw=args.num_workers
    per=args.num_traj//nw; rem=args.num_traj%nw
    counts=[per+(1 if i<rem else 0) for i in range(nw)]
    shard_root=osp.join(outdir,"_shards")
    if osp.exists(shard_root):
        shutil.rmtree(shard_root)
    os.makedirs(shard_root,exist_ok=True)
    shard_dirs=[osp.join(shard_root,f"shard_{i:03d}") for i in range(nw)]
    seeds=[args.seed+i*100000 for i in range(nw)]
    t0=time.time()
    gpu_ids=[int(x) for x in args.gpu_ids.split(",") if x.strip()!=""]
    ng=len(gpu_ids)
    stagger=args.worker_stagger
    # Per-GPU startup counter: workers on DIFFERENT gpus start simultaneously,
    # workers on the SAME gpu stagger to avoid Vulkan init contention.
    gpu_counter={g:0 for g in gpu_ids}
    worker_args=[]
    for i in range(nw):
        g=gpu_ids[i%ng]
        delay=gpu_counter[g]*stagger
        gpu_counter[g]+=1
        worker_args.append((i,counts[i],seeds[i],shard_dirs[i],g,delay))
    print(f"Launching {nw} workers across {ng} GPU(s) {gpu_ids} (stagger={stagger}s per-GPU)")
    ctx=mp.get_context("fork")
    with ctx.Pool(nw) as pool:
        results=pool.starmap(_collect_worker,worker_args)
    el=time.time()-t0
    total_p=sum(r[2] for r in results); total_a=sum(r[3] for r in results)
    sr=f"{total_p/total_a:.1%}" if total_a else "n/a"
    print(f"All workers done in {el:.1f}s. Total: {total_p}/{total_a} ({sr})")
    merge_shards(outdir, shard_dirs)
    print(f"Done! {total_p} episodes, time={el:.1f}s, out={outdir}")



def main():
    p=argparse.ArgumentParser(description="Collect PegInsertionVertical data")
    p.add_argument("--num-traj",type=int,default=500)
    p.add_argument("--output-dir",type=str,default="/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--num-workers",type=int,default=1,help="number of parallel worker processes (each uses 1 CPU core)")
    p.add_argument("--gpu-ids",type=str,default="0",help="comma-separated GPU ids for Vulkan rendering, distributed round-robin across workers (e.g. 0,1,2,3)")
    p.add_argument("--worker-stagger",type=float,default=5.0,help="seconds between worker startups to avoid Vulkan GPU contention")
    args=p.parse_args(); collect_data(args)

if __name__=="__main__": main()
