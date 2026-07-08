#!/usr/bin/env python3
import argparse,json,os,os.path as osp,sys,time,traceback
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
FPS=20;IW=224;IH=224;RW=640;RH=480
SDIM=8;ADIM=7;CSIZE=1000

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
        self.od=osp.abspath(outdir); self.task=task
        self.eps=[]; self.dfs=[]; self.gi=0
        os.makedirs(self.od,exist_ok=True)
        os.makedirs(osp.join(self.od,"data","chunk-000"),exist_ok=True)
        os.makedirs(osp.join(self.od,"meta","episodes","chunk-000"),exist_ok=True)
        for c in ["top","wrist","render"]:
            os.makedirs(osp.join(self.od,"videos",f"observation.images.{c}","chunk-000"),exist_ok=True)
    def add_episode(self,acts,states,bf,wf,rf):
        ei=len(self.eps); T=len(acts); ts=np.arange(T,dtype=np.float32)/FPS
        df=pd.DataFrame({"action":[r.tolist() for r in acts],"observation.state":[r.tolist() for r in states],"timestamp":ts,"frame_index":np.arange(T,dtype=np.int64),"episode_index":np.full(T,ei,dtype=np.int64),"index":np.arange(self.gi,self.gi+T,dtype=np.int64),"task_index":np.zeros(T,dtype=np.int64),"task":[self.task]*T})
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
        sch=pa.schema([pa.field("action",pa.list_(pa.float32())),pa.field("observation.state",pa.list_(pa.float32())),pa.field("timestamp",pa.float32()),pa.field("frame_index",pa.int64()),pa.field("episode_index",pa.int64()),pa.field("index",pa.int64()),pa.field("task_index",pa.int64()),pa.field("task",pa.string())])
        pq.write_table(pa.Table.from_pandas(cdf,schema=sch),osp.join(self.od,"data","chunk-000","file-000.parquet"))
        edf=pd.DataFrame(self.eps)
        pq.write_table(pa.Table.from_pandas(edf),osp.join(self.od,"meta","episodes","chunk-000","file-000.parquet"))
        tdf=pd.DataFrame({"task_index":[0]},index=[self.task]); tdf.index.name=None
        tdf.to_parquet(osp.join(self.od,"meta","tasks.parquet"),index=True)
        stats=self._stats(cdf)
        with open(osp.join(self.od,"meta","stats.json"),"w") as f: json.dump(stats,f,indent=2)
        TF=len(cdf); TE=len(self.eps)
        feat={"action":{"dtype":"float32","shape":[ADIM],"names":[f"action_{i}" for i in range(ADIM)],"fps":float(FPS)},"observation.state":{"dtype":"float32","shape":[SDIM],"names":[f"joint_{i}" for i in range(SDIM)],"fps":float(FPS)},"timestamp":{"dtype":"float32","shape":[1],"names":None,"fps":float(FPS)},"frame_index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"episode_index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"task_index":{"dtype":"int64","shape":[1],"names":None,"fps":float(FPS)},"task":{"dtype":"string","shape":[1],"names":None,"fps":float(FPS)}}
        for c,h,w in [("top",IH,IW),("wrist",IH,IW),("render",RH,RW)]:
            feat[f"observation.images.{c}"]={"dtype":"video","shape":[h,w,3],"names":["height","width","channels"],"info":{"video.fps":float(FPS),"video.height":h,"video.width":w,"video.channels":3,"video.codec":"mp4v","video.pix_fmt":"yuv420p","video.is_depth_map":False,"has_audio":False}}
        dsz=sum(f.stat().st_size for f in Path(self.od).rglob("data/*.parquet"))
        info={"codebase_version":"v3.0","robot_type":"panda_wristcam","total_episodes":TE,"total_frames":TF,"total_tasks":1,"total_videos":TE*3,"total_chunks":1,"chunks_size":CSIZE,"fps":FPS,"data_files_size_in_mb":int(dsz/(1024*1024)),"splits":{"train":f"0:{TE}"},"data_path":"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet","video_path":"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4","features":feat}
        with open(osp.join(self.od,"meta","info.json"),"w") as f: json.dump(info,f,indent=2)
        print(f"Dataset: {TE} eps, {TF} frames -> {self.od}")
    def _stats(self,df):
        s={}; a=np.stack(df["action"].values)
        s["action"]={"mean":a.mean(0).tolist(),"std":a.std(0).tolist(),"max":a.max(0).tolist(),"min":a.min(0).tolist(),"count":[len(a)]}
        st=np.stack(df["observation.state"].values)
        s["observation.state"]={"mean":st.mean(0).tolist(),"std":st.std(0).tolist(),"max":st.max(0).tolist(),"min":st.min(0).tolist(),"count":[len(st)]}
        for fld in ["timestamp","frame_index","episode_index","index","task_index"]:
            v=df[fld].values
            s[fld]={"mean":[float(v.mean())],"std":[float(v.std())],"max":[int(v.max())] if fld!="timestamp" else [float(v.max())],"min":[int(v.min())] if fld!="timestamp" else [float(v.min())],"count":[len(v)]}
        return s

def collect_data(args):
    outdir=osp.abspath(args.output_dir); os.makedirs(outdir,exist_ok=True)
    writer=LeRobotWriter(outdir); seed=args.seed
    passed=0; attempts=0; fmp=0; fsucc=0; t0=time.time()
    # Create env ONCE and reuse for all episodes
    env=gym.make("PegInsertionVertical-v1",num_envs=1,obs_mode="rgb",robot_uids="panda_wristcam",control_mode="pd_joint_pos",sim_backend="cpu",render_mode="all",reward_mode="normalized_dense",max_episode_steps=600,sensor_configs=dict(shader_pack="default"),human_render_camera_configs=dict(shader_pack="default"))
    recorder=ObservationRecorder(env)
    pbar=tqdm(total=args.num_traj,desc="Collecting")
    while passed<args.num_traj:
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
                seed+=1; recorder.stop()
                continue
        except Exception as e:
            print(f"Error seed={seed}: {e}"); traceback.print_exc(); fmp+=1; seed+=1; recorder.stop()
            continue
        recorder.stop()
        records=recorder.records
        if not records:
            print(f"No records seed={seed}"); seed+=1; continue
        acts=compute_ee_delta_actions(records)
        states=np.stack([r["pre"]["state"] for r in records])
        bf=np.stack([r["pre"]["base_camera_rgb"] for r in records])
        wf=np.stack([r["pre"]["hand_camera_rgb"] for r in records])
        rf=np.stack([r["pre"]["render_rgb"] for r in records])
        writer.add_episode(acts,states,bf,wf,rf)
        passed+=1; pbar.update(1)
        el=time.time()-t0; rate=passed/el if el>0 else 0
        pbar.set_postfix(dict(succ=f"{passed/attempts:.2%}",fmp=fmp,fs=fsucc,epl=len(records),r=f"{rate:.1f}e/s"))
        seed+=1
    pbar.close(); writer.finalize()
    try: env.close()
    except: pass
    el=time.time()-t0
    print(f"Done! {passed}/{attempts} ({passed/attempts:.1%}), time={el:.1f}s, out={outdir}")

def main():
    p=argparse.ArgumentParser(description="Collect PegInsertionVertical data")
    p.add_argument("--num-traj",type=int,default=500)
    p.add_argument("--output-dir",type=str,default="/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical")
    p.add_argument("--seed",type=int,default=0)
    args=p.parse_args(); collect_data(args)

if __name__=="__main__": main()
