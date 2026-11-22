"""Prepare resumable 2048-point / 80-frame official trajectories in background."""
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import requests
import torch
from dataset.constants import INDEX_OBJ,OBJPART
from utils.geometry_utils import farthest_point_sampling_indices
from scripts.prepare_quality_smoke import fetch,layout,read_positions,REVISION
from scripts.continuous_common import DATA,STATE,atomic_json,read_json,event,setup,StopFlag


def api_list(path):
    url=f"https://huggingface.co/api/datasets/lishiqianhugh/GSCollision/tree/{REVISION}/{path}?limit=1000"
    for attempt in range(5):
        try:
            r=requests.get(url,timeout=(15,30));r.raise_for_status();return r.json()
        except Exception as exc:
            event("data","api_retry",path=path,attempt=attempt+1,error=type(exc).__name__)
            time.sleep(min(2**attempt,8))
    raise RuntimeError(f"cannot list {path}")


def make_plan():
    existing=read_json(DATA/"plan.json")
    if existing:return existing
    folders={x["path"].split("/")[-1] for x in api_list("mpm") if x["type"]=="directory"}
    groups=[f"3_{i}" for i in range(9) if f"3_{i}" in folders]
    train=[];val=[];group_counts={}
    for group in groups:
        target=DATA/"source_configs"/f"{group}.json"
        if not target.exists():target.write_bytes(fetch(f"scene_configs/{group}.json")[0])
        configs=json.loads(target.read_text())
        available={x["path"].split("/")[-1] for x in api_list("mpm/"+group) if x["type"]=="directory"}
        names=sorted(set(configs)&available,key=lambda name:int(name.split("_")[0]))
        rng=random.Random(REVISION+group)
        if group=="3_0":
            old_val=[name for name in names if int(name.split("_")[0]) in (8,9)]
            other=[name for name in names if int(name.split("_")[0])>=10]
            validation=set(old_val+rng.sample(other,4-len(old_val)))
        else:validation=set(rng.sample(names,min(4,len(names)//10)))
        group_counts[group]={"configured":len(configs),"uploaded":len(names),"validation":len(validation),
                             "train":len(names)-len(validation)}
        for name in names:
            objs=[INDEX_OBJ[i] for i in configs[name]["scene_object_idxs"]]
            entry={"group":group,"name":name,"source":f"mpm/{group}/{name}",
                   "objects":objs,"counts":[OBJPART[obj] for obj in objs],
                   "split":"validation" if name in validation else "train"}
            (val if name in validation else train).append(entry)
    # Commit the entire split before any long training. Former smoke validation
    # scenes stay excluded from the training pool.
    random.Random(0).shuffle(train)
    train.sort(key=lambda x:0 if x["group"]=="3_0" and x["name"].split("_")[0] in ("0","1") else 1)
    val.sort(key=lambda x:0 if x["group"]=="3_0" and x["name"].split("_")[0] in ("8","9") else 1)
    priority=[]
    for _ in range(2):priority += [val.pop(0),train.pop(0)]
    while train or val:
        if val:priority.append(val.pop(0))
        priority.extend(train[:4]);train=train[4:]
    plan={"revision":REVISION,"available_training_groups":groups,
          "group_counts":group_counts,
          "missing_paper_training_groups":[f"3_{i}" for i in range(9) if f"3_{i}" not in folders],
          "formal_test_groups_excluded":["3_9","4","6"],
          "previously_inspected_development_scenes":["2700_duck_bowl_ball","2701_bowl_pillow_can"],
          "entries":priority}
    atomic_json(DATA/"plan.json",plan)
    return plan


def prepare(entry,config,stop):
    stem=entry["group"]+"_"+entry["name"]
    cache=DATA/"cache"/(stem+".npz");meta=cache.with_suffix(".json")
    if cache.exists() and meta.exists():return read_json(meta)
    part=DATA/"partial"/stem;part.mkdir(parents=True,exist_ok=True)
    frames=list(range(config["frames"]));count=config["keypoints_per_object"]
    spec=read_json(part/"layout.json")
    if spec is None:
        spec=layout(entry["source"]+"/0000.h5")
        assert spec==layout(entry["source"]+f"/{frames[-1]:04d}.h5")
        assert sum(entry["counts"])==spec["shape"][0],"object partition mismatch"
        atomic_json(part/"layout.json",spec)
    progress=read_json(part/"progress.json",{"completed":{},"indices_ready":False})
    if not progress["indices_ready"]:
        raw,digest=read_positions(entry["source"]+"/0000.h5",spec)
        ids=torch.repeat_interleave(torch.arange(3),torch.tensor(entry["counts"])).unsqueeze(-1)
        seed=int(hashlib.sha256(entry["source"].encode()).hexdigest()[:8],16)
        torch.manual_seed(seed)
        indices=farthest_point_sampling_indices(torch.tensor(raw.astype(np.float32)),ids,count).numpy()
        np.save(part/"indices.npy",indices)
        tracks=np.lib.format.open_memmap(part/"tracks.npy",mode="w+",dtype=np.float32,shape=(len(frames),3,count,3))
        tracks[0]=raw[indices].reshape(3,count,3);tracks.flush()
        progress={"completed":{"0":digest},"indices_ready":True}
        atomic_json(part/"progress.json",progress)
    else:
        indices=np.load(part/"indices.npy")
        tracks=np.lib.format.open_memmap(part/"tracks.npy",mode="r+")
    missing=[frame for frame in frames if str(frame) not in progress["completed"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["download_workers"]) as pool:
        futures={pool.submit(read_positions,entry["source"]+f"/{frame:04d}.h5",spec):frame for frame in missing}
        for future in concurrent.futures.as_completed(futures):
            frame=futures[future];raw,digest=future.result()
            tracks[frame]=raw[indices].reshape(3,count,3)
            progress["completed"][str(frame)]=digest
            if len(progress["completed"])%5==0 or stop.requested():
                tracks.flush();atomic_json(part/"progress.json",progress)
                status={"status":"downloading","current":entry["source"],
                        "completed_frames":len(progress["completed"]),"frames":len(frames),"time":time.time()}
                atomic_json(STATE/"data_status.json",status)
                event("data","frames",**status)
            if stop.requested():
                for pending in futures:pending.cancel()
                return None
    tracks.flush();atomic_json(part/"progress.json",progress)
    assert len(progress["completed"])==len(frames) and np.isfinite(tracks).all()
    assert np.linalg.norm(tracks[-1]-tracks[0],axis=-1).mean()>1e-4
    tmp=cache.with_suffix(".npz.tmp")
    with tmp.open("wb") as file:
        np.savez_compressed(file,points=tracks,indices=indices,frames=np.array(frames),dt=np.float32(.02))
    os.replace(tmp,cache)
    metadata={**entry,"cache":str(cache.relative_to(ROOT)),"revision":REVISION,
              "frames":len(frames),"keypoints_per_object":count,"sampled_dt":.02,
              "cache_sha256":hashlib.sha256(cache.read_bytes()).hexdigest(),
              "sample_indices_sha256":hashlib.sha256(indices.tobytes()).hexdigest(),
              "sampling":"upstream per-object FPS; seed=first 32 bits of SHA256(source)",
              "position_block_sha256":progress["completed"]}
    atomic_json(meta,metadata)
    del tracks
    shutil.rmtree(part)
    event("data","scene_ready",source=entry["source"],split=entry["split"])
    return metadata


def main():
    config=setup();torch.set_num_threads(4);stop=StopFlag()
    (DATA/"cache").mkdir(exist_ok=True);(DATA/"source_configs").mkdir(exist_ok=True)
    plan=make_plan()
    failures=read_json(DATA/"failures.json",{})
    while not stop.requested():
        for entry in plan["entries"]:
            if stop.requested():break
            if shutil.disk_usage(ROOT).free<20*1024**3:
                atomic_json(STATE/"data_status.json",{"status":"waiting_for_disk_space","time":time.time()})
                stop.sleep(60);continue
            try:
                result=prepare(entry,config,stop)
                if result is not None:failures.pop(entry["source"],None)
            except Exception as exc:
                failures[entry["source"]]={"error_type":type(exc).__name__,"time":time.time()}
                event("data","scene_retry_later",source=entry["source"],error=type(exc).__name__)
                atomic_json(DATA/"failures.json",failures)
                stop.sleep(5)
        if stop.requested():break
        ready=[read_json(p) for p in (DATA/"cache").glob("*.json")]
        atomic_json(STATE/"data_status.json",{"status":"cache_pass_complete","time":time.time(),
                    "train_ready":sum(x["split"]=="train" for x in ready if x),
                    "validation_ready":sum(x["split"]=="validation" for x in ready if x),
                    "failures":len(failures),"missing_groups":plan["missing_paper_training_groups"]})
        stop.sleep(300)
    status=read_json(STATE/"data_status.json",{})
    status.update(status="stopped",time=time.time())
    atomic_json(STATE/"data_status.json",status)
    event("data","stopped")


if __name__=="__main__":main()
