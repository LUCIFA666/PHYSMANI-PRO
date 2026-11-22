"""Shared paths and atomic status files for the unattended dynamics run."""
import json
import os
import signal
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/"runtime/continuous"
STATE=RUNTIME/"state"
DATA=ROOT/"data/continuous"
OUT=ROOT/"artifacts/continuous"
STOP=STATE/"STOP"
CONFIG={
    "seed":0,"hidden_dim":200,"num_layers":4,"keypoints_per_object":2048,
    "frames":80,"sampled_dt":0.02,"ode_step_size":0.02,"batch_size":32,
    "lr":1e-5,"min_lr":1e-7,"decay_reference_epochs":1001,
    "max_epochs":None,"validation_interval_seconds":180,
    "checkpoint_interval_seconds":60,"gradient_clip_norm":1.,
    "bootstrap_updates":400,"bootstrap_lr":1e-4,
    "minimum_canonical_training_scenes":2,"minimum_canonical_validation_scenes":2,
    "memory_cap_gib":26,"download_workers":8,
    "data_revision":"4db8e745c1c4f77cdfd74334110ba1156ac8270e",
    "initialization":"new 200-wide model; coarse pretraining on the cached 8-scene split",
    "formal_test":"3_9, 4 and 6 excluded from training; previously inspected 2700/2701 are development examples",
}


def setup():
    for p in (RUNTIME,STATE,DATA,OUT,ROOT/"logs/continuous"):
        p.mkdir(parents=True,exist_ok=True)
    path=RUNTIME/"config.json"
    if not path.exists():atomic_json(path,CONFIG)
    return read_json(path,CONFIG)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write("\n")
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return default


def event(role,kind,**details):
    row={"time":time.time(),"role":role,"event":kind,**details}
    print(json.dumps(row,ensure_ascii=False),flush=True)


class StopFlag:
    def __init__(self):
        self.signaled=False
        def handler(*_):self.signaled=True
        signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    def requested(self):return self.signaled or STOP.exists()
    def sleep(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end and not self.requested():time.sleep(min(1,end-time.monotonic()))
