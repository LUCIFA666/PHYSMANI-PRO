"""Require a full actual-state proof for the same next local block command."""
from pathlib import Path
from copy import deepcopy
from .block_fresh_tail_v448 import SmoothBlockEnvironment as OriginalBoundary
from .next_sample_actual_v455 import check_first,binding as sample_binding
from .evaluation import file_sha
from ..contracts import Reachability
SOURCES=('our_mani/multitask/block_next_sample_v456.py','our_mani/multitask/next_sample_actual_v455.py')

def binding(root):
 return dict(schema='r10_block_exact_next_actual_local_and_brake_v456',next_sample=sample_binding(root),
  sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
  old_complete_dual_paths_exact_collision_and_counted_fresh_tail_retained=True,
  native_motion_guaranteed=False,formal_VAL_TEST=False)

class ActualFirstGate:
 def __init__(self,original,boundary):self.original,self.boundary=original,boundary
 def __getattr__(self,name):return getattr(self.original,name)
 def check_joints(self,robot,q,v,duration):
  full=self.original.check_joints(robot,q,v,duration)
  if not full.feasible:return full
  b=self.boundary
  proof=check_first(self.original,robot,b.reference_q,b.reference_v,q,v,duration,float(b.env.ledger.dt))
  b.next_sample_events.append(dict(time=float(robot.timestamp),passed=proof.feasible,
   reason=proof.reason,evidence=deepcopy(proof.evidence),full_original_actual_path_retained=True))
  return full if proof.feasible else Reachability(False,'block_exact_next_actual_'+proof.reason,evidence=proof.evidence)

class SmoothBlockEnvironment(OriginalBoundary):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.next_sample_events=[]
 def _gates(self):
  actual,reference=super()._gates()
  return ActualFirstGate(actual,self),reference
 def receipt(self):return dict(super().receipt(),block_next_sample_v456=dict(
  binding=binding(Path(__file__).resolve().parents[2]),events=deepcopy(self.next_sample_events)))
