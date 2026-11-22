"""Cup paths additionally prove the exact next public actual-state command."""
from pathlib import Path
from .cup_contact_speed_executor import CupContactSpeedBackend,verify as previous_verify
from .next_position_actual_v473 import install,binding as sample_binding
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
from .cup_reference_reserve import TASKS
SOURCES=('our_mani/multitask/cup_next_position_executor_v478.py','our_mani/multitask/next_position_actual_v473.py')
def binding(root,config,task):
 return dict(supersedes455_only_nominal_derivative_as_actual_endpoint_assumption=True,task=task,next_sample=sample_binding(root),sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},formal_VAL_TEST=False)
def verify(root,prepared):
 previous_verify(root,prepared);expected=binding(root,prepared['config'],prepared['args'].task)
 if prepared['config'].get('cup_next_position_v478')!=expected:raise ValueError('freeze same next command actual-state proof')
 checked_subset(root,prepared['files'],{str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
 return expected
class CupNextPositionBackend(CupContactSpeedBackend):
 def actor(self,prepared,env,descriptions):
  actor,identity=super().actor(prepared,env,descriptions);install(actor);return actor,identity
