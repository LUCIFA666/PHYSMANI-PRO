"""Cup executor with continuously repaired same-deadline joint references."""
from pathlib import Path
from .cup_contact_speed_executor import CupContactSpeedBackend,verify as previous_verify
from .receding_position_bridge_v484 import install,binding as bridge_binding
from .cup_reference_reserve import TASKS
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
SOURCES=('our_mani/multitask/cup_receding_bridge_v485.py',
 'our_mani/multitask/receding_position_bridge_v484.py','our_mani/multitask/next_position_actual_v473.py')
def binding(root,config,task):
 return dict(task=task,bridge=bridge_binding(root),sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
  previous_extra455_and473_rejection_only_policies_not_installed=True,formal_VAL_TEST=False)
def verify(root,prepared):
 previous_verify(root,prepared);expected=binding(root,prepared['config'],prepared['args'].task)
 if prepared['config'].get('cup_receding_bridge_v485')!=expected:raise ValueError('freeze same deadline measured-state bridge')
 checked_subset(root,prepared['files'],{str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
 return expected
class CupRecedingBridgeBackend(CupContactSpeedBackend):
 def actor(self,prepared,env,descriptions):
  actor,identity=super().actor(prepared,env,descriptions);install(actor);return actor,identity
