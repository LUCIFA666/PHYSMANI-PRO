"""Unchanged remove completion geometry; zero local twist at command endpoints."""
from pathlib import Path
from .remove_uncertain_translation_executor import RemoveUncertainTranslationBackend,verify as previous_verify
from .remove_reference_reserve import TASKS
from .canonical_terminal_curve import install
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
PHASES=('approach','acquire','detach')
SOURCES=('our_mani/multitask/remove_terminal_executor_v470.py','our_mani/multitask/canonical_terminal_curve.py')
def binding(root,config,task):
 if task not in TASKS:raise ValueError('explicit remove task only')
 return dict(task=task,phases=list(PHASES),sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
  original_duration_completion_canonical_author_wp1_guard_and_physical_grasp_retained=True,
  external_frame_velocity_preserved=True,formal_VAL_TEST=False)
def verify(root,prepared):
 previous_verify(root,prepared);expected=binding(root,prepared['config'],prepared['args'].task)
 if prepared['config'].get('remove_terminal_v470')!=expected:raise ValueError('freeze remove canonical terminal control')
 checked_subset(root,prepared['files'],{str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
 return expected
class RemoveTerminalBackend(RemoveUncertainTranslationBackend):
 def actor(self,prepared,env,descriptions):
  actor,identity=super().actor(prepared,env,descriptions);install(actor,phases=PHASES);return actor,identity
