"""Remove executor with current translation-only cup registration fallback."""
from pathlib import Path
from .remove_continuous_executor import RemoveContinuousBackend,verify as verify_original
from .remove_reference_reserve import TASKS
from .cup_translation_uncertainty import binding as fallback_binding,install
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset
SOURCES=('our_mani/multitask/remove_uncertain_translation_executor.py','our_mani/multitask/cup_translation_uncertainty.py','our_mani/multitask/translation_uncertainty_mesh.py','our_mani/multitask/lossless_case_archive.py')


def binding(root,config,task):
    if task not in TASKS:raise ValueError('explicit two remove tasks')
    return dict(remove_uncertain_translation_executor=dict(task=task,fallback=fallback_binding(root),
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},previous_extra440_spread_rejection_superseded_by_explicit_enlarged_collision_geometry=True,formal_VAL_TEST=False))


def verify(root,prepared):
    verify_original(root,prepared);expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('remove_uncertain_translation_executor')!=expected['remove_uncertain_translation_executor']:
        raise ValueError('freeze current translation-only registration fallback')
    checked_subset(root,prepared['files'],{str(Path(root)/p):file_sha(Path(root)/p) for p in SOURCES})
    return expected


class RemoveUncertainTranslationBackend(RemoveContinuousBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions);install(actor)
        return actor,identity
