"""Original cup control with an additional finger contact corridor."""
from pathlib import Path
from .cup_contact_speed_executor import CupContactSpeedBackend, verify as previous_verify
from .grasp_contact_corridor_v502 import install, binding as contact_binding
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha

SOURCES = ('our_mani/multitask/cup_contact_corridor_v503.py',
           'our_mani/multitask/grasp_contact_corridor_v502.py')


def binding(root, config, task):
    return dict(task=task, contact=contact_binding(root, config['baseline_repo']),
        sources_sha256={p: file_sha(Path(root)/p) for p in SOURCES},
        original443_control_and_models_retained=True,
        extra455_473_484_target_constraints_not_installed=True, formal_VAL_TEST=False)


def verify(root, prepared):
    previous_verify(root, prepared)
    expected = binding(root, prepared['config'], prepared['args'].task)
    if prepared['config'].get('cup_contact_corridor_v503') != expected:
        raise ValueError('freeze additional public finger contact corridor')
    subset = {str(Path(root)/p): sha for p, sha in expected['sources_sha256'].items()}
    subset.update(expected['contact']['native_static_API_sources_sha256'])
    checked_subset(root, prepared['files'], subset)
    return expected


class CupContactCorridorBackend(CupContactSpeedBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        install(actor, env, native_repo=prepared['config']['baseline_repo'])
        return actor, identity
