"""Canonical JSON consumer for the explicit v107 initial-sequence schema.

JSON object key order carries no policy meaning. Reconstruct the declared
partition order before the unchanged v107 proof checker, without changing any
curve, observation, label, failed partition or canonical payload fingerprint.
"""
from copy import deepcopy
from .normal_cup_primary_fitting_v107 import PrimarySequenceLibrary as ProducerLibrary
from .interaction import InteractionArtifactError


class PrimarySequenceLibrary(ProducerLibrary):
    def __init__(self,payload,*,allow_synthetic=False):
        normalized=deepcopy(payload);m=normalized['metadata'];scope=m['initial_sequence_policy_scope']
        keys=list(scope['admitted_partition_keys'])+[row['key'] for row in scope['unadmitted_partitions']]
        fits=m['completed_skill_fitting']['phase_fits']
        if len(keys)!=len(set(keys)) or set(keys)!=set(fits):
            raise InteractionArtifactError('declared policy must retain every original fit partition exactly')
        m['completed_skill_fitting']['phase_fits']={key:fits[key] for key in keys}
        super().__init__(normalized,allow_synthetic=allow_synthetic)
