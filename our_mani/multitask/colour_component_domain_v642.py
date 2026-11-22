"""Frozen q90 colour bytes with independent full16 runtime qualification.

Retain the original candidate's truthful metadata and every original evidence
check. The new external capability admits its runtime use; it neither mutates
that artifact into a legacy production model nor calls a TRAIN-purpose actor.
"""
from .colour_distribution import DevelopmentQuantileRoleResolver
from .rml_component_domain_v640 import ComponentDomain


class QualifiedQuantileRoleResolver(DevelopmentQuantileRoleResolver):
    def __init__(self,task_spec,artifact,*,root,shared_semantic_permission,domain):
        if not isinstance(domain,ComponentDomain):raise ValueError('sealed component domain required')
        purpose=domain.require(task_spec.task_name)
        # This flag loads unchanged candidate material and its original proof;
        # evaluation authorization is supplied by the separate sealed domain.
        super().__init__(task_spec,artifact,root=root,shared_semantic_permission=shared_semantic_permission,
            allow_development_candidate=True)
        self.runtime_component_domain=domain.receipt()
        self.formal_runtime_admission=purpose=='evaluation'
