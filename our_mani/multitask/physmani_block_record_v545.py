"""Translate actual legacy parent reset metrics into the common case format."""
from copy import deepcopy
from .evaluation import native_completion_record


def completion_record(identity,*,initial_conditions,**kwargs):
    # The legacy metric collector predates the explicit policy-use flags. The
    # caller collected these values in the parent and never forwarded them to
    # the public frontend. Preserve the entire measured payload verbatim.
    parent_metrics=dict(metric_only=True,policy_use_allowed=False,
        legacy_parent_initial_conditions=deepcopy(initial_conditions),
        format_adapter='r10_original_legacy_parent_metrics_v545')
    return native_completion_record(identity,initial_conditions=parent_metrics,**kwargs)
