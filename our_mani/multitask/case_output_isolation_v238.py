"""Reject the observed cross-method output collision before any case starts."""
from pathlib import Path


def require_development_output(root, output, *, task, raw, method):
    root = Path(root).resolve()
    output = (root/output).resolve()
    base = root/'artifacts/framework/multitask_r10'
    if (method not in ('cv_geometry', 'context_geometry', 'physmani')
            or not output.is_relative_to(base)
            or 'development_demo_only' not in output.relative_to(base).parts
            or output.parts[-3:] != (task, raw, method)):
        raise ValueError('development output must uniquely encode the actual task, source and method')
    return output
