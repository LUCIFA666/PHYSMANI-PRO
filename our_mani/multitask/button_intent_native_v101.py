"""Unchanged v95 policy with native button success-component diagnostics."""
from pathlib import Path
from .button_intent_native_v92 import ButtonIntentNativeBackend as OriginalBackend
from .button_intent_native_v92 import model_preflight as original_preflight
from .button_success_metrics_v100 import snapshot
from .evaluation import file_sha


def model_preflight(root,prepared):
    from scripts.evaluate_multitask_case_r10 import locked_digest
    result=original_preflight(root,prepared);root=Path(root)
    sources={name:file_sha(root/name) for name in (
        'our_mani/multitask/button_intent_native_v101.py',
        'our_mani/multitask/button_success_metrics_v100.py')}
    for name,checksum in sources.items():
        if locked_digest(root/name,prepared['files'])!=checksum:raise ValueError('freeze native metric observer sources')
    return dict(result,button_success_components_metric_only_sources=sources,
        policy_inputs_changed=False,policy_commands_changed=False)


class ButtonIntentNativeBackend(OriginalBackend):
    def _capture(self,env,stage):
        row=super()._capture(env,stage)
        if stage=='after_dispatch':row['button_success_components_v100']=snapshot(env)
        return row
