"""Read-only capture of the frozen network's existing K8 alternative bank."""
import numpy as np
from ..multitask_executor_cycle_20261112_v6.runtime import PublicContinuousHistoryPolicy
from ..world_tcp_command_warmstart_20261104_v3.runtime_adapter import bank_from_v3_model_output,select_generic_reference
from ..trained_world_reference_live_20261104_v1.runtime import E2_SHA256


class FrozenCommandBankPolicy(PublicContinuousHistoryPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._bank=None
        # The observer returns None, preserving the exact original forward
        # output and every weight/buffer. It introduces no extra forward pass.
        self._capture_handle=self.model.register_forward_hook(self._capture_bank)

    def _capture_bank(self,module,args,output):
        self._bank=bank_from_v3_model_output(output)

    def predict_candidates(self,packet):
        primary=super().predict_reference(packet)
        bank=self._bank
        order=np.argsort(-bank.demo_logits,kind='stable')
        if int(order[0])!=primary.selected_index:raise RuntimeError('Captured bank differs from original primary proposal')
        references=[select_generic_reference(bank,timestamp=packet['timestamp'],policy_sha256=self.checkpoint_sha256,
                      e_sha256=E2_SHA256,selection_rule='caller_selected',selected_index=int(i))for i in order]
        self.last_diagnostic.update(existing_K8_alternatives_captured=True,extra_NN_forward_calls=0,
            demo_likelihood_order=order.tolist(),demo_logits=bank.demo_logits.tolist(),
            proposal_score_is_task_success_probability=False)
        return references
