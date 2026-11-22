"""The public demo bank contract is shared; model/checkpoint family is v3."""
from ..world_tcp_command_warmstart_20261104_v2.runtime_adapter import (
    DemoCommandBankV2,bank_from_model_output,generic_v1_proposals,select_generic_reference,
)


def bank_from_v3_model_output(raw,*,batch_index=0):
    # No model-version relabeling: this function only converts the shared
    # unqualified public command schema. Policy wholeSHA is supplied separately.
    return bank_from_model_output(raw,batch_index=batch_index)
