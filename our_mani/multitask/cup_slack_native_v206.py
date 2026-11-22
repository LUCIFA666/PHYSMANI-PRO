from pathlib import Path
from .cup_transit_native_v181 import NormalCupTransitBackend,HighCupTransitBackend
from .cup_transit_native_v181 import model_preflight as original_preflight
from . import cup_phase_slack_v205 as policy
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def model_preflight(root,prepared):
    result=original_preflight(root,prepared);task=prepared['args'].task
    expected=policy.binding(root,prepared['config'],task)
    if prepared['config'].get('cup_phase_slack_v205',{}).get(task)!=expected:raise ValueError('actual phase allowance binding differs')
    files={str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()}
    files[str(Path(__file__))]=file_sha(__file__);checked_subset(root,prepared['files'],files)
    return dict(result,cup_phase_slack_v205=expected)


class NormalCupSlackBackend(NormalCupTransitBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        policy.install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity


class HighCupSlackBackend(HighCupTransitBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        policy.install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
