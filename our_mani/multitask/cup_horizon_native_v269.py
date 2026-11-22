from pathlib import Path
from .cup_author_native_v252 import model_preflight as original_preflight
from .cup_author_native_v252 import NormalCupSlackBackend as OriginalNormal,HighCupSlackBackend as OriginalHigh
from . import cup_transit_horizon_v268 as policy
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def model_preflight(root,prepared):
    result=original_preflight(root,prepared);task=prepared['args'].task
    expected=policy.binding(root,prepared['config'],task)
    if prepared['config'].get('cup_transit_horizon_v268',{}).get(task)!=expected:raise ValueError('actual longer prefix differs')
    files={str(Path(root)/name):sha for name,sha in expected['sources_sha256'].items()}
    files[str(Path(__file__))]=file_sha(__file__);checked_subset(root,prepared['files'],files)
    return dict(result,cup_transit_horizon_v268=expected)


class NormalCupSlackBackend(OriginalNormal):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        policy.install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity


class HighCupSlackBackend(OriginalHigh):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        policy.install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
