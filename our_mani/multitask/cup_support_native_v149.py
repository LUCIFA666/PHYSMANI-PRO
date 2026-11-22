"""Two explicit cup-placement native consumers with public support measurement."""
from pathlib import Path
from .normal_cup_diagonal_native_v123 import NormalCupNativeBackend
from .high_cup_native_v134 import HighCupNativeBackend
from .normal_cup_preflight_v139 import model_preflight as normal_preflight
from .high_cup_native_v134 import model_preflight as high_preflight
from . import cup_public_support_v148 as support
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def model_preflight(root,prepared):
    task=prepared['args'].task
    if task not in support.TASKS:raise ValueError('explicit two-cup TRAIN consumers only')
    result=(normal_preflight if task==support.TASKS[0] else high_preflight)(root,prepared)
    expected=support.binding(root,prepared['config'],task)
    if prepared['config'].get('cup_public_support_v148',{}).get(task)!=expected:raise ValueError('actual support binding differs')
    sources={str(Path(root)/name):sha for name,sha in expected['sources_sha256'].items()}
    for name in ('our_mani/multitask/cup_support_native_v149.py','our_mani/multitask/normal_cup_preflight_v139.py'):
        sources[str(Path(root)/name)]=file_sha(Path(root)/name)
    checked_subset(root,prepared['files'],sources)
    return dict(result,cup_public_support_v148=expected)


class NormalCupSupportBackend(NormalCupNativeBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        support.install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity


class HighCupSupportBackend(HighCupNativeBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        support.install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
