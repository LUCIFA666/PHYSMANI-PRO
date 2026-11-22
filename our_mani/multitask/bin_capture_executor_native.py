"""High-bin author approach followed by a collision-checked physical capture region."""
from pathlib import Path
from .bin_author_executor_native import BinAuthorBackend,binding as parent_binding,verify as verify_parent
from .bin_capture_region import binding as region_binding,install
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
SOURCES=('our_mani/multitask/bin_capture_executor_native.py','our_mani/multitask/bin_capture_region.py')


def binding(root,config,task):
    return dict(schema='r10_high_bin_author_approach_original_canonical_capture_region_executor_v1',
        parent=parent_binding(root,config,task),capture=region_binding(root,config,task),
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},formal_VAL_TEST=False)


def verify(root,prepared):
    verify_parent(root,prepared);expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('bin_capture_executor')!=expected:raise ValueError('freeze capture-region executor')
    if prepared['config'].get('bin_capture_region')!=expected['capture']:raise ValueError('freeze original physical canonical capture')
    checked_subset(root,prepared['files'],{str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
    return expected


class BinCaptureBackend(BinAuthorBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
