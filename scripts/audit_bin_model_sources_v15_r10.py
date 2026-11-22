"""Same complete model gate as the prior audit, with audit-only SHA deduplication."""
import json,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import evaluation
from our_mani.multitask.audit_sha_cache_v12 import AuditSHACache
from our_mani.multitask.bin_geometry_task_release_v15 import audit_task_model_coverage
from our_mani.multitask import acquisition_lineage,reference_time,reference_overrides
# These independent historical readers define byte-identical SHA helpers.
originals=(evaluation.file_sha,acquisition_lineage.file_sha,reference_time.sha,reference_overrides.sha)
cache=AuditSHACache()
# Already imported aliases and future local imports use the same guarded cache.
for name,module in list(sys.modules.items()):
 if name.startswith(('our_mani','scripts')):
  for attribute in ('file_sha','sha'):
   function=getattr(module,attribute,None)
   if any(function is original for original in originals):setattr(module,attribute,cache)
B=ROOT/'artifacts/framework/multitask_r10/bin_model_readiness_v15'
B.mkdir(exist_ok=True)
if (B/'coverage_initial.json').exists():raise RuntimeError('Preserve existing completed model audit')
stop=threading.Event()
def progress():
 while not stop.wait(10):evaluation.atomic_json(B/'audit_progress.json',cache.receipt(),overwrite=True)
thread=threading.Thread(target=progress,daemon=True);thread.start()
try:
 c=json.loads((ROOT/'artifacts/framework/multitask_r10/bin_executor_v15_setup/configuration.json').read_text())
 view=ROOT/'artifacts/framework/multitask_r10/context_training_view_full_v6/manifest.json'
 result=audit_task_model_coverage(ROOT,'put_rubbish_in_moving_bin',c,context_training_view=dict(path=str(view.relative_to(ROOT)),sha256=cache(view)))
 for p in (Path(__file__),ROOT/'our_mani/multitask/audit_sha_cache_v12.py',ROOT/'tests/test_audit_sha_cache_v12.py'):
  result['input_files_sha256'][str(p.relative_to(ROOT))]=cache(p)
 result['audit_only_hash_cache']=cache.receipt();evaluation.atomic_json(B/'coverage_initial.json',result)
 print(json.dumps(dict(passed=result['formal_model_coverage_passed'],errors=result['errors'],frozen_files=len(result['input_files_sha256']),cache=cache.receipt())),flush=True)
finally:stop.set();thread.join()
