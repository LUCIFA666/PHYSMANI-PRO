"""Explicit TRAIN-only validation-edge wrapper; original basketball policy intact."""
from pathlib import Path
import contextlib
import io
import json
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.frozen_input_edges_v463 import declared_edge_scope
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.evaluation import atomic_json,file_sha


def main():
 argv=list(sys.argv[1:])
 def value(flag):
  if argv.count(flag)!=1:raise ValueError('exactly one '+flag+' required')
  return argv[argv.index(flag)+1]
 original=(ROOT/value('--original-worker')).resolve()
 if original!=ROOT/'scripts/run_task_executor_smoke_basketball_r10.py':raise ValueError('explicit frozen basketball worker only')
 if (value('--split')!='train' or value('--method') not in ('cv_geometry','context_geometry')
  or value('--task') not in ('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')
  or '--development-demo' not in argv or value('--manifest')!='artifacts/framework/protocol/train_manifest.json'):
  raise ValueError('original TRAIN basketball development only')
 files=json.loads((ROOT/value('--frozen-inputs')).read_text())['files'];verify_inputs(ROOT,files)
 sources=(Path(__file__),ROOT/'our_mani/multitask/frozen_input_edges_v463.py',original)
 if any(files.get(str(p))!=file_sha(p) for p in sources):raise ValueError('freeze validation wrapper and original policy sources')
 witness=dict(schema='r10_declared_input_edges_TRAIN_validation_wrapper_v464',
  sources_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in sources},
  original_controller_config_models_actuation_unchanged=True,
  canonical_exactly_one_match_and_strong_byte_SHA_retained=True,unrelated_outputs_are_not_inputs=True,
  formal_VAL_TEST=False,native_retry=False)
 out=(ROOT/value('--out')).resolve();check='--check' in argv
 i=argv.index('--original-worker');del argv[i:i+2];oldargv=sys.argv;sys.argv=[str(original),*argv]
 try:
  with declared_edge_scope():
   if check:
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):runpy.run_path(str(original),run_name='__main__')
    proof=json.loads(stream.getvalue())
    if not proof['complete'] or proof['native_started'] or proof['GPU_started']:raise ValueError('actual check did not remain preflight only')
    proof['input_lookup_scope']=witness;print(json.dumps(proof),flush=True)
   else:
    runpy.run_path(str(original),run_name='__main__')
    atomic_json(out/'declared_input_edges_execution_v464.json',dict(complete=True,**witness))
 finally:sys.argv=oldargv
if __name__=='__main__':main()
