"""Freeze instrumentation only; preserve every v3 policy/model/physics input."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs
from our_mani.multitask.native_actuator_diagnostic_v4 import diagnostic_binding


def main():
    base=ROOT/'artifacts/framework/multitask_r10/geometry_continuous_reference_smoke_v3_3_setup'
    out=ROOT/'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v4_setup'
    if out.exists(): raise RuntimeError('Preserve prior setup; no overwrite')
    old=json.loads((base/'frozen_inputs.json').read_text())
    verify_inputs(ROOT,old['files'])
    config=json.loads((base/'configuration.json').read_text())
    binding=diagnostic_binding(ROOT,config['baseline_repo'])
    config['native_metric_diagnostics']=binding
    out.mkdir()
    atomic_json(out/'configuration.json',config)
    files=deepcopy(old['files'])
    for path in (out/'configuration.json',Path(__file__),
        ROOT/'scripts/run_geometry_native_actuator_diagnostic_v4_r10.py',
        ROOT/'scripts/run_native_actuator_diagnostic_pair_v4_r10.py',
        ROOT/binding['source']['path'],*(Path(p) for p in binding['native_API_sources_sha256'])):
        files[str(path.resolve())]=file_sha(path)
    lock=dict(old,schema='r10_native_actuator_readonly_nonfitting_TRAIN_actual_lock_v4',
        created_utc=datetime.now(timezone.utc).isoformat(),files=files,
        scope='Instrumentation only around frozen v3: fixed normal basketball TRAIN104 CV/Context pair; no policy input/command/model/physics/deadline changes. No fitting/VAL/TEST/video/retry.',native_started=False)
    atomic_json(out/'frozen_inputs.json',lock)
    verify_inputs(ROOT,files)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    rows=[]
    for i,(name,sha) in enumerate(sorted(files.items())):
        path=Path(name)
        if path.suffix not in ('.py','.sh'): continue
        target=snapshot/f'{i:04d}_{path.name}';shutil.copyfile(path,target)
        if file_sha(target)!=sha: raise RuntimeError('Archive copy SHA differs')
        rows.append(dict(original_path=name,sha256=sha,archived_path=str(target.relative_to(ROOT))))
    verify_inputs(ROOT,files)
    atomic_json(out/'source_snapshot.json',dict(schema='r10_fixed_TRAIN_smoke_actual_executable_archive_v1',
        files=rows,all_frozen_inputs_verified_before_and_after_copy=True,
        original_files_preserved=len(old['files']),current_frozen_input_count=len(files),native_started=False))
    atomic_json(out/'setup_receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        previous_setup=str(base.relative_to(ROOT)),previous_config_sha256=file_sha(base/'configuration.json'),
        previous_lock_sha256=file_sha(base/'frozen_inputs.json'),configuration_sha256=file_sha(out/'configuration.json'),
        frozen_lock_sha256=file_sha(out/'frozen_inputs.json'),diagnostic_binding=binding,
        source_snapshot_sha256=file_sha(out/'source_snapshot.json'),
        original_frozen_files_preserved_exact=all(files.get(k)==v for k,v in old['files'].items()),
        original_frozen_input_count=len(old['files']),current_frozen_input_count=len(files),
        policy_overlay_and_continuation_identities_unchanged=True,
        fixed_original_TRAIN_parent='episode104',task='moving_basketball_in_hoop',methods=['cv_geometry','context_geometry'],
        original_dt=.05,original_task_seconds=10.,feedback_slack_seconds=config['policy_overlay']['configurations']['feedback']['phase_tracking_slack_seconds'],
        original_effective_absolute_deadlines_unchanged=True,formal_metric=False,fitting_source=False,native_started=False))
    print(json.dumps(dict(setup=str(out),configuration_sha256=file_sha(out/'configuration.json'),lock_sha256=file_sha(out/'frozen_inputs.json'))))


if __name__=='__main__': main()
