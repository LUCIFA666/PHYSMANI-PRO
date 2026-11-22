"""Freeze a new TRAIN-only controller; preserve all prior sources and models."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs
from our_mani.multitask.native_actuator_diagnostic_v5 import diagnostic_binding
from our_mani.multitask.continuous_reference_actor_backend_v5 import (
    persistent_reference_policy_overlay, persistent_reference_continuation_identity)
from our_mani.multitask.continuous_reference_feedforward_v5 import SCHEMA
from our_mani.multitask.interaction import InteractionLibrary, CandidateConfig, MPCConfig


def main():
    base=ROOT/'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v4_setup'
    out=ROOT/'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5_setup'
    if out.exists(): raise RuntimeError('Preserve prior setup; no overwrite')
    old=json.loads((base/'frozen_inputs.json').read_text())
    verify_inputs(ROOT,old['files'])
    config=json.loads((base/'configuration.json').read_text())
    configurations=config['policy_overlay']['configurations']
    reference=dict(configurations['persistent_reference'],version=SCHEMA)
    config['policy_overlay']=persistent_reference_policy_overlay(ROOT,
        configurations['feedback'],reference,configurations['causal_retarget'])
    task='moving_basketball_in_hoop'
    library=InteractionLibrary.load(config['learned_artifacts'][task]['interaction']['path'])
    candidate=CandidateConfig(**config['candidate_config']);mpc=MPCConfig(**config['mpc_config'])
    identities={p:persistent_reference_continuation_identity(ROOT,config,task,p,library,candidate,mpc)
        for p in ('cv','context')}
    config['continuation_identities']={task:{p:i['sha256']for p,i in identities.items()}}
    binding=diagnostic_binding(ROOT,config['baseline_repo'])
    config['native_metric_diagnostics']=binding
    out.mkdir()
    atomic_json(out/'configuration.json',config)
    files=deepcopy(old['files'])
    for path in (out/'configuration.json',Path(__file__),
        ROOT/'scripts/run_geometry_native_actuator_diagnostic_v5_r10.py',
        ROOT/'scripts/run_native_actuator_diagnostic_pair_v5_r10.py',
        ROOT/'tests/test_continuous_reference_feedforward_v5.py',
        ROOT/'tests/test_continuous_reference_actor_backend_v5.py',
        ROOT/'tests/test_continuous_direct_counted_setpoint_v5.py',
        ROOT/'tests/test_native_actuator_diagnostic_v5.py',
        *(ROOT/source['path'] for source in config['policy_overlay']['sources'].values()),
        ROOT/binding['source']['path'],*(Path(p) for p in binding['native_API_sources_sha256'])):
        files[str(path.resolve())]=file_sha(path)
    lock=dict(old,schema='r10_native_actuator_readonly_nonfitting_TRAIN_actual_lock_v5',
        created_utc=datetime.now(timezone.utc).isoformat(),files=files,
        scope='Fixed normal basketball TRAIN104 CV/Context: same-tick public acquire-to-lift continuation and TRAIN gripper curves; current causal inputs, full dual reachability/collision gates, prior effective deadlines, slack2 and native physics preserved. No fitting/VAL/TEST/video/retry.',native_started=False)
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
        policy_overlay_and_continuation_identities_unchanged=False,
        policy_overlay=config['policy_overlay'],identities=identities,
        original_models_and_physics_unchanged=True,
        fixed_original_TRAIN_parent='episode104',task='moving_basketball_in_hoop',methods=['cv_geometry','context_geometry'],
        original_dt=.05,original_task_seconds=10.,feedback_slack_seconds=config['policy_overlay']['configurations']['feedback']['phase_tracking_slack_seconds'],
        original_effective_absolute_deadlines_unchanged=True,formal_metric=False,fitting_source=False,native_started=False))
    print(json.dumps(dict(setup=str(out),configuration_sha256=file_sha(out/'configuration.json'),lock_sha256=file_sha(out/'frozen_inputs.json'))))


if __name__=='__main__': main()
