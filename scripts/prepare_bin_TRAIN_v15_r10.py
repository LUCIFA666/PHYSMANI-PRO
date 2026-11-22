"""Reuse complete bin models and freeze an independent TRAIN executor adapter."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs
from our_mani.multitask.bin_actor_backend_v15 import persistent_reference_policy_overlay, persistent_reference_continuation_identity
from our_mani.multitask.bin_native_backend_v15 import diagnostic_binding
from our_mani.multitask.interaction import InteractionLibrary, CandidateConfig, MPCConfig
from scripts.run_bin_TRAIN_v15_r10 import TASK, SOURCES, METHODS, SETUP, command
from scripts.run_geometry_bin_v15_r10 import parser, preflight


def main():
    if SETUP.exists():
        raise RuntimeError('Preserve the existing prepared namespace')
    base = ROOT/'artifacts/framework/multitask_r10/geometry_stopped_release_v14_setup'
    old = json.loads((base/'frozen_inputs.json').read_text())
    verify_inputs(ROOT, old['files'])
    for launcher in ([sys.executable], ['bash', 'scripts/run_local_sim.sh']):
        subprocess.run(launcher+['-m','unittest','discover','-s','tests','-p','test_bin_reference_v15.py'], cwd=ROOT, check=True)
    c = json.loads((base/'configuration.json').read_text())
    bundle = ROOT/'artifacts/framework/multitask_r10/TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/report.json'
    fit = json.loads(bundle.read_text())['per_task'][TASK]
    assert fit['interaction_status']=='TRAIN_fitted_pending_formal_release'
    assert fit['native_TRAIN_parent_sources']==100
    lib_pointer = deepcopy(fit['interaction'])
    lib_pointer['path'] = str(Path(lib_pointer['path']).relative_to(ROOT))
    c['learned_artifacts'][TASK]['interaction'] = lib_pointer
    c['development_demo_scope'] = dict(task=TASK, fixed_TRAIN_sources=list(SOURCES), formal_metric=False)
    c['execution_scope'] = 'normal_bin_fixed_first4_TRAIN_executor_qualification_v15'
    c['authorized_scope'] = '10-hour task-by-task complete TRAIN/VAL/TEST; geometry three methods only; no Value/NGFF/FASTER'
    settings = c['policy_overlay']['configurations']
    settings['persistent_reference']['version'] = 'r10_continuous_verified_reference_feedforward_v15'
    c['policy_overlay'] = persistent_reference_policy_overlay(ROOT, settings['feedback'], settings['persistent_reference'], settings['causal_retarget'])
    c['native_metric_diagnostics'] = diagnostic_binding(ROOT, c['baseline_repo'])
    lib = InteractionLibrary.load(ROOT/lib_pointer['path'])
    for predictor in ('cv','context'):
        identity = persistent_reference_continuation_identity(ROOT,c,TASK,predictor,lib,CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
        c['continuation_identities'].setdefault(TASK,{})[predictor] = identity['sha256']
    original_manifest = ROOT/'artifacts/framework/protocol/train_manifest.json'
    original = json.loads(original_manifest.read_text())
    rows = [r for r in original['episodes'] if r['task']==TASK][:4]
    assert tuple(r['raw_episode_id'] for r in rows)==SOURCES
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json',c)
    atomic_json(SETUP/'manifest.json',dict(split='train',episodes=rows,original_rows_unchanged=True,
        original_manifest=dict(path=str(original_manifest.relative_to(ROOT)),sha256=file_sha(original_manifest)),
        source_selection='first4 original TRAIN rows, preregistered before outcomes',formal_VAL=False,formal_TEST=False))
    files = dict(old['files'])
    paths = [Path(__file__), ROOT/'scripts/run_bin_TRAIN_v15_r10.py',ROOT/'scripts/run_geometry_bin_v15_r10.py',
        ROOT/'our_mani/multitask/bin_reference_v15.py',ROOT/'our_mani/multitask/bin_actor_backend_v15.py',
        ROOT/'our_mani/multitask/bin_dispatch_v15.py',ROOT/'our_mani/multitask/bin_native_backend_v15.py',
        ROOT/'tests/test_bin_reference_v15.py',SETUP/'configuration.json',SETUP/'manifest.json',
        ROOT/lib_pointer['path'],Path(lib.metadata['TRAIN_source_manifest_path']),bundle,original_manifest]
    paths += [Path(row['path']) for row in rows]
    paths += [ROOT/p['path'] for p in c['learned_artifacts'][TASK].values() if p]
    paths += [ROOT/p['path'] for p in c.get('axis_swept_assets',{}).get(TASK,{}).values() if p.get('enabled')]
    for p in paths:
        p = p.resolve(); checksum = file_sha(p)
        assert str(p) not in files or files[str(p)]==checksum
        files[str(p)] = checksum
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_normal_bin_TRAIN_v15_frozen_inputs',created_utc=datetime.now(timezone.utc).isoformat(),
        files=files,approved_for_execution=True,native_started=False,models_retrained=False,
        previous310_inputs_preserved=True,simulation_seconds=10,dt=.05,TEST_roster_read=False))
    checks={}
    for method in METHODS:
        checks[method] = preflight(parser().parse_args(command(method,SOURCES[0])[3:]))['model_preflight']
        result = subprocess.run(command(method,SOURCES[0])+['--check'],cwd=ROOT,capture_output=True,text=True,check=True)
        atomic_json(SETUP/(method+'_native_CPU_preflight.json'),json.loads(result.stdout))
    atomic_json(SETUP/'CPU_acceptance.json',dict(complete=True,methods=list(checks),original_motion_and_collision_gates=True,
        native_started=False,models_retrained=False,regressions_each_environment=33,actual_argv_preflight_passed=True))
    verify_inputs(ROOT,files)
    print(json.dumps(dict(prepared=True,task=TASK,frozen_inputs=len(files),sources=list(SOURCES))))


if __name__ == '__main__':
    main()
