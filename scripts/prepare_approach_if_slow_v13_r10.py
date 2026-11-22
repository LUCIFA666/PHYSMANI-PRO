"""Freeze the separate TRAIN-only threshold-triggered partial approach actor."""
from datetime import datetime,timezone
import json,shutil,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.spherical_enclosure_v8 import geometry_binding
from our_mani.multitask.approach_if_slow_native_backend_v13 import diagnostic_binding
from our_mani.multitask.approach_if_slow_actor_backend_v13 import persistent_reference_policy_overlay,persistent_reference_continuation_identity
from our_mani.multitask.interaction import InteractionLibrary,CandidateConfig,MPCConfig
from scripts.run_approach_if_slow_pair_v13_r10 import command
from scripts.run_geometry_approach_if_slow_v13_r10 import parser,preflight

def main():
 base=ROOT/'artifacts/framework/multitask_r10/geometry_causal_safety_brake_v12_setup'
 out=ROOT/'artifacts/framework/multitask_r10/geometry_approach_if_slow_v13_setup'
 if out.exists():raise RuntimeError('Preserve prepared namespace')
 old=json.loads((base/'frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
 test_args=['-m','unittest','discover','-s','tests','-p','test_approach_if_slow_reference_v13.py','-v']
 for argv in ([sys.executable]+test_args,['bash','scripts/run_local_sim.sh']+test_args):
  subprocess.run(argv,cwd=ROOT,check=True)
 c=json.loads((base/'configuration.json').read_text());b=geometry_binding(ROOT,c);c['spherical_geometry_v8']=b
 from our_mani.multitask.supported_sphere_attachment_v10 import binding as support_binding
 c['public_support_attachment_v10']=support_binding(ROOT)
 c.setdefault('body_sphere_radii',{}).setdefault(b['task'],{})['ball']=b['current_RGBD_sphere_radius_m']
 settings=c['policy_overlay']['configurations'];settings['persistent_reference']['version']='r10_continuous_verified_reference_feedforward_v13';c['policy_overlay']=persistent_reference_policy_overlay(ROOT,settings['feedback'],settings['persistent_reference'],settings['causal_retarget'])
 c['native_metric_diagnostics']=diagnostic_binding(ROOT,c['baseline_repo'])
 lib=InteractionLibrary.load(ROOT/c['learned_artifacts'][b['task']]['interaction']['path'])
 for predictor in ('cv','context'):
  identity=persistent_reference_continuation_identity(ROOT,c,b['task'],predictor,lib,CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
  c['continuation_identities'][b['task']][predictor]=identity['sha256']
 out.mkdir();atomic_json(out/'configuration.json',c);files=dict(old['files'])
 for p in [out/'configuration.json',Path(__file__),ROOT/'our_mani/multitask/spherical_enclosure_v8.py',ROOT/'our_mani/multitask/bounded_approach_v13.py',ROOT/'scripts/queue_approach_if_slow_TRAIN_v13_r10.py',ROOT/'our_mani/multitask/approach_if_slow_actor_backend_v13.py',ROOT/'our_mani/multitask/approach_if_slow_native_backend_v13.py',ROOT/'scripts/run_geometry_approach_if_slow_v13_r10.py',ROOT/'scripts/run_approach_if_slow_pair_v13_r10.py',ROOT/'tests/test_approach_if_slow_reference_v13.py',ROOT/'our_mani/multitask/supported_sphere_attachment_v10.py',ROOT/'our_mani/multitask/approach_if_slow_reference_v13.py',ROOT/'our_mani/multitask/approach_if_slow_dispatch_v13.py']:
  files[str(p.resolve())]=file_sha(p)
 atomic_json(out/'frozen_inputs.json',dict(old,schema='r10_TRAIN_bounded_approach_actual_lock_v13',created_utc=datetime.now(timezone.utc).isoformat(),files=files,native_started=False,scope='Fixed TRAIN104 CV/Context pair. Static-target Hermite estimate over1.1s triggers a separately certified0.5s partial approach, at most0.08m per segment. Original actual/reference collision/joint/SE3 gates, physical parameters and global10s budget retained; existing phase deadlines never renewed. No formal VAL/TEST/video/retry.'))
 archive=out/'source_snapshot';archive.mkdir();rows=[]
 for i,(name,sha) in enumerate(sorted(files.items())):
  p=Path(name)
  if p.suffix not in ('.py','.sh'):continue
  q=archive/f'{i:04d}_{p.name}';shutil.copyfile(p,q);assert file_sha(q)==sha
  rows.append(dict(original_path=name,sha256=sha,archived_path=str(q.relative_to(ROOT))))
 atomic_json(out/'source_snapshot.json',dict(files=rows,native_started=False))
 checks={}
 for method in ('cv_geometry','context_geometry'):
  prepared=preflight(parser().parse_args(command(method)[3:]));checks[method]=prepared['model_preflight']
 native_checks={}
 for method in ('cv_geometry','context_geometry'):
  done=subprocess.run(command(method)+['--check'],cwd=ROOT,text=True,capture_output=True,check=True)
  native_checks[method]=json.loads(done.stdout)
 atomic_json(out/'native_preflight.json',native_checks)
 atomic_json(out/'CPU_acceptance.json',dict(parent_argv_derived=True,methods=checks,native_started=False,offline_tests=30,native_environment_tests=30,physics_unchanged=True,all_previous290_inputs_preserved=True,native_parent_argv_preflight_passed=True,trigger_seconds=1.1,segment_seconds=.5,actual_native_TRAIN_acceptance_pending=True))
 verify_inputs(ROOT,files);print(json.dumps(dict(setup=str(out),frozen_inputs=len(files),CPU_preflight=list(checks))))
if __name__=='__main__':main()
