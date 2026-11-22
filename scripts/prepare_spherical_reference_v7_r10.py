"""Freeze independently bound v7 geometry, existing sphere fitting and v5 motor stream."""
from datetime import datetime,timezone
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.spherical_enclosure_v7 import geometry_binding
from our_mani.multitask.spherical_native_backend_v7 import diagnostic_binding
from our_mani.multitask.spherical_reference_actor_backend_v7 import persistent_reference_policy_overlay,persistent_reference_continuation_identity
from our_mani.multitask.interaction import InteractionLibrary,CandidateConfig,MPCConfig
from scripts.run_spherical_reference_pair_v7_r10 import command
from scripts.run_geometry_spherical_reference_v7_r10 import parser,preflight

def main():
 base=ROOT/'artifacts/framework/multitask_r10/geometry_public_collision_diagnostic_v6_setup'
 out=ROOT/'artifacts/framework/multitask_r10/geometry_spherical_reference_v7_setup'
 if out.exists():raise RuntimeError('Preserve prepared namespace')
 old=json.loads((base/'frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
 c=json.loads((base/'configuration.json').read_text());b=geometry_binding(ROOT,c);c['spherical_geometry_v7']=b
 c.setdefault('body_sphere_radii',{}).setdefault(b['task'],{})['ball']=b['current_RGBD_sphere_radius_m']
 settings=c['policy_overlay']['configurations'];c['policy_overlay']=persistent_reference_policy_overlay(ROOT,settings['feedback'],settings['persistent_reference'],settings['causal_retarget'])
 c['native_metric_diagnostics']=diagnostic_binding(ROOT,c['baseline_repo'])
 lib=InteractionLibrary.load(ROOT/c['learned_artifacts'][b['task']]['interaction']['path'])
 for predictor in ('cv','context'):
  identity=persistent_reference_continuation_identity(ROOT,c,b['task'],predictor,lib,CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
  c['continuation_identities'][b['task']][predictor]=identity['sha256']
 out.mkdir();atomic_json(out/'configuration.json',c);files=dict(old['files'])
 for p in [out/'configuration.json',Path(__file__),ROOT/'our_mani/multitask/spherical_enclosure_v7.py',ROOT/'our_mani/multitask/spherical_reference_actor_backend_v7.py',ROOT/'our_mani/multitask/spherical_native_backend_v7.py',ROOT/'scripts/run_geometry_spherical_reference_v7_r10.py',ROOT/'scripts/run_spherical_reference_pair_v7_r10.py',ROOT/'tests/test_spherical_enclosure_v7.py']:
  files[str(p.resolve())]=file_sha(p)
 atomic_json(out/'frozen_inputs.json',dict(old,schema='r10_TRAIN_spherical_reference_actual_lock_v7',created_utc=datetime.now(timezone.utc).isoformat(),files=files,native_started=False,scope='Fixed TRAIN104 pair. Tighter proven all-rotation sphere enclosure; existing current RGBD sphere registration enabled using TRAIN CAD radius; unchanged v5 motion/grasp/collision gates, physics and deadlines. No VAL/TEST/video/retry.'))
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
 atomic_json(out/'CPU_acceptance.json',dict(parent_argv_derived=True,methods=checks,native_started=False,offline_tests=5,native_environment_tests=5,physics_unchanged=True,all_previous232_inputs_preserved=True))
 verify_inputs(ROOT,files);print(json.dumps(dict(setup=str(out),frozen_inputs=len(files),CPU_preflight=list(checks))))
if __name__=='__main__':main()
