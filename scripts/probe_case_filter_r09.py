"""Two independent native TRAIN processes verify one non-first exact source."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unchanged_episode_body():
    def body(path):
        tree = ast.parse(path.read_text())
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name=='run_episode')
    old, new = body(ROOT/'scripts/evaluate_aligned_r08.py'), body(ROOT/'scripts/evaluate_aligned_r09.py')
    class NormalizeStop(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str): node.value = node.value.replace('success_r09_STOP', 'success_r08_STOP')
            return node
    return ast.dump(old)==ast.dump(NormalizeStop().visit(new))


def condition_acceptance(a, b, tolerances):
    errors = []
    pa, pb = np.asarray(a['robot_pose']), np.asarray(b['robot_pose'])
    position_delta = float(np.linalg.norm(pa[:3]-pb[:3]))
    rotation_delta = float((Rotation.from_quat(pa[3:]).inv()*Rotation.from_quat(pb[3:])).magnitude())
    if position_delta > tolerances['robot_position_m']: errors.append('robot_position')
    if rotation_delta > tolerances['robot_orientation_rad']: errors.append('robot_orientation')
    if not np.allclose(a['robot_joints'], b['robot_joints'], rtol=0, atol=tolerances['robot_joints_rad']): errors.append('robot_joints')
    if a['information_setting']!=b['information_setting'] or a['sensor_protocol']!=b['sensor_protocol']:
        errors.append('sensor_or_information_protocol')
    if a['cameras'].keys()!=b['cameras'].keys(): errors.append('camera_set')
    for name in a['cameras'].keys() & b['cameras'].keys():
        for field in ('intrinsics', 'camera_to_world'):
            if not np.allclose(a['cameras'][name][field], b['cameras'][name][field], rtol=0,
                    atol=tolerances['camera_calibration_component']): errors.append('camera_'+name+'_'+field)
    ma, mb = {r['name']:r for r in a['static_map']}, {r['name']:r for r in b['static_map']}
    maximum_fixture_delta = 0.
    if ma.keys()!=mb.keys(): errors.append('static_map_shapes')
    for name in ma.keys() & mb.keys():
        delta = float(np.max(np.abs(np.asarray(ma[name]['pose'])-np.asarray(mb[name]['pose']))))
        maximum_fixture_delta = max(maximum_fixture_delta, delta)
        if delta > tolerances['fixture_pose_component']: errors.append('fixture_pose_'+name)
        if ma[name]['mesh_sha256']!=mb[name]['mesh_sha256']: errors.append('fixture_mesh_'+name)
    return dict(accepted=not errors, errors=errors, robot_position_delta_m=position_delta,
        robot_orientation_delta_rad=rotation_delta, maximum_fixture_pose_component_delta=maximum_fixture_delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/framework/aligned_protocol_r09.json')
    parser.add_argument('--task', default='pick_moving_target_on_the_table')
    parser.add_argument('--raw-episode-id', default='episode1')
    args = parser.parse_args()
    source = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    task_rows = [row for row in source if row['task']==args.task]
    selected = [row for row in task_rows if row['raw_episode_id']==args.raw_episode_id]
    if len(selected)!=1 or task_rows[0]['raw_episode_id']==args.raw_episode_id:
        raise ValueError('smoke must select one unique non-first TRAIN source')
    base = ROOT/'artifacts/framework/aligned_r09/case_filter_smoke'
    if (base/'report.json').exists(): raise FileExistsError('smoke report is immutable')
    base.mkdir(parents=True, exist_ok=True)
    records = []; identities = []
    for repeat in (1, 2):
        out = base/f'run{repeat}'
        if out.exists() and any(out.glob('pick*__episode*.json')):
            raise FileExistsError('smoke must execute fresh cases rather than resume existing records: '+str(out))
        with (ROOT/f'logs/framework/aligned_r09_case_filter_smoke_run{repeat}.log').open('w') as log:
            subprocess.run(['bash', 'scripts/run_local_sim.sh', 'scripts/evaluate_aligned_r09.py',
                '--config', args.config, '--predictor', 'cv', '--ranking', 'geometry', '--split', 'train',
                '--task', args.task, '--raw-episode-id', args.raw_episode_id,
                '--episodes-per-task', '1', '--out', str(out.relative_to(ROOT))],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        records.append(json.loads((out/f'{args.task}__{args.raw_episode_id}.json').read_text()))
        identities.append(json.loads((out/'inputs.json').read_text()))
    a, b = records
    source_ok = all(row['task']==args.task and row['raw_episode_id']==args.raw_episode_id
        and row['raw_episode_filter']==args.raw_episode_id and not row['formal_test_used'] for row in records)
    identity_ok = all(row['raw_episode_id']==args.raw_episode_id and row['raw_episode_task']==args.task for row in identities)
    independent = a['evaluation_process_pid']!=b['evaluation_process_pid']
    conditions_equal = a['public_initial_conditions']==b['public_initial_conditions']
    tolerance_path = ROOT/'artifacts/framework/aligned_r09/acceptance_tolerances.json'
    tolerances = json.loads(tolerance_path.read_text())
    conditions_check = condition_acceptance(a['public_initial_conditions'], b['public_initial_conditions'], tolerances)
    unchanged = unchanged_episode_body()
    clocks = [row['alignment_receipt'] for row in records]
    clock_ok = all(abs(r['clock']['simulation_seconds']-r['actual_physics_seconds'])<1e-4
        and abs(r['clock']['simulation_seconds']-r['actual_task_seconds'])<1e-4 for r in clocks)
    config = json.loads((ROOT/args.config).read_text())
    same_control = a['controller_sha256']==b['controller_sha256']
    seeds_ok = all(row['reset_seed']==config['seed'] for row in records)
    filter_passed = source_ok and identity_ok and independent and unchanged and clock_ok and same_control and seeds_ok
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        passed=filter_passed and conditions_check['accepted'],
        case_filter_process_control_smoke_passed=filter_passed,
        paired_source_parity_passed=conditions_check['accepted'], initial_conditions_acceptance=conditions_check,
        scope='native TRAIN CV geometry repeated exact non-first source; no TEST, no fitting, no success-rate claim',
        formal_test_used=False, predictor='cv', ranking='geometry', task=args.task,
        raw_episode_id=args.raw_episode_id, source_path=selected[0]['path'],
        raw_filter_bypasses_first_episode_truncation=source_ok, identity_binds_exact_task_and_source=identity_ok,
        evaluation_process_pids=[r['evaluation_process_pid'] for r in records], independent_processes=independent,
        public_initial_conditions_bitwise_equal=conditions_equal,
        initial_rgbd_bitwise_equal=a['public_initial_conditions']['rgbd_sha256']==b['public_initial_conditions']['rgbd_sha256'],
        run_episode_AST_unchanged_except_STOP=unchanged, native_clock_acceptance=clock_ok,
        official_task_successes=[row['official_task_success'] for row in records],
        candidate_attempt_counts=[len(row['candidate_attempts']) for row in records],
        controller_sha256=[row['controller_sha256'] for row in records],
        reset_seeds=[row['reset_seed'] for row in records],
        acceptance_tolerances=tolerances,
        alignment_receipts=clocks,
        files={str(path.relative_to(ROOT)):sha(path) for path in (
            Path(__file__), ROOT/'scripts/evaluate_aligned_r09.py', ROOT/args.config, tolerance_path)})
    (base/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report['passed']: raise RuntimeError('r09 independent native case-filter smoke failed')


if __name__=='__main__':
    main()
