"""New public ring seating estimate, distinct from Native task success.

Keep the1121 actuator, preclose qualification, targets, forecasting and grasp
compensation. Replace only insert completion and the actual closed-to-open
release permission. Engineering bounds come from the frozen TRAIN CAD; they
are not calibrated public-pose uncertainty bounds or Native success sensors.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, MethodType
from unittest.mock import patch
import numpy as np

from ..contracts import Pose, RobotState, ObjectState
from .interaction import PolicyState
from .rotation_equivalence import CurrentRotationEquivalence

SCHEMA = 'r10_new_public_ring_mesh_seating_policy_v1129'
SOURCE = 'our_mani/multitask/public_ring_seating_policy_v1129.py'
TASKS = ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed')
INPUTS = {
    TASKS[0]: {
        'interaction': ('artifacts/framework/multitask_r10/TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/insert_onto_rotating_peg/interaction.json', 'c5907dedfe8ed4c43bf5e0a70ccf91995170cded3d861516a4210fd0b48d5524'),
        'prior': ('artifacts/framework/multitask_r10/ring_grasp_prior_v419/insert_onto_rotating_peg/model.json', '7a9122ad14e89d3ed83a230f487242d29771f8b41d4307555df9c6893f8319a2'),
        'D4': ('artifacts/framework/multitask_r10/rotation_equivalence_runtime_v4/insert_onto_rotating_peg__ring_D4_certificate.json', '5ca19865519eb911b692177f3562774a160651bfaad7990191ee362344d673cc'),
        'CAD': ('artifacts/framework/multitask_r10/TRAIN_smoke_collection_v7/infrastructure_attempts/2a45bebe85f6fb64/0/output/cases/train/insert_onto_rotating_peg/2a45bebe85f6fb64/TRAIN_geometry.json', 'c1894b905d4fc724b4eb727dc60ceeacf42ab84f7b4848e7458259ca34f89767'),
    },
    TASKS[1]: {
        'interaction': ('artifacts/framework/multitask_r10/high_peg_primary_minimax_v232/interaction.json', '724fe2f057cf81d56c0477ccc28fb4fa7c826b537c9d15bb1f4584e81220fce3'),
        'prior': ('artifacts/framework/multitask_r10/ring_grasp_prior_v419/insert_onto_rotating_peg_high_speed/model.json', '85050e0fb34ce710d5b28b2d672c1fdd2d65acd723aea0301d4532134fb3f58a'),
        'D4': ('artifacts/framework/multitask_r10/rotation_equivalence_runtime_v4/insert_onto_rotating_peg_high_speed__ring_D4_certificate.json', 'ad3e2011c1cdc7085afd4337da92a169b88b4dc13da4445ef1ec823f4227eda0'),
        'CAD': ('artifacts/framework/multitask_r10/TRAIN_smoke_collection_v7/infrastructure_attempts/11c1b4bed2910fc5/0/output/cases/train/insert_onto_rotating_peg_high_speed/11c1b4bed2910fc5/TRAIN_geometry.json', '2a2c3ccaf4e058b75f126f85f76608381eee5655ca919f656f5dc29f671361bc'),
    },
}


def require(value, message):
    if not value: raise ValueError('ring1129: '+message)


def _load(root, ref):
    name, checksum = ref; path = Path(root)/name
    require(path.resolve() == path, 'canonical frozen TRAIN input required')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == checksum, 'frozen TRAIN input changed: '+name)
    return json.loads(data)


def _compose(a, b):
    return Pose(a.position+a.rotation@b.position, a.rotation@b.rotation)


def _array(value):
    result = np.array(value, dtype=float, copy=True)
    require(np.isfinite(result).all(), 'finite frozen mesh geometry required')
    result.setflags(write=False); return result


def _edges(faces):
    return tuple(sorted({tuple(sorted((int(a),int(b)))) for face in faces
                         for a,b in zip(face,np.roll(face,-1))}))


def _parts(vertices, faces):
    parent = list(range(len(vertices)))
    def find(index):
        while parent[index] != index: index = parent[index]
        return index
    for face in faces:
        for index in face[1:]: parent[find(int(index))] = find(int(face[0]))
    roots = [find(i) for i in range(len(vertices))]
    return [vertices[np.array(roots) == root] for root in sorted(set(roots))]


@dataclass(frozen=True)
class RingSeatRule:
    task: str
    nominal_ring_to_peg: Pose
    slab: tuple
    aperture: tuple
    peg_vertices: object
    peg_edges: object
    certificate_sha256: str
    epsilon: float
    provenance: dict


def derive_ring_seat_rule(root, config, task):
    require(task in TASKS, 'explicit ring task required')
    root = Path(root).resolve(); refs = INPUTS[task]
    for actual, name in ((config['learned_artifacts'][task]['interaction'],'interaction'),
        (config['public_ring_compensation_v580'][task]['prior'],'prior'),
        (config['finite_rotation_symmetries'][task]['ring'],'D4')):
        require(actual == dict(path=refs[name][0],sha256=refs[name][1]), 'exact original task input binding: '+name)
    model, prior, cert, cad = (_load(root,refs[key]) for key in ('interaction','prior','D4','CAD'))
    require(model['metadata'].get('task_name') == task, 'original task interaction model required')
    require(cert['CAD_file'] == dict(path=refs['CAD'][0],sha256=refs['CAD'][1])
        and cert.get('TRAIN_only') is True and cert.get('VAL_used') is False and cert.get('TEST_used') is False,
        'frozen TRAIN-only D4/CAD dependency required')
    require(prior.get('task') == task and prior.get('original_declared_parent_count') == 100
        and prior.get('captured_parent_count') == 100 and prior.get('all_captured_parents_used_equally') is True
        and prior.get('native_success_used_for_selection') is False and prior.get('VAL_used') is False
        and prior.get('TEST_used') is False, 'unchanged all100 TRAIN grasp prior required')
    require(cad.get('task') == task and cad.get('split') == 'train'
        and cad.get('world_pose_or_motion_schedule_in_asset') is False, 'body-local original TRAIN CAD required')
    templates = [t for t in model['templates'] if t['phase'] == 'insert']
    require(len(templates) == 1 and templates[0].get('reference_time_basis') == 'command_goal_reference',
            'one original insert command-goal canonical geometry required')
    template = templates[0]
    nominal = _compose(Pose.from_xyzw(template['completion_geometry_xyzw']), Pose.from_xyzw(prior['tool_to_ring_xyzw']))
    ring = _array(cad['bodies']['ring']['collision_vertices_body'])
    faces = np.asarray(cad['bodies']['ring']['collision_triangles'],int)
    require(ring.shape == (32,3) and faces.ndim == 2 and faces.shape[1] == 3
        and faces.min() >= 0 and faces.max() < len(ring), 'source four-bar ring mesh required')
    epsilon = float(64*np.finfo(float).eps*max(1.,float(np.abs(ring).max())))
    groups = _parts(ring,faces); require(len(groups) == 4 and all(x.shape == (8,3) for x in groups), 'four original connected ring bars required')
    limits = {}
    for points in groups:
        center = points.mean(0); axis = 1 if abs(center[1]) > abs(center[2]) else 2
        sign = 1 if center[axis] > 0 else -1
        require((axis,sign) not in limits, 'one bar per aperture side required')
        bound = float(points[:,axis].min() if sign > 0 else points[:,axis].max())
        require(bound*sign > epsilon, 'bar must lie outside the central aperture')
        limits[axis,sign] = bound
    aperture = tuple((limits[axis,-1],limits[axis,1]) for axis in (1,2))
    slab = (float(ring[:,0].min()),float(ring[:,0].max()))
    require(slab[0] < 0 < slab[1] and all(a < 0 < b for a,b in aperture), 'source aperture/slab axes required')
    rotations = np.asarray(cert['group_rotations_body'],float)
    require(rotations.shape == (8,3,3) and np.allclose(abs(rotations[:,:,0]),np.array([1.,0.,0.]),atol=epsilon,rtol=0),
            'original D4 normal is ring local X, never full orientation promotion')
    vertices = {}; edges = {}
    for name, body in cad['bodies'].items():
        if not name.startswith('peg'): continue
        points = _array(body['collision_vertices_body']); tri = np.asarray(body['collision_triangles'],int)
        require(points.shape == (8,3) and tri.shape == (12,3), 'original selected-peg box mesh required')
        vertices[name] = points; edges[name] = _edges(tri)
    require(vertices, 'original peg meshes required')
    provenance = dict(schema=SCHEMA, task=task,
        inputs={key:dict(path=name,sha256=checksum) for key,(name,checksum) in refs.items()},
        nominal_ring_to_peg_xyzw=nominal.xyzw().tolist(), ring_slab_x_m=list(slab), aperture_yz_m=[list(x) for x in aperture],
        engineering_test='peg intersects both ring slab faces and its clipped convex cross-sections fit inside the central aperture',
        floating_epsilon_m=epsilon, epsilon_source='64*float64_epsilon*max(1,max_abs_frozen_ring_vertex); numerical_only',
        all100_prior_used_without_success_selection=True, target_or_forecast_or_grasp_TF_update_changed=False,
        interaction_completed_insert_parent_counts=[value.get('completed_parent_count') for key,value in
            model['metadata'].get('completed_skill_fitting',{}).get('phase_fits',{}).items() if key.startswith('insert|')],
        original_template_position_tolerance_m=template['position_tolerance_m'],
        original_template_orientation_tolerance_rad=template['orientation_tolerance_rad'],
        original_template_tolerance_scope=model['metadata'].get('command_goal_tolerance_scope'),
        original_public_completion_verifier_calibrated=model['metadata'].get('public_completion_verifier_calibrated'),
        template_tolerances_used_as_calibrated_public_seating_bounds=False,
        public_pose_uncertainty_calibrated=False, nominal_geometric_estimate_only=True,
        native_success_certificate=False, native_sensor_or_task_success_input=False,
        phase_completion_is_not_native_success=True, formal_VAL_TEST_qualification=False)
    return RingSeatRule(task,nominal,slab,aperture,MappingProxyType(vertices),MappingProxyType(edges),
                        cert['certificate_sha256'],epsilon,provenance)


def binding(root, config, task):
    rule = derive_ring_seat_rule(root,config,task)
    return dict(rule.provenance,source_sha256=hashlib.sha256((Path(root)/SOURCE).read_bytes()).hexdigest(),
        changed_factor='public_ring_mesh_seating_for_insert_completion_and_first_release_open',
        release_done_after_open='original1121_source_rule', already_open_never_reclosed_by_geometry=True,
        native_chain_and_preclose_and_targets_unchanged=True)


def _current_evidence_matches(evidence, ring, now, rule, age):
    return isinstance(evidence,CurrentRotationEquivalence) and evidence.matches(body_id='ring',timestamp=ring.timestamp,
        pose=ring.pose,now=now,maximum_age=age,certificate_sha256=rule.certificate_sha256)


def _clip_peg(rule, reference_id, ring_pose, peg_pose):
    points = ring_pose.inverse_apply(peg_pose.apply(rule.peg_vertices[reference_id]))
    sections = []; clipped = [p for p in points if rule.slab[0]-rule.epsilon <= p[0] <= rule.slab[1]+rule.epsilon]
    for plane in rule.slab:
        section = [p for p in points if abs(p[0]-plane) <= rule.epsilon]
        for a,b in rule.peg_edges[reference_id]:
            x,y = points[a],points[b]; dx,dy = x[0]-plane,y[0]-plane
            if dx*dy < 0: section.append(x+(y-x)*(dx/(dx-dy)))
        sections.append(section); clipped.extend(section)
    if not clipped: return False,dict(axial_faces_intersect=[False,False],minimum_aperture_margin_m=None)
    clipped = np.asarray(clipped); margins = []
    for index,(lower,upper) in enumerate(rule.aperture,1):
        margins.extend((float(clipped[:,index].min()-lower),float(upper-clipped[:,index].max())))
    axial = [len(section) >= 3 for section in sections]
    return bool(all(axial) and min(margins) >= -rule.epsilon),dict(
        axial_faces_intersect=axial,minimum_aperture_margin_m=min(margins),
        aperture_margins_m=margins,clipped_peg_vertices_in_ring=clipped.tolist())


def evaluate_public_ring_seating(source_policy, rule):
    """Only already supplied public DTOs/current D4 evidence; no Native getters."""
    def unknown(reason): return dict(known=False,seated=False,reason=reason,nominal_estimate_only=True)
    goal = source_policy.selected; state = source_policy.frontend.state
    if type(goal) is not dict or type(state) is not PolicyState or type(state.robot) is not RobotState: return unknown('current_public_intent_and_robot_required')
    if goal.get('phase') not in ('insert','release'): return unknown('current_insert_or_release_required')
    now = float(state.robot.timestamp); age = float(64*np.finfo(float).eps*max(1.,abs(now)))
    if tuple(goal.get('manipulated_ids',())) != ('ring',) or goal.get('reference_id') not in rule.peg_vertices: return unknown('same_selected_ring_and_peg_required')
    ring = state.bodies.get('ring'); peg = state.bodies.get(goal['reference_id'])
    if type(ring) is not ObjectState or type(peg) is not ObjectState: return unknown('original_public_body_DTOs_required')
    if type(ring.pose) is not Pose or type(peg.pose) is not Pose: return unknown('original_public_Pose_DTOs_required')
    if not ring.visible or not peg.visible or abs(now-ring.timestamp)>age or abs(now-peg.timestamp)>age: return unknown('fresh_visible_ring_and_peg_required')
    metadata = state.perception_metadata; bodies = metadata.get('bodies',{}); info = bodies.get('ring',{})
    invalid = set(metadata.get('invalid_mode_ids',()))
    if not state.robot.contact or info.get('motion_mode') != 'attached' or invalid.intersection(('ring',goal['reference_id'])): return unknown('current_public_held_ring_required')
    witness = source_policy.close_witness
    if (type(witness) is not dict or tuple(witness.get('manipulated_ids',())) != ('ring',)
        or witness.get('phase_index') != tuple(source_policy.phase_order).index('acquire')
        or type(witness.get('confirmed_after')) not in (float,int) or witness['confirmed_after']>now+age): return unknown('own_confirmed_ring_close_required')
    if not _current_evidence_matches(info.get('rotation_equivalence_evidence'),ring,now,rule,age): return unknown('current_typed_D4_geometry_required')
    seated,geometry = _clip_peg(rule,goal['reference_id'],ring.pose,peg.pose)
    relative = peg.pose.inverse_apply(ring.pose.position)
    return dict(known=True,seated=seated,reason='public_mesh_seating_estimate' if seated else 'public_mesh_not_seated',
        timestamp=now,reference_id=goal['reference_id'],ring_relative_position_m=relative.tolist(),
        nominal_ring_target_position_m=rule.nominal_ring_to_peg.position.tolist(),
        nominal_position_error_m=float(np.linalg.norm(relative-rule.nominal_ring_to_peg.position)),
        geometry=geometry,nominal_estimate_only=True,native_success_claimed=False)


class RingGuardedReleaseWitness:
    def __init__(self, parent, rule, *, max_events=1000):
        require(type(max_events) is int and 1<=max_events<=10000,'bounded guard trace required')
        self.parent_dispatch = parent; self.source_policy = parent.source_policy; self.rule = rule
        self.max_events = max_events; self.events = []; self.dropped_events = 0

    def __getattr__(self,name): return getattr(self.parent_dispatch,name)

    def record(self,kind,result):
        if len(self.events)<self.max_events:self.events.append(dict(kind=kind,**deepcopy(result)))
        else:self.dropped_events+=1

    def qualify_opening(self,requested,current_requested):
        opening,proof,proposed = self.parent_dispatch.qualify_opening(requested,current_requested)
        goal = self.source_policy.selected
        if (type(goal) is dict and goal.get('phase')=='release' and current_requested<.5
                and opening>=.5):
            result = evaluate_public_ring_seating(self.source_policy,self.rule); self.record('closed_to_open_permission',result)
            if not result['known'] or not result['seated']:
                return 0.,dict(proof,public_ring_seating_v1129=result,
                    qualified_opening=0.,reason='defer_first_release_open_until_public_ring_seating'),None
        return opening,proof,proposed

    def confirm_opening(self,*args,**kwargs): return self.parent_dispatch.confirm_opening(*args,**kwargs)

    def receipt(self):
        return dict(self.parent_dispatch.receipt(),public_ring_seating_v1129=dict(
            binding=deepcopy(self.rule.provenance),events=deepcopy(self.events),dropped_events=self.dropped_events,
            insert_done_uses_public_ring_geometry=True,release_done_original=True,
            native_sensor_or_task_success_input=False,already_open_never_reclosed_by_geometry=True))


@contextmanager
def owned_ring_seating_scope(source_policy, rule, *, record=None):
    require(source_policy.task == rule.task,'owned original ring policy/task required')
    original = source_policy._done
    def done(owned,goal):
        if goal['phase']!='insert':return original(goal)
        result = evaluate_public_ring_seating(owned,rule)
        if record is not None:record('insert_completion',result)
        return bool(result['known'] and result['seated'])
    with patch.object(source_policy,'_done',MethodType(done,source_policy)):yield source_policy


@contextmanager
def integrated_ring_seating_scope(root,config,task,frontend,dispatched,executor):
    rule = derive_ring_seat_rule(root,config,task)
    source = dispatched.source_policy
    require(source.frontend is frontend and executor.skill_policy is dispatched,'one owned1121 frontend/policy/executor required')
    guarded = RingGuardedReleaseWitness(dispatched,rule)
    with ExitStack() as scopes:
        scopes.enter_context(owned_ring_seating_scope(source,rule,record=guarded.record))
        scopes.enter_context(patch.object(executor,'skill_policy',guarded))
        yield frontend,guarded,executor
