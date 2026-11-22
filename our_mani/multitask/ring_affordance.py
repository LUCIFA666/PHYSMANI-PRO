"""Explicit TRAIN permissions for the ring's finite D4 pose quotient.

This permission is restricted to ring-relative approach/acquire. It never
changes peg semantics, declares full orientation, grants target contact to an
approach, or bypasses robot/held-object/world collision checks. Native grasps
are proximity-triggered attachments; this is simulator-policy permission,
not evidence of frictional grasp quality or success of all transformed poses.
"""
import ast
from dataclasses import asdict,dataclass,field
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from ..contracts import Pose
from .evaluation import digest,file_sha
from .rotation_equivalence import VerifiedFiniteSymmetry,CurrentRotationEquivalence

SCHEMA='r10_TRAIN_ring_D4_phase_affordance_permission_v2'
EXACT_ORBIT_SCHEMA='r10_exact_rational_measured_TCP_action_orbit_v1'
_TOKEN=object()
TASKS=('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed')
PHASES=('approach','acquire')


def require(value,message):
    if not value:raise ValueError(message)


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    return value


def template_sha256(template):return digest(_plain(asdict(template)))


def _rational(value):return Fraction.from_float(float(value))


def _vector(values):return tuple(_rational(value) for value in values)


def _matrix(values):return tuple(_vector(row) for row in values)


def _transpose(matrix):return tuple(zip(*matrix))


def _multiply(a,b):
    columns=_transpose(b)
    return tuple(tuple(sum((x*y for x,y in zip(row,column)),Fraction()) for column in columns) for row in a)


def _apply(matrix,vector):
    return tuple(sum((x*y for x,y in zip(row,vector)),Fraction()) for row in matrix)


def _quaternion_matrix(values):
    """Exact normalized-quaternion rotation; no sqrt, LAPACK or SciPy."""
    x,y,z,w=_vector(values);norm=x*x+y*y+z*z+w*w
    require(norm>Fraction(1,10**16),'finite nonzero measured TCP quaternion required')
    two=2/norm
    return ((1-two*(y*y+z*z),two*(x*y-z*w),two*(x*z+y*w)),
            (two*(x*y+z*w),1-two*(x*x+z*z),two*(y*z-x*w)),
            (two*(x*z-y*w),two*(y*z+x*w),1-two*(x*x+y*y)))


def _canonical_pose(position,rotation):
    def pair(value):return [value.numerator,value.denominator]
    return dict(position_m=[pair(value) for value in position],
        rotation=[[pair(value) for value in row] for row in rotation])


def _display_pose(position,rotation):
    # Python's Fraction -> binary64 conversion is correctly rounded; the
    # immutable rational witness remains authoritative, not this display.
    return dict(position_m=list(map(float,position)),rotation=[list(map(float,row)) for row in rotation])


def exact_measured_action_orbit(raw_samples,group,center):
    """Reproduce action-orbit hashes with exact arithmetic on frozen floats.

    Each sample is (recorded body position, recorded body rotation matrix,
    recorded measured TCP xyzw). The body-local map is explicitly the recorded
    matrix transpose; no quaternion roundtrip, interpolation, Slerp or fitted
    pose enters this witness. Geometry equivalence still comes from the
    separately verified finite CAD certificate, not from this arithmetic.
    """
    group=np.asarray(group,float);center=np.asarray(center,float);samples=tuple(raw_samples)
    require(samples and group.shape==(8,3,3) and center.shape==(3,) and np.isfinite(group).all()
        and np.isfinite(center).all(),'explicit finite D4 group, center and raw measured samples required')
    curves=[];raw=[]
    for body_position,body_rotation,tcp in samples:
        body=Pose(body_position,body_rotation);tcp=np.asarray(tcp,float)
        require(tcp.shape==(7,) and np.isfinite(tcp).all(),'exact orbit requires original finite measured TCP xyzw')
        bp,br,tp=_vector(body.position),_matrix(body.rotation),_vector(tcp[:3])
        inverse=_transpose(br)
        curves.append((_apply(inverse,tuple(a-b for a,b in zip(tp,bp))),_multiply(inverse,_quaternion_matrix(tcp[3:]))))
        raw.append(dict(body_position=body.position.tolist(),body_rotation=body.rotation.tolist(),measured_TCP_xyzw=tcp.tolist()))
    c=_vector(center);orbit=[]
    for index,value in enumerate(group):
        g=_matrix(value);gc=_apply(g,c);offset=tuple(a-b for a,b in zip(c,gc));transformed=[]
        for p,r in curves:
            gp=_apply(g,p);transformed.append((tuple(a+b for a,b in zip(offset,gp)),_multiply(g,r)))
        canonical=[_canonical_pose(p,r) for p,r in transformed]
        orbit.append(dict(group_index=index,curve_hash_schema=EXACT_ORBIT_SCHEMA,
            curve_sha256=digest(dict(schema=EXACT_ORBIT_SCHEMA,poses=canonical)),sample_count=len(canonical),
            source_raw_samples_sha256=digest(raw),
            start_relative_TCP_exact=canonical[0],final_relative_TCP_exact=canonical[-1],
            start_relative_TCP_display=_display_pose(*transformed[0]),
            final_relative_TCP_display=_display_pose(*transformed[-1])))
    return orbit


def _native_rule(gripper_source,task_source):
    """Audit the actual author implementation, retaining its full file SHA."""
    gp=Path(gripper_source).resolve();tp=Path(task_source).resolve()
    tree=ast.parse(gp.read_text());gripper=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Gripper')
    grasp=next(n for n in gripper.body if isinstance(n,ast.FunctionDef) and n.name=='grasp')
    # Compare executable statements rather than comments/docstrings. An added
    # orientation/material/hidden-condition branch invalidates the permit.
    actual=ast.Module(body=grasp.body[1:] if isinstance(grasp.body[0],ast.Expr) and
        isinstance(grasp.body[0].value,ast.Constant) and isinstance(grasp.body[0].value.value,str) else grasp.body,type_ignores=[])
    expected=ast.parse('''detected = self._proximity_sensor.is_detected(obj)
if detected and obj not in self._grasped_objects:
    self._grasped_objects.append(obj)
    self._old_parents.append(obj.get_parent())
    obj.set_parent(self._attach_point, keep_in_place=True)
return detected
''')
    require(ast.dump(actual,include_attributes=False)==ast.dump(expected,include_attributes=False),
        'native grasp rule differs from the audited proximity-only keep-in-place attachment')
    tasktree=ast.parse(tp.read_text());classes=[n for n in tasktree.body if isinstance(n,ast.ClassDef)]
    init=next(n for c in classes for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='init_task')
    names=[n for n in ast.walk(init) if isinstance(n,ast.Call)]
    grasps=[n for n in names if isinstance(n.func,ast.Attribute) and n.func.attr=='register_graspable_objects']
    expected_roles=ast.parse('self.register_graspable_objects([self._square_ring])').body[0].value
    require(len(grasps)==1 and ast.dump(grasps[0],include_attributes=False)==ast.dump(expected_roles,include_attributes=False),
        'native ring task does not register exactly the square ring as graspable')
    expert=next(n for n in tasktree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    expected_grasp=ast.parse('is_grasp = len(task.robot.gripper.get_grasped_objects()) > 0').body[0]
    require(any(isinstance(n,ast.Assign) and ast.dump(n,include_attributes=False)==ast.dump(expected_grasp,include_attributes=False)
        for n in ast.walk(expert)),'native teacher acquire transition no longer measures grasped objects')
    expected_transition=ast.parse("if stage == 'wp1-grasp' and is_grasp:\n    stage = 'wp2'").body[0]
    require(any(isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==ast.dump(expected_transition.test,include_attributes=False)
        and len(n.body)==1 and ast.dump(n.body[0],include_attributes=False)==ast.dump(expected_transition.body[0],include_attributes=False)
        for n in ast.walk(expert)),'native teacher acquire-to-lift transition no longer proves a grasp')
    return dict(gripper_file=dict(path=str(gp),sha256=file_sha(gp)),
        task_file=dict(path=str(tp),sha256=file_sha(tp)),
        grasp_executable_AST_sha256=digest(ast.dump(actual,include_attributes=False)),
        rule='proximity_sensor.is_detected(obj); set_parent(attach_point, keep_in_place=True)',
        acquire_to_lift_requires_native_grasped_object=True,only_registered_graspable_object='square_ring',
        orientation_or_material_grasp_predicate=False,contact_friction_quality_proven=False)


def _ring_library_fit_key(template):
    require(getattr(template,'reference_time_basis',None)=='current',
        'ring finite permission requires current-reference interaction semantics')
    return '|'.join((template.phase,template.anchor_id,template.reference_role,
        template.reference_kind,template.manipulated_role,template.reference_mode,template.reference_time_basis))


def build_train_permission(root,symmetry_pointer,TRAIN_manifest_pointer,*,
        native_gripper_source,native_task_source,library_pointer=None,development_only=False):
    """Reproduce acquired TRAIN curves and optionally bind a fitted library.

    A one-parent engineering proof is saved as pending. Production requires at
    least eight independently completed parents *per phase* and the real fitted
    library's matching terminal crossfit support. No development result can
    silently become a production permission.
    """
    from .artifact_fitting import load_train_cases,interaction_source
    from .interaction import InteractionLibrary
    root=Path(root).resolve()
    def local(pointer):
        path=(root/pointer['path']).resolve()
        require(path.is_relative_to(root) and file_sha(path)==pointer['sha256'],'frozen TRAIN permission input SHA differs')
        return path
    require(type(development_only) is bool,'explicit development-only flag required')
    symmetry=VerifiedFiniteSymmetry.load(root,symmetry_pointer);cert=symmetry.certificate
    require(cert['task'] in TASKS and cert['body_id']=='ring','finite ring permission cannot name another role or family')
    manifest_path=local(TRAIN_manifest_pointer);manifest,cases=load_train_cases(root,manifest_path)
    native=_native_rule(native_gripper_source,native_task_source)
    require(Path(native_task_source).name==cert['task']+'.py','native task source belongs to another task')
    group=np.asarray(cert['group_rotations_body']);center=np.asarray(cert['center_body'])
    rows=[];support={phase:set() for phase in PHASES};inputs={
        str(local(symmetry_pointer).relative_to(root)):symmetry_pointer['sha256'],
        str(manifest_path.relative_to(root)):TRAIN_manifest_pointer['sha256'],
        str(Path(__file__).resolve().relative_to(root)):file_sha(__file__)}
    for name in ('artifact_fitting.py','interaction.py','specs.py'):
        source=Path(__file__).resolve().with_name(name)
        inputs[str(source.relative_to(root))]=file_sha(source)
    for case in cases:
        entry=case['entry']
        if entry['task']!=cert['task'] or not entry.get('teacher_used'):continue
        source,_=interaction_source(case);a=case['arrays'];ring=list(map(str,a['body_ids'])).index('ring')
        events={e.phase:e for e in source.completion_events if e.phase in PHASES}
        if set(events)!=set(PHASES):continue
        if any(e.binding.reference_id!='ring' or e.binding.reference_role!='ring'
                or e.binding.manipulated_role!='ring' or e.binding.reference_kind!='observed_se3'
                or e.reference_mode!='external' for e in events.values()):continue
        acquire=events['acquire'];end=acquire.attempt_end_time
        attached=np.flatnonzero((a['times']>=end-1e-6)&(a['times']<=end+.100001)&
            (a['modes'][:,ring]==1)&a['mode_valid'][:,ring]&a['robot_contact'])
        command=next(r for r in source.command_ledger if r['command_id']==acquire.origin_command_id)
        if not len(attached) or command.get('applied_opening')!=0.:continue
        case_row=dict(TRAIN_parent_source_group=source.grouping_source,source_id=source.source_id,
            case_id=case['record']['case_id'],actual_ring_attached_with_public_contact_time=float(a['times'][attached[0]]),
            native_ring_grasp_identity_evidence='author acquire-to-lift requires get_grasped_objects and only square_ring is registered',
            acquire_command_applied_opening=0.,phases={})
        for phase,event in events.items():
            indices=np.flatnonzero((source.command_ids==event.origin_command_id)&
                (source.times>=event.attempt_start_time-1e-8)&(source.times<event.attempt_end_time-1e-8))
            curves=[];accepted_indices=[]
            for i in indices:
                try:
                    source.body_tracks['ring'].at(source.times[i],orientation_required=True,
                        same_mode_as='external',not_before=source.times[i])
                except ValueError:continue
                # at() above checks mask, clock, mode and orientation labels.
                # Only the exact raw recorded sample enters the witness.
                curves.append((a['target_positions'][i,ring],a['target_rotations'][i,ring],a['measured_tcp_xyzw'][i]))
                accepted_indices.append(int(i))
            if not curves:continue
            orbit=exact_measured_action_orbit(curves,group,center)
            case_row['phases'][phase]=dict(origin_command_id=event.origin_command_id,
                actual_completed_command=True,ordered_transition=event.to_phase,
                measured_executed_TCP_samples=len(curves),transformed_action_orbit=orbit,
                exact_witness_schema=EXACT_ORBIT_SCHEMA,source_cache=entry['cache'],source_cache_sha256=entry['cache_sha256'],
                source_frame_indices=accepted_indices,
                body_local_mapping='exact rational recorded target_rotations transpose; source matrix unchanged',
                measured_TCP_rotation='exact rational normalized raw xyzw products divided by quaternion norm squared',
                relative_SE3_action='(center-S@center, S) compose measured ring-relative TCP',
                every_runtime_orbit_requires_fresh_IK_robot_and_held_world_collision_checks=True)
        if set(case_row['phases'])!=set(PHASES):continue
        for phase in PHASES:support[phase].add(source.grouping_source)
        for name,checksum in case['record']['files_sha256'].items():inputs[str((root/name).resolve().relative_to(root))]=checksum
        inputs[str(case['path'].relative_to(root))]=file_sha(case['path']);rows.append(case_row)
    minimum=1 if development_only else 8
    phase_support={p:dict(completed_parent_groups=sorted(support[p]),completed_parent_count=len(support[p]),
        minimum_completed_parents=minimum,passed=len(support[p])>=minimum) for p in PHASES}
    bound=None;templates=[];library_ok=False
    if library_pointer is not None:
        path=local(library_pointer);library=InteractionLibrary.load(path)
        require(library.task_spec.task_name==cert['task'],'ring permission library task differs')
        require(library.metadata.get('synthetic') is False,'synthetic library cannot receive a real TRAIN permission')
        require(library.metadata.get('TRAIN_source_manifest_sha256')==TRAIN_manifest_pointer['sha256'],
            'ring permission library used another TRAIN acquisition')
        development_library=library.metadata.get('development_support_override') is True
        require(development_only or not development_library,'development interaction library cannot receive production permission')
        for phase in PHASES:
            for template in library.for_phase(phase):
                require(template.reference_role==template.manipulated_role=='ring'
                    and template.reference_kind=='observed_se3' and template.reference_mode=='external',
                    'ring finite permission cannot exchange other reference/contact semantics')
                key=_ring_library_fit_key(template)
                fit=library.metadata.get('source_grouped_lag_selection',{}).get(key,{})
                parents=set(fit.get('completed_parent_sources',[]))
                require(fit.get('passed') is True and len(parents)>=minimum and parents<=support[phase]
                    and fit.get('completed_parent_count')==len(parents),
                    'matching library terminal-crossfit TRAIN support is absent from the native grasp proof')
                templates.append(dict(phase=phase,anchor_id=template.anchor_id,template_sha256=template_sha256(template)))
        bound=dict(file=library_pointer,payload_sha256=library.sha256)
        inputs[str(path.relative_to(root))]=library_pointer['sha256'];library_ok=bool(templates)
    authorized=library_ok and all(v['passed'] for v in phase_support.values())
    value=dict(schema=SCHEMA,exact_action_orbit_schema=EXACT_ORBIT_SCHEMA,
        all_permission_fields_replayed_exactly=True,permission_numerical_replay_tolerance=0.,
        task=cert['task'],body_id='ring',allowed_reference_phases=list(PHASES),
        finite_symmetry=symmetry_pointer,certificate_sha256=cert['certificate_sha256'],
        TRAIN_manifest=TRAIN_manifest_pointer,native_gripper_source=str(Path(native_gripper_source).resolve()),
        native_task_source=str(Path(native_task_source).resolve()),native_grasp_semantics=native,
        library=bound,templates=templates,phase_support=phase_support,actual_TRAIN_grasp_rows=rows,
        development_only=development_only,policy_skill_authorized=authorized,
        status='authorized_development_only' if authorized and development_only else
            'authorized_for_frozen_library' if authorized else 'pending_library_or_completed_parent_support',
        full_orientation_observable=False,TRAIN_only=True,VAL_used=False,TEST_used=False,
        collision_permission='geometry-only quotient; unchanged phase-specific contact rules and complete held/world checks',
        extrapolation_requires_current_typed_RGBD_evidence=True,inputs_sha256=inputs,
        transformed_grasp_success_guaranteed=False,
        limitations=['Finite CAD symmetry is conditional on its explicit micrometre surface tolerance.',
            'Native proximity attachments do not establish physical contact/friction grasp quality.',
            'TRAIN demonstrated grasps plus conditional action equivariance do not prove every D4 orbit succeeds.',
            'Peg/frame/cup affordances and their semantic directions are outside this permission.',
            'The quotient does not supply a physical angular-velocity measurement.'])
    value['permission_sha256']=digest(value);return value


@dataclass(frozen=True)
class VerifiedRingAffordance:
    permission:dict
    symmetry:VerifiedFiniteSymmetry=field(repr=False,compare=False)
    _token:object=field(repr=False,compare=False)

    def __post_init__(self):
        require(self._token is _TOKEN,'ring phase permission must be reproduced from frozen TRAIN data')
        self._check()

    def _check(self):
        p=self.permission
        require(p.get('schema')==SCHEMA and p.get('permission_sha256')==digest({k:v for k,v in p.items() if k!='permission_sha256'})
            and p.get('certificate_sha256')==self.symmetry.certificate['certificate_sha256'],
            'loaded ring phase permission was mutated')

    @classmethod
    def load(cls,root,pointer,*,allow_development=False):
        root=Path(root).resolve();path=(root/pointer['path']).resolve()
        require(path.is_relative_to(root) and file_sha(path)==pointer['sha256'],'ring phase permission file SHA differs')
        value=json.loads(path.read_text())
        require(value.get('schema')==SCHEMA and value.get('permission_sha256')==digest({k:v for k,v in value.items() if k!='permission_sha256'}),
            'ring phase permission schema/fingerprint differs')
        for name,checksum in value['inputs_sha256'].items():
            source=(root/name).resolve()
            require(source.is_relative_to(root) and file_sha(source)==checksum,'ring phase permission frozen input SHA differs')
        require(allow_development or value.get('development_only') is False,'development ring permission requires explicit opt-in')
        expected=build_train_permission(root,value['finite_symmetry'],value['TRAIN_manifest'],
            native_gripper_source=value['native_gripper_source'],native_task_source=value['native_task_source'],
            library_pointer=value['library']['file'] if value.get('library') else None,
            development_only=value['development_only'])
        require(value==expected,'ring phase permission cannot be reproduced from frozen TRAIN acquisition/library')
        return cls(value,VerifiedFiniteSymmetry.load(root,value['finite_symmetry']),_TOKEN)

    def permits_reference(self,*,task,body_id,phase,template,library_sha256,evidence,
            observed_pose,observed_timestamp,now,maximum_age,allow_development=False):
        self._check();p=self.permission
        if (p.get('policy_skill_authorized') is not True or task!=p['task'] or body_id!='ring'
                or phase not in PHASES or (p['development_only'] and not allow_development)
                or library_sha256!=p.get('library',{}).get('payload_sha256')
                or not isinstance(evidence,CurrentRotationEquivalence)):
            return False
        if dict(phase=phase,anchor_id=template.anchor_id,template_sha256=template_sha256(template)) not in p['templates']:
            return False
        return evidence.matches(body_id=body_id,timestamp=observed_timestamp,pose=observed_pose,
            now=now,maximum_age=maximum_age,certificate_sha256=p['certificate_sha256'])

    def permits_geometry(self,evidence,*,body_id,observed_pose,observed_timestamp,now,maximum_age):
        # Geometry equivalence itself needs no fitted skill library; it grants
        # neither a reference/anchor nor a robot contact bypass.
        self._check()
        return isinstance(evidence,CurrentRotationEquivalence) and evidence.matches(
            body_id=body_id,timestamp=observed_timestamp,pose=observed_pose,now=now,
            maximum_age=maximum_age,certificate_sha256=self.permission['certificate_sha256'])
