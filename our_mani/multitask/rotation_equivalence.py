"""Finite TRAIN CAD symmetries and conditional current-RGBD pose quotients.

This independent prototype does not promote a template to full orientation,
authorize a skill, or recover a native/GT pose.  A geometric symmetry is not
an affordance, texture, contact, or task-success symmetry.  Currently only a
single visual-and-physical native component decomposable into closed convex
triangle surfaces is certifiable; other CADs fail closed.
"""
from collections import Counter
from dataclasses import dataclass,field
import ast
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from .evaluation import digest,file_sha
from .training_geometry import _check_asset

SCHEMA='r10_TRAIN_finite_surface_symmetry_certificate_v2'
OBSERVATION_SCHEMA='r10_current_RGBD_finite_rotation_equivalence_v1'
ENVELOPE_SCHEMA='r10_typed_current_RGBD_finite_rotation_envelope_v1'
NUMERICAL_REPLAY_SCHEMA='r10_finite_surface_derived_metric_replay_v1'
SURFACE_LENGTH_REPLAY_GUARD_M=1e-14
SURFACE_AREA_REPLAY_GUARD_M2=1e-15
_COVERAGE_LENGTH_FIELDS=frozenset(('native_to_auxiliary_surface_Hausdorff_upper_bound_m',))
_COVERAGE_AREA_FIELDS=frozenset(('triangle_area_m2','hull_area_m2','area_residual_m2',
    'maximum_coplanar_overlap_m2','area_tolerance_m2'))
_MATCH_LENGTH_FIELDS=frozenset(('maximum_auxiliary_vertex_bijection_residual_m',
    'original_surface_Hausdorff_upper_bound_m'))
_VERIFIED_TOKEN=object()
_CURRENT_TOKEN=object()


def require(value,message):
    if not value:raise ValueError(message)


def numerical_replay_policy():
    return dict(schema=NUMERICAL_REPLAY_SCHEMA,
        surface_coverage_length_fields=sorted(_COVERAGE_LENGTH_FIELDS),
        surface_coverage_area_fields=sorted(_COVERAGE_AREA_FIELDS),
        group_surface_matches_length_fields=sorted(_MATCH_LENGTH_FIELDS),
        absolute_length_guard_m=SURFACE_LENGTH_REPLAY_GUARD_M,
        absolute_area_guard_m2=SURFACE_AREA_REPLAY_GUARD_M2,
        all_other_fields_and_discrete_structure_exact=True,relative_tolerance=0.,
        each_geometry_and_group_predicate_recomputed=True,
        stored_file_digest_and_all_source_SHA_exact=True,
        recomputed_certificate_digest_is_not_a_stored_evidence_field=True,
        guarded_metrics_still_require_margin_to_their_original_geometric_limits=True)


def _metric_margins(certificate):
    """Every allowed numerical guard fits inside the original proof limits."""
    length,area=SURFACE_LENGTH_REPLAY_GUARD_M,SURFACE_AREA_REPLAY_GUARD_M2
    tolerance=certificate['surface_tolerance_m']
    for row in certificate['surface_coverage']:
        require(row['native_to_auxiliary_surface_Hausdorff_upper_bound_m']+length<=tolerance/4,
            'auxiliary surface proof has no margin for the explicit numerical replay guard')
        require(row['area_residual_m2']+2*area<=row['area_tolerance_m2']
            and row['maximum_coplanar_overlap_m2']+area<=row['area_tolerance_m2']
            and abs(row['triangle_area_m2']-row['hull_area_m2'])+2*area<=row['area_tolerance_m2'],
            'surface area/overlap proof has no margin for the explicit numerical replay guard')
    require(all(row['original_surface_Hausdorff_upper_bound_m']+length<=tolerance
        for row in certificate['group_surface_matches']),
        'finite surface symmetry has no margin for the explicit numerical replay guard')


def replay_train_certificate(stored,replayed):
    """Compare exact evidence plus two explicitly dimensioned metric budgets.

    This is intentionally not a recursive allclose. Only named *derived*
    surface-length and surface-area diagnostics can differ. All CAD/group
    arrays, permutations, source SHA, booleans, thresholds and other numbers
    remain exact. The full proof builder must have passed in this environment
    before this function is used by the loader.
    """
    require(stored.get('schema')==replayed.get('schema')==SCHEMA
        and stored.get('numerical_replay_policy')==replayed.get('numerical_replay_policy')==numerical_replay_policy(),
        'finite surface numerical replay schema/policy differs')
    for value in (stored,replayed):
        require(value.get('certificate_sha256')==digest({k:v for k,v in value.items() if k!='certificate_sha256'}),
            'finite symmetry certificate fingerprint differs')
        _metric_margins(value)
    differences=[];checked=0
    def compare(a,b,path=()):
        nonlocal checked
        require(type(a) is type(b),'finite symmetry evidence type differs at '+str(path))
        if isinstance(a,dict):
            require(set(a)==set(b),'finite symmetry evidence structure differs at '+str(path))
            for key in a:
                if path==() and key=='certificate_sha256':continue
                compare(a[key],b[key],path+(key,))
        elif isinstance(a,list):
            require(len(a)==len(b),'finite symmetry evidence list structure differs at '+str(path))
            for i,(left,right) in enumerate(zip(a,b)):compare(left,right,path+(i,))
        else:
            unit=None;guard=None
            if len(path)==3 and path[0]=='surface_coverage' and type(path[1]) is int:
                if path[2] in _COVERAGE_LENGTH_FIELDS:unit,guard='m',SURFACE_LENGTH_REPLAY_GUARD_M
                elif path[2] in _COVERAGE_AREA_FIELDS:unit,guard='m2',SURFACE_AREA_REPLAY_GUARD_M2
            elif len(path)==3 and path[0]=='group_surface_matches' and type(path[1]) is int and path[2] in _MATCH_LENGTH_FIELDS:
                unit,guard='m',SURFACE_LENGTH_REPLAY_GUARD_M
            if guard is None:
                require(a==b,'finite symmetry exact evidence differs at '+str(path));return
            require(type(a) is float and np.isfinite([a,b]).all() and min(a,b)>=0
                and abs(a-b)<=guard,'finite symmetry derived metric exceeds explicit '+unit+' replay guard at '+str(path))
            checked+=1
            if a!=b:differences.append(dict(path=list(path),stored=a,recomputed=b,
                absolute_difference=abs(a-b),unit=unit,absolute_guard=guard))
    compare(stored,replayed)
    return dict(schema=NUMERICAL_REPLAY_SCHEMA,accepted=True,all_source_hashes_and_discrete_evidence_exact=True,
        full_geometric_predicates_recomputed=True,original_proof_limits_preserved_with_guard_margin=True,
        stored_certificate_sha256=stored['certificate_sha256'],recomputed_certificate_sha256=replayed['certificate_sha256'],
        absolute_length_guard_m=SURFACE_LENGTH_REPLAY_GUARD_M,absolute_area_guard_m2=SURFACE_AREA_REPLAY_GUARD_M2,
        checked_derived_float_fields=checked,different_derived_float_fields=len(differences),differences=differences)


def finite_group(rotations,*,name='D4',algebra_tolerance=1e-8):
    """Validate proper SO(3), distinctness, identity, inverses and closure."""
    g=np.asarray(rotations,float)
    require(name=='D4' and g.shape==(8,3,3) and np.isfinite(g).all()
        and 0<algebra_tolerance<=1e-6,'explicit finite eight-element D4 rotations required')
    eye=np.eye(3)
    require(np.max(np.abs(g@np.swapaxes(g,1,2)-eye))<=algebra_tolerance
        and np.max(np.abs(np.linalg.det(g)-1))<=algebra_tolerance,'symmetries must be proper SO(3), not reflections')
    distances=np.linalg.norm(g[:,None]-g,axis=(2,3));np.fill_diagonal(distances,np.inf)
    require(distances.min()>algebra_tolerance,'finite group contains duplicate rotations')
    def match(value):
        d=np.linalg.norm(g-value,axis=(1,2));i=int(d.argmin())
        require(d[i]<=algebra_tolerance,'finite rotations do not contain identity/inverses/closure')
        return i
    identity=match(eye);inverses=[match(value.T) for value in g]
    closure=[[match(a@b) for b in g] for a in g]
    orders=[]
    for i in range(len(g)):
        index=identity
        for n in range(1,9):
            index=closure[index][i]
            if index==identity:orders.append(n);break
        else:raise ValueError('finite group element has invalid order')
    require(Counter(orders)=={1:1,2:5,4:2} and any(closure[i][j]!=closure[j][i] for i in range(8) for j in range(8)),
        'group is not D4: a named eight-element group alone is insufficient')
    return g,dict(name=name,order=8,proper_SO3=True,identity_index=identity,inverse_indices=inverses,
        multiplication_table=closure,element_orders=orders,algebra_tolerance=algebra_tolerance)


def _cross(a,b):return float(a[0]*b[1]-a[1]*b[0])


def _area(points):
    return abs(sum(_cross(a,b) for a,b in zip(points,np.roll(points,-1,axis=0))))*.5 if len(points)>=3 else 0.


def _triangle_intersection(a,b):
    """Exact planar triangle clipping in a shared 2D frame, up to floats."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    if _cross(b[1]-b[0],b[2]-b[0])<0:b=b[::-1]
    polygon=list(a)
    for edge,end in zip(b,np.roll(b,-1,axis=0)):
        if not polygon:break
        result=[];old=polygon[-1];d_old=_cross(end-edge,old-edge)
        for point in polygon:
            d=_cross(end-edge,point-edge)
            if (d>=0)!=(d_old>=0):
                result.append(old+(point-old)*(d_old/(d_old-d)))
            if d>=0:result.append(point)
            old,d_old=point,d
        polygon=result
    return _area(np.asarray(polygon))


def _box_facets(points):
    hull=ConvexHull(points);facets={frozenset(np.flatnonzero(np.abs(points@n+d)<=1e-8))
        for n,d in zip(hull.equations[:,:3],hull.equations[:,3])}
    require(len(facets)==6 and all(len(face)==4 for face in facets),'auxiliary component is not a six-face affine box')
    return facets


def _mesh_parts(vertices,triangles,precision_budget):
    """Prove each native surface is close to a fully covered convex box.

    Area alone is insufficient: also require a closed manifold, hull-supported
    triangles and no overlapping coplanar interiors.  Thus alternate valid
    face diagonals may match without treating vertex-only evidence as a surface
    proof.  Native float meshes can make a nominal quad slightly nonplanar.
    An auxiliary affine eight-corner box is fitted without altering the CAD;
    a vertex-wise bound applies to every point of each corresponding triangle
    in both directions.  Its complete box-surface coverage is checked below.
    This deliberately conservative prototype rejects other connected meshes.
    """
    v=np.asarray(vertices,float);f=np.asarray(triangles)
    require(v.ndim==2 and v.shape[1]==3 and len(v)>=4 and np.isfinite(v).all()
        and f.ndim==2 and f.shape[1]==3 and len(f)>=4 and np.issubdtype(f.dtype,np.integer)
        and f.min()>=0 and f.max()<len(v),'finite indexed triangular surfaces required')
    require(set(f.ravel())==set(range(len(v))),'unused CAD vertices cannot establish surface equivalence')
    parent=np.arange(len(v))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    for face in f:
        for i in face[1:]:parent[find(int(i))]=find(int(face[0]))
    roots=np.array([find(i) for i in range(len(v))]);parts=[];evidence=[]
    for root in sorted(set(roots)):
        indices=np.flatnonzero(roots==root);faces=f[np.isin(f[:,0],indices)]
        edge_counts=Counter(tuple(sorted((int(a),int(b)))) for face in faces for a,b in zip(face,np.roll(face,-1)))
        require(all(n==2 for n in edge_counts.values()),'CAD component is not a closed two-manifold triangle surface')
        native=v[indices]
        require(len(native)==8,'ring prototype requires eight-corner connected native box components')
        axes=np.linalg.eigh((native-native.mean(0)).T@(native-native.mean(0)))[1]
        coordinates=(native-native.mean(0))@axes;signs=np.where(coordinates>=0,1.,-1.)
        require(len(set(map(tuple,signs)))==8,'native component does not identify eight distinct affine-box corners')
        design=np.column_stack((np.ones(8),signs));points=design@np.linalg.lstsq(design,native,rcond=None)[0]
        approximation=float(np.linalg.norm(native-points,axis=1).max())
        require(approximation<=precision_budget/4,'native surface is too far from its auxiliary convex box')
        # Keep exact source triangle indices.  Every barycentric point maps to
        # the corresponding auxiliary triangle with error <= approximation.
        local_faces=np.searchsorted(indices,faces);hull=ConvexHull(points);_box_facets(points)
        require(hull.volume>1e-12,'CAD component must have nonzero volume')
        tri=points[local_faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);areas=np.linalg.norm(cross,axis=1)*.5
        require(areas.min()>1e-12,'degenerate CAD triangle cannot certify a surface')
        plane_tolerance=1e-8;area_tolerance=max(1e-11,float(hull.area)*1e-7)
        for face in tri:
            supported=np.max(np.abs(face@hull.equations[:,:3].T+hull.equations[:,3]),axis=0)
            require(supported.min()<=plane_tolerance,'nonconvex/internal triangle cannot certify convex surface coverage')
        require(abs(areas.sum()-hull.area)<=area_tolerance,'CAD triangle area does not cover its convex hull')
        normals=cross/(2*areas[:,None]);maximum_overlap=0.
        for i in range(len(tri)):
            for j in range(i):
                if (abs(float(normals[i]@normals[j]))<1-1e-8
                        or np.max(np.abs((tri[j]-tri[i,0])@normals[i]))>plane_tolerance):continue
                u=tri[i,1]-tri[i,0];u/=np.linalg.norm(u);basis=np.stack((u,np.cross(normals[i],u)),axis=1)
                overlap=_triangle_intersection((tri[i]-tri[i,0])@basis,(tri[j]-tri[i,0])@basis)
                maximum_overlap=max(maximum_overlap,overlap)
                require(overlap<=area_tolerance,'overlapping triangles cannot be counted as complete surface coverage')
        parts.append((points,approximation))
        evidence.append(dict(vertices=len(indices),triangles=len(faces),closed_two_manifold=True,
            auxiliary_geometry='least-squares affine eight-corner box; original CAD unchanged',
            native_to_auxiliary_surface_Hausdorff_upper_bound_m=approximation,
            correspondence='same original triangle indices, barycentric vertex-displacement bound in both directions',
            all_auxiliary_triangles_supported_by_convex_hull=True,triangle_area_m2=float(areas.sum()),hull_area_m2=float(hull.area),
            area_residual_m2=float(abs(areas.sum()-hull.area)),maximum_coplanar_overlap_m2=maximum_overlap,
            plane_tolerance_m=plane_tolerance,area_tolerance_m2=area_tolerance))
    return parts,evidence


def certify_train_geometry(asset,body_id,rotations,center,*,surface_tolerance_m=2.5e-6,algebra_tolerance=1e-8):
    """Return deterministic geometry-only proof; never authorize actor skills."""
    _check_asset(asset)
    require(asset['split']=='train' and asset['source_group'].startswith('train:')
        and body_id=='ring' and asset['task'] in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed')
        and body_id in asset['bodies'],'prototype symmetry CAD must be a real explicit TRAIN ring role')
    g,algebra=finite_group(rotations,algebra_tolerance=algebra_tolerance)
    c=np.asarray(center,float)
    require(c.shape==(3,) and np.isfinite(c).all() and 0<surface_tolerance_m<=1e-5,'finite CAD center and explicit micrometre surface tolerance required')
    body=asset['bodies'][body_id];visual=body['visual_component_names'];physical=body['collision_component_names']
    require(len(visual)==1 and visual==physical==[row['name'] for row in body['components']],
        'all visual/physical surfaces must be the same single exported native component')
    annotations={row['name']:row['flags'] for row in body['owned_component_annotations']}
    require(annotations[visual[0]]['renderable'] is True
        and (annotations[visual[0]]['collidable'] is True or annotations[visual[0]]['respondable'] is True),
        'common native visual-and-physical component flags are missing')
    parts,coverage=_mesh_parts(body['collision_vertices_body'],body['collision_triangles'],surface_tolerance_m);matches=[]
    for rotation in g:
        transformed=[(points-c)@rotation.T+c for points,approx in parts];cost=np.full((len(parts),len(parts)),np.inf);vertex_maps={};vertex_cost={}
        for i,a in enumerate(transformed):
            for j,(b,approximation) in enumerate(parts):
                if a.shape!=b.shape:continue
                d=np.linalg.norm(a[:,None]-b[None],axis=2);rows,columns=linear_sum_assignment(d)
                if {frozenset(columns[list(face)]) for face in _box_facets(a)}!=_box_facets(b):continue
                vertex_cost[i,j]=float(d[rows,columns].max())
                cost[i,j]=vertex_cost[i,j]+parts[i][1]+approximation;vertex_maps[i,j]=columns.tolist()
        rows,columns=linear_sum_assignment(cost)
        residual=float(cost[rows,columns].max())
        require(np.isfinite(residual) and residual<=surface_tolerance_m,
            'finite group does not preserve the complete physical/visual convex-part surfaces')
        matches.append(dict(convex_part_permutation=columns.tolist(),maximum_auxiliary_vertex_bijection_residual_m=
            max(vertex_cost[i,int(j)] for i,j in zip(rows,columns)),
            vertex_bijections=[vertex_maps[i,int(j)] for i,j in zip(rows,columns)],
            original_surface_Hausdorff_upper_bound_m=residual,
            bound_includes_both_native_to_auxiliary_errors=True,convex_quadrilateral_face_bijection=True))
    return dict(schema=SCHEMA,TRAIN_only=True,VAL_used=False,TEST_used=False,
        numerical_replay_policy=numerical_replay_policy(),
        task=asset['task'],body_id=body_id,TRAIN_source_group=asset['source_group'],CAD_asset_sha256=asset['asset_sha256'],
        center_body=c.tolist(),group_rotations_body=g.tolist(),group_validation=algebra,
        surface_tolerance_m=surface_tolerance_m,complete_visual_and_physical_geometric_surface_evidence=True,
        visual_physical_component_identity=visual[0],surface_coverage=coverage,group_surface_matches=matches,
        surface_equivalence_kind='finite convex-part surface bijection with both native-to-auxiliary errors bounded within TRAIN CAD precision tolerance',
        exact_mathematical_mesh_symmetry_claimed=False,texture_or_material_equivalence_certified=False,
        full_orientation_observable=False,orientation_equivalence_requires_current_RGBD_candidate_evidence=True,
        TRAIN_phase_affordance_permission_required=True,policy_skill_authorized=False,
        limitations=['Prototype is limited to this TRAIN ring family and certified eight-corner affine-box components.',
            'Surface geometry equivalence does not prove grasp/contact/approach or task-success equivariance.',
            'Current ICP quotient consistency does not prove absolute registration accuracy or exhaustive global identifiability.'])


def build_train_certificate(root,asset_pointer,body_id,rotations,center,*,
        TRAIN_manifest_pointer,TRAIN_receipt_pointer,surface_tolerance_m=2.5e-6):
    root=Path(root).resolve();path=(root/asset_pointer['path']).resolve()
    require(path.is_relative_to(root) and file_sha(path)==asset_pointer['sha256'],'frozen TRAIN CAD file SHA differs')
    asset=json.loads(path.read_text());certificate=certify_train_geometry(asset,body_id,rotations,center,
        surface_tolerance_m=surface_tolerance_m)
    source=Path(__file__).resolve();exporter=source.parent/'training_geometry.py';flags=source.parent/'component_flags.py'
    reader=source.parent/'safe_native_mesh.py';lineage={}
    def read_pointer(pointer):
        p=(root/pointer['path']).resolve()
        require(p.is_relative_to(root) and file_sha(p)==pointer['sha256'],'TRAIN acquisition lineage file SHA differs')
        lineage[str(p.relative_to(root))]=pointer['sha256'];return json.loads(p.read_text())
    manifest=read_pointer(TRAIN_manifest_pointer);record=read_pointer(TRAIN_receipt_pointer)
    entry=record.get('entry',{});process=record.get('parent_process_receipt',{})
    require(manifest.get('complete') is True and manifest.get('completed')==manifest.get('expected')
        and not manifest.get('missing_case_ids') and manifest.get('TEST_used') is False
        and entry in manifest.get('entries',[]) and dict(case_id=record.get('case_id'),**TRAIN_receipt_pointer) in manifest.get('case_receipts',[]) 
        and record.get('status')=='completed' and record.get('split')==entry.get('split')=='train'
        and record.get('task')==entry.get('task')==asset['task'] and entry.get('parent_source_group')==asset['source_group']
        and record.get('plan_sha256')==manifest.get('plan_sha256') and record.get('policy_history_contains_native_GT') is False
        and record.get('files_sha256',{}).get(str(path.relative_to(root)))==asset_pointer['sha256']
        and record.get('native_reset_receipt',{}).get('source_hdf5_sha256')==asset['source_hdf5_sha256'],
        'CAD must belong to the exact completed native TRAIN acquisition and manifest')
    child=read_pointer(dict(path=process['child_case_path'],sha256=process['child_case_sha256']))
    require(child=={k:v for k,v in record.items() if k!='parent_process_receipt'}
        and child.get('process_id')==record.get('process_id') and type(record.get('process_id')) is int
        and record['process_id']>0 and record.get('case_isolation')=='fresh_simulator_process_per_episode',
        'TRAIN CAD acquisition is not the exact fresh-process raw child')
    launcher=(root/process['launcher']).resolve()
    require(launcher.is_relative_to(root) and file_sha(launcher)==process['launcher_sha256'],'TRAIN acquisition launcher changed')
    lineage[str(launcher.relative_to(root))]=process['launcher_sha256'];command=process.get('command',[])
    require('--plan' in command,'TRAIN acquisition plan receipt missing')
    plan_path=(root/command[command.index('--plan')+1]).resolve()
    require(plan_path.is_relative_to(root),'TRAIN acquisition plan must be a frozen project file')
    plan=read_pointer(dict(path=str(plan_path.relative_to(root)),sha256=file_sha(plan_path)))
    require(plan.get('plan_sha256')==record['plan_sha256'],'TRAIN CAD acquisition used a different plan')
    export_source_evidence={}
    for code in (exporter,flags,reader):
        declared=plan.get('inputs',{}).get(str(code)) or plan.get('inputs',{}).get(str(code.relative_to(root)))
        executed=code
        if declared!=file_sha(code):
            executed=(root/TRAIN_manifest_pointer['path']).parent/'executed_source_snapshot'/code.relative_to(root)
            require(executed.is_file() and file_sha(executed)==declared,'frozen executed CAD source snapshot is missing')
            # Later consumers may evolve, but the actual raw exporter used to
            # establish visual=physical surface identity must still be audited.
            require(code==exporter,'native flags/mesh reader differ from the acquisition source lock')
            def functions(p):return {node.name:ast.dump(node,include_attributes=False)
                for node in ast.parse(p.read_text()).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
            old,new=functions(executed),functions(code)
            require(all(old.get(name)==new.get(name) and name in old for name in
                ('export_train_geometry','_owned_shapes','_surface_points')),
                'native raw CAD exporter logic changed after the frozen acquisition')
        require(declared==file_sha(executed),'native CAD exporter/flags/reader differ from the acquisition source lock')
        lineage[str(executed.relative_to(root))]=declared
        export_source_evidence[code.name]=dict(executed_path=str(executed.relative_to(root)),executed_sha256=declared,
            current_path=str(code.relative_to(root)),current_sha256=file_sha(code),
            raw_export_function_AST_unchanged=True)
    certificate['inputs_sha256']={str(path.relative_to(root)):file_sha(path),
        str(source.relative_to(root)):file_sha(source),str(exporter.relative_to(root)):file_sha(exporter),
        str(flags.relative_to(root)):file_sha(flags),str(reader.relative_to(root)):file_sha(reader),**lineage}
    certificate['CAD_file']=dict(path=str(path.relative_to(root)),sha256=file_sha(path))
    certificate['TRAIN_acquisition_manifest']=TRAIN_manifest_pointer
    certificate['TRAIN_acquisition_receipt']=TRAIN_receipt_pointer
    certificate['native_CAD_source_evidence']=export_source_evidence
    certificate['certificate_sha256']=digest(certificate)
    return certificate


@dataclass(frozen=True)
class VerifiedFiniteSymmetry:
    certificate:dict
    _token:object=field(repr=False,compare=False)
    replay_receipt:dict=field(default_factory=dict,repr=False,compare=False)

    def __post_init__(self):
        require(self._token is _VERIFIED_TOKEN,'finite symmetry evidence must be reproduced by the file loader')

    @classmethod
    def load(cls,root,pointer):
        root=Path(root).resolve();path=(root/pointer['path']).resolve()
        require(path.is_relative_to(root) and file_sha(path)==pointer['sha256'],'finite symmetry certificate file SHA differs')
        value=json.loads(path.read_text())
        require(value.get('schema')==SCHEMA and value.get('certificate_sha256')==digest({k:v for k,v in value.items() if k!='certificate_sha256'}),
            'finite symmetry certificate fingerprint differs')
        for name,checksum in value['inputs_sha256'].items():
            source=(root/name).resolve()
            require(source.is_relative_to(root) and file_sha(source)==checksum,'finite symmetry CAD/source input changed')
        expected=build_train_certificate(root,value['CAD_file'],value['body_id'],value['group_rotations_body'],
            value['center_body'],TRAIN_manifest_pointer=value['TRAIN_acquisition_manifest'],
            TRAIN_receipt_pointer=value['TRAIN_acquisition_receipt'],surface_tolerance_m=value['surface_tolerance_m'])
        receipt=replay_train_certificate(value,expected)
        return cls(value,_VERIFIED_TOKEN,receipt)


def current_orientation_equivalence(symmetry,candidate_poses,candidate_rms_m,*,
        registration_valid,directly_visible,all_retained_hypotheses_provided,
        rms_slack_m=.0005,maximum_registration_rms_m=.012,
        quotient_rotation_tolerance_rad=.04,center_translation_tolerance_m=.003):
    """Judge all current near-optimal ICP hypotheses modulo a verified group.

    No GT, task result, expert, future observation or native handle is accepted.
    The statement is explicitly conditional on the supplied retained ICP search.
    A symmetry-consistent hypothesis can still be inaccurate in absolute pose.
    """
    require(isinstance(symmetry,VerifiedFiniteSymmetry),'load and reproduce a frozen TRAIN surface certificate first')
    certificate=symmetry.certificate
    require(certificate['certificate_sha256']==digest({k:v for k,v in certificate.items() if k!='certificate_sha256'}),
        'loaded finite symmetry evidence was mutated')
    g=np.asarray(certificate['group_rotations_body']);c=np.asarray(certificate['center_body'])
    poses=tuple(candidate_poses);rms=np.asarray(candidate_rms_m,float)
    require(rms.shape==(len(poses),) and np.isfinite(rms).all() and (rms>=0).all()
        and all(isinstance(p,Pose) for p in poses),'current finite ICP hypotheses and RMS evidence required')
    require(all(np.isfinite(p.position).all() and np.isfinite(p.rotation).all()
        and np.linalg.norm(p.rotation.T@p.rotation-np.eye(3))<1e-6 and abs(np.linalg.det(p.rotation)-1)<1e-6 for p in poses),
        'current hypotheses must be finite proper SE3 poses')
    require(min(rms_slack_m,maximum_registration_rms_m,quotient_rotation_tolerance_rad,center_translation_tolerance_m)>0
        and quotient_rotation_tolerance_rad<np.pi/8,'explicit finite registration/quotient tolerances required')
    result=dict(schema=OBSERVATION_SCHEMA,body_id=certificate['body_id'],
        certificate_sha256=certificate['certificate_sha256'],orientation_equivalence_observable=False,
        full_orientation_observable=False,policy_skill_authorized=False,TRAIN_phase_affordance_permission_required=True,
        representative_pose_source='current_RGBD_ICP_only',absolute_registration_accuracy_proven=False,
        exhaustive_global_identifiability_claimed=False,candidate_generation_scope='all supplied retained ICP hypotheses',
        tested_hypotheses=len(poses),near_optimal_hypotheses=0,reason=None)
    if not (registration_valid is True and directly_visible is True and all_retained_hypotheses_provided is True):
        result['reason']='current direct valid registration and every retained ICP hypothesis required';return result
    if len(poses)<2 or rms.min()>maximum_registration_rms_m:
        result['reason']='at least two finite sufficiently accurate current ICP hypotheses required';return result
    best=int(rms.argmin());near=np.flatnonzero(rms<=rms[best]+rms_slack_m);best_pose=poses[best]
    if len(near)<2:
        result['reason']='insufficient near-optimal evidence for the finite quotient';return result
    rows=[]
    for index in near:
        pose=poses[int(index)];relative=best_pose.rotation.T@pose.rotation
        angles=Rotation.from_matrix(relative[None]@np.swapaxes(g,1,2)).magnitude();group_index=int(angles.argmin())
        delta=float(np.linalg.norm((pose.position+pose.rotation@c)-(best_pose.position+best_pose.rotation@c)))
        rows.append(dict(hypothesis_index=int(index),group_index=group_index,
            quotient_rotation_residual_rad=float(angles[group_index]),world_symmetry_center_delta_m=delta))
    accepted=all(row['quotient_rotation_residual_rad']<=quotient_rotation_tolerance_rad
        and row['world_symmetry_center_delta_m']<=center_translation_tolerance_m for row in rows)
    result.update(orientation_equivalence_observable=accepted,near_optimal_hypotheses=len(near),best_hypothesis_index=best,
        near_optimal_evidence=rows,reason='all near-optimal retained poses agree modulo certified D4' if accepted else
        'a near-optimal pose differs outside the certified finite SE3 quotient',
        representative_pose=dict(position=best_pose.position.tolist(),rotation=best_pose.rotation.tolist()),
        tolerances=dict(rms_slack_m=rms_slack_m,maximum_registration_rms_m=maximum_registration_rms_m,
            quotient_rotation_tolerance_rad=quotient_rotation_tolerance_rad,center_translation_tolerance_m=center_translation_tolerance_m))
    return result


@dataclass(frozen=True)
class CurrentRotationEquivalence:
    """Non-serializable current-observation capability, never a full-R flag.

    The public constructor below composes each *retained* root hypothesis with
    the frozen role-to-root transform. Consumers must match the exact observed
    body, timestamp and representative SE3; a copied JSON flag cannot grant a
    geometry or skill permission. Confidence is conditional on the retained
    registration search and is not an absolute pose-accuracy certificate.
    """
    body_id:str
    timestamp:float
    representative_pose:Pose
    symmetry:VerifiedFiniteSymmetry=field(repr=False,compare=False)
    evidence:dict=field(repr=False,compare=False)
    evidence_sha256:str
    _token:object=field(repr=False,compare=False)

    def __post_init__(self):
        require(self._token is _CURRENT_TOKEN,'current quotient evidence must come from retained RGBD hypotheses')
        self._check()

    def _check(self):
        require(isinstance(self.symmetry,VerifiedFiniteSymmetry) and isinstance(self.representative_pose,Pose)
            and self.body_id==self.symmetry.certificate['body_id'] and np.isfinite(self.timestamp)
            and self.evidence_sha256==digest(self.evidence)
            and self.evidence.get('body_id')==self.body_id
            and self.evidence.get('timestamp')==self.timestamp
            and self.evidence.get('certificate_sha256')==self.symmetry.certificate['certificate_sha256']
            and self.symmetry.certificate['certificate_sha256']==digest({k:v for k,v in
                self.symmetry.certificate.items() if k!='certificate_sha256'}),
            'current finite quotient capability or frozen certificate was mutated')
        pose=self.evidence.get('representative_pose')
        if pose is not None:
            require(np.array_equal(self.representative_pose.position,np.asarray(pose['position']))
                and np.array_equal(self.representative_pose.rotation,np.asarray(pose['rotation'])),
                'current quotient representative pose was changed')

    @classmethod
    def from_root_hypotheses(cls,symmetry,*,body_id,timestamp,root_candidate_poses,
            candidate_rms_m,body_in_root,registration_valid,directly_visible,
            all_retained_hypotheses_provided,representative_body_pose=None,**tolerances):
        require(isinstance(symmetry,VerifiedFiniteSymmetry) and isinstance(body_in_root,Pose)
            and body_id==symmetry.certificate['body_id'] and np.isfinite(timestamp),
            'current role and frozen root-to-role transform are required')
        roots=tuple(root_candidate_poses)
        require(roots and all(isinstance(p,Pose) for p in roots),'retained root SE3 hypotheses required')
        poses=tuple(Pose(p.position+p.rotation@body_in_root.position,p.rotation@body_in_root.rotation) for p in roots)
        result=current_orientation_equivalence(symmetry,poses,candidate_rms_m,
            registration_valid=registration_valid,directly_visible=directly_visible,
            all_retained_hypotheses_provided=all_retained_hypotheses_provided,**tolerances)
        selected=poses[int(np.argmin(candidate_rms_m))]
        if representative_body_pose is not None:
            require(isinstance(representative_body_pose,Pose) and
                np.linalg.norm(selected.position-representative_body_pose.position)<=1e-9 and
                np.linalg.norm(selected.rotation-representative_body_pose.rotation)<=1e-9,
                'quotient representative must be the actual current observed registration pose')
            selected=representative_body_pose
        if result.get('representative_pose') is not None:
            result['representative_pose']=dict(position=selected.position.tolist(),rotation=selected.rotation.tolist())
        result.update(envelope_schema=ENVELOPE_SCHEMA,timestamp=float(timestamp),
            finite_certificate_replay_receipt=json.loads(json.dumps(symmetry.replay_receipt)),
            root_hypothesis_count=len(roots),body_in_root=dict(position=body_in_root.position.tolist(),
                rotation=body_in_root.rotation.tolist()),
            retained_root_hypotheses_sha256=digest([dict(position=p.position.tolist(),rotation=p.rotation.tolist(),
                rms_m=float(r)) for p,r in zip(roots,candidate_rms_m)]))
        return cls(body_id,float(timestamp),selected,symmetry,result,digest(result),_CURRENT_TOKEN)

    @property
    def certificate_sha256(self):
        self._check();return self.evidence['certificate_sha256']

    @property
    def observable(self):
        self._check();return self.evidence.get('orientation_equivalence_observable') is True

    def matches(self,*,body_id,timestamp,pose,now=None,maximum_age=None,certificate_sha256=None):
        self._check()
        if (body_id!=self.body_id or not isinstance(pose,Pose) or not np.isfinite(timestamp)
                or abs(float(timestamp)-self.timestamp)>1e-9
                or not np.array_equal(pose.position,self.representative_pose.position)
                or not np.array_equal(pose.rotation,self.representative_pose.rotation)
                or (certificate_sha256 is not None and certificate_sha256!=self.certificate_sha256)):
            return False
        if now is not None:
            require(maximum_age is not None and np.isfinite([now,maximum_age]).all()
                and maximum_age>=0,'explicit current evidence age bound required')
            if not -1e-8<=float(now)-self.timestamp<=float(maximum_age):return False
        return self.observable

    def receipt(self):
        self._check();return json.loads(json.dumps(self.evidence))
