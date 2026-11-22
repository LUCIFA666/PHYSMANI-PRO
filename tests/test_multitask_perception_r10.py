"""CPU synthetic RGB-D checks: world SE3, symmetry, occlusion and prior boundaries."""
from dataclasses import replace

import numpy as np
import unittest
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.perception import (
    BodyTemplate, GeometryComponent, MultiBodySE3Tracker, PriorProvenance,
    RegistrationConfig, RigidChildConstraint, decode_instance_mask,
)
from our_mani.multitask.specs import get_task_spec


CAD = PriorProvenance('declared_cad', 'synthetic CAD fixture, shared by both methods',
                      'metres in explicitly supplied body frame', True)
TRAIN_LAYOUT = PriorProvenance('train_calibrated_rigid_layout', 'synthetic TRAIN layout',
                               'child-local to parent-local', True, 'train',
                               ('train:synthetic:episode0',), 'a'*64)


def box(center, dimensions, resolution=13):
    points = []
    center, dimensions = np.asarray(center), np.asarray(dimensions)
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            a, b = np.meshgrid(np.linspace(-.5,.5,resolution), np.linspace(-.5,.5,resolution))
            face = np.zeros((resolution**2,3))
            face[:,axis] = sign*.5
            face[:,others[0]], face[:,others[1]] = a.ravel(),b.ravel()
            points.append(face*dimensions+center)
    return np.unique(np.concatenate(points),axis=0)


def cameras():
    result = {}
    for name,position in zip(('front','8','16','36'),
                             ([1.4,0,1.3],[0,1.3,1.3],[-1.3,0,1.25],[0,-1.3,1.3])):
        position=np.asarray(position,float)
        forward=np.array([0.,0.,1.])-position;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
        up=np.cross(forward,right)
        rotation=np.column_stack((right,up,forward))
        focal=-600. if name=='front' else 600.
        k=np.array([[focal,0,128.],[0,focal,128.],[0,0,1.]])
        result[name]=CameraCalibration(k,Pose(position,rotation))
    return result


def render(templates, poses, timestamp, *, omit_ids=()):
    calibrations=cameras();depths={};masks={}
    for name,calibration in calibrations.items():
        depth=np.full((256,256),np.inf);labels=np.zeros((256,256),dtype=np.int64)
        for template in templates:
            for component in template.components:
                label=component.instance_ids[0]
                if label in omit_ids:continue
                world=poses[template.role_id].apply(component.points_body)
                points=calibration.camera_to_world.inverse_apply(world)
                uv=points@calibration.intrinsics.T;uv=np.rint(uv[:,:2]/uv[:,2:]).astype(int)
                good=(points[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<256)&(uv[:,1]>=0)&(uv[:,1]<256)
                for pixel,z in zip(uv[good],points[good,2]):
                    x,y=pixel
                    if z<depth[y,x]:depth[y,x],labels[y,x]=z,label
        depth[~np.isfinite(depth)]=0
        depths[name]=depth;masks[name]=labels
    return dict(timestamp=timestamp,**{'observation/depths':depths,
        'observation/instance_masks':masks,'observation/calibrations':calibrations})


def multi_component_body(role='frame'):
    return BodyTemplate(role,tuple(GeometryComponent(str(i),(i+1,),box(center,size))
        for i,(center,size) in enumerate([([-.055,0,0],[.035,.025,.025]),
                                       ([.045,.025,.02],[.025,.04,.04]),
                                       ([.005,-.05,-.015],[.02,.025,.035])])),CAD)


def test_stationary_center_rotation_and_world_twist_from_four_views():
    template=multi_component_body();tracker=MultiBodySE3Tracker([template]);objects=None;meta=None
    center=np.array([.02,-.01,1.]);omega=.7
    for tick in range(5):
        time=tick*.05;pose=Pose(center,Rotation.from_euler('z',.28+omega*time).as_matrix())
        objects,meta=tracker.observe(render([template],{'frame':pose},time),modes={'frame':'external'})
    state=objects[0]
    assert np.linalg.norm(state.pose.position-center)<.006
    assert (Rotation.from_matrix(state.pose.rotation).inv()*Rotation.from_matrix(pose.rotation)).magnitude()<.08
    assert np.linalg.norm(state.linear_velocity)<.04
    assert abs(state.angular_velocity[2]-omega)<.2
    assert meta['bodies']['frame']['orientation_observable']
    assert meta['bodies']['frame']['twist_observed']


def test_sphere_position_without_claiming_orientation_or_spin():
    phi=np.linspace(0,np.pi,24);theta=np.linspace(0,2*np.pi,48,endpoint=False)
    theta,phi=np.meshgrid(theta,phi)
    points=.04*np.c_[np.sin(phi).ravel()*np.cos(theta).ravel(),np.sin(phi).ravel()*np.sin(theta).ravel(),np.cos(phi).ravel()]
    template=BodyTemplate('ball',(GeometryComponent('sphere',(51,),points),),CAD,
                          orientation_observability='none',sphere_radius=.04)
    tracker=MultiBodySE3Tracker([template]);center=np.array([.0,.0,1.])
    for tick in range(3):
        center=np.array([tick*.004,0.,1.]);pose=Pose(center,Rotation.from_euler('xyz',[.4,.5,tick*.7]).as_matrix())
        states,meta=tracker.observe(render([template],{'ball':pose},tick*.05))
    assert np.linalg.norm(states[0].pose.position-center)<.004
    assert np.allclose(states[0].pose.rotation,np.eye(3))
    assert np.allclose(states[0].angular_velocity,0)
    assert not meta['bodies']['ball']['orientation_observable']
    assert meta['bodies']['ball']['angular_velocity_observability']=='none'


def test_rigid_children_supply_parent_angular_phase_and_occlusion_is_explicit():
    parent=BodyTemplate('frame',(GeometryComponent('base',(100,),box([0,0,0],[.05,.05,.02])),),CAD)
    child1=BodyTemplate('peg0',(GeometryComponent('peg',(101,),box([0,0,0],[.018,.018,.05])),),CAD)
    child2=BodyTemplate('peg1',(GeometryComponent('peg',(102,),box([0,0,0],[.018,.018,.04])),),CAD)
    child3=BodyTemplate('peg2',(GeometryComponent('peg',(103,),box([0,0,0],[.018,.018,.03])),),CAD)
    templates=[parent,child1,child2,child3];offsets=[[-.06,0,.04],[.04,.035,.035],[.02,-.055,.025]]
    constraints=[RigidChildConstraint('frame',child.role_id,Pose(offset),TRAIN_LAYOUT) for child,offset in zip(templates[1:],offsets)]
    tracker=MultiBodySE3Tracker(templates,rigid_children=constraints)
    root=Pose(np.array([0.,0.,1.]),Rotation.from_euler('z',.55).as_matrix())
    poses={'frame':root,**{child.role_id:Pose(root.apply(offset),root.rotation) for child,offset in zip(templates[1:],offsets)}}
    states,meta=tracker.observe(render(templates,poses,0.,omit_ids=(100,)))
    states={s.object_id:s for s in states}
    assert not states['frame'].visible
    assert meta['rigid_groups'][0]['registration_valid']
    assert meta['rigid_groups'][0]['rigidity_consistency_tested']
    assert np.linalg.norm(states['frame'].pose.position-root.position)<.006
    assert Rotation.from_matrix(states['frame'].pose.rotation@root.rotation.T).magnitude()<.1
    # Losing the whole group after an indirectly initialized parent must not
    # invent an observation or crash for a never-directly-visible role.
    hidden=render(templates,poses,.05,omit_ids=(100,101,102,103))
    states,meta=tracker.observe(hidden)
    assert all(not state.visible for state in states)
    assert all(not body['twist_observed'] for body in meta['bodies'].values())


def test_no_future_frame_or_timestamp_and_mode_change_resets_twist():
    template=multi_component_body();tracker=MultiBodySE3Tracker([template]);pose=Pose([0,0,1])
    observation=render([template],{'frame':pose},0.)
    tracker.observe(observation,modes={'frame':'external'})
    with unittest.TestCase().assertRaisesRegex(ValueError,'timestamps must increase'):tracker.observe(observation)
    future=render([template],{'frame':pose},.05)
    future['observation/camera_timestamps']={'8':.051}
    with unittest.TestCase().assertRaisesRegex(ValueError,'future or stale'):tracker.observe(future)
    _,meta=tracker.observe(render([template],{'frame':pose},.05),modes={'frame':'attached'})
    assert not meta['bodies']['frame']['twist_observed']
    assert meta['bodies']['frame']['mode_provenance']=='caller_controller_touch_action_proprioception'


def test_template_permissions_and_articulated_constraints():
    with unittest.TestCase().assertRaisesRegex(ValueError,'equal information'):replace(CAD,shared_with_baseline=False)
    with unittest.TestCase().assertRaisesRegex(ValueError,'VAL/TEST'):PriorProvenance('declared_cad','file','local',True,'test')
    template=multi_component_body('button')
    with unittest.TestCase().assertRaisesRegex(ValueError,'TRAIN source'):BodyTemplate.from_train_geometry('button',template.components,
        source_episodes=[dict(split='test',task='x',raw_episode_id='episode1')],evidence_sha256='a'*64,
        source_reference='bad',frame_definition='body')
    plate=BodyTemplate('plate',(GeometryComponent('plate',(222,),box([0,0,0],[.02,.02,.01])),),CAD)
    constraint=RigidChildConstraint('button','plate',Pose([0,0,.02]),TRAIN_LAYOUT)
    with unittest.TestCase().assertRaisesRegex(ValueError,'articulated'):MultiBodySE3Tracker([template,plate],
        task_spec=get_task_spec('push_moving_button'),rigid_children=[constraint])


def test_color_coded_mask_bytes_agree_with_integer_ids():
    ids=np.array([[1,65537],[256,16777215]],dtype=np.int64)
    rgb=np.stack((ids%256,(ids//256)%256,(ids//65536)%256),axis=-1)
    assert np.array_equal(decode_instance_mask(rgb.astype(np.uint8)),ids)
    assert np.array_equal(decode_instance_mask(rgb.astype(float)/255),ids)


def test_conflicting_rigid_prior_falls_back_instead_of_forcing_child_pose():
    parent=multi_component_body('frame')
    child=BodyTemplate('peg0',(GeometryComponent('peg',(201,),box([0,0,0],[.02,.025,.045])),),CAD)
    constraint=RigidChildConstraint('frame','peg0',Pose([.09,0,.03]),TRAIN_LAYOUT)
    root_pose=Pose([0,0,1],Rotation.from_euler('z',.2).as_matrix())
    moved_child=Pose(root_pose.apply([.22,.0,.03]),root_pose.rotation)
    tracker=MultiBodySE3Tracker([parent,child],rigid_children=[constraint])
    states,meta=tracker.observe(render([parent,child],{'frame':root_pose,'peg0':moved_child},0.))
    states={s.object_id:s for s in states}
    assert not meta['rigid_groups'][0]['registration_valid']
    assert meta['bodies']['peg0']['rigid_root'] is None
    assert np.linalg.norm(states['peg0'].pose.position-moved_child.position)<.008
    assert np.linalg.norm(states['frame'].pose.position-root_pose.position)<.008


def test_axis_symmetric_shape_reports_axis_and_does_not_invent_axial_spin():
    theta,z=np.meshgrid(np.linspace(0,2*np.pi,60,endpoint=False),np.linspace(-.06,.06,24))
    points=np.c_[.025*np.cos(theta).ravel(),.025*np.sin(theta).ravel(),z.ravel()]
    template=BodyTemplate('cylinder',(GeometryComponent('side',(310,),points),),CAD,
                          orientation_observability='axis')
    tracker=MultiBodySE3Tracker([template]);pose=None
    for tick in range(3):
        pose=Pose([0,0,1],Rotation.from_euler('x',.3+tick*.015).as_matrix()@
                  Rotation.from_euler('z',tick*.6).as_matrix())
        states,meta=tracker.observe(render([template],{'cylinder':pose},tick*.05))
    state=states[0];axis=state.pose.rotation[:,2]
    assert abs(np.dot(axis,pose.rotation[:,2]))>.99
    assert abs(np.dot(state.angular_velocity,axis))<1e-9
    assert not meta['bodies']['cylinder']['orientation_observable']
    assert meta['bodies']['cylinder']['angular_velocity_observability']=='perpendicular_to_symmetry_axis_only'


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
        for name, value in globals().items() if name.startswith("test_") and callable(value))


def test_nonrenderable_role_is_never_directly_visible_and_needs_rigid_rgbd_support():
    with unittest.TestCase().assertRaises(ValueError):
        BodyTemplate('hidden', (), CAD)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'TRAIN evidence'):
        BodyTemplate('hidden', (), CAD, visual_geometry_status='native_nonrenderable')
    hidden = BodyTemplate('hidden', (), TRAIN_LAYOUT, visual_geometry_status='native_nonrenderable')
    parent = multi_component_body('frame')
    pose = Pose([0., 0., 1.], Rotation.from_euler('z', .3).as_matrix())
    alone = MultiBodySE3Tracker([hidden])
    states, meta = alone.observe(render([hidden], {'hidden': pose}, 0.))
    assert states == [] and not meta['bodies']['hidden']['visible']
    assert not meta['bodies']['hidden']['registration_valid']
    constraint = RigidChildConstraint('frame', 'hidden', Pose([.07, 0., .02]), TRAIN_LAYOUT)
    tracker = MultiBodySE3Tracker([parent, hidden], rigid_children=[constraint])
    states, meta = tracker.observe(render([parent], {'frame': pose}, 0.))
    state = {row.object_id: row for row in states}['hidden']
    assert not state.visible
    assert np.linalg.norm(state.pose.position - pose.apply([.07, 0., .02])) < .006
    assert meta['bodies']['hidden']['points_source'] == 'current_rigid_group_RGBD_support_not_child_surface'
    assert meta['bodies']['hidden']['observed_point_count'] == 0
    states, meta = tracker.observe(render([parent], {'frame': pose}, .05, omit_ids=(1, 2, 3)))
    assert all(not row.visible for row in states)
    assert meta['bodies']['hidden']['occlusion'] == 'TRAIN_native_nonrenderable_no_direct_visual_surface'
