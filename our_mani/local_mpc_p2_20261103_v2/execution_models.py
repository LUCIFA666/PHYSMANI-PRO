"""Pluggable NumPy execution model protocol for an independent P2 prototype.

No Native imports, stepping, training or live-controller writes. ``q``/``dq``
are measured feedback; reference arrays are actual controller-internal state,
not measured joint motion. A virtual h is never a next-plan feedback source.
"""
from copy import deepcopy
import numpy as np
from ..local_mpc_20261102.batch_mpc import reference,norm,unit,cross,qmul,qinv,qexp,qlog,rotate,limited


def _joint_vector(value,name,joints=None):
    array=np.asarray(value,dtype=np.float64)
    if array.ndim!=1 or (joints is not None and len(array)!=joints) or not np.isfinite(array).all():
        raise ValueError('finite actual '+name+'[J] required')
    return array.copy()


def validate_feedback_state(actual):
    """Preserve joint measurements alongside the frozen P1 public-state keys."""
    state=reference.validate_state(actual)
    if ('q' in actual)!=('dq' in actual):raise ValueError('q and dq must be supplied together')
    if 'q' in actual:
        state['q']=_joint_vector(actual['q'],'q')
        if not len(state['q']):raise ValueError('at least one joint required')
        state['dq']=_joint_vector(actual['dq'],'dq',len(state['q']))
    if 'finger_open_amounts' in actual:
        fingers=_joint_vector(actual['finger_open_amounts'],'finger_open_amounts',2)
        if np.any((fingers<0)|(fingers>1)):raise ValueError('actual two finger openings must be in [0,1]')
        state['finger_open_amounts']=fingers
    return state


def initialize_execution_batch(actual,batch_size,*,requires_joint_state=False,p1_compat=False):
    """Freshly broadcast the current measured state and actual reference h.

    Controller aliases:
    reference_positions -> q_ref, reference_velocities -> dq_ref,
    reference_accelerations -> ddq_ref, target_joints -> goal_q/last_goal.
    Absent/invalid reference uses measured q/dq and zero a as placeholders with
    valid=False; it is not promoted to an existing controller reference.
    """
    state=validate_feedback_state(actual)
    if type(batch_size) is not int or batch_size<=0:raise ValueError('positive integer batch size required')
    if requires_joint_state and 'q' not in state:raise ValueError('this execution model requires actual q/dq')
    keys=('tcp_position','tcp_quaternion','tcp_velocity','tcp_angular_velocity','object_position','object_quaternion',
          'object_linear_velocity','object_angular_velocity','opening','held','time')
    x={key:np.broadcast_to(state[key],(batch_size,)+np.shape(state[key])).copy() for key in keys}
    if 'q' in state:
        x.update(q=np.broadcast_to(state['q'],(batch_size,len(state['q']))).copy(),
                 dq=np.broadcast_to(state['dq'],(batch_size,len(state['q']))).copy())
    if 'finger_open_amounts' in state:
        x['finger_open_amounts']=np.broadcast_to(state['finger_open_amounts'],(batch_size,2)).copy()
    actuator=state.get('actuator_state',{})
    fallback=0. if state['held'] else state['opening']
    if not p1_compat:fallback=actuator.get('gripper_target',fallback)
    requested=float(state.get('requested_opening',actuator.get('requested_opening',fallback)))
    # P1 parity: an explicit state.requested_opening remains authoritative.
    if not np.isfinite(requested) or not 0<=requested<=1:raise ValueError('finite actual requested opening required')
    x['requested_opening']=np.full(batch_size,requested)
    j=len(state.get('q',[]));valid=bool(actuator.get('reference_valid',actuator.get('valid',False)))
    actual_a=actuator.get('ddq_ref',actuator.get('reference_accelerations'))
    initialized=bool(actuator.get('initialized',actuator.get('reference_initialized',actual_a is not None)))
    if (valid or initialized) and j==0:raise ValueError('initialized joint reference without measured q/dq is not a P2 state')
    if valid or initialized:
        q_value=actuator.get('q_ref',actuator.get('reference_positions'))
        dq_value=actuator.get('dq_ref',actuator.get('reference_velocities'))
        if valid and (q_value is None or dq_value is None):raise ValueError('valid reference requires actual reference q/dq')
        q_ref=_joint_vector(state['q'] if q_value is None else q_value,'reference q',j)
        dq_ref=_joint_vector(state['dq'] if dq_value is None else dq_value,'reference dq',j)
        ddq_ref=_joint_vector(actuator.get('ddq_ref',actuator.get('reference_accelerations')),'reference ddq',j)
        goal_value=actuator.get('goal_q',actuator.get('last_goal',actuator.get('target_joints')))
        if valid and goal_value is None:raise ValueError('valid reference requires its actual last goal')
        goal=_joint_vector(state['q'] if goal_value is None else goal_value,'last goal',j)
    else:
        q_ref=state.get('q',np.empty(0));dq_ref=state.get('dq',np.empty(0));ddq_ref=np.zeros(j);goal=q_ref.copy()
    h={name:np.broadcast_to(value,(batch_size,j)).copy() for name,value in
       (('q_ref',q_ref),('dq_ref',dq_ref),('ddq_ref',ddq_ref),('goal_q',goal),('last_goal',goal))}
    h.update(valid=np.full(batch_size,valid),reference_valid=np.full(batch_size,valid),
             initialized=np.full(batch_size,initialized or valid),requested_opening=np.full(batch_size,requested),
             object_in_tcp=rotate(qinv(x['tcp_quaternion']),x['object_position']-x['tcp_position']),
             object_quaternion_tcp=qmul(qinv(x['tcp_quaternion']),x['object_quaternion']),
             held_anchor=np.full(batch_size,state['held'],dtype=bool),
             source='fresh_actual_feedback_and_controller_snapshot',predicted=False,
             feedback_time=float(state['time']),reference_is_measured_state=False)
    h['gripper_target']=np.full(batch_size,float(actuator.get('gripper_target',requested)))
    return x,h


def validate_step_batch(previous,action,nxt,h_next,info,dt):
    """Cheap shared contract guard; failures cannot become fake contact/clock."""
    n=len(previous['time'])
    vector_sizes={'tcp_position':3,'tcp_quaternion':4,'tcp_velocity':3,'tcp_angular_velocity':3,
                  'object_position':3,'object_quaternion':4,'object_linear_velocity':3,'object_angular_velocity':3}
    for name,size in vector_sizes.items():
        value=np.asarray(nxt.get(name))
        if value.shape!=(n,size) or not np.isfinite(value).all():raise ValueError('invalid predicted '+name)
    for name in ('time','opening','requested_opening','held'):
        if np.asarray(nxt.get(name)).shape!=(n,):raise ValueError('invalid predicted '+name)
    if np.asarray(nxt['held']).dtype!=np.bool_:raise ValueError('predicted held must remain explicit bool')
    if np.any((~previous['held'])&nxt['held']):raise ValueError('implicit predicted attachment is forbidden in P1/P2 joint-response models')
    if not np.allclose(nxt['time'],previous['time']+dt,atol=1e-9,rtol=0):raise ValueError('execution model endpoint clock differs')
    if np.any((nxt['opening']<0)|(nxt['opening']>1)) or not np.isfinite(nxt['opening']).all():raise ValueError('invalid predicted actual aperture')
    if not np.isfinite(nxt['time']).all() or not np.isfinite(nxt['requested_opening']).all():raise ValueError('invalid predicted clock/command')
    for key in ('tcp_quaternion','object_quaternion'):
        if not np.allclose(norm(nxt[key]),1.,atol=1e-8,rtol=0):raise ValueError('predicted quaternion must be unit')
    for key in ('q','dq'):
        if key in previous:
            if np.asarray(nxt.get(key)).shape!=previous[key].shape or not np.isfinite(nxt[key]).all():raise ValueError('lost/nonfinite predicted joint '+key)
    if 'finger_open_amounts' in previous:
        fingers=np.asarray(nxt.get('finger_open_amounts'))
        if fingers.shape!=(n,2) or not np.isfinite(fingers).all() or np.any((fingers<0)|(fingers>1)):
            raise ValueError('finite predicted finger_open_amounts[N,2] in [0,1] required')
    if not isinstance(h_next,dict) or not isinstance(info,dict):raise ValueError('explicit h_next/info dictionaries required')
    if info.get('microsteps')!=2:raise ValueError('execution model must disclose two 50ms microsteps')
    if 'feasible' not in info:raise ValueError('execution model must disclose its feasibility mask')
    feasible=np.asarray(info['feasible'])
    if feasible.shape!=(n,) or feasible.dtype!=np.bool_:raise ValueError('aligned boolean model feasibility required')
    return feasible


class SimpleBatchExecutionModel:
    """Exact batched P1 Cartesian approximation, now behind step_batch.

    Joint q/dq/reference arrays are passed through and explicitly NOT predicted
    by this P1 model. A joint-residual+FK replacement must update them itself.
    """
    requires_joint_state=False
    def __init__(self,config=None):self.config=config or reference.MPCConfig()
    def initialize_batch(self,actual,batch_size):
        return initialize_execution_batch(actual,batch_size,requires_joint_state=self.requires_joint_state,p1_compat=True)
    def step_batch(self,x,u,h,dt=.1):
        c=self.config
        if dt!=c.dt:raise ValueError('two-tick .1s macrostep required')
        s={key:value.copy() for key,value in x.items()}
        # Reference arrays are immutable inputs here; only requested_opening and
        # provenance are replaced, so copying their whole storage is unnecessary.
        hidden=dict(h)
        ap,aq,opening=u['tcp_position'],u['tcp_quaternion'],u['opening']
        previous_opening=hidden['requested_opening']
        if not np.isin(opening,(0.,1.)).all():raise ValueError('binary requested Native opening required')
        for micro in range(2):
            before=s['object_position'].copy()
            wanted=limited(c.position_gain*(ap-s['tcp_position']),c.max_tcp_speed)
            next_v=s['tcp_velocity']+limited(wanted-s['tcp_velocity'],c.max_tcp_acceleration*c.native_dt)
            s['tcp_position']+=.5*(s['tcp_velocity']+next_v)*c.native_dt;s['tcp_velocity']=next_v
            error=qlog(qmul(aq,qinv(s['tcp_quaternion'])))
            wanted_w=limited(c.rotation_gain*error,c.max_angular_speed)
            next_w=s['tcp_angular_velocity']+limited(wanted_w-s['tcp_angular_velocity'],c.max_angular_acceleration*c.native_dt)
            delta=.5*(s['tcp_angular_velocity']+next_w)*c.native_dt
            s['tcp_quaternion']=unit(qmul(qexp(delta),s['tcp_quaternion']));s['tcp_angular_velocity']=next_w
            requested=previous_opening if micro*c.native_dt<c.opening_command_delay-1e-9 else opening
            s['opening']+=np.clip(requested-s['opening'],-c.opening_rate*c.native_dt,c.opening_rate*c.native_dt)
            s['held']&=requested<.5
            held=s['held'];free=~held
            if np.any(held):
                s['object_position'][held]=s['tcp_position'][held]+rotate(s['tcp_quaternion'][held],hidden['object_in_tcp'][held])
                s['object_quaternion'][held]=unit(qmul(s['tcp_quaternion'][held],hidden['object_quaternion_tcp'][held]))
                s['object_linear_velocity'][held]=(s['object_position'][held]-before[held])/c.native_dt
                s['object_angular_velocity'][held]=next_w[held]
            if np.any(free):
                s['object_position'][free]+=c.native_dt*s['object_linear_velocity'][free]
                s['object_quaternion'][free]=unit(qmul(qexp(c.native_dt*s['object_angular_velocity'][free]),s['object_quaternion'][free]))
            s['time']+=c.native_dt
        s['requested_opening']=opening.copy();hidden['requested_opening']=opening.copy()
        hidden['gripper_target']=opening.copy()
        hidden.update(source='virtual_rollout_prediction_only',predicted=True)
        return s,hidden,dict(model='P1_simple_batch_exact_reference',microsteps=2,feasible=np.ones(len(s['time']),bool),
            joint_prediction='not_modeled_measurements_passed_through',actuator_reference_prediction='not_modeled_passed_through',
            Native_steps=0,velocity_reference_actuated=False,attachment_inferred_from_closed_command=False)


class JointResidualFKExecutionModel:
    """P2-ready adapter; it supplies no learned weights or fictional training.

    reference_step(actual_q,actual_dq,joint_goal,h,limits,dt=.05) produces
    controller references. transition_batch(q,dq,q_ref,dq_ref,ddq_ref,latent,dt)
    must return finite q/dq, optional latent and feasible[N]. fk_batch(q) returns
    world tcp_position/tcp_quaternion. Both physical ticks call transition and
    FK; the next reference call receives predicted actual q/dq, never q_ref as
    a substitute measurement. No Native IK/FK or Cartesian fallback is used.
    """
    requires_joint_state=True
    requires_joint_goal=True
    def __init__(self,reference_step,limits,transition_batch,fk_batch,*,config=None,latent_initializer=None,model_id='untrained_adapter_only'):
        if (not callable(reference_step) or not callable(fk_batch)
                or not (callable(transition_batch) or callable(getattr(transition_batch,'predict_execution',None)))):
            raise TypeError('explicit reference, joint-transition and FK batch callbacks required')
        self.reference_step,self.limits,self.transition_batch,self.fk_batch=reference_step,limits,transition_batch,fk_batch
        self.config=config or reference.MPCConfig()
        self.latent_initializer,self.model_id=latent_initializer,str(model_id)
    def initialize_batch(self,actual,batch_size):
        x,h=initialize_execution_batch(actual,batch_size,requires_joint_state=True)
        h['latent']=None if self.latent_initializer is None else self.latent_initializer(deepcopy(actual),batch_size)
        return x,h
    def step_batch(self,x,u,h,dt=.1):
        c=self.config;n=len(x['time'])
        if dt!=c.dt:raise ValueError('two .05s physical-model ticks required')
        if 'joint_goal' not in u:raise ValueError('explicit joint_goal required; no Native IK or Cartesian fallback')
        goal=np.asarray(u['joint_goal'],float)
        if goal.shape!=x['q'].shape or not np.isfinite(goal).all():raise ValueError('finite joint_goal[N,J] required')
        s={key:value.copy() for key,value in x.items()};hidden=deepcopy(h)
        previous_opening=hidden['requested_opening'].copy()
        feasible=np.ones(n,bool);applied=[];changes=[];actual_inputs=[];predictions=[]
        for micro in range(2):
            actual_inputs.append(s['q'].copy())
            ref=self.reference_step(s['q'],s['dq'],goal,hidden,self.limits,dt=c.native_dt)
            hidden=ref['next_h'];ref_ok=np.asarray(ref['feasible'],bool)&np.asarray(ref['applied'],bool)
            applied.append(np.asarray(ref['applied']).copy());changes.append(np.asarray(ref['target_changed']).copy())
            # Failed samples are still finite virtual rows, explicitly masked
            # infeasible. No failed reference is promoted to an applied target.
            commanded=np.where(ref_ok[:,None],ref['q_ref'],s['q'])
            rich_transition=callable(getattr(self.transition_batch,'predict_execution',None))
            if rich_transition:
                if 'finger_open_amounts' not in s:raise ValueError('learned execution requires actual two finger_open_amounts; mean cannot substitute')
                references=dict(q_ref=commanded,dq_ref=ref['dq_ref'].copy(),ddq_ref=ref['ddq_ref'].copy(),
                                gripper_target=np.asarray(u['opening']).copy())
                predicted=self.transition_batch.predict_execution({key:value.copy() for key,value in s.items()},
                    references,deepcopy(hidden.get('latent')),c.native_dt,held_anchor=hidden['held_anchor'].copy())
            else:
                predicted=self.transition_batch(s['q'].copy(),s['dq'].copy(),commanded,
                    ref['dq_ref'].copy(),ref['ddq_ref'].copy(),deepcopy(hidden.get('latent')),c.native_dt)
            q,dq=np.asarray(predicted['q'],float),np.asarray(predicted['dq'],float)
            if q.shape!=s['q'].shape or dq.shape!=s['dq'].shape or not np.isfinite(q).all() or not np.isfinite(dq).all():
                raise ValueError('joint transition must return finite predicted actual q/dq')
            if 'feasible' not in predicted:raise ValueError('joint-transition must disclose feasible[N]')
            transition_ok=np.asarray(predicted['feasible'])
            if transition_ok.shape!=(n,) or transition_ok.dtype!=np.bool_:raise ValueError('joint-transition feasible[N] bool required')
            feasible&=ref_ok&transition_ok
            if hasattr(self.limits,'lower'):
                feasible&=np.all((q>=self.limits.lower-1e-8)&(q<=self.limits.upper+1e-8),axis=-1)
            s['q'],s['dq']=q,dq;hidden['latent']=predicted.get('latent',hidden.get('latent'))
            fk=self.fk_batch(q.copy())
            p,rotation=np.asarray(fk['tcp_position'],float),np.asarray(fk['tcp_quaternion'],float)
            if p.shape!=(n,3) or rotation.shape!=(n,4) or not np.isfinite(p).all() or not np.isfinite(rotation).all() or np.any(norm(rotation)<1e-12):
                raise ValueError('FK must return finite nonzero world TCP pose')
            rotation=unit(rotation)
            s['tcp_velocity']=(p-s['tcp_position'])/c.native_dt
            s['tcp_angular_velocity']=qlog(qmul(rotation,qinv(s['tcp_quaternion'])))/c.native_dt
            s['tcp_position'],s['tcp_quaternion']=p,rotation
            requested=previous_opening if micro*c.native_dt<c.opening_command_delay-1e-9 else u['opening']
            if rich_transition:
                fingers=np.asarray(predicted.get('finger_open_amounts'),float)
                if fingers.shape!=(n,2) or not np.isfinite(fingers).all() or np.any((fingers<0)|(fingers>1)):
                    raise ValueError('learned transition must return two valid finger_open_amounts')
                s['finger_open_amounts']=fingers
                s['opening']=np.mean(fingers,axis=-1)
            else:
                s['opening']+=np.clip(requested-s['opening'],-c.opening_rate*c.native_dt,c.opening_rate*c.native_dt)
            s['held']&=requested<.5
            before=s['object_position'].copy();held=s['held'];free=~held
            if np.any(held):
                s['object_position'][held]=p[held]+rotate(rotation[held],hidden['object_in_tcp'][held])
                s['object_quaternion'][held]=unit(qmul(rotation[held],hidden['object_quaternion_tcp'][held]))
                s['object_linear_velocity'][held]=(s['object_position'][held]-before[held])/c.native_dt
                s['object_angular_velocity'][held]=s['tcp_angular_velocity'][held]
            if np.any(free):
                s['object_position'][free]+=c.native_dt*s['object_linear_velocity'][free]
                s['object_quaternion'][free]=unit(qmul(qexp(c.native_dt*s['object_angular_velocity'][free]),s['object_quaternion'][free]))
            s['time']+=c.native_dt
            hidden['requested_opening']=np.asarray(u['opening']).copy()
            hidden['gripper_target']=np.asarray(u['opening']).copy()
            predictions.append(q.copy())
        s['requested_opening']=np.asarray(u['opening']).copy()
        hidden.update(last_goal=hidden['goal_q'].copy(),reference_valid=hidden['valid'].copy(),
                      source='virtual_joint_residual_FK_rollout_only',predicted=True)
        return s,hidden,dict(model=self.model_id,microsteps=2,feasible=feasible,joint_prediction='injected_transition_plus_FK',
            actual_q_inputs=np.stack(actual_inputs,axis=1),predicted_actual_q=np.stack(predictions,axis=1),
            reference_applied=np.stack(applied,axis=1),target_changed=np.stack(changes,axis=1),
            Native_steps=0,gripper_model=('injected_two_finger_response_held_anchor_fixed' if rich_transition else 'P1_rate_assumption_not_learned'),
            held_input='actual_anchor_fixed_no_future_contact_label',attachment_inferred_from_closed_command=False)
