"""Unqualified completion-latch finger response approximation, CPU only.

Wrap an explicit rich joint/two-finger transition. A completed real command
with an unchanged virtual target uses constant current finger coordinates as
a zero-motor approximation. No predicted completion, attachment, contact or
passive-load model is created. The base q/dq/feasibility are unchanged.
"""
from copy import deepcopy
from uuid import uuid4

import numpy as np


SCHEMA='actual_completion_anchor_zero_motor_finger_approximation_20261103_v1'
_KEY='completion_response_state'


def _array(value,shape,name):
    result=np.asarray(value,dtype=float)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError('finite '+name+str(shape)+' required')
    return result.copy()


class CompletionAwareFingerResponse:
    """Fresh real-plan initializer plus the existing rich transition API.

    Use ``latent_initializer=response.initialize_latent`` when constructing a
    new JointResidualFKExecutionModel. Never pass a previous virtual h as the
    next real ``actual`` state. Known virtual provenance is rejected; this
    interface does not authenticate an arbitrarily relabelled caller dict.
    """
    def __init__(self,base_response,*,base_latent_initializer=None):
        if not callable(getattr(base_response,'predict_execution',None)):
            raise TypeError('explicit rich base joint/two-finger response required')
        if base_latent_initializer is not None and not callable(base_latent_initializer):
            raise TypeError('explicit callable base initializer required')
        self.base_response=base_response;self.base_latent_initializer=base_latent_initializer
        self._real_anchor=None;self._tokens=set()
        self.evidence=dict(schema=SCHEMA,base_response_evidence=deepcopy(getattr(base_response,'evidence',{})),
            base_joint_predictions_unchanged=True,completion_read_from_real_actuator_snapshot_only=True,
            constant_fingers_are_zero_motor_approximation_not_passive_dynamics_model=True,
            completion_implies_held=False,completion_implies_contact=False,predicted_completion_inferred=False,
            actuator_done_is_not_a_target_reached_certificate=True,
            execution_reversal_contract='unfinished old target reversed: first Native doneTrue may be brake/cache-clear; rearm once at same physical time before latching',
            virtual_target_change_unlatches_without_predicting_new_completion=True,
            new_protocol_qualified=False,globally_qualified=False,formal_TEST_used=False,
            Native_started=False,parameters_fitted=False)

    def initialize_latent(self,actual,batch_size):
        if not isinstance(actual,dict) or type(batch_size) is not int or batch_size<=0:
            raise ValueError('fresh real state dictionary and positive batch size required')
        if actual.get('predicted') is True or _KEY in actual:
            raise ValueError('virtual state cannot initialize a new real completion anchor')
        h=actual.get('actuator_state')
        if (not isinstance(h,dict) or h.get('predicted') is True or _KEY in h or 'latent' in h
                or str(h.get('source','')).startswith('virtual')):
            raise ValueError('fresh actual actuator snapshot required; no old virtual h')
        now=actual.get('time')
        if isinstance(now,(bool,np.bool_)) or not np.isscalar(now) or not np.isfinite(now) or now<0:
            raise ValueError('finite real anchor time required')
        if 'gripper_done' not in h or 'gripper_target' not in h:
            raise ValueError('explicit actual gripper_done and target required')
        done=h['gripper_done'];target=h['gripper_target']
        if done is not None and not isinstance(done,(bool,np.bool_)):
            raise ValueError('actual gripper_done must be boolean or explicitly unknown None')
        if isinstance(target,(bool,np.bool_)) or not np.isscalar(target) or not np.isfinite(target) or target not in (0.,1.):
            raise ValueError('actual known binary gripper target required')
        clock=h.get('reference_clock')
        if done is not None or h.get('reference_valid'):
            if clock is None or not np.isfinite(clock) or abs(float(clock)-float(now))>1e-8:
                raise ValueError('actuator snapshot clock must match the real anchor')
        elif clock is not None and (not np.isfinite(clock) or abs(float(clock)-float(now))>1e-8):
            raise ValueError('uninitialized actuator clock is stale')
        fingers=_array(actual.get('finger_open_amounts'),(2,),'actual two fingers')
        if np.any((fingers<0)|(fingers>1)):raise ValueError('actual fingers outside[0,1]')
        now=float(now)
        if self._real_anchor is not None and now<self._real_anchor-1e-8:
            raise ValueError('real completion anchor cannot move backwards; use a new response per episode')
        if self._real_anchor is None or abs(now-self._real_anchor)>1e-8:
            self._tokens.clear();self._real_anchor=now
        token=uuid4().hex;self._tokens.add(token)
        base=None if self.base_latent_initializer is None else self.base_latent_initializer(deepcopy(actual),batch_size)
        return dict(base_latent=deepcopy(base),**{_KEY:dict(schema=SCHEMA,binding=token,feedback_time=now,
            virtual_time=np.full(batch_size,now),done=np.full(batch_size,done is not None and bool(done)),
            done_known=np.full(batch_size,done is not None),target=np.full(batch_size,float(target)),
            source='fresh_actual_gripper_done_and_target_not_virtual_feedback',predicted=False)})

    def predict_execution(self,x,references,latent,dt,*,held_anchor):
        if dt!=.05:raise ValueError('one50ms rich transition required')
        if not isinstance(latent,dict) or not isinstance(latent.get(_KEY),dict):
            raise ValueError('initialize_latent must read fresh actual completion status first')
        memory=latent[_KEY];n=len(x['q'])
        anchor=np.asarray(held_anchor)
        if anchor.shape!=(n,) or anchor.dtype!=np.bool_:
            raise ValueError('base-model held feature must remain explicit bool[N]; it is not a completion condition')
        if memory.get('schema')!=SCHEMA or memory.get('binding') not in self._tokens:
            raise ValueError('stale/foreign completion latent cannot replace a real-plan initializer')
        if abs(float(memory.get('feedback_time',np.inf))-self._real_anchor)>1e-8:
            raise ValueError('completion latent belongs to a different real anchor')
        done=np.asarray(memory.get('done'));known=np.asarray(memory.get('done_known'))
        if done.shape!=(n,) or known.shape!=(n,) or done.dtype!=np.bool_ or known.dtype!=np.bool_:
            raise ValueError('aligned explicit boolean completion memory required')
        previous=_array(memory.get('target'),(n,),'previous virtual target')
        target=_array(references.get('gripper_target'),(n,),'known virtual gripper target')
        if not np.isin(previous,(0.,1.)).all() or not np.isin(target,(0.,1.)).all():
            raise ValueError('binary actuator targets required; no silent rounding')
        times=_array(x.get('time'),(n,),'actual/predicted state time')
        expected=_array(memory.get('virtual_time'),(n,),'completion virtual time')
        if not np.allclose(times,expected,atol=1e-8,rtol=0) or np.any(times<self._real_anchor-1e-8):
            raise ValueError('completion and predicted state clocks differ')
        fingers=_array(x.get('finger_open_amounts'),(n,2),'current two fingers')
        if np.any((fingers<0)|(fingers>1)):raise ValueError('current fingers outside[0,1]')
        changed=target!=previous;hold=known&done&(~changed)
        # Copies prevent an injected base model from mutating the caller or
        # changing the current fingers used by the explicit hold approximation.
        predicted=self.base_response.predict_execution(deepcopy(x),deepcopy(references),
            deepcopy(latent.get('base_latent')),dt,held_anchor=anchor.copy())
        base_fingers=_array(predicted.get('finger_open_amounts'),(n,2),'base predicted fingers')
        if np.any((base_fingers<0)|(base_fingers>1)):raise ValueError('base predicted fingers outside[0,1]')
        for key in ('q','dq'):
            value=np.asarray(predicted.get(key))
            if value.shape!=np.shape(x[key]) or not np.isfinite(value).all():raise ValueError('invalid base '+key)
        feasible=np.asarray(predicted.get('feasible'))
        if feasible.shape!=(n,) or feasible.dtype!=np.bool_:raise ValueError('base feasible[N] bool required')
        next_fingers=base_fingers.copy();next_fingers[hold]=fingers[hold]
        next_memory=deepcopy(memory)
        next_memory.update(done=done&(~changed),done_known=known.copy(),target=target.copy(),
            virtual_time=times+dt,predicted=True,source='virtual_completion_memory_not_next_real_feedback')
        next_latent=dict(base_latent=deepcopy(predicted.get('latent',latent.get('base_latent'))),**{_KEY:next_memory})
        result={key:value for key,value in predicted.items() if key not in ('held','attached','contact','task_success')}
        result.update(finger_open_amounts=next_fingers,latent=next_latent,
            completion_hold_mask=hold.copy(),completion_target_changed=changed.copy(),
            completion_response_schema=SCHEMA,base_q_dq_unchanged=True,
            zero_motor_finger_approximation=True,completion_implies_contact=False,
            new_protocol_qualified=False,globally_qualified=False)
        return result
