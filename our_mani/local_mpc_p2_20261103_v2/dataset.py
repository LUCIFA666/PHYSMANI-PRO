"""P2 real-dispatched-reference datasets; no old advice or simulator imports.

Episodes are assigned by explicit parent identity BEFORE sliding windows exist.
Only closed P1 truth-diagnostic runs are read. Failed runs retain their strictly
aligned executed prefix, including short tails; P0 is retained in the admission
ledger as excluded, never silently used for fitting.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DT, DOF, FINGERS = .05, 7, 2
STATE_DIM, REFERENCE_DIM, FEATURE_DIM = 17, 22, 39
SPLITS = ("train", "validation", "holdout")
SCHEMA = "local_mpc_P2_actual_servo_dataset_20261103_v2"


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _lines(path):
    return [_json_line(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def _json_line(line):
    return json.loads(line)


def _vector(value, length, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"finite {name}[{length}] required")
    return result.copy()


def _state(row):
    if row.get("truth_diagnostic_only") is not True or row.get("official_public_input_evaluation") is not False:
        raise ValueError("explicit truth-only P1 state required")
    if type(row.get("held")) is not bool:
        raise ValueError("actual native held boolean required; gripper command is insufficient")
    opening = _vector(row["finger_open_amounts"], FINGERS, "actual native finger openings")
    if np.any(opening < -1e-5) or np.any(opening > 1+1e-5):
        raise ValueError("native aperture outside declared normalized range")
    return np.r_[_vector(row["q"], DOF, "actual q"), _vector(row["dq"], DOF, "actual dq"),
                 opening, float(row["held"])]


def _tcp(row):
    p, q = _vector(row["tcp_position"], 3, "TCP position"), _vector(row["tcp_quaternion"], 4, "TCP xyzw")
    norm = np.linalg.norm(q)
    if norm < 1e-8 or not np.isclose(norm, 1., atol=1e-4):
        raise ValueError("actual TCP unit quaternion required")
    x,y,z,w = q/norm
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    matrix[:3, 3] = p
    return matrix


def canonical_h(value):
    """Preserve initialized versus valid: a failed goal may retain old a."""
    value = value or {}
    names = dict(goal_q="target_joints", q_ref="reference_positions",
                 dq_ref="reference_velocities", ddq_ref="reference_accelerations")
    result = {}
    for short, long in names.items():
        raw = value.get(short, value.get(long))
        result[short] = np.zeros(DOF) if raw is None else _vector(raw, DOF, short)
    result["valid"] = bool(value.get("valid", value.get("reference_valid", False)))
    result["initialized"] = bool(value.get("initialized", value.get("q_ref", value.get("reference_positions")) is not None))
    result["gripper_target"] = float(value.get("gripper_target", 1.))
    return result


def _tick_id(value):
    index = int(round(float(value)/DT))
    if abs(float(value)-index*DT) > 1e-7:
        raise ValueError("Native clock is not on the declared 50ms grid")
    return index


def read_episode(directory):
    """Validate source fingerprints and join every actually executed tick."""
    directory = Path(directory).resolve(strict=True)
    required = [directory/name for name in ("run_inputs.json", "worker_exit.json", "initial_snapshot.json")]
    if not all(p.is_file() for p in required):
        raise ValueError("only reaped/closed episodes with immutable admission evidence are allowed")
    run = _json(required[0])
    config = run.get("config", {})
    if run.get("TEST_read") is not False:
        raise ValueError("formal TEST guard precedes every state or diagnostic-prefix read")
    if run.get("mode") == "p0":
        return dict(excluded_reason="P0_is_execution_contract_only", transitions=0, files={str(p):file_sha(p) for p in required})
    if (run.get("mode") != "p1" or run.get("TEST_read") is not False
            or config.get("formal_evaluation") is not False
            or config.get("physics_dt") != DT or config.get("control_dt") != .1
            or config.get("decision_ticks") != 2 or not run.get("source_sha256")):
        raise ValueError("closed source-bound P1 truth episodes only; no formal TEST/old advice")
    exited = _json(required[1])
    if exited.get("native_closed") is not True or type(exited.get("exit_code")) is not int:
        raise ValueError("actual closed Native worker status with an explicit exit code required")
    initial = _json(required[2])
    paths = required + [directory/name for name in ("transitions.jsonl", "native_ticks.jsonl", "result.json") if (directory/name).is_file()]
    before_hash = {str(p):file_sha(p) for p in paths}
    controls, native = _lines(directory/"transitions.jsonl"), _lines(directory/"native_ticks.jsonl")
    native = [row.get("state", row) for row in native]
    truth = {int(row["counted_tick"]): row for row in native}
    if len(truth) != len(native) or sorted(truth) != list(range(1, len(native)+1)):
        raise ValueError("Native tick ledger must be unique and contiguous; no gap fabrication")
    for tick, row in truth.items():
        if _tick_id(row["time"]) != tick:
            raise ValueError("Native tick/time disagreement")
    state = initial["state"]
    if _tick_id(state["time"]) != 0:
        raise ValueError("initial state must be at post-staging tick zero")
    states, tcp = [_state(state)], [_tcp(state)]
    for i in range(1, len(truth)+1):
        states.append(_state(truth[i])); tcp.append(_tcp(truth[i]))
    # New collections expose an independent native servo readback. It audits
    # the setter-dispatched action and never substitutes for actual q/dq.
    readback_rows = [state] + [truth[i] for i in range(1, len(truth)+1)]
    readback_available = ['servo_target_positions' in row for row in readback_rows]
    if any(readback_available) and not all(readback_available):
        raise ValueError('servo readback availability changed inside one episode')
    servo_readbacks = (np.stack([_vector(row['servo_target_positions'], DOF, 'native servo target readback')
                                for row in readback_rows]) if all(readback_available) else None)
    maximum_readback_error = 0.
    applied, goals, openings, memories, cycle_ids, consumed = [], [], [], [], [], []
    expected_before = 0
    for cycle, row in enumerate(controls):
        receipt = row["executor"]
        h = canonical_h(row.get("actuator_state_before", row.get("before",{}).get("actuator_state")))
        if _tick_id(row["before"]["time"]) != expected_before:
            raise ValueError("100ms transition prefix has a missing/duplicate action")
        goal = _vector(receipt["target_joints"], DOF, "actual post-IK native joint goal")
        completed = [part for part in receipt["ticks"] if part.get("tick_completed") is True]
        if len(completed) != receipt.get("executed_ticks") or len(completed) not in (1,2):
            raise ValueError("executed prefix must contain one/two actual ticks, never an unexecuted label")
        for part in completed:
            first, last = _tick_id(part["clock_before"]), _tick_id(part["clock_after"])
            if first != expected_before or last != first+1 or last not in truth:
                raise ValueError("executor/Native clock join failed")
            for key, field, target in (("measured_positions_before", "q", states[first][:DOF]),
                                       ("measured_velocities_before", "dq", states[first][DOF:2*DOF]),
                                       ("measured_positions_after", "q", states[last][:DOF]),
                                       ("measured_velocities_after", "dq", states[last][DOF:2*DOF])):
                if not np.allclose(_vector(part[key],DOF,key), target, atol=2e-6, rtol=0):
                    raise ValueError("executor measured state disagrees with joined Native truth: "+field)
            qref = _vector(part["reference_positions"], DOF, "dispatched qref")
            if servo_readbacks is not None:
                error = float(np.max(np.abs(qref-servo_readbacks[last])))
                maximum_readback_error = max(maximum_readback_error, error)
                if error > 2e-6:
                    raise ValueError('dispatched reference disagrees with independent native servo readback')
            vref = _vector(part["reference_velocities"], DOF, "known reference velocity")
            aref = _vector(part["reference_accelerations"], DOF, "known reference acceleration")
            grip = float(part["gripper_target"])
            if grip not in (0.,1.):
                raise ValueError("explicit binary dispatched gripper command required")
            memories.append(deepcopy(h)); applied.append(np.r_[qref,vref,aref,grip])
            goals.append(goal.copy()); openings.append(grip); cycle_ids.append(cycle); consumed.append(last)
            h = dict(q_ref=qref,dq_ref=vref,ddq_ref=aref,goal_q=goal.copy(),valid=True,initialized=True,gripper_target=grip)
            expected_before = last
        if _tick_id(row["after"]["time"]) != expected_before:
            raise ValueError("control after-state time differs from its actual last tick")
    # A worker may crash after a physical tick but before its control row was
    # appended. Preserve the episode and expose those unusable unmatched ticks.
    # They have no dispatched-reference evidence and never become labels.
    unmatched = sorted(set(truth)-set(consumed))
    if unmatched and unmatched != list(range(expected_before+1,len(truth)+1)):
        raise ValueError("unmatched Native ticks are not a documented trailing failure prefix")
    if unmatched and exited.get("exit_code") == 0:
        raise ValueError("successful worker omitted dispatched-reference evidence")
    for p, digest in before_hash.items():
        if file_sha(p) != digest:
            raise RuntimeError("closed episode changed during data ingestion")
    count = len(applied)
    family=None
    if all(key in config for key in ('seed','initial_target_position','calibration_train_episode')):
        reset={key:config.get(key) for key in ('seed','task','initial_target_position','calibration_train_episode',
                                              'native_task_scene_sha256')}
        reset['scene']=run.get('scene')
        if run.get('scene')!='static':reset['uniform_velocity']=config.get('uniform_velocity')
        if run.get('scene')=='speed_change':
            reset.update(speed_change_at_tick=config.get('speed_change_at_tick'),changed_velocity=config.get('changed_velocity'))
        family=hashlib.sha256(json.dumps(reset,sort_keys=True).encode()).hexdigest()
    seed_scene_family=None
    if 'seed' in config:
        # Height/geometry variants of one seeded scenario are one parent,
        # even when initial z or corrected geometry differs.
        seed_family=dict(seed=config['seed'],scene=run.get('scene'),task=config.get('task'),
                         calibration_train_episode=config.get('calibration_train_episode'))
        seed_scene_family=hashlib.sha256(json.dumps(seed_family,sort_keys=True).encode()).hexdigest()
    raw_limits=initial.get('joint_limits')
    reference_limits=None
    if raw_limits is not None:
        acceleration=config.get('joint_reference_acceleration',raw_limits['controller_reference_acceleration'])
        jerk=config.get('joint_reference_jerk',raw_limits['controller_reference_jerk'])
        if (acceleration!=raw_limits['controller_reference_acceleration']
                or jerk!=raw_limits['controller_reference_jerk']):
            raise ValueError('native snapshot and actual executor reference configuration disagree')
        reference_limits=dict(lower=_vector(raw_limits['lower'],DOF,'native lower').tolist(),
                              upper=_vector(raw_limits['upper'],DOF,'native upper').tolist(),
                              velocity=_vector(raw_limits['velocity'],DOF,'native velocity').tolist(),
                              acceleration=float(acceleration),jerk=float(jerk),reference_velocity_cap=3.)
    return dict(states=np.asarray(states[:count+1],np.float64),tcp=np.asarray(tcp[:count+1],np.float64),
                references=np.asarray(applied,np.float64).reshape(count,REFERENCE_DIM),
                command_goal_q=np.asarray(goals,np.float64).reshape(count,DOF),
                command_opening=np.asarray(openings,np.float64),h=memories,
                cycle_id=np.asarray(cycle_ids,np.int64),transitions=count,
                native_ticks=len(native),unmatched_trailing_ticks=unmatched,
                failed=exited.get("exit_code")!=0,files=before_hash,
                source_sha256=run["source_sha256"],scene=run.get("scene"),config=config,reset_family_fingerprint=family,
                seed_scene_family_fingerprint=seed_scene_family,reference_limits=reference_limits,
                servo_target_readback_available=servo_readbacks is not None,
                servo_target_readback_max_error_rad=maximum_readback_error if servo_readbacks is not None else None,
                servo_target_readbacks=None if servo_readbacks is None else servo_readbacks[:count+1],
                observed_but_not_model_features=[key for key in ('finger_positions','finger_velocities','gripper_actuator_history')
                                               if key in state],
                dispatched_reference_semantics="setter_dispatched_reference_independently_audited_when_native_readback_exists_never_actual_q",
                excluded_reason=None)


def build_episode_manifest(episode_specs, split_by_parent):
    """Explicit user-owned parent assignment, no random window-level split."""
    if any(split not in SPLITS for split in split_by_parent.values()):
        raise ValueError("only train/validation/holdout diagnostic splits allowed; formal TEST is never loaded")
    records, seen_ids, seen_paths, families, seed_families = [], set(), set(), {}, {}
    for spec in episode_specs:
        episode_id, parent_id = spec["episode_id"], spec["parent_id"]
        if not all(isinstance(x,str) and x for x in (episode_id,parent_id)) or parent_id not in split_by_parent:
            raise ValueError("explicit nonempty episode/parent identities and prior split assignment required")
        path = str(Path(spec["path"]).resolve(strict=True))
        if episode_id in seen_ids or path in seen_paths:
            raise ValueError("duplicate episode identity or replayed same output path")
        seen_ids.add(episode_id);seen_paths.add(path)
        data = read_episode(path)
        family=data.get('reset_family_fingerprint')
        if family is not None:
            if family in families and families[family]!=parent_id:
                raise ValueError('same seeded reset/scenario restart family assigned different parent identities')
            families[family]=parent_id
        seed_family=data.get('seed_scene_family_fingerprint')
        if seed_family is not None:
            if seed_family in seed_families and seed_families[seed_family]!=parent_id:
                raise ValueError('same seed/scenario across height variants assigned different parent identities')
            seed_families[seed_family]=parent_id
        records.append(dict(episode_id=episode_id,parent_id=parent_id,path=path,
                            split=split_by_parent[parent_id],transitions=data["transitions"],
                            excluded_reason=data["excluded_reason"],files=data["files"],
                            reset_family_fingerprint=family,seed_scene_family_fingerprint=seed_family,
                            failed=data.get("failed",False),unmatched_trailing_ticks=data.get("unmatched_trailing_ticks",[])))
    return dict(schema=SCHEMA,episode_records=records,split_by_parent=dict(split_by_parent),
                split_before_windows=True,DT=DT,formal_TEST_read=False,old_expert_advice_read=False,
                P0_used_for_fitting=False,failed_prefixes_retained=True,
                same_seed_scene_height_variants_share_parent=True)


class ServoWindowDataset(Dataset):
    def __init__(self, manifest, split, *, horizon_ticks=16):
        if manifest.get("schema") != SCHEMA or split not in SPLITS or manifest.get("formal_TEST_read") is not False:
            raise ValueError("explicit P2 diagnostic manifest/split required")
        if type(horizon_ticks) is not int or not 1<=horizon_ticks<=16:
            raise ValueError("horizon must be 1..16 Native ticks (50..800ms)")
        self.manifest,self.split,self.horizon_ticks=manifest,split,horizon_ticks
        all_parents={}
        for row in manifest["episode_records"]:
            if manifest["split_by_parent"].get(row["parent_id"]) != row["split"]:
                raise ValueError("episode assignment differs from the prior parent split ledger")
            if row["parent_id"] in all_parents and all_parents[row["parent_id"]]!=row["split"]:
                raise ValueError("parent leaked across episode splits")
            all_parents[row["parent_id"]]=row["split"]
        self.episodes,self.rows,self.anchors=[],[],[]
        for row in manifest["episode_records"]:
            if row["split"]!=split or row["excluded_reason"] is not None:continue
            data=read_episode(row["path"])
            if data["files"]!=row["files"] or data["transitions"]!=row["transitions"]:
                raise ValueError("source-bound episode changed since split/manifest freeze")
            e=len(self.episodes);self.episodes.append(data);self.rows.append(row)
            self.anchors.extend((e,i) for i in range(data["transitions"]))
        parent_episodes=Counter(row["parent_id"] for row in self.rows if row["transitions"])
        self.weights=np.asarray([1/(parent_episodes[self.rows[e]["parent_id"]]*self.episodes[e]["transitions"])
                                 for e,_ in self.anchors],np.float64)
        self.summary=dict(split=split,episodes=len(self.rows),parents=len(parent_episodes),windows=len(self.anchors),
                          failed_episodes=sum(x["failed"] for x in self.episodes),
                          zero_transition_episodes=sum(not x["transitions"] for x in self.episodes),
                          unmatched_trailing_ticks=sum(len(x["unmatched_trailing_ticks"]) for x in self.episodes),
                          independent_servo_readback_episodes=sum(x['servo_target_readback_available'] for x in self.episodes),
                          maximum_servo_readback_error_rad=max((x['servo_target_readback_max_error_rad'] or 0.
                                                               for x in self.episodes),default=0.),
                          reference_regeneration_training_supported=all(x["reference_limits"] is not None for x in self.episodes),
                          actual_state_schema='q7_dq7_measured_finger_openings2_native_held1',
                          finger_positions_velocities_and_gripper_cache_used_as_model_features=False,
                          horizon_ticks=horizon_ticks,formal_TEST_read=False,source_filter_by_success=False,
                          rollout_scope="recorded_reference_auxiliary_plus_regenerated_reference_known_command_free_actual_state")

    def __len__(self):return len(self.anchors)

    def __getitem__(self,index):
        e,start=self.anchors[index];data,row=self.episodes[e],self.rows[e]
        length=min(self.horizon_ticks,data["transitions"]-start)
        def padded(array):
            out=np.zeros((self.horizon_ticks,*array.shape[1:]),dtype=array.dtype)
            out[:length]=array[start:start+length]
            return out
        state=data["states"]
        tcp=np.broadcast_to(np.eye(4),(self.horizon_ticks,4,4)).copy()
        tcp[:length]=data["tcp"][start+1:start+1+length]
        h=data["h"][start]
        return dict(initial_state=state[start].astype(np.float32),
                    initial_actuator_state={k:np.asarray(v,dtype=np.bool_ if k in ("valid","initialized") else np.float32) for k,v in h.items()},
                    teacher_states=padded(state[:-1]).astype(np.float32),
                    references=padded(data["references"]).astype(np.float32),
                    command_goal_q=padded(data["command_goal_q"]).astype(np.float32),
                    command_opening=padded(data["command_opening"]).astype(np.float32),
                    target_states=padded(state[1:]).astype(np.float32),target_tcp=tcp.astype(np.float32),
                    valid=np.arange(self.horizon_ticks)<length,weight=np.float32(self.weights[index]),
                    episode_id=row["episode_id"],parent_id=row["parent_id"],start_tick=start,split=self.split)


def features_numpy(state,reference):
    return np.concatenate((state[...,:DOF],state[...,DOF:2*DOF],state[...,2*DOF:],
                           reference[...,:DOF]-state[...,:DOF],reference[...,DOF:]),axis=-1)


def train_statistics(dataset):
    if dataset.split!="train" or not len(dataset):
        raise ValueError("nonempty TRAIN-only dataset required to fit input scales and residual bounds")
    x, rq,rv=[],[],[]
    for episode in dataset.episodes:
        s=episode["states"]
        if episode["transitions"]:
            x.append(features_numpy(s[:-1],episode["references"]))
            rq.append(s[1:,:DOF]-episode["references"][:,:DOF])
            rv.append(s[1:,DOF:2*DOF]-episode["references"][:,DOF:2*DOF])
    x=np.concatenate(x)
    return dict(feature_mean=x.mean(0).tolist(),feature_std=np.maximum(x.std(0),.01).tolist(),
                residual_q_bound=np.maximum(np.abs(np.concatenate(rq)).max(0)*1.25,.001).tolist(),
                residual_dq_bound=np.maximum(np.abs(np.concatenate(rv)).max(0)*1.25,.05).tolist(),
                source_split="train",parent_ids=sorted({r["parent_id"] for r in dataset.rows}),
                episode_ids=sorted(r['episode_id'] for r in dataset.rows),
                source_transition_count=sum(e['transitions'] for e in dataset.episodes),
                feature_inputs='current_actual_state_and_current_dispatched_reference_only',
                held_next_is_not_in_features=True,labels_passed_as_model_inputs=False,
                residual_bounds_fit_from_TRAIN_next_state_labels=True,
                residual_base="dispatched_reference_q_and_dq",zero_initial_joint_residual=True,
                validation_or_holdout_used_for_scaling=False)


def common_reference_limits(dataset):
    """Require actual native/executor bounds, never limits estimated from labels."""
    from .reference import ReferenceLimits
    values=[episode['reference_limits'] for episode in dataset.episodes]
    if not values or any(value is None for value in values):
        raise ValueError('every episode needs actual native joint limits and controller reference configuration')
    if any(value!=values[0] for value in values[1:]):
        raise ValueError('mixed executor reference limits need explicitly separate batches')
    return ReferenceLimits(**values[0]),deepcopy(values[0])
