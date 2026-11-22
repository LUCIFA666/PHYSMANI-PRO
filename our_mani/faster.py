"""FASTER streaming integration, absolute control ticks, and stale-goal rejection.

This file does not substitute a scripted controller for an untrained VLA.
Inference requires explicit provenance from a HAS-trained, goal-conditioned model.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time

import numpy as np


CONDITIONING_VERSION = "our_mani_region_prompt_v1"
ACTION_SPACE = "world_ee_xyz_xyzw_gripper_open"


@dataclass(frozen=True)
class StreamRequest:
    epoch: int
    goal_id: int
    start_tick: int
    observation_time: float
    deadline: float
    prefix_length: int


class ActionBuffer:
    """One inference connection, concurrent controller consumption, atomic invalidation.

    Final responses can contain the same actions as prior partial messages. They
    fill unexecuted slots only, never append a duplicate chunk to the queue.
    """
    def __init__(self, *, action_dim=8, max_observation_age=0.3):
        self.action_dim, self.max_observation_age = action_dim, max_observation_age
        self._lock = threading.RLock()
        self.epoch, self.goal_id, self.last_tick = 0, None, -1
        self._actions = {}
        self.rejected_actions = 0

    def activate(self, goal_id):
        with self._lock:
            self.epoch += 1
            self.goal_id = goal_id
            self._actions.clear()

    def invalidate(self):
        self.activate(None)

    def request(self, goal, *, start_tick, prefix_length=0, observation_time=None):
        with self._lock:
            if goal.goal_id != self.goal_id:
                raise ValueError("activate goal before requesting FASTER actions")
            if start_tick <= self.last_tick:
                raise ValueError("inference cannot start at an already executed control tick")
            return StreamRequest(self.epoch, goal.goal_id, start_tick,
                                 goal.observation_time if observation_time is None else observation_time,
                                 goal.arrival_time+0.2, prefix_length)

    def accept(self, request, actions, *, first_index, now):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim or not np.isfinite(actions).all():
            raise ValueError("invalid FASTER action chunk")
        with self._lock:
            if request.epoch != self.epoch or request.goal_id != self.goal_id or now > request.deadline or now-request.observation_time > self.max_observation_age:
                self.rejected_actions += len(actions)
                return 0
            accepted = 0
            for index, action in enumerate(actions, start=first_index):
                tick = request.start_tick+index
                if index < request.prefix_length or tick <= self.last_tick:
                    continue
                if np.linalg.norm(action[3:7]) < 1e-6 or not 0 <= action[7] <= 1:
                    raise ValueError("invalid EE quaternion/gripper command")
                # Latest refinement replaces only not-yet-executed slots.
                expiry = min(request.deadline, request.observation_time+self.max_observation_age)
                self._actions[tick] = (action.copy(), expiry)
                accepted += 1
            return accepted

    def pop(self, tick, *, now):
        with self._lock:
            if tick <= self.last_tick:
                raise ValueError("control ticks must increase")
            self.last_tick = tick
            for old in [t for t in self._actions if t < tick]:
                self._actions.pop(old)
            item = self._actions.pop(tick, None)
            if item is None:
                return None
            action, expiry = item
            if now > expiry:
                self.rejected_actions += 1
                return None
            return action

    def prefix(self, start_tick, delay, *, now=None):
        with self._lock:
            slots = [self._actions.get(t) for t in range(start_tick, start_tick+delay)]
            if any(item is None for item in slots):
                raise ValueError("cannot invent missing committed action prefix")
            if now is not None and any(item[1] < now for item in slots):
                raise ValueError("cannot condition on expired committed actions")
            return np.asarray([item[0] for item in slots], dtype=np.float32).reshape(delay, self.action_dim)


def validate_metadata(metadata):
    expected = {"our_mani_conditioning": CONDITIONING_VERSION, "action_space": ACTION_SPACE,
                "has_trained": True}
    missing = {key: value for key, value in expected.items() if metadata.get(key) != value}
    if missing or not metadata.get("checkpoint_sha256"):
        raise ValueError(f"FASTER server must identify compatible trained weights: {missing}")


def goal_condition(goal, now):
    """Fixed schema shared with training. xyzw, m, m/s, rad/s, seconds."""
    return {"pose": goal.pose.xyzw().astype(np.float32),
            "linear_velocity": np.asarray(goal.linear_velocity, dtype=np.float32),
            "angular_velocity": np.asarray(goal.angular_velocity, dtype=np.float32),
            "radii": np.asarray(goal.region.radii, dtype=np.float32),
            "remaining_seconds": np.float32(max(0., goal.arrival_time-now))}


class FasterApproach:
    def __init__(self, client, buffer, *, clock=time.monotonic):
        validate_metadata(client.get_server_metadata())
        self.client, self.buffer, self.clock = client, buffer, clock
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="our-mani-faster")
        self.future = None
        self.timings = []

    def submit(self, observation, goal, *, now, start_tick, delay=0):
        if self.future is not None and not self.future.done():
            return False  # one socket must not interleave requests
        if self.future is not None:
            self.future.result()  # surface previous server failures
        prefix = self.buffer.prefix(start_tick, delay, now=now) if delay else np.empty((0, 8), np.float32)
        observation_time = float(observation["timestamp"])
        if now-observation_time > self.buffer.max_observation_age or observation_time > now+1e-6:
            raise ValueError("FASTER inference requires a fresh causal observation")
        request = self.buffer.request(goal, start_tick=start_tick, prefix_length=delay,
                                      observation_time=observation_time)
        data = dict(observation)
        data["handover_goal"] = goal_condition(goal, now)
        data["action_prefix"], data["delay"] = prefix, np.int32(delay)
        def infer():
            cursor, saw_partial = delay, False
            started = time.perf_counter()
            first_action_ms = None
            def on_partial(actions):
                nonlocal cursor, saw_partial, first_action_ms
                # Official FASTER callback carries only a contiguous array.
                # Prefix actions are already clean and are NOT emitted.
                saw_partial = True
                accepted = self.buffer.accept(request, actions, first_index=cursor, now=self.clock())
                if accepted and first_action_ms is None:
                    first_action_ms = (time.perf_counter()-started)*1000
                cursor += len(actions)
                return accepted
            result = self.client.infer_streaming(data, on_actions_ready=on_partial)
            if "actions" in result:
                # Early-stop servers return a full x_t containing unfinished
                # noisy future actions. Only finalized partial indices are usable.
                final = result["actions"][:cursor] if saw_partial else result["actions"]
                self.buffer.accept(request, final, first_index=0, now=self.clock())
            self.timings.append({"goal_id": goal.goal_id, "observation_time": observation_time,
                                 "first_accepted_action_ms": first_action_ms,
                                 "inference_total_ms": (time.perf_counter()-started)*1000,
                                 "finalized_action_count": cursor-delay})
            return result
        self.future = self.executor.submit(infer)
        return True

    def close(self):
        self.buffer.invalidate()
        self.executor.shutdown(wait=True)
