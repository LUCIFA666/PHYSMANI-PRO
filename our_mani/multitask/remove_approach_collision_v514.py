"""Read-only public-ghost collision partners around the stalled TRAIN approach."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np
from .remove_terminal_executor_v470 import RemoveTerminalBackend, verify as previous_verify
from .all_phase_collision_diagnostic_v17 import CollisionObserver
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset

SOURCES = ('our_mani/multitask/remove_approach_collision_v514.py',
           'our_mani/multitask/all_phase_collision_diagnostic_v17.py')


def binding(root, config, task):
    api = Path(config['baseline_repo'])/'third_party/PyRep/pyrep/robots/arms/arm.py'
    return dict(task=task, sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        native_arm_static_API=dict(path=str(api),sha256=file_sha(api)),
        window_seconds=[1.,1.6], maximum_records=64,
        extra_read_only_public_ghost_partner_queries=True,
        policy_commands_and_collision_results_unchanged=True,
        extra_physics_or_camera_queries=False, native_dynamic_truth_read=False,
        formal_VAL_TEST=False)


def verify(root, prepared):
    previous_verify(root, prepared)
    expected = binding(root, prepared['config'], prepared['args'].task)
    if prepared['config'].get('remove_approach_collision_v514') != expected:
        raise ValueError('freeze read-only remove collision diagnostic')
    files = {str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()}
    files[expected['native_arm_static_API']['path']] = expected['native_arm_static_API']['sha256']
    checked_subset(root, prepared['files'], files)
    return expected


class Observer(CollisionObserver):
    def _instrument(self):
        original_collision = self.actor.robot_model.collision
        original_arm = self.env.arm.check_arm_collision
        service = self.service
        ghosts = list(service.static_ghosts)
        for body in service.estimated_bodies.values():
            ghosts.extend(body.ghosts())
        seen = set()
        def sample(instance, q, offset):
            previous = self.sample
            self.sample = dict(future_offset=float(offset),sampled_joints=np.asarray(q).tolist())
            try:
                return original_collision(q, offset)
            finally:
                self.sample = previous
        self.actor.robot_model.collision = MethodType(sample, self.actor.robot_model)
        def arm(*args, **kwargs):
            result = original_arm(*args, **kwargs)
            port = self.actor.collision_port
            if (not result or self.sample is None or port.frame is None or port.bound is None
                    or port.bound['phase']!='approach' or len(self.events)>=64):
                return result
            now = float(port.frame[0].robot.timestamp)
            if not 1.-1e-8<=now<=1.6+1e-8:
                return result
            key = (now,self.sample['future_offset'],tuple(self.sample['sampled_joints']))
            if key in seen:
                return result
            seen.add(key)
            clock = service._clock()
            record = dict(time=now,sample=deepcopy(self.sample),binding=deepcopy(port.bound),
                          attachments=sorted(service.estimated_attachments),public_ghost_partners=[])
            for ghost in ghosts:
                if ghost.is_collidable():
                    self.extra_pair_queries += 1
                    if original_arm(ghost):
                        record['public_ghost_partners'].append(self.geometry(ghost))
            service._clock_check(clock,'read-only remove collision partner diagnostic')
            self.events.append(record)
            return result
        self.env.arm.check_arm_collision = arm

    def receipt(self):
        return dict(super().receipt(),schema='r10_remove_approach_public_collision_window_v514',
                    window_seconds=[1.,1.6],maximum_rejection_records=64)


class RemoveApproachDiagnosticBackend(RemoveTerminalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        observer = Observer(actor, env)
        previous = actor.metrics_report
        def metrics(instance):
            return dict(previous(),remove_approach_collision_v514=observer.receipt())
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
