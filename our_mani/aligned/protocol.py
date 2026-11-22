"""One counted physics step per task step, including gripper motion."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass
class TickLedger:
    dt: float = .05
    budget: float = 10.
    ticks: int = 0
    phases: dict = field(default_factory=dict)
    grasp_requests: int = 0
    release_requests: int = 0

    @property
    def time(self):
        return self.ticks*self.dt

    @property
    def remaining(self):
        return max(0., self.budget-self.time)

    def before(self):
        if self.remaining < self.dt-1e-8:
            raise TimeoutError('aligned physics budget exhausted')

    def after(self, physics_delta, task_delta, phase, physics_tolerance=2e-5):
        if abs(physics_delta-self.dt) > physics_tolerance:
            raise RuntimeError(f'unaccounted physics step: {physics_delta}')
        if abs(task_delta-self.dt) > 2e-5:
            raise RuntimeError(f'task/physics clocks diverged: {task_delta}')
        self.ticks += 1
        self.phases[phase] = self.phases.get(phase, 0)+1

    def receipt(self):
        return dict(ticks=self.ticks, simulation_seconds=self.time,
                    budget_seconds=self.budget, dt=self.dt, phases=dict(self.phases),
                    grasp_requests=self.grasp_requests, release_requests=self.release_requests)


def controller_hash(root, config, predictor):
    """Bind Value probabilities to actuator, physics and predictor, not r04 hashes."""
    root = Path(root)
    names = ['our_mani/local_capture.py', 'our_mani/control.py',
             'our_mani/benchmark_perception.py', 'our_mani/goal_perception.py',
             'our_mani/aligned/protocol.py', 'our_mani/aligned/simulation.py']
    if predictor == 'context':
        names += ['our_mani/context_control.py', 'our_mani/context_dynamics.py', config['dynamics_checkpoint']]
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode()); digest.update((root/name).read_bytes())
    keys = ('dt', 'simulation_seconds', 'gripper_velocity', 'gripper_semantics',
            'local_servo_lead', 'local_max_speed', 'local_max_acceleration', 'map_access',
            'current_instance_masks','information_setting')
    digest.update(json.dumps({key:config[key] for key in keys}, sort_keys=True).encode())
    return digest.hexdigest()
