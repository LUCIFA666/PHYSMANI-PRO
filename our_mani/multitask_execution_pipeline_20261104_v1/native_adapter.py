"""Metric-only native outcome adapter; no geometric replacement success rules."""
from dataclasses import dataclass, asdict
import hashlib
import inspect
from pathlib import Path
from .registry import TASKS


@dataclass(frozen=True)
class NativeOutcome:
    success: bool
    terminate: bool
    terminal_failure: bool
    physical_timestamp: float
    metric_only: bool = True
    policy_use_allowed: bool = False


class NativeOutcomeAdapter:
    """Bind an actual RLBench task after launch; never launch/reset/step it here.

    The parent's physical ledger supplies timestamps. First terminal wins; final
    instantaneous success is retained separately from the ever-success latch.
    """
    def __init__(self, task, contract):
        if contract.get('task') not in TASKS:
            raise ValueError('Exact registered task contract required')
        cls = type(task)
        if cls.__module__ != contract['native_module'] or cls.__name__ != contract['native_class']:
            raise TypeError('Actual native task type differs from its frozen contract')
        source = Path(inspect.getfile(cls)).resolve()
        expected = contract['native_source']
        if source != Path(expected['path']).resolve() or hashlib.sha256(source.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('Native task source differs from the registry')
        for name, checksum in contract['native_condition_sources'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Native task/condition implementation changed')
        self._task, self._success_function = task, task.success.__func__
        self._terminal = self._current = None
        self._ever_success = False

    def observe_after_physics(self, physical_timestamp):
        import math
        timestamp = float(physical_timestamp)
        if (not math.isfinite(timestamp) or timestamp < 0
                or self._current is not None and timestamp <= self._current.physical_timestamp):
            raise ValueError('A unique increasing actual physics-ledger timestamp is required')
        if self._task.success.__func__ is not self._success_function:
            raise TypeError('Bound native success implementation changed')
        value = self._task.success()
        # np.bool_ is accepted without importing NumPy during default preparation.
        is_bool = lambda x: type(x) is bool or (type(x).__module__ == 'numpy' and type(x).__name__ == 'bool_')
        if not isinstance(value, (tuple, list)) or len(value) != 2 or not all(map(is_bool, value)):
            raise TypeError('Native task.success() must return two real booleans')
        success, terminate = map(bool, value)
        self._current = NativeOutcome(success, terminate, terminate and not success, timestamp)
        self._ever_success |= success
        if self._terminal is None and terminate:
            self._terminal = self._current
        return self._current

    def metric_receipt(self):
        return dict(native_success_source='task.success()',
            native_current=None if self._current is None else asdict(self._current),
            native_first_terminal=None if self._terminal is None else asdict(self._terminal),
            native_ever_success=self._ever_success, metric_only=True, policy_use_allowed=False)
