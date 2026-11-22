"""Bounded CPU-only revalidation after the original canonical-index race.

Only the original two exact topology-instability errors may be revalidated.
All original canonical matching, SHA, model and source gates run again. This
helper must never wrap a native action, reset, GPU call or episode runner.
"""
import time
from .protocol import MultitaskProtocolError

TRANSIENT_MESSAGES=frozenset((
    'canonical input topology remained unstable across three index checks',
    'canonical batch topology remained unstable across three index validations',
))


def revalidate_cpu_preflight(callback,*,maximum_attempts=6,timeout_seconds=45.):
    if not 1<=maximum_attempts<=6 or not 0<timeout_seconds<=45.:raise ValueError('bounded CPU-only revalidation required')
    began=time.monotonic();events=[]
    for attempt in range(1,maximum_attempts+1):
        try:return callback(),events
        except MultitaskProtocolError as error:
            elapsed=time.monotonic()-began
            if str(error) not in TRANSIENT_MESSAGES or attempt==maximum_attempts or elapsed>=timeout_seconds:raise
            events.append(dict(attempt=attempt,reason=str(error),elapsed_seconds=elapsed,
                original_all_input_model_gates_will_be_revalidated=True,native_or_GPU_or_rollout_retried=False))
    raise AssertionError('unreachable CPU revalidation state')
