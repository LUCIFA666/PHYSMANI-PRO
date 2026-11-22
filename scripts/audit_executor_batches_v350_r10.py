"""Read-only audit of the original TRAIN target-memory smoke pair."""
from pathlib import Path
import runpy


def main():
    source = Path(__file__).with_name('audit_executor_batches_v344_r10.py')
    code = source.read_text()
    code = code.replace("BATCHES = {'high_peg_support_v330_setup': 2}",
                        "BATCHES = {'high_peg_target_memory_v342_setup': 2}")
    code = code.replace('executor_fixed_TRAIN_complete_audit_v344',
                        'executor_fixed_TRAIN_complete_audit_v350')
    code = code.replace('r10_completed_fixed_TRAIN_batch_audit_v344',
                        'r10_completed_fixed_TRAIN_batch_audit_v350')
    assert 'high_peg_support_v330_setup' not in code
    exec(compile(code, str(Path(__file__)), 'exec'),
         {'__file__': str(Path(__file__).resolve()), '__name__': '__main__'})


if __name__ == '__main__':
    main()
