"""Read a new compact case or verify and load its complete original trace."""
from pathlib import Path
import gzip
import json
from .evaluation import file_sha
from .lossless_case_archive import uncompressed_sha


def read_case(path,*,full=False):
    path=Path(path).resolve();case=json.loads(path.read_text());pointer=case.get('full_trace_archive')
    if not full or pointer is None:return case
    archive=Path(pointer['path']).resolve()
    if archive.parent!=path.parent or pointer.get('schema')!='r10_lossless_original_evaluator_trace_gzip_v1':
        raise ValueError('archive must be the same case original trace')
    if file_sha(archive)!=pointer['sha256'] or uncompressed_sha(archive)!=pointer['uncompressed_sha256']:
        raise ValueError('compressed or original trace digest changed')
    with gzip.open(archive,'rt',encoding='utf-8') as stream:original=json.load(stream)
    for key in ('identity','official_task_success','native_terminal_failure','counted_clock','cleanup_completed'):
        if original.get(key)!=case.get(key):raise ValueError('summary changed a native outcome, identity or clock')
    return original
